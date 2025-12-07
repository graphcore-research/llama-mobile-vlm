#include "benchmarking.hpp"

#include <arm_neon.h>
#include <cassert>
#include <random>
#include <thread>

// -------------------------------------------------------------------------------------------------
// Basic types

using Symbol = uint8_t;
using Histogram = std::vector<uint64_t>;

constexpr size_t NSymbols = 1 << (8 * sizeof(Symbol));

struct Transcoder {
    virtual std::vector<char> encode(const std::vector<Symbol>& data) const = 0;
    virtual uint32_t decode_sum(const char* data) const = 0;

    virtual ~Transcoder() = default;
};

struct Format {
    std::string name;
    std::function<std::unique_ptr<Transcoder>(const Histogram&)> factory;
};

// -------------------------------------------------------------------------------------------------
// Data generation

std::vector<double> generate_student_t(size_t n, double df, uint64_t seed) {
    std::default_random_engine rng(seed);
    std::student_t_distribution<double> dist(df);
    std::vector<double> data;
    data.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        data.push_back(dist(rng));
    }
    return data;
}

std::vector<Symbol> quantise(const std::vector<double>& data, double delta) {
    std::vector<Symbol> symbols;
    symbols.reserve(data.size());
    for (const auto& v : data) {
        int q = std::clamp(static_cast<int>(std::round(v / delta)) + 128, 0, 255);
        symbols.push_back(static_cast<Symbol>(q));
    }
    return symbols;
}

Histogram count_histogram(const std::vector<Symbol>& symbols) {
    Histogram hist(NSymbols, 0);
    for (const auto& s : symbols) {
        hist[s]++;
    }
    return hist;
}

double calculate_entropy(const Histogram& hist) {
    uint64_t total = std::accumulate(hist.begin(), hist.end(), 0ull);
    double entropy = 0.0;
    for (const auto& freq : hist) {
        if (freq > 0) {
            double p = static_cast<double>(freq) / static_cast<double>(total);
            entropy -= p * std::log2(p);
        }
    }
    return entropy;
}

struct DataSettings {
    double dof;
    double delta;
};

// -------------------------------------------------------------------------------------------------
// Implementation helpers

struct Code {
    uint64_t value;
    uint8_t length;
};

struct SymbolWithLength {
    Symbol symbol;
    uint8_t length;
};

struct Node {
    uint64_t freq;
    std::optional<Symbol> symbol;
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;

    Node() : freq(0), symbol(std::nullopt), left(nullptr), right(nullptr) {}

    Node(Symbol s, uint64_t f)
        : freq(f), symbol(std::make_optional(s)), left(nullptr), right(nullptr) {}

    Node(std::shared_ptr<Node> l, std::shared_ptr<Node> r)
        : freq(l->freq + r->freq), symbol(std::nullopt), left(l), right(r) {}
};

bool operator<(const SymbolWithLength& a, const SymbolWithLength& b) {
    if (a.length != b.length) return a.length < b.length;
    return a.symbol < b.symbol;
}

using CanonicalCodebook = std::vector<SymbolWithLength>;

void dump(const CanonicalCodebook& codebook) {
    uint group_length = 0;
    std::vector<Symbol> group;
    auto flush_group = [&]() {
        if (!group.empty()) {
            std::cerr << "length: " << static_cast<uint>(group_length) << " (" << group.size()
                      << " symbols): ";
            for (const auto& s : group) {
                std::cerr << static_cast<uint>(s) << " ";
            }
            std::cerr << std::endl;
            group.clear();
        }
    };
    for (const auto& entry : codebook) {
        if (entry.length != group_length) {
            flush_group();
            group_length = entry.length;
        }
        group.push_back(entry.symbol);
    }
    flush_group();
}

CanonicalCodebook build_huffman_limited(const Histogram& hist, uint max_depth) {
    std::vector<std::shared_ptr<Node>> leaves;
    for (auto i = 0u; i < hist.size(); ++i) {
        if (hist[i] > 0) {
            leaves.push_back(std::make_shared<Node>(static_cast<Symbol>(i), hist[i]));
        }
    }
    if (leaves.empty()) return {};

    // Count bits for each symbol
    auto cmp = [](const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
        if (a->freq != b->freq) return a->freq < b->freq;
        if (a->symbol && b->symbol) return a->symbol.value() < b->symbol.value();
        if (a->symbol) return true;
        if (b->symbol) return false;
        return false;
    };
    std::sort(leaves.begin(), leaves.end(), cmp);
    std::vector<std::shared_ptr<Node>> current_list = leaves;
    for (auto d = 0u; d < max_depth - 1; ++d) {
        std::vector<std::shared_ptr<Node>> packages;
        for (size_t i = 0; i + 1 < current_list.size(); i += 2) {
            packages.push_back(std::make_shared<Node>(current_list[i], current_list[i + 1]));
        }
        current_list.clear();
        std::merge(leaves.begin(), leaves.end(), packages.begin(), packages.end(),
                   std::back_inserter(current_list), cmp);
    }

    // Map symbols to their lengths
    std::vector<uint8_t> lengths(hist.size(), 0);
    std::function<void(const std::shared_ptr<Node>&)> traverse;
    traverse = [&](const std::shared_ptr<Node>& item) {
        if (item->symbol) {
            lengths[item->symbol.value()]++;
        } else {
            if (item->left) traverse(item->left);
            if (item->right) traverse(item->right);
        }
    };
    auto items_to_select = std::min(2 * (leaves.size() - 1), current_list.size());
    for (uint i = 0; i < items_to_select; ++i) {
        traverse(current_list[i]);
    }

    // Collect and sort (symbol, length) pairs
    CanonicalCodebook codebook;
    for (size_t i = 0; i < lengths.size(); ++i) {
        if (lengths[i] > 0) {
            codebook.push_back({static_cast<Symbol>(i), lengths[i]});
        }
    }
    std::sort(codebook.begin(), codebook.end());
    return codebook;
}

template <class T>
T reverse_bits(T bits, uint8_t n_bits) {
    T reversed = 0;
    for (uint8_t i = 0; i < n_bits; ++i) {
        reversed <<= 1;
        reversed |= (bits & 1);
        bits >>= 1;
    }
    return reversed;
}

std::vector<std::optional<Code>> build_encoding_table(const CanonicalCodebook& codebook) {
    std::vector<std::optional<Code>> table;
    table.resize(NSymbols, std::nullopt);
    uint64_t code = 0;
    uint8_t code_length = 0;
    for (const auto& [symbol, length] : codebook) {
        while (length > code_length) {
            code <<= 1;
            code_length++;
        }
        table[symbol] = {reverse_bits(code, length), length};
        code++;
    }
    return table;
}

std::vector<SymbolWithLength> build_decoding_table(const CanonicalCodebook& codebook,
                                                   size_t table_index_bits) {
    std::vector<SymbolWithLength> table(1u << table_index_bits, {0, 0});
    uint32_t code = 0;
    uint8_t code_length = 0;
    for (const auto& [symbol, length] : codebook) {
        while (length > code_length) {
            code <<= 1;
            code_length++;
        }
        uint32_t entry_count = 1u << (table_index_bits - length);
        for (uint32_t i = 0; i < entry_count; ++i) {
            uint32_t index = (code << (table_index_bits - length)) + i;
            table[reverse_bits(index, table_index_bits)] = {symbol, length};
        }
        code++;
    }
    return table;
}

/*
 * Write an LSB-first bitstream. E.g. with uint8_t
 *                              data[0]   data[1]
 *  write_bits(0b101, 3)   -> 0000'0101
 *  write_bits(0b11, 2)    -> 0001'1101
 *  write_bits(0b10011, 5) -> 0111'1101 0000'0010
 */
template <class T>
struct LsbBitStreamWriter {
    using Element = T;
    constexpr static uint8_t T_bits = sizeof(T) * 8;
    std::vector<T> data;
    uint8_t bits_filled;

    LsbBitStreamWriter() : bits_filled(0) { data.push_back(0); }

    size_t size_bits() const { return (data.size() - 1) * T_bits + bits_filled; }
    size_t size_bytes() const { return data.size() * sizeof(T); }

    void write_bits(T bits, uint8_t n_bits) {
        bits &= ~T(0) >> (sizeof(T) * 8 - n_bits);
        auto to_write = std::min<uint8_t>(n_bits, T_bits - bits_filled);
        data.back() |= (bits << bits_filled);
        bits_filled += to_write;
        if (bits_filled == T_bits) {
            data.push_back(0);
            bits_filled = 0;
        }
        // write any remaining bits
        bits >>= to_write;
        n_bits -= to_write;
        if (n_bits > 0) {
            write_bits(bits, n_bits);
        }
    }
};

std::ostream& operator<<(std::ostream& out, const LsbBitStreamWriter<uint32_t>& bsw) {
    for (auto i = 0u; i < bsw.data.size(); ++i) {
        auto n_bits = (i + 1 == bsw.data.size()) ? bsw.bits_filled : 32u;
        for (auto j = 0u; j < n_bits; ++j) {
            out << ((bsw.data[i] >> j) & 1);
        }
    }
    return out;
}

// -------------------------------------------------------------------------------------------------

namespace no_compression {

struct Impl : Transcoder {
    std::vector<char> encode(const std::vector<Symbol>& data) const override {
        std::vector<char> result(sizeof(uint64_t) + data.size());
        *reinterpret_cast<uint64_t*>(&result[0]) = data.size();
        std::memcpy(&result[sizeof(uint64_t)], data.data(), data.size());
        return result;
    }

    uint32_t decode_sum(const char* data) const override {
        const uint64_t n_symbols = *reinterpret_cast<const uint64_t*>(data);
        uint32_t sum = 0;
        const Symbol* symbols = reinterpret_cast<const Symbol*>(data + sizeof(uint64_t));
#pragma omp parallel for reduction(+ : sum)
        for (uint64_t i = 0; i < n_symbols; ++i) {
            sum += symbols[i];
        }
        return sum;
    }
};

std::unique_ptr<Transcoder> create(const Histogram&) {
    return std::make_unique<Impl>();
}

}  // namespace no_compression

// -------------------------------------------------------------------------------------------------

namespace huffman_limited_memory {

struct Impl : Transcoder {
    uint64_t table_index_bits;
    std::vector<std::optional<Code>> encoding_table;
    std::vector<SymbolWithLength> decoding_table;

    Impl(const CanonicalCodebook& codebook, uint64_t table_index_bits)
        : table_index_bits(table_index_bits),
          encoding_table(build_encoding_table(codebook)),
          decoding_table(build_decoding_table(codebook, table_index_bits)) {}

    std::vector<char> encode(const std::vector<Symbol>& data) const override {
        LsbBitStreamWriter<uint32_t> bsw;
        for (const auto& s : data) {
            auto code = encoding_table[s];
            assert(code.has_value());
            bsw.write_bits(code->value, code->length);
        }
        std::vector<char> result(sizeof(uint64_t) + bsw.size_bytes());
        *reinterpret_cast<uint64_t*>(&result[0]) = data.size();
        std::memcpy(&result[sizeof(uint64_t)], bsw.data.data(), bsw.size_bytes());
        return result;
    }

    uint32_t decode_sum(const char* data) const override {
        const uint64_t n_symbols = *reinterpret_cast<const uint64_t*>(data);
        const uint32_t* data_ptr = reinterpret_cast<const uint32_t*>(data + sizeof(uint64_t));
        const uint32_t index_mask = (1u << table_index_bits) - 1;
        constexpr uint32_t data_bits = sizeof(uint32_t) * 8;
        uint32_t sum = 0;
        uint64_t current = *data_ptr++;
        uint32_t current_bits = data_bits;
        for (size_t count = 0; count < n_symbols; ++count) {
            if (current_bits < table_index_bits) {
                current |= static_cast<uint64_t>(*data_ptr++) << current_bits;
                current_bits += data_bits;
            }
            auto index = current & index_mask;
            auto entry = decoding_table[index];
            sum += entry.symbol;
            current >>= entry.length;
            current_bits -= entry.length;
        }
        return sum;
    }
};

std::unique_ptr<Transcoder> create(const Histogram& hist, size_t table_index_bits) {
    auto codebook = build_huffman_limited(hist, table_index_bits);
    return std::make_unique<Impl>(codebook, table_index_bits);
}

}  // namespace huffman_limited_memory

// -------------------------------------------------------------------------------------------------
// Driver

struct DecodeBenchmark {
    Transcoder& coder;
    std::vector<char> encoded_data;
    volatile uint64_t _sum;

    size_t bytes_per_run() const { return encoded_data.size(); }
    void runonce() { _sum = coder.decode_sum(encoded_data.data()); }
};

void evaluate_format(const Format& format,
                     const DataSettings& settings,
                     const std::vector<size_t>& nthreads) {
    const auto samples = 1 << 22;
    std::cerr << "\n# " << format.name << " | ν: " << settings.dof << ", Δ: " << settings.delta
              << std::endl;

    const auto symbols =
        quantise(generate_student_t(samples, settings.dof, /*seed*/ 42), settings.delta);
    const auto hist = count_histogram(symbols);
    double entropy = calculate_entropy(hist);
    auto coder = format.factory(hist);
    if (!coder) {
        std::cerr << "  (skipped)" << std::endl;
        return;
    }

    // Test correctness
    for (auto n_sub : std::initializer_list<size_t>{10, 100, symbols.size() / 3, symbols.size()}) {
        auto expected_sum = std::accumulate(symbols.begin(), symbols.begin() + n_sub, 0ull);
        auto encoded = coder->encode(std::vector<Symbol>(symbols.begin(), symbols.begin() + n_sub));
        auto decoded_sum = coder->decode_sum(encoded.data());
        if (expected_sum != decoded_sum) {
            std::cerr << "  bad decode for " << n_sub << " elements, expected: " << expected_sum
                      << ", decoded: " << decoded_sum << std::endl;
        }
    }

    // Compression ratio
    auto encoded = coder->encode(symbols);
    auto ratio =
        static_cast<double>(encoded.size()) / static_cast<double>(symbols.size() * sizeof(Symbol));
    auto bits_per_symbol =
        8.0 * static_cast<double>(encoded.size()) / static_cast<double>(symbols.size());
    std::cerr << "  compression ratio :: " << ratio << " (" << bits_per_symbol << " bits, entropy "
              << entropy << " bits)" << std::endl;

    // Benchmark
    for (auto nthread : nthreads) {
        omp_set_num_threads(nthread);
        DecodeBenchmark benchmark{*coder, encoded, 0};
        auto measurement = run_benchmark(benchmark, 10, /*pre_runs*/ 5);
        std::cerr << "  decode @ " << nthread << " threads :: " << measurement << "B/s"
                  << std::endl;
    }
}

int main() {
    std::cerr << "[encodings]" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<Format> formats = {
        {"no_compression", no_compression::create},
        {"huffman_limited_memory[8]",
         [](const Histogram& hist) {
             return huffman_limited_memory::create(hist, /*table_bits*/ 8);
         }},
        {"huffman_limited_memory[12]",
         [](const Histogram& hist) {
             return huffman_limited_memory::create(hist, /*table_bits*/ 12);
         }},
        {"huffman_limited_memory[16]",
         [](const Histogram& hist) {
             return huffman_limited_memory::create(hist, /*table_bits*/ 16);
         }},
    };
    std::vector<DataSettings> data_settings = {
        {3.0, 0.745},
        {5.0, 0.645},
        {7.0, 0.605},
        {100.0, 0.525},
    };
    std::vector<size_t> nthreads{
        1,
        // std::thread::hardware_concurrency(),
    };
    for (const auto& settings : data_settings) {
        for (const auto& format : formats) {
            evaluate_format(format, settings, nthreads);
        }
    }

    std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
    std::cerr << "\n[encodings] finished in " << elapsed.count() << " seconds" << std::endl;
    return 0;
}
