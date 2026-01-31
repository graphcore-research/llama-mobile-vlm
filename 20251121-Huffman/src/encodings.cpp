#include "benchmarking.hpp"

#include <arm_neon.h>
#include <cassert>
#include <random>
#include <thread>

// -------------------------------------------------------------------------------------------------
// Basic types

using Symbol = int8_t;
using Histogram = std::vector<uint64_t>;

constexpr int32_t MinSymbol = std::numeric_limits<Symbol>::min();
constexpr size_t NSymbols = 1 << (8 * sizeof(Symbol));

struct Transcoder {
    virtual std::vector<char> encode(const std::vector<Symbol>& data) const = 0;
    virtual int32_t decode_sum(const char* data) const = 0;

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
        int q = std::clamp(static_cast<int>(std::round(v / delta)), -128, 127);
        symbols.push_back(static_cast<Symbol>(q));
    }
    return symbols;
}

Histogram count_histogram(const std::vector<Symbol>& symbols) {
    Histogram hist(NSymbols, 0);
    for (const auto& s : symbols) {
        hist[static_cast<int>(s) - MinSymbol]++;
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

struct EncodingTable {
    std::vector<std::optional<Code>> table;

    const Code& operator[](Symbol s) const {
        auto index = s - MinSymbol;
        if (!table[index].has_value()) {
            std::ostringstream oss;
            oss << "Symbol " << static_cast<int>(s) << " not in encoding table";
            throw std::runtime_error(oss.str());
        }
        return table[index].value();
    }
};

using CanonicalCodebook = std::vector<SymbolWithLength>;

std::ostream& operator<<(std::ostream& out, const Code& code) {
    for (int i = code.length - 1; i >= 0; --i) {
        out << ((code.value >> i) & 1);
    }
    return out;
}

bool operator<(const SymbolWithLength& a, const SymbolWithLength& b) {
    if (a.length != b.length) return a.length < b.length;
    return a.symbol < b.symbol;
}

void dump(const CanonicalCodebook& codebook) {
    uint group_length = 0;
    std::vector<Symbol> group;
    auto flush_group = [&]() {
        if (!group.empty()) {
            std::cerr << "length: " << static_cast<uint>(group_length) << " (" << group.size()
                      << " symbols): ";
            for (const auto& s : group) {
                std::cerr << static_cast<int>(s) << " ";
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

void dump(const EncodingTable& table) {
    for (auto i = 0u; i < table.table.size(); ++i) {
        if (table.table[i].has_value()) {
            std::cerr << (static_cast<int>(i) + MinSymbol) << " -> " << *table.table[i]
                      << std::endl;
        }
    }
}

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

CanonicalCodebook build_huffman_limited(const Histogram& hist, uint max_depth) {
    std::vector<std::shared_ptr<Node>> leaves;
    for (auto i = 0u; i < hist.size(); ++i) {
        if (hist[i] > 0) {
            leaves.push_back(std::make_shared<Node>(static_cast<Symbol>(i + MinSymbol), hist[i]));
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
            lengths[item->symbol.value() - MinSymbol]++;
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
            codebook.push_back({static_cast<Symbol>(i + MinSymbol), lengths[i]});
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

EncodingTable build_encoding_table(const CanonicalCodebook& codebook) {
    std::vector<std::optional<Code>> table;
    table.resize(NSymbols, std::nullopt);
    uint64_t code = 0;
    uint8_t code_length = 0;
    for (const auto& [symbol, length] : codebook) {
        while (length > code_length) {
            code <<= 1;
            code_length++;
        }
        table[symbol - MinSymbol] = {reverse_bits(code, length), length};
        code++;
    }
    return {std::move(table)};
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

// -------------------------------------------------------------------------------------------------
// Writers & readers

enum class BitStreamMode { Lsb, Msb };

/*
 * Write a bitstream with configurable word fill direction. Default is LSB-first (bits fill up
 * from the least-significant bit). With Mode::Msb bits fill down from the most-significant bit.
 *
 * Example (uint8_t, Mode::Lsb):
 *                              data[0]   data[1]
 *  write_bits(0b101, 3)   -> 0000'0101
 *  write_bits(0b11, 2)    -> 0001'1101
 *  write_bits(0b10011, 5) -> 0111'1101 0000'0010
 */
template <class T, BitStreamMode _Mode = BitStreamMode::Lsb>
struct BitStreamWriter {
    using Element = T;
    constexpr static BitStreamMode Mode = _Mode;
    constexpr static uint8_t T_bits = sizeof(T) * 8;

    std::vector<T> data;
    uint8_t bits_filled;

    BitStreamWriter() : bits_filled(0) { data.push_back(0); }

    size_t size_bits() const { return (data.size() - 1) * T_bits + bits_filled; }
    size_t size_bytes() const { return data.size() * sizeof(T); }

    void write_bits(T bits, uint8_t n_bits) {
        auto lower_mask = [](uint8_t count) -> T { return (T(1) << count) - 1; };
        auto upper_mask = [&](uint8_t count) -> T {
            return ((T(1) << count) - 1) << (n_bits - count);
        };
        while (n_bits) {
            const auto available = static_cast<uint8_t>(T_bits - bits_filled);
            const auto to_write = std::min<uint8_t>(n_bits, available);
            if (Mode == BitStreamMode::Lsb) {
                data.back() |= (bits & lower_mask(to_write)) << bits_filled;
                bits >>= to_write;
            } else {
                if (n_bits < available) {
                    data.back() |= (bits & upper_mask(to_write)) << (available - n_bits);
                } else {
                    data.back() |= (bits & upper_mask(to_write)) >> (n_bits - available);
                }
            }
            n_bits -= to_write;
            bits_filled += to_write;
            if (bits_filled == T_bits) {
                data.push_back(0);
                bits_filled = 0;
            }
        }
    }

    void write_bits(const Code& code) { write_bits(code.value, code.length); }
};

template <class T, BitStreamMode Mode>
std::ostream& operator<<(std::ostream& out, const BitStreamWriter<T, Mode>& bsw) {
    using BSW = std::remove_reference_t<decltype(bsw)>;
    for (auto i = 0u; i < bsw.data.size(); ++i) {
        auto n_bits = (i + 1 == bsw.data.size()) ? bsw.bits_filled : BSW::T_bits;
        if (Mode == BitStreamMode::Lsb) {
            for (auto j = 0u; j < n_bits; ++j) {
                out << ((bsw.data[i] >> j) & 1);
            }
        } else {
            for (auto j = 0u; j < n_bits; ++j) {
                out << ((bsw.data[i] >> (BSW::T_bits - 1 - j)) & 1);
            }
        }
    }
    return out;
}

template <class T, class TBuffer, BitStreamMode Mode = BitStreamMode::Lsb>
struct BitStreamReader {
    constexpr static uint8_t TBuffer_bits = sizeof(TBuffer) * 8;
    constexpr static uint32_t T_bits = sizeof(T) * 8;
    static_assert(sizeof(TBuffer) > sizeof(T), "TBuffer must be wider than T to avoid overflow");

    const T* data;
    TBuffer buffer;
    uint32_t buffer_bits;

    BitStreamReader() : data(nullptr), buffer(0), buffer_bits(0) {}
    explicit BitStreamReader(const T* data) : data(data), buffer(0), buffer_bits(0) {}

    TBuffer& read(uint8_t min_bits) {
        if (buffer_bits < min_bits) {
            auto next = static_cast<TBuffer>(*data++);
            if constexpr (Mode == BitStreamMode::Lsb) {
                buffer |= next << buffer_bits;
            } else {
                buffer |= next << (TBuffer_bits - T_bits - buffer_bits);
            }
            buffer_bits += T_bits;
        }
        return buffer;
    }

    void advance(uint8_t n_bits) {
        if constexpr (Mode == BitStreamMode::Lsb) {
            buffer >>= n_bits;
        } else {
            buffer <<= n_bits;
        }
        buffer_bits -= n_bits;
    }
};

struct Writer {
    std::vector<char> data;

    template <class T>
    void write(const T* ptr, size_t n) {
        size_t i = data.size();
        data.resize(i + n * sizeof(T));
        std::memcpy(&data[i], ptr, n * sizeof(T));
    }

    template <class T>
    void write(T value) {
        write(&value, 1);
    }

    template <class T, BitStreamMode Mode>
    void write_bitstream(const BitStreamWriter<T, Mode>& bsw) {
        write(bsw.data.data(), bsw.data.size());
    }
};

struct Reader {
    const char* data;

    explicit Reader(const char* data) : data(data) {}

    template <class T>
    const T* get(size_t byte_offset) {
        return reinterpret_cast<const T*>(data + byte_offset);
    }

    template <class T>
    T read(size_t byte_offset) {
        return *get<T>(byte_offset);
    }

    template <class T, class TBuffer, BitStreamMode Mode = BitStreamMode::Lsb>
    BitStreamReader<T, TBuffer, Mode> read_bitstream(size_t byte_offset) {
        return BitStreamReader<T, TBuffer, Mode>(get<T>(byte_offset));
    }
};

// -------------------------------------------------------------------------------------------------

namespace no_compression {

struct Impl : Transcoder {
    std::vector<char> encode(const std::vector<Symbol>& data) const override {
        Writer w;
        w.write<uint64_t>(data.size());
        w.write(data.data(), data.size());
        return w.data;
    }

    int32_t decode_sum(const char* data) const override {
        const uint64_t n_symbols = *reinterpret_cast<const uint64_t*>(data);
        int32_t sum = 0;
        const Symbol* symbols = reinterpret_cast<const Symbol*>(data + sizeof(uint64_t));
#pragma omp parallel for reduction(+ : sum)
        for (auto i = 0u; i < n_symbols; ++i) {
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
    EncodingTable encoding_table;
    std::vector<SymbolWithLength> decoding_table;

    Impl(const CanonicalCodebook& codebook, uint64_t table_index_bits)
        : table_index_bits(table_index_bits),
          encoding_table(build_encoding_table(codebook)),
          decoding_table(build_decoding_table(codebook, table_index_bits)) {}

    std::vector<char> encode(const std::vector<Symbol>& data) const override {
        BitStreamWriter<uint32_t> stream;
        for (const auto& s : data) {
            stream.write_bits(encoding_table[s]);
        }
        Writer w;
        w.write<uint64_t>(data.size());
        w.write_bitstream(stream);
        return w.data;
    }

    int32_t decode_sum(const char* data) const override {
        const uint32_t index_mask = (1u << table_index_bits) - 1;
        Reader reader(data);
        const auto n_symbols = reader.read<uint64_t>(0);
        auto stream = reader.read_bitstream<uint32_t, uint64_t>(sizeof(uint64_t));
        int32_t sum = 0;
        for (auto count = 0u; count < n_symbols; ++count) {
            auto index = stream.read(table_index_bits) & index_mask;
            auto entry = decoding_table[index];
            sum += entry.symbol;
            stream.advance(entry.length);
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

namespace huffman_limited_memory_multistream {

template <size_t N>
struct Impl : Transcoder {
    uint64_t table_index_bits;
    EncodingTable encoding_table;
    std::vector<SymbolWithLength> decoding_table;

    Impl(const CanonicalCodebook& codebook, uint64_t table_index_bits)
        : table_index_bits(table_index_bits),
          encoding_table(build_encoding_table(codebook)),
          decoding_table(build_decoding_table(codebook, table_index_bits)) {}

    std::vector<char> encode(const std::vector<Symbol>& data) const override {
        // Distribute symbols across N streams in round-robin fashion
        std::array<BitStreamWriter<uint32_t>, N> streams;
        for (size_t i = 0; i < data.size(); ++i) {
            streams[i % N].write_bits(encoding_table[data[i]]);
        }
        Writer w;
        w.write<uint64_t>(data.size());  // n_symbols
        // N* offset
        uint64_t offset = sizeof(uint64_t) + N * sizeof(uint64_t);
        for (const auto& stream : streams) {
            w.write<uint64_t>(offset);
            offset += stream.size_bytes();
        }
        // N* bitstreams
        for (const auto& stream : streams) {
            w.write_bitstream(stream);
        }
        return w.data;
    }

    int32_t decode_sum(const char* data) const override {
        const uint32_t index_mask = (1u << table_index_bits) - 1;
        Reader reader(data);

        // Setup streams using stored offsets
        std::array<BitStreamReader<uint32_t, uint64_t>, N> streams;
        for (size_t i = 0; i < N; ++i) {
            auto offset = reader.read<uint64_t>(sizeof(uint64_t) + i * sizeof(uint64_t));
            streams[i] = reader.read_bitstream<uint32_t, uint64_t>(offset);
        }

        // Decode symbols from all streams in parallel until exhausted
        int32_t sum = 0;
        uint64_t remaining = reader.read<uint64_t>(0);
        while (true) {
            for (auto& stream : streams) {
                auto index = stream.read(table_index_bits) & index_mask;
                auto entry = decoding_table[index];
                sum += entry.symbol;
                stream.advance(entry.length);
                if (--remaining == 0) return sum;
            }
        }
    }
};

template <size_t N>
std::unique_ptr<Transcoder> create(const Histogram& hist, size_t table_index_bits) {
    auto codebook = build_huffman_limited(hist, table_index_bits);
    return std::make_unique<Impl<N>>(codebook, table_index_bits);
}

}  // namespace huffman_limited_memory_multistream

// -------------------------------------------------------------------------------------------------

namespace unary_fallback {

struct Impl : Transcoder {
    static constexpr int N = 16;
    EncodingTable encoding_table;

    Impl() {
        encoding_table.table.resize(NSymbols);
        for (auto i = 0u; i < NSymbols; ++i) {
            auto s = static_cast<Symbol>(i + MinSymbol);
            if (s == 0) {
                encoding_table.table[i] = Code{0b1, 1};
            } else if (std::abs(s) <= 6) {
                // [...0*abs(s), 1, signbit]
                auto sign = s < 0 ? 1u : 0u;
                encoding_table.table[i] = Code{0b10 | sign, static_cast<uint8_t>(std::abs(s) + 2)};
            } else {
                // [0000'0000, s]
                encoding_table.table[i] = Code{static_cast<uint8_t>(s), 16};
            }
        }
    }

    std::vector<char> encode(const std::vector<Symbol>& data) const override {
        assert(data.size() % N == 0 && "Data size must be multiple of N for SIMD decoding");
        std::array<BitStreamWriter<uint32_t, BitStreamMode::Msb>, N> streams;
        for (size_t i = 0; i < data.size(); ++i) {
            streams[i % N].write_bits(encoding_table[data[i]]);
        }
        // std::cerr << "streams[ ]: ";
        // for (size_t i = 0; i < streams[0].size_bits(); ++i) {
        //     std::cerr << (i % 64 ? ' ' : '*');
        // }
        // std::cerr << std::endl;
        // std::cerr << "streams[0]: " << streams[0] << std::endl;
        Writer w;
        w.write<uint64_t>(data.size());  // n_symbols
        // N* offset
        uint64_t offset = sizeof(uint64_t) + N * sizeof(uint64_t);
        for (const auto& stream : streams) {
            w.write<uint64_t>(offset);
            offset += stream.size_bytes();
        }
        // N* bitstreams
        for (const auto& stream : streams) {
            w.write_bitstream(stream);
        }
        return w.data;
    }

    static std::tuple<int8x16_t, uint8x16_t> decode(uint8x16_t bits0, uint8x16_t bits1) {
        uint8x16_t nzeros = vclzq_u8(bits0);
        int8x16_t values = vreinterpretq_s8_u8(nzeros);

        uint8x16_t signs = vtstq_u8(vshlq_u8(bits0, nzeros), vdupq_n_u8(0b0100'0000));
        values = vbslq_s8(signs, vnegq_s8(values), values);

        uint8x16_t fallback = vceqq_u8(bits0, vdupq_n_u8(0));
        values = vbslq_s8(fallback, bits1, values);

        // Could be better as a tbl lookup? (so far 8 inst, this adds +8 inst)
        // uint8x16_t bits_to_skip = nzeros;
        uint8x16_t is_zero = vceqq_u8(nzeros, vdupq_n_u8(0));
        uint8x16_t n_extra_bits = vbslq_u8(is_zero, vdupq_n_u8(1), vdupq_n_u8(2));
        uint8x16_t bits_to_skip = vaddq_u8(nzeros, n_extra_bits);
        bits_to_skip = vbslq_u8(fallback, vdupq_n_u8(16), bits_to_skip);

        return std::make_tuple(values, bits_to_skip);
    }

    int32_t decode_sum(const char* data) const override {
        // Note: test reading with 16(32)-bit or 32(64)-bit buffers
        // Note: try a table lookup for bits_to_read

        Reader reader(data);
        const auto n_symbols = reader.read<uint64_t>(0);

        // Setup streams using stored offsets
        std::array<BitStreamReader<uint32_t, uint64_t, BitStreamMode::Msb>, N> streams;
        for (size_t i = 0; i < N; ++i) {
            auto offset = reader.read<uint64_t>(sizeof(uint64_t) + i * sizeof(uint64_t));
            streams[i] = reader.read_bitstream<uint32_t, uint64_t, BitStreamMode::Msb>(offset);
        }

        int32_t sum = 0;
        for (auto count = 0u; count < n_symbols / N; ++count) {
            // Sketch:
            //  - read from 16 streams into two uint8x16_t for bits [0-7] [8-15]
            //  - count leading zeros
            //  - extract and apply sign
            //  - test and copy large values from 2nd register
            //  - compute & apply shift for original streams

            uint8_t _bits0[16], _bits1[16];
            for (int i = 0; i < N; ++i) {
                auto value = streams[i].read(16);
                _bits0[i] = static_cast<uint8_t>(value >> 56);
                _bits1[i] = static_cast<uint8_t>(value >> 48);
            }
            for (int i = N; i < 16; ++i) {
                _bits0[i] = 0;
                _bits1[i] = 0;
            }
            uint8x16_t bits0 = vld1q_u8(_bits0);  // bits [0-7]
            uint8x16_t bits1 = vld1q_u8(_bits1);  // bits [8-15]
            auto d = decode(bits0, bits1);

            sum += vaddlvq_s8(std::get<0>(d));

            uint8_t _bits_to_advance[16];
            vst1q_u8(_bits_to_advance, std::get<1>(d));
            for (int i = 0; i < N; ++i) {
                auto bits_to_advance = _bits_to_advance[i];
                streams[i].advance(bits_to_advance);
            }

            // // Debug
            // int8_t debug_values[N];
            // vst1q_s8(debug_values, std::get<0>(d));
            // std::cerr << "Decoded values: ";
            // for (int i = 0; i < N; ++i) {
            //     std::cerr << static_cast<int>(debug_values[i]) << " ";
            // }
            // std::cerr << std::endl;

            // vst1q_s8(debug_values, std::get<1>(d));
            // std::cerr << "Bits to skip: ";
            // for (int i = 0; i < N; ++i) {
            //     std::cerr << static_cast<int>(debug_values[i]) << " ";
            // }
            // std::cerr << "\n" << std::endl;
        }
        return sum;
    }
};

std::unique_ptr<Transcoder> create(const Histogram&) {
    return std::make_unique<Impl>();
}

}  // namespace unary_fallback

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
    for (auto n_sub : std::initializer_list<size_t>{16, 256, symbols.size() / 2, symbols.size()}) {
        auto expected_sum = std::accumulate(symbols.begin(), symbols.begin() + n_sub, 0);
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
        {"unary_fallback", unary_fallback::create},
        {"huffman_limited_memory[8]",
         [](const Histogram& hist) {
             return huffman_limited_memory::create(hist, /*table_bits*/ 8);
         }},
        {"huffman_limited_memory_multistream[1][8]",
         [](const Histogram& hist) {
             return huffman_limited_memory_multistream::create<1>(hist, /*table_bits*/ 8);
         }},
        {"huffman_limited_memory_multistream[2][8]",
         [](const Histogram& hist) {
             return huffman_limited_memory_multistream::create<2>(hist, /*table_bits*/ 8);
         }},
        {"huffman_limited_memory_multistream[4][8]",
         [](const Histogram& hist) {
             return huffman_limited_memory_multistream::create<4>(hist, /*table_bits*/ 8);
         }},
        {"huffman_limited_memory_multistream[8][8]",
         [](const Histogram& hist) {
             return huffman_limited_memory_multistream::create<8>(hist, /*table_bits*/ 8);
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
        // {3.0, 0.745},
        // {5.0, 0.645},
        {7.0, 0.605},
        // {100.0, 0.525},
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
