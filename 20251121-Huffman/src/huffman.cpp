#include "common.hpp"

#include <bitset>
#include <fstream>

// -------------------------------------------------------------------------------------------------
// Basic types

using Symbol = uint8_t;

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

struct Code {
    uint64_t bits;
    uint8_t length;
};

std::ostream& operator<<(std::ostream& os, const Code& code) {
    for (int8_t i = code.length - 1; i >= 0; --i) {
        os << ((code.bits >> i) & 1);
    }
    return os;
}

using Histogram = std::vector<uint64_t>;
using Codebook = std::vector<std::optional<Code>>;

// -------------------------------------------------------------------------------------------------
// Utilities

uint log2_ceil(uint64_t n) {
    uint l = 0;
    uint64_t v = 1;
    while (v < n) {
        v <<= 1;
        l++;
    }
    return l;
}

Codebook generate_huffman_codes(const std::shared_ptr<Node>& root) {
    Codebook codes;
    std::function<void(const std::shared_ptr<Node>&, uint64_t, uint8_t)> traverse;
    traverse = [&](const std::shared_ptr<Node>& node, uint64_t bits, uint8_t length) {
        if (node->symbol.has_value()) {
            if (codes.size() <= node->symbol.value()) {
                codes.resize(node->symbol.value() + 1);
            }
            codes[node->symbol.value()] = Code{bits, length};
            assert(node->left == nullptr && node->right == nullptr);
        } else {
            traverse(node->left, (bits << 1), length + 1);
            traverse(node->right, (bits << 1) | 1, length + 1);
        }
    };
    traverse(root, 0, 0);
    return codes;
}

void print_tree(const std::shared_ptr<Node>& node, const std::string& prefix = "") {
    if (!node) return;
    if (node->symbol.has_value()) {
        std::cout << prefix << "Symbol: " << static_cast<uint>(node->symbol.value())
                  << " Freq: " << node->freq << "\n";
    } else {
        std::cout << prefix << "Node Freq: " << node->freq << "\n";
        print_tree(node->left, prefix + "  ");
        print_tree(node->right, prefix + "  ");
    }
}

double code_space_usage(const Codebook& codes) {
    double usage = 0.0;
    for (const auto& code_opt : codes) {
        if (code_opt.has_value()) {
            const auto& code = *code_opt;
            usage += 1.0 / (1ull << code.length);
        }
    }
    return usage;
}

double bits_per_symbol(const Histogram& hist, const Codebook& codes) {
    double total_bits = 0.0;
    uint64_t total_symbols = 0;
    for (auto s = 0u; s < hist.size(); ++s) {
        if (hist[s]) {
            assert(codes.at(s).has_value() &&
                   "Code must exist for any symbol with non-zero frequency");
            total_bits += hist[s] * codes[s]->length;
            total_symbols += hist[s];
        }
    }
    return total_bits / total_symbols;
}

void show_codes(const Histogram& hist, const Codebook& codes, uint max_length) {
    for (auto s = 0u; s < codes.size(); ++s) {
        if (codes[s] && codes[s]->length <= max_length) {
            std::cout << " " << s << " -> " << *codes[s] << " ("
                      << static_cast<int>(codes[s]->length) << ")" << "\n";
        }
    }
    std::cout << "Average bits per symbol: " << bits_per_symbol(hist, codes) << " (usage "
              << code_space_usage(codes) << ")" << "\n";
}

// -------------------------------------------------------------------------------------------------
// Building Huffman codes

std::shared_ptr<Node> build_huffman_tree(const Histogram& hist) {
    std::vector<std::shared_ptr<Node>> nodes;
    for (auto s = 0u; s < hist.size(); ++s) {
        if (hist[s] > 0) {
            nodes.push_back(std::make_shared<Node>(static_cast<Symbol>(s), hist[s]));
        }
    }
    auto cmp = [](const auto& a, const auto& b) { return a->freq > b->freq; };
    std::make_heap(nodes.begin(), nodes.end(), cmp);
    while (nodes.size() > 1) {
        std::pop_heap(nodes.begin(), nodes.end(), cmp);
        auto left = nodes.back();
        nodes.pop_back();
        std::pop_heap(nodes.begin(), nodes.end(), cmp);
        auto right = nodes.back();
        nodes.pop_back();
        nodes.push_back(std::make_shared<Node>(left, right));
        std::push_heap(nodes.begin(), nodes.end(), cmp);
    }
    return nodes.front();
}

std::shared_ptr<Node> build_huffman_tree_limited(const Histogram& hist, uint max_depth) {
    std::vector<std::shared_ptr<Node>> leaves;
    for (auto i = 0u; i < hist.size(); ++i) {
        if (hist[i] > 0) {
            leaves.push_back(std::make_shared<Node>(static_cast<Symbol>(i), hist[i]));
        }
    }
    if (leaves.empty()) return nullptr;
    if (leaves.size() == 1) return leaves.front();

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

    std::vector<uint> lengths(hist.size(), 0);
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

    // Collect symbols and their lengths
    struct SymbolLen {
        Symbol symbol;
        uint length;
    };
    std::vector<SymbolLen> sym_lens;
    for (size_t i = 0; i < lengths.size(); ++i) {
        if (lengths[i] > 0) {
            sym_lens.push_back({static_cast<Symbol>(i), lengths[i]});
        }
    }
    std::sort(sym_lens.begin(), sym_lens.end(), [](const SymbolLen& a, const SymbolLen& b) {
        if (a.length != b.length) return a.length < b.length;
        return a.symbol < b.symbol;
    });

    // Build the tree
    auto root = std::make_shared<Node>();
    uint64_t code = 0;
    uint current_len = 0;
    for (const auto& sl : sym_lens) {
        while (current_len < sl.length) {
            code <<= 1;
            current_len++;
        }
        std::shared_ptr<Node> curr = root;
        for (int i = current_len - 1; i >= 0; --i) {
            bool bit = (code >> i) & 1;
            if (bit == 0) {
                if (!curr->left) {
                    curr->left = std::make_shared<Node>();
                }
                curr = curr->left;
            } else {
                if (!curr->right) {
                    curr->right = std::make_shared<Node>();
                }
                curr = curr->right;
            }
        }
        curr->symbol = sl.symbol;
        curr->freq = hist[sl.symbol];
        code++;
    }

    // Count up the frequencies from the leaves
    std::function<uint64_t(std::shared_ptr<Node>)> fix_freq;
    fix_freq = [&](std::shared_ptr<Node> node) {
        if (!node) return 0ul;
        if (node->symbol.has_value()) return node->freq;
        node->freq = fix_freq(node->left) + fix_freq(node->right);
        return node->freq;
    };
    fix_freq(root);

    return root;
}

Codebook build_codes_fallback(const Histogram& hist, uint max_length) {
    // Build a regular full Huffman tree
    auto original_root = build_huffman_tree(hist);
    auto original_codes = generate_huffman_codes(original_root);

    // Merge symbols with long codes
    auto merged_hist = hist;
    Symbol fallback_symbol = 0;
    auto fallback_symbol_allocated = false;
    for (auto s = 0u; s < original_codes.size(); ++s) {
        auto long_code = original_codes[s] && original_codes[s]->length > max_length;
        if (!fallback_symbol_allocated && (!original_codes[s] || !long_code)) {
            fallback_symbol = s;
            fallback_symbol_allocated = true;
        } else if (long_code) {
            merged_hist[fallback_symbol] += hist[s];
            merged_hist[s] = 0;
        }
    }
    if (!fallback_symbol_allocated) {
        return original_codes;
    }
    assert(std::accumulate(merged_hist.begin(), merged_hist.end(), 0ull) ==
           std::accumulate(hist.begin(), hist.end(), 0ull));

    // We want to make sure the merged symbol fits in max_length
    // (this is a bit hacky, as it might cause other symbols to exceed max_length again)
    merged_hist[fallback_symbol] = std::max<uint64_t>(
        merged_hist[fallback_symbol], *std::min_element(merged_hist.begin(), merged_hist.end()));

    // Build Huffman tree with low-freq symbols merged
    auto merged_root = build_huffman_tree(merged_hist);
    auto merged_codes = generate_huffman_codes(merged_root);
    merged_codes.resize(hist.size());
    for (auto s = 0u; s < merged_codes.size(); ++s) {
        assert(!(merged_codes[s] && merged_codes[s]->length > max_length));
    }
    assert(merged_codes[fallback_symbol].has_value());

    // Assign fallback codes
    auto fallback_bits = log2_ceil(hist.size());
    const auto fallback_code = *merged_codes[fallback_symbol];
    for (auto s = 0u; s < hist.size(); ++s) {
        if (hist[s] > 0 && (merged_hist[s] == 0 || s == fallback_symbol)) {
            auto code = fallback_code;
            code.length += fallback_bits;
            code.bits = (code.bits << fallback_bits) | s;
            merged_codes[s] = code;
        }
    }
    merged_codes[fallback_symbol] = std::nullopt;
    return merged_codes;
}

// -------------------------------------------------------------------------------------------------
// Decoders

/*
 * Write an LSB-first bitstream. E.g. with uint8_t
 *                              data[0]   data[1]
 *  write_bits(0b101, 3)   -> 0000'0101
 *  write_bits(0b11, 2)    -> 0001'1101
 *  write_bits(0b10011, 5) -> 0111'1101 0000'0010
 */
template <class T>
struct BitStreamWriter {
    constexpr static uint8_t T_bits = sizeof(T) * 8;
    std::vector<T> data;
    uint8_t bits_filled;

    BitStreamWriter() : bits_filled(0) { data.push_back(0); }

    size_t size() const { return (data.size() - 1) * T_bits + bits_filled; }

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

std::ostream& operator<<(std::ostream& out, const BitStreamWriter<uint32_t>& bsw) {
    for (auto i = 0u; i < bsw.data.size(); ++i) {
        auto n_bits = (i + 1 == bsw.data.size()) ? bsw.bits_filled : 32u;
        for (auto j = 0u; j < n_bits; ++j) {
            out << ((bsw.data[i] >> j) & 1);
        }
    }
    return out;
}

struct TableEntry {
    Symbol value;
    uint8_t n_bits;
};

template <uint32_t table_index_bits>
__attribute__((noinline)) uint32_t decode_basic(const uint32_t* data,
                                                size_t total_bits,
                                                const TableEntry* table) {
    constexpr uint32_t index_mask = (1u << table_index_bits) - 1;
    auto sum = 0u;
    const uint32_t* data_ptr = data;
    uint64_t current = *data_ptr++;
    uint32_t current_bits = 32;
    for (size_t offset = 0; offset < total_bits;) {
        if (current_bits < table_index_bits) {
            current |= static_cast<uint64_t>(*data_ptr++) << current_bits;
            current_bits += 32;
        }
        auto index = current & index_mask;
        auto entry = table[index];
        sum += entry.value;
        offset += entry.n_bits;
        current >>= entry.n_bits;
        current_bits -= entry.n_bits;
    }
    return sum;
}

// -------------------------------------------------------------------------------------------------
// Driver scripts

void test_huffman_codes() {
    // Read histogram (see generate_hist.py)
    std::vector<uint64_t> hist(256);
    std::ifstream hist_file("build/hist.bin", std::ios::binary);
    hist_file.read(reinterpret_cast<char*>(hist.data()), hist.size() * sizeof(uint64_t));
    if (!hist_file) {
        throw std::runtime_error("Failed to read histogram data");
    }
    std::cerr << "[huffman.cpp] Read histogram of " << hist.size() << " symbols." << std::endl;

    // Build Huffman trees
    std::cout << "--- Standard Huffman ---" << std::endl;
    auto root = build_huffman_tree(hist);
    auto codes = generate_huffman_codes(root);
    show_codes(hist, codes, /*max_length*/ 8);

    std::cout << "\n--- Limited Huffman (max_depth=8) ---" << std::endl;
    auto root_limited = build_huffman_tree_limited(hist, 8);
    auto codes_limited = generate_huffman_codes(root_limited);
    show_codes(hist, codes_limited, /*max_length*/ 8);

    std::cout << "\n--- Two-stage Huffman (max_code_length=8) ---" << std::endl;
    auto codes_2stage = build_codes_fallback(hist, 8);
    show_codes(hist, codes_2stage, /*max_length*/ 8);
}

void test_decode_basic() {
    std::vector<TableEntry> table({
        {10, 1},  // 000
        {20, 2},  // 001
        {10, 1},  // 010
        {30, 3},  // 011
        {10, 1},  // 100
        {20, 2},  // 101
        {10, 1},  // 110
        {40, 3}   // 111
    });

    BitStreamWriter<uint32_t> bsw;
    for (int i = 0; i < 10; ++i) {
        bsw.write_bits(0b0, 1);    // 10
        bsw.write_bits(0b01, 2);   // 20
        bsw.write_bits(0b011, 3);  // 30
        bsw.write_bits(0b111, 3);  // 40
    }
    assert(bsw.size() == 10 * 9);
    std::cerr << bsw << std::endl;

    auto sum = decode_basic<3>(bsw.data.data(), bsw.size(), table.data());
    assert(sum == (10 + 20 + 30 + 40) * 10);
}

uint32_t reverse_bits(uint32_t bits, uint8_t n_bits) {
    uint32_t reversed = 0;
    for (uint8_t i = 0; i < n_bits; ++i) {
        reversed <<= 1;
        reversed |= (bits & 1);
        bits >>= 1;
    }
    return reversed;
}

template <uint32_t table_index_bits>
struct BenchmarkDecodeBasic {
    BitStreamWriter<uint32_t> src;
    std::vector<TableEntry> table;
    uint32_t sum;

    explicit BenchmarkDecodeBasic(const BitStreamWriter<uint32_t>& bsw,
                                  const std::vector<TableEntry>& tbl)
        : src(bsw), table(tbl), sum(0) {}
    uint64_t bytes_per_run() const { return src.size() / 8; }
    std::string name() const {
        std::ostringstream s;
        s << "BenchmarkDecodeBasic<" << table_index_bits << ">";
        return s.str();
    }
    __attribute__((noinline)) void runonce() {
        sum = decode_basic<table_index_bits>(src.data.data(), src.size(), table.data());
    }
};

void benchmark_decode_basic() {
    const std::vector<std::pair<Symbol, uint8_t>> example_canonical_codes = {
        {127, 2}, {128, 2}, {126, 3}, {129, 3}, {130, 3}, {125, 5}, {131, 5},
        {124, 6}, {132, 6}, {123, 7}, {133, 7}, {122, 8}, {134, 8}};
    const size_t n_symbols = 10 * 1024 * 1024;  // 10 * 1024 * 1024;
    constexpr uint32_t table_index_bits = 8;

    // Build a decoding table
    std::vector<TableEntry> table(1u << table_index_bits, {0, 0});
    uint32_t code = 0;
    uint8_t code_length = 0;
    for (const auto& [symbol, length] : example_canonical_codes) {
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

    // Generate data (sampling from the table is like sampling from the codes)
    BitStreamWriter<uint32_t> bsw;
    std::default_random_engine rng(12345);
    std::uniform_int_distribution<size_t> dist(0, table.size() - 1);
    std::vector<Symbol> symbols;
    while (symbols.size() < n_symbols) {
        auto index = dist(rng);
        auto entry = table[index];
        if (entry.n_bits == 0) continue;
        bsw.write_bits(index, entry.n_bits);
        symbols.push_back(static_cast<Symbol>(entry.value));
    }

    auto sum = decode_basic<table_index_bits>(bsw.data.data(), bsw.size(), table.data());
    uint32_t expected_sum = 0;
    for (const auto& s : symbols) {
        expected_sum += s;
    }
    std::cerr << "Decoded sum: " << sum << ", expected sum: " << expected_sum << "\n";
    assert(sum == expected_sum);

    BenchmarkDecodeBasic<table_index_bits> benchmark(bsw, table);
    auto measurement = run_benchmark(benchmark, 50, /*pre_runs*/ 5);
    std::cerr << benchmark.name() << " :: " << measurement << "B/s" << std::endl;
}

int main() {
    // omp_set_num_threads(omp_get_max_threads());
    omp_set_num_threads(1);

    std::cerr << "[huffman.cpp] running on " << omp_get_max_threads() << " threads" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // test_huffman_codes();
    // test_decode_basic();
    benchmark_decode_basic();

    std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
    std::cerr << "[huffman.cpp] finished in " << elapsed.count() << " seconds" << std::endl;
    return 0;
}
