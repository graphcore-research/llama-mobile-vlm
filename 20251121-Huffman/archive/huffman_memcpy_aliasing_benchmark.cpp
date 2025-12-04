#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

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

void flushCache() {
    // Allocate a buffer larger than the largest cache
    using Element = uint64_t;
    const size_t cacheFlushSize = 512 * 1024 * 1024 / sizeof(Element);
    static std::vector<Element> cacheFlushBuffer(cacheFlushSize);
    for (auto n = 0; n < 3; ++n) {
        // Read-modify-write seems better than write-only for flushing caches
#pragma omp parallel for schedule(static)
        for (auto i = 0ull; i < cacheFlushBuffer.size(); ++i) {
            cacheFlushBuffer[i] += 1;
        }
    }
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
// Benchmarks

struct Measurement {
    std::vector<double> samples;

    double mean() const {
        double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
        return sum / samples.size();
    }
    double standard_error() const {
        double m = mean();
        double sum_sq = 0.0;
        for (double s : samples) {
            sum_sq += (s - m) * (s - m);
        }
        return std::sqrt(sum_sq / (samples.size() - 1)) / std::sqrt(samples.size());
    }
};
std::ostream& operator<<(std::ostream& out, const Measurement& m) {
    auto mean = m.mean();
    auto error = m.standard_error();
    std::string units = " ";
    auto divisor = 1.0;
    if (mean >= 1e9) {
        divisor = 1e9;
        units = " G";
    } else if (mean >= 1e6) {
        divisor = 1e6;
        units = " M";
    } else if (mean >= 1e3) {
        divisor = 1e3;
        units = " k";
    }
    return out << mean / divisor << " ± " << error / divisor << units;
}

template <class Benchmark>
Measurement benchmark_transfer(Benchmark&& benchmark, uint runs, uint pre_runs) {
    std::vector<double> results;
    for (uint i = 0; i < runs + pre_runs; ++i) {
        flushCache();
        auto start = std::chrono::high_resolution_clock::now();
        benchmark.runonce();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        results.push_back(benchmark.bytes_per_run() / elapsed.count());
    }
    results.erase(results.begin(), results.begin() + pre_runs);
    return Measurement{results};
}

enum class byte : uint8_t {};
static_assert(sizeof(byte) == 1, "byte must be 1 byte");

template <class Element = uint64_t>
struct BenchmarkMemcpy {
    std::vector<Element> src;
    std::vector<Element> dst;

    BenchmarkMemcpy(size_t size) : src(size / sizeof(Element)), dst(size / sizeof(Element)) {}

    uint64_t bytes_per_run() const { return 2 * src.size() * sizeof(Element); }

    __attribute__((noinline)) void runonce() {
        const auto count = src.size();
        const Element* src_ptr = src.data();
        Element* dst_ptr = dst.data();
#pragma omp parallel for
        for (size_t i = 0u; i < count; ++i) {
            dst_ptr[i] = src_ptr[i];
        }
    }
};

template <class Element = uint64_t>
struct BenchmarkIncrement {
    std::vector<Element> src;
    std::vector<Element> dst;

    BenchmarkIncrement(size_t size) : src(size / sizeof(Element)), dst(size / sizeof(Element)) {
        std::iota(src.begin(), src.end(), 0);
    }

    uint64_t bytes_per_run() const { return 2 * src.size() * sizeof(Element); }

    __attribute__((noinline)) void runonce() {
#pragma omp parallel for
        for (auto i = 0u; i < src.size(); i++) {
            dst[i] = src[i] + 1;
        }
    }
};

template <class Element = uint64_t>
struct BenchmarkTableLookup {
    std::vector<Element> src;
    std::vector<Element> dst;
    std::vector<Element> table;

    BenchmarkTableLookup(size_t size)
        : src(size / sizeof(Element)), dst(size / sizeof(Element)), table(256) {
        std::iota(src.begin(), src.end(), 0);
    }

    uint64_t bytes_per_run() const { return 2 * src.size() * sizeof(Element); }

    __attribute__((noinline)) void runonce() {
#pragma omp parallel for
        for (auto i = 0u; i < src.size(); i++) {
            dst[i] = src[i] + 1;
        }
    }
};

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

int main() {
    std::cerr << "[huffman.cpp] running" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // test_huffman_codes();
    auto run = [](auto&& benchmark, const std::string& name) {
        auto measurement = benchmark_transfer(benchmark, 40, /*pre_runs*/ 10);
        std::cout << name << " :: " << measurement << "B/s" << std::endl;
    };

    run(BenchmarkMemcpy<uint64_t>(10 * 1024 * 1024), "Memcpy (uint64_t) ");
    run(BenchmarkMemcpy<uint32_t>(10 * 1024 * 1024), "Memcpy (uint32_t) ");
    run(BenchmarkMemcpy<uint16_t>(10 * 1024 * 1024), "Memcpy (uint16_t) ");
    run(BenchmarkMemcpy<uint8_t>(10 * 1024 * 1024), "Memcpy (uint8_t)  ");
    run(BenchmarkMemcpy<int8_t>(10 * 1024 * 1024), "Memcpy (int8_t)   ");
    run(BenchmarkMemcpy<std::byte>(10 * 1024 * 1024), "Memcpy (std::byte)");
    run(BenchmarkMemcpy<char8_t>(10 * 1024 * 1024), "Memcpy (char8_t)  ");
    run(BenchmarkMemcpy<byte>(10 * 1024 * 1024), "Memcpy (byte)     ");
    // run(BenchmarkIncrement<uint64_t>(10 * 1024 * 1024), "Increment (uint64_t)");
    // run(BenchmarkIncrement<uint32_t>(10 * 1024 * 1024), "Increment (uint32_t)");
    // run(BenchmarkIncrement<uint8_t>(10 * 1024 * 1024), "Increment (uint8_t)");

    std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
    std::cerr << "[huffman.cpp] finished in " << elapsed.count() << " seconds" << std::endl;
    return 0;
}
