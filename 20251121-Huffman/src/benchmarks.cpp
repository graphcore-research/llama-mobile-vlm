#include "common.hpp"

template <class Element>
struct BenchmarkMemcpy {
    std::vector<Element> src;
    std::vector<Element> dst;

    explicit BenchmarkMemcpy(size_t size)
        : src(size / sizeof(Element)), dst(size / sizeof(Element)) {}
    uint64_t bytes_per_run() const { return 2 * src.size() * sizeof(Element); }
    std::string name() const { return "Memcpy<" + type_name<Element>() + ">"; }
    void runonce() { benchmark_memcpy(dst.data(), src.data(), src.size()); }

    __attribute__((noinline)) static void benchmark_memcpy(Element* __restrict__ dst,
                                                           const Element* __restrict__ src,
                                                           size_t count) {
#pragma omp parallel for
        for (size_t i = 0u; i < count; ++i) {
            dst[i] = src[i];
        }
    }
};

template <class Element>
struct BenchmarkIncrement {
    std::vector<Element> src;
    std::vector<Element> dst;

    explicit BenchmarkIncrement(size_t size)
        : src(size / sizeof(Element)), dst(size / sizeof(Element)) {}
    uint64_t bytes_per_run() const { return 2 * src.size() * sizeof(Element); }
    std::string name() const { return "Increment<" + type_name<Element>() + ">"; }
    void runonce() { benchmark_increment(dst.data(), src.data(), src.size()); }

    __attribute__((noinline)) static void benchmark_increment(Element* __restrict__ dst,
                                                              const Element* __restrict__ src,
                                                              size_t count) {
#pragma omp parallel for
        for (size_t i = 0u; i < count; ++i) {
            dst[i] = static_cast<Element>(static_cast<uint>(src[i]) + 1);
        }
    }
};

template <class Element, class Accumulator = uint32_t>
struct BenchmarkReduceSum {
    std::vector<Element> src;
    Accumulator dst;

    explicit BenchmarkReduceSum(size_t size) : src(size / sizeof(Element)), dst(0) {}
    std::string name() const {
        return "ReduceSum<" + type_name<Element>() + ", " + type_name<Accumulator>() + ">";
    }
    uint64_t bytes_per_run() const { return src.size() * sizeof(Element) + sizeof(Accumulator); }
    void runonce() { benchmark_reduce_sum(&dst, src.data(), src.size()); }

    __attribute__((noinline)) static void benchmark_reduce_sum(Accumulator* __restrict__ dst,
                                                               const Element* src,
                                                               size_t count) {
        Accumulator sum = 0;
#pragma omp parallel for reduction(+ : sum)
        for (size_t i = 0u; i < count; ++i) {
            sum += static_cast<Accumulator>(src[i]);
        }
        *dst = sum;
    }
};

enum class LookupOp { Memory, Tbl1, Tbl4, Tbl4x2 };

std::ostream& operator<<(std::ostream& out, LookupOp op) {
    switch (op) {
        case LookupOp::Memory:
            return out << "Memory";
        case LookupOp::Tbl1:
            return out << "Tbl1";
        case LookupOp::Tbl4:
            return out << "Tbl4";
        case LookupOp::Tbl4x2:
            return out << "Tbl4x2";
        default:
            assert(false && "Unknown LookupOp");
            return out;
    }
}

template <class Index, class Element>
struct BenchmarkTableLookup {
    std::vector<Index> src;
    std::vector<Element> table;
    uint32_t dst;
    LookupOp op;

    explicit BenchmarkTableLookup(size_t indices, size_t table_entries, LookupOp op)
        : src(indices), table(table_entries, Element(1)), dst(0), op(op) {
        std::default_random_engine rng(12345);
        std::uniform_int_distribution<uint32_t> dist(0, table_entries - 1);
        for (auto& index : src) {
            index = static_cast<Index>(dist(rng));
        }
        if (op == LookupOp::Tbl1) {
            assert(sizeof(Index) == 1 && "Tbl1 only supports Index of size 1 byte");
            assert(table_entries == 16 && "Tbl1 only supports 16 table entries");
        }
        if (op == LookupOp::Tbl4) {
            assert(sizeof(Index) == 1 && "Tbl4 only supports Index of size 1 byte");
            assert(table_entries == 64 && "Tbl4 only supports 64 table entries");
        }
        if (op == LookupOp::Tbl4x2) {
            assert(sizeof(Index) == 1 && "Tbl4x2 only supports Index of size 1 byte");
            assert(table_entries == 128 && "Tbl4x2 only supports 128 table entries");
        }
    }
    std::string name() const {
        std::ostringstream ss;
        ss << "TableLookup(" << table.size() << ", Idx:" << type_name<Index>()
           << ", Val:" << type_name<Element>() << ", " << op << ")";
        return ss.str();
    }
    uint64_t bytes_per_run() const {
        return src.size() * sizeof(Index) + table.size() * sizeof(Element) + sizeof(uint32_t);
    }
    void runonce() { benchmark_table_lookup(&dst, src.data(), table.data(), src.size(), op); }

    static uint32_t benchmark_table_lookup_thread(const Index* __restrict__ src,
                                                  const Element* __restrict__ table,
                                                  size_t count,
                                                  LookupOp op) {
        if (op == LookupOp::Memory) {
            uint32_t sum = 0;
            for (size_t i = 0; i < count; ++i) {
                sum += static_cast<uint32_t>(table[static_cast<size_t>(src[i])]);
            }
            return sum;
        }
        if (op == LookupOp::Tbl1) {
            auto table_ptr = reinterpret_cast<const uint8_t*>(table);
            uint8x16_t vtable = vld1q_u8(&table_ptr[0]);
            uint8x16_t vsum = vdupq_n_u8(0);
            for (size_t i = 0; i < count; i += 16) {
                uint8x16_t vindices = vld1q_u8(reinterpret_cast<const uint8_t*>(&src[i]));
                uint8x16_t vresult = vqtbl1q_u8(vtable, vindices);
                vsum = vaddq_u8(vsum, vresult);
            }
            return vaddvq_u8(vsum);
        }
        if (op == LookupOp::Tbl4) {
            auto table_ptr = reinterpret_cast<const uint8_t*>(table);
            uint8x16x4_t vtable = {vld1q_u8(&table_ptr[0]), vld1q_u8(&table_ptr[16]),
                                   vld1q_u8(&table_ptr[32]), vld1q_u8(&table_ptr[48])};
            uint8x16_t vsum = vdupq_n_u8(0);
            for (size_t i = 0; i < count; i += 16) {
                uint8x16_t vindices = vld1q_u8(reinterpret_cast<const uint8_t*>(&src[i]));
                uint8x16_t vresult = vqtbl4q_u8(vtable, vindices);
                vsum = vaddq_u8(vsum, vresult);
            }
            return vaddvq_u8(vsum);
        }
        if (op == LookupOp::Tbl4x2) {
            auto table_ptr = reinterpret_cast<const uint8_t*>(table);
            uint8x16x4_t vtable1 = {vld1q_u8(&table_ptr[0]), vld1q_u8(&table_ptr[16]),
                                    vld1q_u8(&table_ptr[32]), vld1q_u8(&table_ptr[48])};
            uint8x16x4_t vtable2 = {vld1q_u8(&table_ptr[64]), vld1q_u8(&table_ptr[80]),
                                    vld1q_u8(&table_ptr[96]), vld1q_u8(&table_ptr[112])};
            uint8x16_t vsum = vdupq_n_u8(0);
            for (size_t i = 0; i < count; i += 16) {
                uint8x16_t vindices = vld1q_u8(reinterpret_cast<const uint8_t*>(&src[i]));
                uint8x16_t vresult1 = vqtbl4q_u8(vtable1, vindices);
                uint8x16_t vresult2 = vqtbl4q_u8(vtable2, vindices);
                vsum = vaddq_u8(vsum, vresult1);
                vsum = vaddq_u8(vsum, vresult2);
            }
            return vaddvq_u8(vsum);
        }
        assert(false && "Unknown LookupOp");
        return 0;
    }

    __attribute__((noinline)) static void benchmark_table_lookup(uint32_t* __restrict__ dst,
                                                                 const Index* __restrict__ src,
                                                                 const Element* __restrict__ table,
                                                                 size_t count,
                                                                 LookupOp op) {
        assert(count % omp_get_max_threads() == 0);
        auto chunk_size = count / omp_get_max_threads();
        uint32_t sum = 0;
#pragma omp parallel
        {
            uint32_t local_sum = benchmark_table_lookup_thread(
                &src[omp_get_thread_num() * chunk_size], table, chunk_size, op);
#pragma omp atomic
            sum += local_sum;
        }
        *dst = sum;
    }
};

int main() {
    omp_set_num_threads(omp_get_max_threads());
    // omp_set_num_threads(1);

    std::cerr << "[benchmark] running on " << omp_get_max_threads() << " threads" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    auto run = [](auto&& benchmark) {
        auto measurement = run_benchmark(benchmark, 40, /*pre_runs*/ 10);
        std::cerr << std::setw(60) << benchmark.name() << " :: " << measurement << "B/s"
                  << std::endl;
    };

    const size_t data_size = 100 * 1024 * 1024;
    run(BenchmarkMemcpy<uint32_t>(data_size));
    run(BenchmarkMemcpy<uint8_t>(data_size));
    run(BenchmarkMemcpy<byte>(data_size));

    run(BenchmarkIncrement<uint32_t>(data_size));
    run(BenchmarkIncrement<uint8_t>(data_size));
    run(BenchmarkIncrement<byte>(data_size));

    run(BenchmarkReduceSum<uint32_t>(data_size));
    run(BenchmarkReduceSum<uint16_t>(data_size));
    run(BenchmarkReduceSum<uint8_t>(data_size));
    run(BenchmarkReduceSum<byte, uint32_t>(data_size));

    const size_t n_indices = 16 * 1024 * 1024;
    run(BenchmarkTableLookup<uint16_t, uint64_t>(n_indices, 1 << 16, LookupOp::Memory));
    run(BenchmarkTableLookup<uint16_t, uint32_t>(n_indices, 1 << 16, LookupOp::Memory));
    run(BenchmarkTableLookup<uint16_t, uint32_t>(n_indices, 1 << 14, LookupOp::Memory));
    run(BenchmarkTableLookup<uint16_t, uint32_t>(n_indices, 1 << 12, LookupOp::Memory));
    run(BenchmarkTableLookup<uint16_t, uint32_t>(n_indices, 1 << 8, LookupOp::Memory));
    run(BenchmarkTableLookup<uint16_t, uint32_t>(n_indices, 1 << 4, LookupOp::Memory));

    run(BenchmarkTableLookup<uint8_t, uint8_t>(n_indices, 64, LookupOp::Memory));
    run(BenchmarkTableLookup<uint8_t, uint8_t>(n_indices, 16, LookupOp::Tbl1));
    run(BenchmarkTableLookup<uint8_t, uint8_t>(n_indices, 64, LookupOp::Tbl4));
    run(BenchmarkTableLookup<uint8_t, uint8_t>(n_indices, 128, LookupOp::Tbl4x2));

    std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
    std::cerr << "[benchmark] finished in " << elapsed.count() << " seconds" << std::endl;
    return 0;
}
