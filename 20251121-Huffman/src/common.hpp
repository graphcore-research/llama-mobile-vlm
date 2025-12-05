#pragma once

#include <cassert>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <arm_neon.h>
#include <omp.h>

// `byte` is a uint8, which does not have the aliasing rules issues of std::byte or uint8_t
enum class byte : uint8_t {};

static_assert(sizeof(byte) == 1, "byte must be 1 byte");

template <class T>
std::string type_name() {
    if (std::is_same<T, uint8_t>::value) return "uint8_t";
    if (std::is_same<T, uint16_t>::value) return "uint16_t";
    if (std::is_same<T, uint32_t>::value) return "uint32_t";
    if (std::is_same<T, uint64_t>::value) return "uint64_t";
    if (std::is_same<T, int8_t>::value) return "int8_t";
    if (std::is_same<T, int16_t>::value) return "int16_t";
    if (std::is_same<T, int32_t>::value) return "int32_t";
    if (std::is_same<T, int64_t>::value) return "int64_t";
    if (std::is_same<T, std::byte>::value) return "std::byte";
    if (std::is_same<T, char8_t>::value) return "char8_t";
    if (std::is_same<T, byte>::value) return "byte";
    return typeid(T).name();
}

inline void flushCache() {
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

inline std::ostream& operator<<(std::ostream& out, const Measurement& m) {
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
Measurement run_benchmark(Benchmark&& benchmark, uint runs, uint pre_runs) {
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
