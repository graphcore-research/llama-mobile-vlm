#include <omp.h>
#include <iomanip>
#include <iostream>
#include <thread>
#include <tuple>
#include <vector>
#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace acl = arm_compute;

namespace {
void flushCache() {
    // Allocate a buffer larger than the largest cache
    const size_t cacheFlushSize = 512 * 1024 * 1024;
    static std::vector<char> cacheFlushBuffer(cacheFlushSize);
    for (auto n = 0; n < 10; ++n) {
        // Read-modify-write seems better than write-only for flushing caches
#pragma omp parallel for schedule(static)
        for (auto i = 0ull; i < cacheFlushBuffer.size(); ++i) {
            cacheFlushBuffer[i] += 1;
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    std::cerr << "[bench_acl] running benchmarks" << std::endl;
    omp_set_num_threads(std::thread::hardware_concurrency());

    const auto reps = 25u;
    const auto warmupReps = 25u;
    const auto dtype = acl::DataType::F16;

    std::vector<std::tuple<size_t, size_t, size_t, std::string>> cases = {
        // Sizes for 11B (batchSize, dIn, dOut) == (m, k, n)
        {1, 4096, 14336, "text.generate.mlp.up"},     //
        {1, 14336, 4096, "text.generate.mlp.down"},   //
        {1, 4096, 4096, "text.generate.attn.[q,o]"},  //
        {1, 4096, 1024, "text.generate.attn.[k,v]"},  //
        {1, 4096, 128256, "text.generate.predict"},   //
        //
        {128, 4096, 14336, "text.prefill.mlp.up"},     //
        {128, 14336, 4096, "text.prefill.mlp.down"},   //
        {128, 4096, 4096, "text.prefill.attn.[q,o]"},  //
        {128, 4096, 1024, "text.prefill.attn.[k,v]"},  //
        //
        {1601, 1280, 5120, "vision.mlp.up"},          //
        {1601, 5120, 1280, "vision.mlp.down"},        //
        {1601, 1280, 1280, "vision.attn.[q,k,v,o]"},  //
    };

    for (const auto& [m, k, n, name] : cases) {
        acl::Tensor a, b, output;
        a.allocator()->init(acl::TensorInfo(acl::TensorShape{k, m}, 1, dtype));
        b.allocator()->init(acl::TensorInfo(acl::TensorShape{n, k}, 1, dtype));
        output.allocator()->init(acl::TensorInfo(acl::TensorShape{n, m}, 1, dtype));
        a.info()->set_are_values_constant(false);
        b.info()->set_are_values_constant(false);
        output.info()->set_are_values_constant(false);
        const acl::MatMulInfo info;
        const acl::CpuMatMulSettings settings;

        auto status = acl::NEMatMul::validate(a.info(), b.info(), output.info(), info, settings);
        if (status.error_code() != acl::ErrorCode::OK) {
            std::cerr << "[bench_acl] validation error code: "
                      << static_cast<int>(status.error_code())
                      << ", description: " << status.error_description() << std::endl;
            return 1;
        }

        a.allocator()->allocate();
        b.allocator()->allocate();
        output.allocator()->allocate();

        acl::NEMatMul matmul;
        matmul.configure(&a, &b, &output, info, settings);
        std::vector<double> times;
        for (size_t i = 0; i < warmupReps + reps; ++i) {
            flushCache();
            auto start = std::chrono::high_resolution_clock::now();
            matmul.run();
            auto end = std::chrono::high_resolution_clock::now();
            times.push_back(
                std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
        }
        // Report
        auto averageTime = std::accumulate(times.begin() + warmupReps, times.end(), 0.0) /
                           (times.end() - times.begin() - warmupReps);
        auto macs = (m * k * n) / averageTime;
        auto bps = (m * k + k * n + n * m) * acl::data_size_from_type(dtype) / averageTime;
        std::cout << std::right << std::setw(25) << name << ": " << averageTime * 1000 << " ms, "
                  << macs / 1e9 << " GMAC/s, " << bps / double(1u << 30) << " GiB/s" << "    (" << m
                  << ", " << k << ", " << n << ")" << std::endl;
    }
    return 0;
}
