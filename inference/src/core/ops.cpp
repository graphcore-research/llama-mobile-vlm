// Copyright (c) 2026 Graphcore Ltd. All rights reserved.

#include "ops.hpp"

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numbers>
#include <tuple>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif  // __ARM_NEON

namespace squash::ops {

namespace {

int8_t _decode_s3d8(const uint8_t pack, const int8_t* __restrict__ lut, const uint vIdx) {
    switch (vIdx) {
        case 0:
            return lut[0 * 64u + (pack & 0x3Fu)];
        case 1:
            return lut[1 * 64u + ((pack >> 1) & 0x3Fu)];
        case 2:
            return lut[2 * 64u + ((pack & 0x3Fu) ^ (pack >> 7))];
        default:
            return 0;
    }
}

}  // namespace

// -----------------------------------------------------------------------------------------------
// Data copy/convert

void copy(const float* src, uint n, float* dest) {
#pragma omp parallel for
    for (auto i = 0u; i < n; ++i) {
        dest[i] = src[i];
    }
}

void copy(const bf16* src, uint n, bf16* dest) {
#pragma omp parallel for
    for (auto i = 0u; i < n; ++i) {
        dest[i] = src[i];
    }
}

void copy(const int8_t* src_data,
          const bf16* src_scale,
          uint dN,
          uint dK,
          int8_t* dest_data,
          bf16* dest_scale) {
#pragma omp parallel for
    for (auto n = 0u; n < dN; ++n) {
        std::copy_n(src_data + n * dK, dK, dest_data + n * dK);
        dest_scale[n] = src_scale[n];
    }
}

void copyStrided(const bf16* src, uint n, uint d, uint sSrc, uint sDest, bf16* dest) {
    for (uint i = 0; i < n; ++i) {
        for (uint j = 0; j < d; ++j) {
            dest[i * sDest + j] = src[i * sSrc + j];
        }
    }
}

void castFloat(const bf16* in, float* out, uint n) {
#pragma omp parallel for
    for (auto i = 0u; i < n; ++i) {
        out[i] = float(in[i]);
    }
}

void castBf16(const float* in, bf16* out, uint n) {
#pragma omp parallel for
    for (auto i = 0u; i < n; ++i) {
        out[i] = bf16(in[i]);
    }
}

void castChannelInt8(const bf16* in, uint dN, uint dK, int8_t* out_data, bf16* out_scale) {
#pragma omp parallel for
    for (auto n = 0u; n < dN; ++n) {
        auto amax = 0.0f;
        for (auto k = 0u; k < dK; ++k) {
            amax = std::max(amax, std::abs(float(in[n * dK + k])));
        }
        amax = (amax == 0) ? 1.0f : amax;

        auto scale = out_scale[n] = bf16(amax / 127.0f);
        for (auto k = 0u; k < dK; ++k) {
            auto q = std::nearbyint(float(in[n * dK + k]) / float(scale));
            out_data[n * dK + k] = static_cast<int8_t>(std::clamp(q, -127.0f, 127.0f));
        }
    }
}

namespace {

#if defined(__ARM_NEON)

template <uint BlockK>
void _cast_chunk_s3d8(const uint8_t* __restrict__ in,
                      const int8_t* __restrict__ lut,
                      const uint dK,
                      const uint stopN,
                      int8_t* __restrict__ out,
                      const uint outStride) {
    constexpr auto RowsPerPack = 3u;
    static_assert(BlockK % 16 == 0, "BlockK must be a multiple of 16");

    int8x16x4_t luts[RowsPerPack];
#pragma unroll
    for (auto row = 0u; row < RowsPerPack; ++row) {
        luts[row].val[0] = vld1q_s8(&lut[row * 64u + 0u]);
        luts[row].val[1] = vld1q_s8(&lut[row * 64u + 16u]);
        luts[row].val[2] = vld1q_s8(&lut[row * 64u + 32u]);
        luts[row].val[3] = vld1q_s8(&lut[row * 64u + 48u]);
    }

    const auto kStop = (dK / BlockK) * BlockK;
    for (auto k = 0u; k < kStop; k += BlockK) {
        uint8x16_t idx[RowsPerPack][BlockK / 16];
#pragma unroll
        for (auto iK = 0u; iK < BlockK / 16; ++iK) {
            uint8x16_t packed = vld1q_u8(&in[k + iK * 16]);
            idx[0][iK] = vandq_u8(packed, vdupq_n_u8(0x3Fu));
            idx[1][iK] = vandq_u8(vshrq_n_u8(packed, 1), vdupq_n_u8(0x3Fu));
            idx[2][iK] = veorq_u8(idx[0][iK], vshrq_n_u8(packed, 7));
        }
        if (stopN > 0) {
#pragma unroll
            for (auto iK = 0u; iK < BlockK / 16; ++iK) {
                vst1q_s8(&out[0u * outStride + k + iK * 16], vqtbl4q_s8(luts[0], idx[0][iK]));
            }
        }
        if (stopN > 1) {
#pragma unroll
            for (auto iK = 0u; iK < BlockK / 16; ++iK) {
                vst1q_s8(&out[1u * outStride + k + iK * 16], vqtbl4q_s8(luts[1], idx[1][iK]));
            }
        }
        if (stopN > 2) {
#pragma unroll
            for (auto iK = 0u; iK < BlockK / 16; ++iK) {
                vst1q_s8(&out[2u * outStride + k + iK * 16], vqtbl4q_s8(luts[2], idx[2][iK]));
            }
        }
    }

    for (auto k = kStop; k < dK; ++k) {
        if (stopN > 0) {
            out[0u * outStride + k] = _decode_s3d8(in[k], lut, 0u);
        }
        if (stopN > 1) {
            out[1u * outStride + k] = _decode_s3d8(in[k], lut, 1u);
        }
        if (stopN > 2) {
            out[2u * outStride + k] = _decode_s3d8(in[k], lut, 2u);
        }
    }
}

void _castChannelInt8_s3d8(const uint8_t* __restrict__ in_data,
                           const int8_t* __restrict__ in_lut,
                           const bf16* __restrict__ in_scale,
                           const uint dN,
                           const uint dK,
                           int8_t* __restrict__ out_data,
                           bf16* __restrict__ out_scale) {
    constexpr auto RowsPerPack = 3u;
    constexpr auto BlockK = 128u;
    const auto packedRows = (dN + RowsPerPack - 1) / RowsPerPack;

#pragma omp parallel for
    for (auto p = 0u; p < packedRows; ++p) {
        const auto n0 = p * RowsPerPack;
        const auto stopN = std::min(RowsPerPack, dN - n0);
        _cast_chunk_s3d8<BlockK>(&in_data[p * dK], in_lut, dK, stopN, &out_data[n0 * dK], dK);
        for (auto n = 0u; n < stopN; ++n) {
            out_scale[n0 + n] = in_scale[n0 + n];
        }
    }
}

#else  // !__ARM_NEON

void _castChannelInt8_s3d8(const uint8_t* __restrict__ in_data,
                           const int8_t* __restrict__ in_lut,
                           const bf16* __restrict__ in_scale,
                           const uint dN,
                           const uint dK,
                           int8_t* __restrict__ out_data,
                           bf16* __restrict__ out_scale) {
#pragma omp parallel for
    for (auto n = 0u; n < dN; ++n) {
        out_scale[n] = in_scale[n];
        for (auto k = 0u; k < dK; ++k) {
            out_data[n * dK + k] = _decode_s3d8(in_data[(n / 3u) * dK + k], in_lut, n % 3u);
        }
    }
}

#endif  // __ARM_NEON

}  // namespace

void castChannelInt8(const uint8_t* __restrict__ in_data,
                     const int8_t* __restrict__ in_lut,
                     const bf16* __restrict__ in_scale,
                     const uint dN,
                     const uint dK,
                     int8_t* __restrict__ out_data,
                     bf16* __restrict__ out_scale) {
    _castChannelInt8_s3d8(in_data, in_lut, in_scale, dN, dK, out_data, out_scale);
}

// -----------------------------------------------------------------------------------------------
// Matmuls

namespace {

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) && \
    defined(__ARM_FEATURE_MATMUL_INT8) && defined(__ARM_FEATURE_DOTPROD)

// -----------------------------------------------------------------------------------------------
// ARM BF16 matmul

template <uint BlockN, uint BlockK>
void _mv_chunk_bf16(const __bf16* __restrict__ a,
                    const __bf16* __restrict__ b,
                    const uint dK,
                    __bf16* __restrict__ out) {
    static_assert(BlockK % 8 == 0, "BlockK must be a multiple of 8");

    float32x4_t accs[BlockN * (BlockK / 8)];
#pragma unroll
    for (auto i = 0u; i < BlockN * (BlockK / 8); ++i) {
        accs[i] = vmovq_n_f32(0.0f);
    }

    // Main loop, process [BlockN, BlockK] elements of `b` per iteration.
    const auto kStop = (dK / BlockK) * BlockK;
    for (auto k = 0u; k < kStop; k += BlockK) {
#pragma unroll
        for (auto iK = 0u; iK < BlockK / 8; ++iK) {
            bfloat16x8_t ak = vld1q_bf16(&a[k + iK * 8]);
#pragma unroll
            for (auto n = 0u; n < BlockN; ++n) {
                bfloat16x8_t bk = vld1q_bf16(&b[n * dK + k + iK * 8]);
                accs[n * (BlockK / 8) + iK] = vbfdotq_f32(accs[n * (BlockK / 8) + iK], ak, bk);
            }
        }
    }

#pragma unroll
    for (auto n = 0u; n < BlockN; ++n) {
        float32x4_t& acc_n = accs[n * (BlockK / 8)];
#pragma unroll
        for (auto i = 1u; i < BlockK / 8; ++i) {
            acc_n = vaddq_f32(acc_n, accs[n * (BlockK / 8) + i]);
        }
        float result = vaddvq_f32(acc_n);
        for (auto k = kStop; k < dK; ++k) {
            result += vcvtah_f32_bf16(a[k]) * vcvtah_f32_bf16(b[n * dK + k]);
        }
        out[n] = vcvth_bf16_f32(result);
    }
}

float _dot_bf16(const bf16* __restrict__ a, const bf16* __restrict__ b, const uint dK) {
    bf16 result;
    auto a_ = reinterpret_cast<const __bf16*>(a);
    auto b_ = reinterpret_cast<const __bf16*>(b);
    auto out_ = reinterpret_cast<__bf16*>(&result);
    // Pick a block that divides dK, so no elements fall into the scalar tail (e.g. dK=80)
    if (dK % 64 == 0) {
        _mv_chunk_bf16<1, 64>(a_, b_, dK, out_);
    } else if (dK % 16 == 0) {
        _mv_chunk_bf16<1, 16>(a_, b_, dK, out_);
    } else {
        _mv_chunk_bf16<1, 8>(a_, b_, dK, out_);
    }
    return float(result);
}

template <uint BlockM, uint BlockN>
void _matmulT_chunk_bfmmla(const __bf16* __restrict__ a,
                           const __bf16* __restrict__ b,
                           const uint dK,
                           const uint dN,
                           __bf16* __restrict__ out) {
    // Note: we expect all BlockM, BlockN loops to be unrolled
    static_assert(BlockM % 2 == 0 && BlockN % 2 == 0, "BlockM and BlockN must be even");

    // Each accumulator holds a 2x2 result, accumulated over the full `k` dimension
    float32x4_t accs[(BlockM / 2) * (BlockN / 2)];
#pragma unroll
    for (auto i = 0u; i < (BlockM / 2) * (BlockN / 2); ++i) {
        accs[i] = vmovq_n_f32(0.0f);
    }

    // Main loop, process `(m, k, n) = (BlockM, 8, BlockN)` elements per iteration
    const auto kStop = (dK / 8) * 8;
    for (auto k = 0u; k < kStop; k += 8) {
        bfloat16x8_t aa[BlockM], bb[BlockN];
#pragma unroll
        for (auto m = 0u; m < BlockM; ++m) {
            aa[m] = vld1q_bf16(&a[m * dK + k]);
        }
#pragma unroll
        for (auto n = 0u; n < BlockN; ++n) {
            bb[n] = vld1q_bf16(&b[n * dK + k]);
        }
#pragma unroll
        for (auto m = 0u; m < (BlockM / 2); ++m) {
#pragma unroll
            for (auto n = 0u; n < (BlockN / 2); ++n) {
                float32x4_t& acc = accs[m * (BlockN / 2) + n];
                acc = vbfmmlaq_f32(
                    acc, vcombine_bf16(vget_low_bf16(aa[2 * m]), vget_low_bf16(aa[2 * m + 1])),
                    vcombine_bf16(vget_low_bf16(bb[2 * n]), vget_low_bf16(bb[2 * n + 1])));
                acc = vbfmmlaq_f32(
                    acc, vcombine_bf16(vget_high_bf16(aa[2 * m]), vget_high_bf16(aa[2 * m + 1])),
                    vcombine_bf16(vget_high_bf16(bb[2 * n]), vget_high_bf16(bb[2 * n + 1])));
            }
        }
    }

    // Handle remainder when dK is not a multiple of 8
    for (auto k = kStop; k < dK; ++k) {
#pragma unroll
        for (auto m = 0u; m < (BlockM / 2); ++m) {
#pragma unroll
            for (auto n = 0u; n < (BlockN / 2); ++n) {
                float32x4_t& acc = accs[m * (BlockN / 2) + n];
                float a0 = vcvtah_f32_bf16(a[(2 * m + 0) * dK + k]);
                float a1 = vcvtah_f32_bf16(a[(2 * m + 1) * dK + k]);
                float b0 = vcvtah_f32_bf16(b[(2 * n + 0) * dK + k]);
                float b1 = vcvtah_f32_bf16(b[(2 * n + 1) * dK + k]);
                acc = vmlaq_f32(acc, float32x4_t{a0, a0, a1, a1}, float32x4_t{b0, b1, b0, b1});
            }
        }
    }

    // Store out results, a BlockM x BlockN matrix
#pragma unroll
    for (auto m = 0u; m < (BlockM / 2); ++m) {
#pragma unroll
        for (auto n = 0u; n < (BlockN / 2); ++n) {
            bfloat16x4_t acc_bf16 = vcvt_bf16_f32(accs[m * (BlockN / 2) + n]);
            vst1_lane_bf16(&out[(2 * m + 0) * dN + (2 * n + 0)], acc_bf16, 0);
            vst1_lane_bf16(&out[(2 * m + 0) * dN + (2 * n + 1)], acc_bf16, 1);
            vst1_lane_bf16(&out[(2 * m + 1) * dN + (2 * n + 0)], acc_bf16, 2);
            vst1_lane_bf16(&out[(2 * m + 1) * dN + (2 * n + 1)], acc_bf16, 3);
        }
    }
}

void _matmulT_bf16(const bf16* __restrict__ a_,  // {dM, dK}
                   const bf16* __restrict__ b_,  // {dN, dK}
                   const uint dM,
                   const uint dK,
                   const uint dN,
                   bf16* __restrict__ out_) {  // {dM, dN}
    auto a = reinterpret_cast<const __bf16*>(a_);
    auto b = reinterpret_cast<const __bf16*>(b_);
    auto out = reinterpret_cast<__bf16*>(out_);

    // Matrix-vector cases
    if (dM == 1) {
        constexpr auto BN = 4;
        constexpr auto BK = 8;
        const auto nStop = (dN / BN) * BN;
#pragma omp parallel for
        for (auto n = 0u; n < nStop; n += BN) {
            _mv_chunk_bf16<BN, BK>(a, &b[n * dK], dK, &out[n]);
        }
        for (auto n = nStop; n < dN; ++n) {
            _mv_chunk_bf16<1, 64>(a, &b[n * dK], dK, &out[n]);
        }
        return;
    }
    if (dN == 1) {
        // Cannot transpose & use BN since it would require a strided output write
#pragma omp parallel for
        for (auto m = 0u; m < dM; ++m) {
            _mv_chunk_bf16<1, 64>(&a[m * dK], b, dK, &out[m]);
        }
        return;
    }

    constexpr auto G0 = 16u;  // block size
    constexpr auto G1 = 8u;   // inner block size

    const auto blocksM = (dM + G0 - 1) / G0;
    const auto blocksN = (dN + G0 - 1) / G0;

#pragma omp parallel for
    for (auto i0 = 0u; i0 < blocksM * blocksN; ++i0) {
        auto m0 = G0 * (i0 / blocksN);
        auto m1 = std::min(m0 + G0, dM);
        auto n0 = G0 * (i0 % blocksN);
        auto n1 = std::min(n0 + G0, dN);

        // Main loop
        auto mStop = (m1 / G1) * G1;
        auto nStop = (n1 / G1) * G1;
        for (auto n = n0; n < nStop; n += G1) {
            for (auto m = m0; m < mStop; m += G1) {
                _matmulT_chunk_bfmmla<G1, G1>(&a[m * dK], &b[n * dK], dK, dN, &out[m * dN + n]);
            }
        }
        // Handle remainder when dN is not a multiple of G1, `out[m0:m1, nStop:n1]`
        for (auto n = nStop; n < n1; ++n) {
            for (auto m = m0; m < m1; ++m) {
                _mv_chunk_bf16<1, 64>(&a[m * dK], &b[n * dK], dK, &out[m * dN + n]);
            }
        }
        // Handle remainder when dM is not a multiple of G1, `out[mStop:m1, n0:nStop]`
        // (note: excludes the bottom-right corner which is handled in the loop above)
        for (auto m = mStop; m < m1; ++m) {
            for (auto n = n0; n < nStop; ++n) {
                _mv_chunk_bf16<1, 64>(&a[m * dK], &b[n * dK], dK, &out[m * dN + n]);
            }
        }
    }
}

// -----------------------------------------------------------------------------------------------
// ARM INT8 matmul

template <uint BlockN, uint BlockK>
void _mv_chunk_int8(const int8_t* __restrict__ a,
                    const __bf16 aScale,
                    const int8_t* __restrict__ b,
                    const __bf16* __restrict__ bScale,
                    const uint dK,
                    __bf16* __restrict__ out) {
    static_assert(BlockK % 16 == 0, "BlockK must be a multiple of 16");

    int32x4_t accs[BlockN * (BlockK / 16)];
#pragma unroll
    for (auto i = 0u; i < BlockN * (BlockK / 16); ++i) {
        accs[i] = vmovq_n_s32(0);
    }

    // Main loop, process [BlockN, BlockK] elements of `b` per iteration.
    const auto kStop = (dK / BlockK) * BlockK;
    for (auto k = 0u; k < kStop; k += BlockK) {
#pragma unroll
        for (auto iK = 0u; iK < BlockK / 16; ++iK) {
            int8x16_t ak = vld1q_s8(&a[k + iK * 16]);
#pragma unroll
            for (auto n = 0u; n < BlockN; ++n) {
                int8x16_t bk = vld1q_s8(&b[n * dK + k + iK * 16]);
                accs[n * (BlockK / 16) + iK] = vdotq_s32(accs[n * (BlockK / 16) + iK], ak, bk);
            }
        }
    }

#pragma unroll
    for (auto n = 0u; n < BlockN; ++n) {
        int32x4_t& acc_n = accs[n * (BlockK / 16)];
#pragma unroll
        for (auto i = 1u; i < BlockK / 16; ++i) {
            acc_n = vaddq_s32(acc_n, accs[n * (BlockK / 16) + i]);
        }
        int32_t result = vaddvq_s32(acc_n);
        for (auto k = kStop; k < dK; ++k) {
            result += int32_t(a[k]) * int32_t(b[n * dK + k]);
        }
        out[n] =
            vcvth_bf16_f32(float(result) * vcvtah_f32_bf16(aScale) * vcvtah_f32_bf16(bScale[n]));
    }
}

template <uint BlockM, uint BlockN>
void _matmulT_chunk_smmla(const int8_t* __restrict__ a,
                          const __bf16* __restrict__ aScale,
                          const int8_t* __restrict__ b,
                          const __bf16* __restrict__ bScale,
                          const uint dK,
                          const uint dN,
                          __bf16* __restrict__ out) {
    // Note: we expect all BlockM, BlockN loops to be unrolled
    static_assert(BlockM % 2 == 0 && BlockN % 2 == 0, "BlockM and BlockN must be even");

    // Each accumulator holds a 2x2 result, accumulated over the full `k` dimension
    int32x4_t accs[(BlockM / 2) * (BlockN / 2)];
#pragma unroll
    for (auto i = 0u; i < (BlockM / 2) * (BlockN / 2); ++i) {
        accs[i] = vmovq_n_s32(0);
    }

    // Main loop, process `(m, k, n) = (BlockM, 16, BlockN)` elements per iteration
    const auto kStop = (dK / 16) * 16;
    for (auto k = 0u; k < kStop; k += 16) {
        int8x16_t aa0[BlockM / 2], aa1[BlockM / 2], bb0[BlockN / 2], bb1[BlockN / 2];
#pragma unroll
        for (auto m = 0u; m < (BlockM / 2); ++m) {
            int8x16_t am0 = vld1q_s8(&a[(2 * m + 0) * dK + k]);
            int8x16_t am1 = vld1q_s8(&a[(2 * m + 1) * dK + k]);
            aa0[m] = vcombine_s8(vget_low_s8(am0), vget_low_s8(am1));
            aa1[m] = vcombine_s8(vget_high_s8(am0), vget_high_s8(am1));
        }
#pragma unroll
        for (auto n = 0u; n < (BlockN / 2); ++n) {
            int8x16_t bn0 = vld1q_s8(&b[(2 * n + 0) * dK + k]);
            int8x16_t bn1 = vld1q_s8(&b[(2 * n + 1) * dK + k]);
            bb0[n] = vcombine_s8(vget_low_s8(bn0), vget_low_s8(bn1));
            bb1[n] = vcombine_s8(vget_high_s8(bn0), vget_high_s8(bn1));
        }
#pragma unroll
        for (auto m = 0u; m < (BlockM / 2); ++m) {
#pragma unroll
            for (auto n = 0u; n < (BlockN / 2); ++n) {
                int32x4_t& acc = accs[m * (BlockN / 2) + n];
                acc = vmmlaq_s32(acc, aa0[m], bb0[n]);
                acc = vmmlaq_s32(acc, aa1[m], bb1[n]);
            }
        }
    }

    // Handle remainder when dK is not a multiple of 16
    for (auto k = kStop; k < dK; ++k) {
#pragma unroll
        for (auto m = 0u; m < (BlockM / 2); ++m) {
#pragma unroll
            for (auto n = 0u; n < (BlockN / 2); ++n) {
                int32x4_t& acc = accs[m * (BlockN / 2) + n];
                int32_t a0 = int32_t(a[(2 * m + 0) * dK + k]);
                int32_t a1 = int32_t(a[(2 * m + 1) * dK + k]);
                int32_t b0 = int32_t(b[(2 * n + 0) * dK + k]);
                int32_t b1 = int32_t(b[(2 * n + 1) * dK + k]);
                acc = vaddq_s32(acc, int32x4_t{a0 * b0, a0 * b1, a1 * b0, a1 * b1});
            }
        }
    }

    // Store out results, a BlockM x BlockN matrix
#pragma unroll
    for (auto m = 0u; m < (BlockM / 2); ++m) {
        float aScale0 = vcvtah_f32_bf16(aScale[2 * m + 0]);
        float aScale1 = vcvtah_f32_bf16(aScale[2 * m + 1]);
#pragma unroll
        for (auto n = 0u; n < (BlockN / 2); ++n) {
            int32_t acc[4];
            vst1q_s32(acc, accs[m * (BlockN / 2) + n]);
            float bScale0 = vcvtah_f32_bf16(bScale[2 * n + 0]);
            float bScale1 = vcvtah_f32_bf16(bScale[2 * n + 1]);
            out[(2 * m + 0) * dN + (2 * n + 0)] = vcvth_bf16_f32(float(acc[0]) * aScale0 * bScale0);
            out[(2 * m + 0) * dN + (2 * n + 1)] = vcvth_bf16_f32(float(acc[1]) * aScale0 * bScale1);
            out[(2 * m + 1) * dN + (2 * n + 0)] = vcvth_bf16_f32(float(acc[2]) * aScale1 * bScale0);
            out[(2 * m + 1) * dN + (2 * n + 1)] = vcvth_bf16_f32(float(acc[3]) * aScale1 * bScale1);
        }
    }
}

void _matmulT_int8(const int8_t* __restrict__ lhs,
                   const bf16* __restrict__ lhsScale_,
                   const int8_t* __restrict__ rhs,
                   const bf16* __restrict__ rhsScale_,
                   const uint dM,
                   const uint dK,
                   const uint dN,
                   bf16* __restrict__ out_) {
    auto lhsScale = reinterpret_cast<const __bf16*>(lhsScale_);
    auto rhsScale = reinterpret_cast<const __bf16*>(rhsScale_);
    auto out = reinterpret_cast<__bf16*>(out_);

    if (dM == 1) {
        constexpr auto BN = 4u;
        constexpr auto BK = 16u;
        const auto nStop = (dN / BN) * BN;
#pragma omp parallel for
        for (auto n = 0u; n < nStop; n += BN) {
            _mv_chunk_int8<BN, BK>(lhs, lhsScale[0], &rhs[n * dK], &rhsScale[n], dK, &out[n]);
        }
        for (auto n = nStop; n < dN; ++n) {
            _mv_chunk_int8<1, 64>(lhs, lhsScale[0], &rhs[n * dK], &rhsScale[n], dK, &out[n]);
        }
        return;
    }
    if (dN == 1) {
#pragma omp parallel for
        for (auto m = 0u; m < dM; ++m) {
            _mv_chunk_int8<1, 64>(&lhs[m * dK], lhsScale[m], rhs, rhsScale, dK, &out[m]);
        }
        return;
    }

    constexpr auto G0 = 16u;
    constexpr auto G1 = 8u;

    const auto blocksM = (dM + G0 - 1) / G0;
    const auto blocksN = (dN + G0 - 1) / G0;

#pragma omp parallel for
    for (auto i0 = 0u; i0 < blocksM * blocksN; ++i0) {
        auto m0 = G0 * (i0 / blocksN);
        auto m1 = std::min(m0 + G0, dM);
        auto n0 = G0 * (i0 % blocksN);
        auto n1 = std::min(n0 + G0, dN);

        // Main loop
        auto mStop = m0 + ((m1 - m0) / G1) * G1;
        auto nStop = n0 + ((n1 - n0) / G1) * G1;
        for (auto n = n0; n < nStop; n += G1) {
            for (auto m = m0; m < mStop; m += G1) {
                _matmulT_chunk_smmla<G1, G1>(&lhs[m * dK], &lhsScale[m], &rhs[n * dK], &rhsScale[n],
                                             dK, dN, &out[m * dN + n]);
            }
        }
        // Handle remainder when dN is not a multiple of G1, `out[m0:m1, nStop:n1]`
        for (auto n = nStop; n < n1; ++n) {
            for (auto m = m0; m < m1; ++m) {
                _mv_chunk_int8<1, 64>(&lhs[m * dK], lhsScale[m], &rhs[n * dK], &rhsScale[n], dK,
                                      &out[m * dN + n]);
            }
        }
        // Handle remainder when dM is not a multiple of G1, `out[mStop:m1, n0:nStop]`
        // (note: excludes the bottom-right corner which is handled in the loop above)
        for (auto m = mStop; m < m1; ++m) {
            for (auto n = n0; n < nStop; ++n) {
                _mv_chunk_int8<1, 64>(&lhs[m * dK], lhsScale[m], &rhs[n * dK], &rhsScale[n], dK,
                                      &out[m * dN + n]);
            }
        }
    }
}

// -----------------------------------------------------------------------------------------------
// ARM S3D8 matmul

template <uint BlockN, uint BlockK>
void _mv_chunk_s3d8(const int8_t* __restrict__ a,
                    const __bf16 aScale,
                    const uint8_t* __restrict__ b,
                    const int8_t* __restrict__ bLut,
                    const __bf16* __restrict__ bScale,
                    const uint dK,
                    const uint stopN,
                    __bf16* __restrict__ out) {
    constexpr auto RowsPerPack = 3u;
    static_assert(BlockN % RowsPerPack == 0, "BlockN must be divisible by RowsPerPack");
    static_assert(BlockK % 16 == 0, "BlockK must be a multiple of 16");

    int32x4_t accs[BlockN * (BlockK / 16)];
#pragma unroll
    for (auto i = 0u; i < BlockN * (BlockK / 16); ++i) {
        accs[i] = vmovq_n_s32(0);
    }

    int8x16x4_t luts[RowsPerPack];
#pragma unroll
    for (auto row = 0u; row < RowsPerPack; ++row) {
        luts[row].val[0] = vld1q_s8(&bLut[row * 64u + 0u]);
        luts[row].val[1] = vld1q_s8(&bLut[row * 64u + 16u]);
        luts[row].val[2] = vld1q_s8(&bLut[row * 64u + 32u]);
        luts[row].val[3] = vld1q_s8(&bLut[row * 64u + 48u]);
    }

    const auto kStop = (dK / BlockK) * BlockK;
    for (auto k = 0u; k < kStop; k += BlockK) {
#pragma unroll
        for (auto iK = 0u; iK < BlockK / 16; ++iK) {
            int8x16_t ai = vld1q_s8(&a[k + iK * 16]);
#pragma unroll
            for (auto n = 0u; n < BlockN / RowsPerPack; ++n) {
                uint8x16_t packed = vld1q_u8(&b[n * dK + k + iK * 16]);
                uint8x16_t idx[RowsPerPack];
                idx[0] = vandq_u8(packed, vdupq_n_u8(0x3Fu));
                idx[1] = vandq_u8(vshrq_n_u8(packed, 1), vdupq_n_u8(0x3Fu));
                idx[2] = veorq_u8(idx[0], vshrq_n_u8(packed, 7));
#pragma unroll
                for (auto iP = 0u; iP < RowsPerPack; ++iP) {
                    int32x4_t& acc = accs[(n * RowsPerPack + iP) * (BlockK / 16) + iK];
                    acc = vdotq_s32(acc, ai, vqtbl4q_s8(luts[iP], idx[iP]));
                }
            }
        }
    }

#pragma unroll
    for (auto n = 0u; n < BlockN; ++n) {
        if (n < stopN) {
            int32x4_t& acc_n = accs[n * (BlockK / 16)];
#pragma unroll
            for (auto i = 1u; i < BlockK / 16; ++i) {
                acc_n = vaddq_s32(acc_n, accs[n * (BlockK / 16) + i]);
            }
            int32_t result = vaddvq_s32(acc_n);
            for (auto k = kStop; k < dK; ++k) {
                result += int32_t(a[k]) * int32_t(_decode_s3d8(b[(n / RowsPerPack) * dK + k], bLut,
                                                               n % RowsPerPack));
            }
            out[n] = vcvth_bf16_f32(float(result) * vcvtah_f32_bf16(aScale) *
                                    vcvtah_f32_bf16(bScale[n]));
        }
    }
}

template <uint BlockM, uint BlockN>
void _matmulT_chunk_s3d8_smmla(const int8_t* __restrict__ a,
                               const __bf16* __restrict__ aScale,
                               const uint8_t* __restrict__ b,
                               const int8_t* __restrict__ bLut,
                               const __bf16* __restrict__ bScale,
                               const uint dK,
                               const uint dN,
                               __bf16* __restrict__ out) {
    constexpr auto RowsPerPack = 3u;
    static_assert(BlockM % 2 == 0, "BlockM must be even");
    static_assert(BlockN % 6 == 0, "BlockN must be divisible by 6");

    int32x4_t accs[(BlockM / 2) * (BlockN / 2)];
#pragma unroll
    for (auto i = 0u; i < (BlockM / 2) * (BlockN / 2); ++i) {
        accs[i] = vmovq_n_s32(0);
    }

    int8x16x4_t luts[RowsPerPack];
#pragma unroll
    for (auto row = 0u; row < RowsPerPack; ++row) {
        luts[row].val[0] = vld1q_s8(&bLut[row * 64u + 0u]);
        luts[row].val[1] = vld1q_s8(&bLut[row * 64u + 16u]);
        luts[row].val[2] = vld1q_s8(&bLut[row * 64u + 32u]);
        luts[row].val[3] = vld1q_s8(&bLut[row * 64u + 48u]);
    }

    // Main loop, process `(m, k, n) = (BlockM, 16, BlockN)` elements per iteration.
    const auto kStop = (dK / 16) * 16;
    for (auto k = 0u; k < kStop; k += 16) {
        int8x16_t aa[BlockM], bb[BlockN];
#pragma unroll
        for (auto m = 0u; m < BlockM; ++m) {
            aa[m] = vld1q_s8(&a[m * dK + k]);
        }
#pragma unroll
        for (auto p = 0u; p < BlockN / RowsPerPack; ++p) {
            uint8x16_t packed = vld1q_u8(&b[p * dK + k]);
            uint8x16_t idx0 = vandq_u8(packed, vdupq_n_u8(0x3Fu));
            uint8x16_t idx1 = vandq_u8(vshrq_n_u8(packed, 1), vdupq_n_u8(0x3Fu));
            uint8x16_t idx2 = veorq_u8(idx0, vshrq_n_u8(packed, 7));
            bb[p * RowsPerPack + 0u] = vqtbl4q_s8(luts[0], idx0);
            bb[p * RowsPerPack + 1u] = vqtbl4q_s8(luts[1], idx1);
            bb[p * RowsPerPack + 2u] = vqtbl4q_s8(luts[2], idx2);
        }
#pragma unroll
        for (auto m = 0u; m < (BlockM / 2); ++m) {
            int8x16_t aa0 = vcombine_s8(vget_low_s8(aa[2 * m + 0]), vget_low_s8(aa[2 * m + 1]));
            int8x16_t aa1 = vcombine_s8(vget_high_s8(aa[2 * m + 0]), vget_high_s8(aa[2 * m + 1]));
#pragma unroll
            for (auto n = 0u; n < (BlockN / 2); ++n) {
                int8x16_t bb0 = vcombine_s8(vget_low_s8(bb[2 * n + 0]), vget_low_s8(bb[2 * n + 1]));
                int8x16_t bb1 =
                    vcombine_s8(vget_high_s8(bb[2 * n + 0]), vget_high_s8(bb[2 * n + 1]));
                int32x4_t& acc = accs[m * (BlockN / 2) + n];
                acc = vmmlaq_s32(acc, aa0, bb0);
                acc = vmmlaq_s32(acc, aa1, bb1);
            }
        }
    }

    // Handle remainder when dK is not a multiple of 16
    for (auto k = kStop; k < dK; ++k) {
#pragma unroll
        for (auto m = 0u; m < (BlockM / 2); ++m) {
#pragma unroll
            for (auto n = 0u; n < (BlockN / 2); ++n) {
                int32x4_t& acc = accs[m * (BlockN / 2) + n];
                int32_t a0 = int32_t(a[(2 * m + 0) * dK + k]);
                int32_t a1 = int32_t(a[(2 * m + 1) * dK + k]);
                int32_t b0 = int32_t(_decode_s3d8(b[((2 * n + 0) / RowsPerPack) * dK + k], bLut,
                                                  (2 * n + 0) % RowsPerPack));
                int32_t b1 = int32_t(_decode_s3d8(b[((2 * n + 1) / RowsPerPack) * dK + k], bLut,
                                                  (2 * n + 1) % RowsPerPack));
                acc = vaddq_s32(acc, int32x4_t{a0 * b0, a0 * b1, a1 * b0, a1 * b1});
            }
        }
    }

    // Reduce and store out results, a BlockM x BlockN matrix
#pragma unroll
    for (auto m = 0u; m < (BlockM / 2); ++m) {
        float aScale0 = vcvtah_f32_bf16(aScale[2 * m + 0]);
        float aScale1 = vcvtah_f32_bf16(aScale[2 * m + 1]);
#pragma unroll
        for (auto n = 0u; n < (BlockN / 2); ++n) {
            int32_t acc[4];
            vst1q_s32(acc, accs[m * (BlockN / 2) + n]);
            float bScale0 = vcvtah_f32_bf16(bScale[2 * n + 0]);
            float bScale1 = vcvtah_f32_bf16(bScale[2 * n + 1]);
            out[(2 * m + 0) * dN + (2 * n + 0)] = vcvth_bf16_f32(float(acc[0]) * aScale0 * bScale0);
            out[(2 * m + 0) * dN + (2 * n + 1)] = vcvth_bf16_f32(float(acc[1]) * aScale0 * bScale1);
            out[(2 * m + 1) * dN + (2 * n + 0)] = vcvth_bf16_f32(float(acc[2]) * aScale1 * bScale0);
            out[(2 * m + 1) * dN + (2 * n + 1)] = vcvth_bf16_f32(float(acc[3]) * aScale1 * bScale1);
        }
    }
}

void _matmulT_s3d8(const int8_t* __restrict__ lhs,
                   const bf16* __restrict__ lhsScale_,
                   const uint8_t* __restrict__ rhs,
                   const int8_t* __restrict__ rhsLut,
                   const bf16* __restrict__ rhsScale_,
                   const uint dM,
                   const uint dK,
                   const uint dN,
                   bf16* __restrict__ out_) {
    constexpr auto RowsPerPack = 3u;
    const auto dNP = (dN + RowsPerPack - 1) / RowsPerPack;

    auto lhsScale = reinterpret_cast<const __bf16*>(lhsScale_);
    auto rhsScale = reinterpret_cast<const __bf16*>(rhsScale_);
    auto out = reinterpret_cast<__bf16*>(out_);

    if (dM == 1) {
        constexpr auto BN = 12u;  // multiple of 3
        constexpr auto BK = 16u;
        const auto pStop = (dNP / (BN / RowsPerPack)) * (BN / RowsPerPack);
#pragma omp parallel for
        for (auto p = 0u; p < pStop; p += BN / RowsPerPack) {
            // note: this is safe because a pack is never completely empty (which would cause
            // an out-of-bounds read)
            _mv_chunk_s3d8<BN, BK>(lhs, lhsScale[0], &rhs[p * dK], rhsLut,
                                   &rhsScale[p * RowsPerPack], dK,
                                   std::min(BN, dN - p * RowsPerPack), &out[p * RowsPerPack]);
        }
        for (auto p = pStop; p < dNP; ++p) {
            _mv_chunk_s3d8<RowsPerPack, 64>(
                lhs, lhsScale[0], &rhs[p * dK], rhsLut, &rhsScale[p * RowsPerPack], dK,
                std::min(RowsPerPack, dN - p * RowsPerPack), &out[p * RowsPerPack]);
        }
        return;
    }
    if (dN == 1) {
#pragma omp parallel for
        for (auto m = 0u; m < dM; ++m) {
            _mv_chunk_s3d8<RowsPerPack, 64>(&lhs[m * dK], lhsScale[m], rhs, rhsLut, rhsScale, dK,
                                            1u, &out[m]);
        }
        return;
    }

    constexpr auto G0 = 16u;
    constexpr auto G1M = 8u;
    constexpr auto G1N = 12u;  // multiple of 6

    const auto blocksM = (dM + G0 - 1) / G0;
    const auto blocksP = (dNP + G0 - 1) / G0;
#pragma omp parallel for
    for (auto i0 = 0u; i0 < blocksM * blocksP; ++i0) {
        auto m0 = G0 * (i0 / blocksP);
        auto m1 = std::min(m0 + G0, dM);
        auto p0 = G0 * (i0 % blocksP);
        auto p1 = std::min(p0 + G0, dNP);
        auto n0 = p0 * RowsPerPack;
        auto n1 = std::min(p1 * RowsPerPack, dN);

        const auto mStop = m0 + ((m1 - m0) / G1M) * G1M;
        const auto nStop = n0 + ((n1 - n0) / G1N) * G1N;
        const auto pStop = nStop / RowsPerPack;
        for (auto p = p0; p < pStop; p += G1N / RowsPerPack) {
            for (auto m = m0; m < mStop; m += G1M) {
                _matmulT_chunk_s3d8_smmla<G1M, G1N>(&lhs[m * dK], &lhsScale[m], &rhs[p * dK],
                                                    rhsLut, &rhsScale[p * RowsPerPack], dK, dN,
                                                    &out[m * dN + p * RowsPerPack]);
            }
        }
        // Handle remainder when dN is not a multiple of G1N, `out[m0:m1, (pStop*3):dN]`
        for (auto p = pStop; p < p1; ++p) {
            for (auto m = m0; m < m1; ++m) {
                _mv_chunk_s3d8<RowsPerPack, 64>(
                    &lhs[m * dK], lhsScale[m], &rhs[p * dK], rhsLut, &rhsScale[p * RowsPerPack], dK,
                    std::min(RowsPerPack, dN - p * RowsPerPack), &out[m * dN + p * RowsPerPack]);
            }
        }
        // Handle remainder when dM is not a multiple of G1M, `out[mStop:m1, n0:nStop]`
        // (note: excludes the bottom-right corner which is handled in the loop above)
        for (auto m = mStop; m < m1; ++m) {
            for (auto p = p0; p < pStop; p += G1N / RowsPerPack) {
                _mv_chunk_s3d8<G1N, 64>(
                    &lhs[m * dK], lhsScale[m], &rhs[p * dK], rhsLut, &rhsScale[p * RowsPerPack], dK,
                    std::min(G1N, dN - p * RowsPerPack), &out[m * dN + p * RowsPerPack]);
            }
        }
    }
}

#else  // !(__ARM_NEON && __ARM_FEATURE_BF16_VECTOR_ARITHMETIC && __ARM_FEATURE_MATMUL_INT8 &&
       // __ARM_FEATURE_DOTPROD)

#ifdef __ARM_NEON
#pragma message(                                                                      \
    "ARM NEON detected but required features for optimized matmul are not available," \
    " falling back to generic implementations")
#endif

// -----------------------------------------------------------------------------------------------
// Generic matmul

float _dot_bf16(const bf16* __restrict__ a, const bf16* __restrict__ b, const uint n) {
    float result = 0;
#pragma omp simd reduction(+ : result)
    for (auto i = 0u; i < n; ++i) {
        result += float(a[i]) * float(b[i]);
    }
    return result;
}

void _matmulT_bf16(const bf16* __restrict__ lhs,
                   const bf16* __restrict__ rhs,
                   const uint dM,
                   const uint dK,
                   const uint dN,
                   bf16* __restrict__ out) {
#pragma omp parallel for
    for (auto n = 0u; n < dN; ++n) {
        for (auto m = 0u; m < dM; ++m) {
            out[m * dN + n] = bf16(_dot_bf16(&lhs[m * dK], &rhs[n * dK], dK));
        }
    }
}

int32_t _dot_int8(const int8_t* __restrict__ a, const int8_t* __restrict__ b, const uint n) {
    int32_t result = 0;
#pragma omp simd reduction(+ : result)
    for (auto i = 0u; i < n; ++i) {
        result += int32_t(a[i]) * int32_t(b[i]);
    }
    return result;
}

void _matmulT_int8(const int8_t* __restrict__ lhs,
                   const bf16* __restrict__ lhsScale,
                   const int8_t* __restrict__ rhs,
                   const bf16* __restrict__ rhsScale,
                   const uint dM,
                   const uint dK,
                   const uint dN,
                   bf16* __restrict__ out) {
#pragma omp parallel for
    for (auto n = 0u; n < dN; ++n) {
        auto nScale = float(rhsScale[n]);
        for (auto m = 0u; m < dM; ++m) {
            auto dot = _dot_int8(&lhs[m * dK], &rhs[n * dK], dK);
            out[m * dN + n] = bf16(float(dot) * float(lhsScale[m]) * nScale);
        }
    }
}

int32_t _dot_int8_s3d8(const int8_t* __restrict__ a,
                       const uint8_t* __restrict__ b,
                       const int8_t* __restrict__ bLut,
                       const uint n,
                       const uint dK) {
    int32_t result = 0;
    for (auto k = 0u; k < dK; ++k) {
        result += int32_t(a[k]) * int32_t(_decode_s3d8(b[k], bLut, n % 3));
    }
    return result;
}

void _matmulT_s3d8(const int8_t* __restrict__ lhs,
                   const bf16* __restrict__ lhsScale,
                   const uint8_t* __restrict__ rhs,
                   const int8_t* __restrict__ rhsLut,
                   const bf16* __restrict__ rhsScale,
                   const uint dM,
                   const uint dK,
                   const uint dN,
                   bf16* __restrict__ out) {
#pragma omp parallel for
    for (auto n = 0u; n < dN; ++n) {
        const auto nScale = float(rhsScale[n]);
        for (auto m = 0u; m < dM; ++m) {
            auto dot = _dot_int8_s3d8(&lhs[m * dK], &rhs[(n / 3) * dK], rhsLut, n, dK);
            out[m * dN + n] = bf16(float(dot) * float(lhsScale[m]) * nScale);
        }
    }
}

#endif  // __ARM_NEON && __ARM_FEATURE_BF16_VECTOR_ARITHMETIC && __ARM_FEATURE_MATMUL_INT8 &&
        // __ARM_FEATURE_DOTPROD

}  // namespace

void matmulT(const bf16* __restrict__ lhs,
             const bf16* __restrict__ rhs,
             const uint dM,
             const uint dK,
             const uint dN,
             bf16* __restrict__ out) {
    _matmulT_bf16(lhs, rhs, dM, dK, dN, out);
}

void matmulT(const int8_t* __restrict__ lhs,
             const bf16* __restrict__ lhsScale,
             const int8_t* __restrict__ rhs,
             const bf16* __restrict__ rhsScale,
             const uint dM,
             const uint dK,
             const uint dN,
             bf16* __restrict__ out) {
    _matmulT_int8(lhs, lhsScale, rhs, rhsScale, dM, dK, dN, out);
}

void matmulT(const int8_t* __restrict__ lhs,
             const bf16* __restrict__ lhsScale,
             const uint8_t* __restrict__ rhs,
             const int8_t* __restrict__ rhsLut,
             const bf16* __restrict__ rhsScale,
             const uint dM,
             const uint dK,
             const uint dN,
             bf16* __restrict__ out) {
    _matmulT_s3d8(lhs, lhsScale, rhs, rhsLut, rhsScale, dM, dK, dN, out);
}

// -----------------------------------------------------------------------------------------------
// Gathers

void gather(const bf16* __restrict__ weight,
            const uint* __restrict__ indices,
            const uint nIndices,
            const uint dim,
            bf16* __restrict__ out) {
    for (auto n = 0u; n < nIndices; ++n) {
        for (auto i = 0u; i < dim; ++i) {
            out[n * dim + i] = weight[indices[n] * dim + i];
        }
    }
}

void gather(const int8_t* __restrict__ weight,
            const bf16* __restrict__ weightScale,
            const uint* __restrict__ indices,
            const uint nIndices,
            const uint dim,
            bf16* __restrict__ out) {
#pragma omp parallel for
    for (auto n = 0u; n < nIndices; ++n) {
        auto row = indices[n];
        auto rowScale = float(weightScale[row]);
        for (auto i = 0u; i < dim; ++i) {
            out[n * dim + i] = bf16(float(weight[row * dim + i]) * rowScale);
        }
    }
}

void gather(const uint8_t* __restrict__ weight,
            const int8_t* __restrict__ weightLut,
            const bf16* __restrict__ weightScale,
            const uint* __restrict__ indices,
            const uint nIndices,
            const uint dim,
            bf16* __restrict__ out) {
#pragma omp parallel for
    for (auto n = 0u; n < nIndices; ++n) {
        const auto row = indices[n];
        auto rowScale = float(weightScale[row]);
        for (auto k = 0u; k < dim; ++k) {
            auto v = _decode_s3d8(weight[(row / 3) * dim + k], weightLut, row % 3);
            out[n * dim + k] = bf16(float(v) * rowScale);
        }
    }
}

// -----------------------------------------------------------------------------------------------
// Maths/NN ops

void addInPlace(bf16* __restrict__ x, const bf16* __restrict__ y, const uint n) {
    for (auto i = 0u; i < n; ++i) {
        x[i] = bf16(float(x[i]) + float(y[i]));
    }
}

void broadcastAddInPlace(bf16* __restrict__ x, const bf16* __restrict__ y, uint n, uint d) {
    for (auto i = 0u; i < n; ++i) {
        for (auto j = 0u; j < d; ++j) {
            auto idx = i * d + j;
            x[idx] = bf16(float(x[idx]) + float(y[j]));
        }
    }
}

void geluInPlace(bf16* x, uint n) {
    // As per torch's 'approximate' GELU
    const float c0 = std::sqrt(2.0f / std::numbers::pi_v<float>);
    const float c1 = 0.044715f;
    for (auto i = 0u; i < n; ++i) {
        float xi = float(x[i]);
        float z = std::tanh(c0 * (xi + c1 * xi * xi * xi));
        x[i] = bf16(0.5f * xi * (1.0f + z));
    }
}

void swiGluInPlace(bf16* __restrict__ x, const bf16* __restrict__ gate, const uint n) {
    for (auto i = 0u; i < n; ++i) {
        auto gi = float(gate[i]);
        x[i] = bf16(float(x[i]) * gi / (1 + std::exp(-gi)));
    }
}

void rmsNorm(const bf16* __restrict__ weight,
             const bf16* __restrict__ x,
             const uint batch,
             const uint dim,
             float epsilon,
             bf16* __restrict__ out) {
    for (auto n = 0u; n < batch; ++n) {
        auto xn = x + n * dim;
        float sumSq = 0;
        for (auto i = 0u; i < dim; ++i) {
            auto xi = float(xn[i]);
            sumSq += xi * xi;
        }
        float scale = 1 / std::sqrt(sumSq / float(dim) + epsilon);
        for (auto i = 0u; i < dim; ++i) {
            out[n * dim + i] = bf16(float(xn[i]) * scale * float(weight[i]));
        }
    }
}

void layerNorm(const bf16* __restrict__ weight,
               const bf16* __restrict__ bias,
               const bf16* __restrict__ x,
               uint batch,
               uint dim,
               float epsilon,
               bf16* __restrict__ out) {
    for (auto n = 0u; n < batch; ++n) {
        auto xn = x + n * dim;
        float sum = 0, sumSq = 0;
        for (auto i = 0u; i < dim; ++i) {
            auto xi = float(xn[i]);
            sum += xi;
            sumSq += xi * xi;
        }
        float mean = sum / float(dim);
        float scale = 1 / std::sqrt(sumSq / float(dim) - mean * mean + epsilon);
        for (auto i = 0u; i < dim; ++i) {
            float normed = (float(xn[i]) - mean) * scale;
            out[n * dim + i] = bf16(normed * float(weight[i]) + float(bias[i]));
        }
    }
}

void rotateInPlace(bf16* __restrict__ x,
                   const float* __restrict__ freq,
                   const uint offsetS,
                   const uint dS,
                   const uint dH,
                   const uint dim) {
    for (auto s = 0u; s < dS; ++s) {
        for (auto h = 0u; h < dH; ++h) {
            for (auto i = 0u; i < dim / 2; ++i) {
                auto idxRe = s * (dH * dim) + h * (dim) + i;
                auto idxIm = idxRe + dim / 2;
                auto re = float(x[idxRe]);
                auto im = float(x[idxIm]);
                auto cos = std::cos(freq[i] * float(s + offsetS));
                auto sin = std::sin(freq[i] * float(s + offsetS));
                x[idxRe] = bf16(cos * re - sin * im);
                x[idxIm] = bf16(cos * im + sin * re);
            }
        }
    }
}

namespace {
void softmaxInPlace(float* __restrict__ x, const uint batch, const uint dim) {
    for (auto n = 0u; n < batch; ++n) {
        auto xn = x + n * dim;
        auto max = *std::max_element(xn, xn + dim);
        float sum = 0;
        for (auto i = 0u; i < dim; ++i) {
            xn[i] = std::exp(xn[i] - max);
            sum += xn[i];
        }
        for (auto i = 0u; i < dim; ++i) {
            xn[i] /= sum;
        }
    }
}
}  // namespace

void attentionInPlace(bf16* __restrict__ queryOut,
                      const bf16* __restrict__ key,
                      const bf16* __restrict__ value,
                      const uint dSq,
                      const uint dSkv,
                      const uint dHq,
                      const uint dHkv,
                      const uint dim,
                      const bool causal) {
#pragma omp parallel
    {
        std::unique_ptr<float[]> scores(new float[dSkv]);
        std::unique_ptr<float[]> outTmp(new float[dim]);
#pragma omp for
        for (auto n = 0u; n < dHkv * dSq * dHq; ++n) {
            auto hKv = n / (dSq * dHq);
            auto sQ = (n / dHq) % dSq;
            auto hQ = n % dHq;
            auto dSkv_row = causal ? (dSkv + 1 + sQ - dSq) : dSkv;
            // q @ k.T / sqrt(dim)
            for (auto sKv = 0u; sKv < dSkv_row; ++sKv) {
                float dot =
                    _dot_bf16(&queryOut[sQ * (dHkv * dHq * dim) + hKv * (dHq * dim) + hQ * (dim)],
                              &key[sKv * (dHkv * dim) + hKv * (dim)], dim);
                scores[sKv] = dot / std::sqrt(float(dim));
            }
            softmaxInPlace(scores.get(), 1u, dSkv_row);
            // s @ v
            std::fill_n(outTmp.get(), dim, 0.0f);
            float* __restrict__ out = outTmp.get();
            for (auto sKv = 0u; sKv < dSkv_row; ++sKv) {
                const float score = scores[sKv];
                const bf16* __restrict__ v = &value[sKv * (dHkv * dim) + hKv * (dim)];
                for (auto i = 0u; i < dim; ++i) {
                    out[i] += score * float(v[i]);
                }
            }
            for (auto i = 0u; i < dim; ++i) {
                queryOut[sQ * (dHkv * dHq * dim) + hKv * (dHq * dim) + hQ * (dim) + i] =
                    bf16(outTmp[i]);
            }
        }
    }
}

// -----------------------------------------------------------------------------------------------
// Sampling ops

void randn(bf16* out, ulong n, float stddev, ulong seed) {
#pragma omp parallel
    {
        std::default_random_engine rng(seed + static_cast<ulong>(omp_get_thread_num()));
#pragma omp for schedule(static)
        for (auto i = 0ul; i < n; ++i) {
            out[i] = bf16(std::normal_distribution<float>(0, stddev)(rng));
        }
    }
}

uint sample(const bf16* logits,
            uint n,
            float temperature,
            uint topK,
            float topP,
            std::default_random_engine& rng) {
    // Compute the safe log-softmax normaliser
    std::vector<std::tuple<float, uint>> logitsAndIndices;
    logitsAndIndices.reserve(n);
    auto maxLogit = float(logits[0]);
    for (auto i = 1u; i < n; ++i) {
        maxLogit = std::max(maxLogit, float(logits[i]));
    }
    auto sumExp = 0.f;
    for (auto i = 0u; i < n; ++i) {
        auto x = float(logits[i]) - maxLogit;
        logitsAndIndices.push_back({x, i});
        sumExp += std::exp(x);
    }

    // Sort the logits & indices in descending order
    std::sort(logitsAndIndices.begin(), logitsAndIndices.end(),
              [](const auto& lhs, const auto& rhs) { return std::get<0>(lhs) > std::get<0>(rhs); });

    // Calculate how many alternatives we're actually going to sample from
    auto topKandTopP = 1u;
    auto cumulativeP = std::exp(std::get<0>(logitsAndIndices[0]));
    while (topKandTopP < std::min(topK, n) & cumulativeP < topP * sumExp) {
        cumulativeP += std::exp(std::get<0>(logitsAndIndices[topKandTopP++]));
    }

    // Sample using Gumbel-max
    for (auto i = 0u; i < topKandTopP; ++i) {
        auto noise = std::log(-std::log(std::uniform_real_distribution<float>()(rng)));
        std::get<0>(logitsAndIndices[i]) -= temperature * noise;
    }
    return std::get<1>(
        *std::max_element(logitsAndIndices.begin(), logitsAndIndices.begin() + topKandTopP));
}

}  // namespace squash::ops
