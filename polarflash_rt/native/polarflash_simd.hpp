// POLAR-FLASH: C++20 SIMD Native Inference Runtime Extension
// Implements AVX2 / AVX-512 / NEON vectorized dequantization & fused GEMV kernels
#ifndef POLARFLASH_SIMD_HPP
#define POLARFLASH_SIMD_HPP

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace polarflash::rt {

/**
 * Dequantizes packed 4-bit signed integers [-8, 7] into 32-bit floats.
 * packed: byte array where each byte stores 2 uint4 values: (high << 4) | low
 * out: destination float array (size n)
 * scale: symmetric scale factor
 */
void dequantize_int4_simd(const uint8_t* packed, float* out, size_t n, float scale);

/**
 * Dequantizes packed 2-bit signed integers [-2, 1] into 32-bit floats.
 * packed: byte array where each byte stores 4 2-bit values: (v0 << 6) | (v1 << 4) | (v2 << 2) | v3
 * out: destination float array (size n)
 * scale: symmetric scale factor
 */
void dequantize_int2_simd(const uint8_t* packed, float* out, size_t n, float scale);

/**
 * Dequantizes packed 3-bit signed integers [-4, 3] into 32-bit floats.
 * 8 values packed per 3 bytes (24 bits).
 */
void dequantize_int3_simd(const uint8_t* packed, float* out, size_t n, float scale);

/**
 * Dequantizes packed Polar coordinate pairs:
 * r = r_table[r_idx], theta = theta_table[theta_idx]
 * x1 = r * cos(theta), x2 = r * sin(theta)
 */
void dequantize_polar_simd(
    const uint8_t* packed_bytes,
    float* out,
    size_t num_pairs,
    const float* r_table,
    const float* theta_table,
    uint32_t r_bits,
    uint32_t theta_bits
);

/**
 * Fused Dequantize + Matrix-Vector Multiply (GEMV):
 * y = W * x
 * x: input vector (length K)
 * packed_w: packed weights of shape (M, K)
 * y: output vector (length M)
 */
void fused_gemv_int4(
    const float* x,
    const uint8_t* packed_w,
    float* y,
    size_t M,
    size_t K,
    float scale
);

void fused_gemv_int2(
    const float* x,
    const uint8_t* packed_w,
    float* y,
    size_t M,
    size_t K,
    float scale
);

} // namespace polarflash::rt

#endif // POLARFLASH_SIMD_HPP

