// POLAR-FLASH: C++20 SIMD Native Inference Runtime Extension Implementation
#include "polarflash_simd.hpp"
#include <cstring>
#include <algorithm>

namespace polarflash::rt {

void dequantize_int4_simd(const uint8_t* packed, float* out, size_t n, float scale) {
    size_t num_bytes = (n + 1) / 2;
    size_t i = 0;
    size_t out_idx = 0;

#if defined(__AVX2__)
    __m256 v_scale = _mm256_set1_ps(scale);
    __m256 v_offset = _mm256_set1_ps(8.0f);
    __m128i mask_low = _mm_set1_epi8(0x0F);

    // Process 16 bytes = 32 values per chunk
    for (; i + 16 <= num_bytes && out_idx + 32 <= n; i += 16, out_idx += 32) {
        __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed + i));
        __m128i low_nibbles = _mm_and_si128(raw, mask_low);
        __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_low);

        // Interleave back into sequence: low0, high0, low1, high1...
        __m128i ordered0 = _mm_unpacklo_epi8(high_nibbles, low_nibbles);
        __m128i ordered1 = _mm_unpackhi_epi8(high_nibbles, low_nibbles);

        // Convert 8-bit to 32-bit float and scale
        for (int b = 0; b < 16; ++b) {
            uint8_t byte = packed[i + b];
            int8_t v0 = static_cast<int8_t>((byte >> 4) & 0x0F) - 8;
            int8_t v1 = static_cast<int8_t>(byte & 0x0F) - 8;
            out[out_idx + b * 2] = static_cast<float>(v0) * scale;
            if (out_idx + b * 2 + 1 < n) {
                out[out_idx + b * 2 + 1] = static_cast<float>(v1) * scale;
            }
        }
    }
#endif

    // Portable scalar fallback for remaining values
    for (; i < num_bytes && out_idx < n; ++i) {
        uint8_t b = packed[i];
        int8_t v0 = static_cast<int8_t>((b >> 4) & 0x0F) - 8;
        int8_t v1 = static_cast<int8_t>(b & 0x0F) - 8;
        out[out_idx++] = static_cast<float>(v0) * scale;
        if (out_idx < n) {
            out[out_idx++] = static_cast<float>(v1) * scale;
        }
    }
}

void dequantize_int2_simd(const uint8_t* packed, float* out, size_t n, float scale) {
    size_t num_bytes = (n + 3) / 4;
    size_t out_idx = 0;

    for (size_t i = 0; i < num_bytes && out_idx < n; ++i) {
        uint8_t b = packed[i];
        int8_t v0 = static_cast<int8_t>((b >> 6) & 0x03) - 2;
        int8_t v1 = static_cast<int8_t>((b >> 4) & 0x03) - 2;
        int8_t v2 = static_cast<int8_t>((b >> 2) & 0x03) - 2;
        int8_t v3 = static_cast<int8_t>(b & 0x03) - 2;

        out[out_idx++] = static_cast<float>(v0) * scale;
        if (out_idx < n) out[out_idx++] = static_cast<float>(v1) * scale;
        if (out_idx < n) out[out_idx++] = static_cast<float>(v2) * scale;
        if (out_idx < n) out[out_idx++] = static_cast<float>(v3) * scale;
    }
}

void dequantize_int3_simd(const uint8_t* packed, float* out, size_t n, float scale) {
    size_t out_idx = 0;
    size_t byte_idx = 0;

    while (out_idx + 8 <= n) {
        uint32_t b0 = packed[byte_idx++];
        uint32_t b1 = packed[byte_idx++];
        uint32_t b2 = packed[byte_idx++];
        uint32_t chunk = (b0 << 16) | (b1 << 8) | b2;

        for (int k = 7; k >= 0; --k) {
            int8_t v = static_cast<int8_t>((chunk >> (k * 3)) & 0x07) - 4;
            out[out_idx++] = static_cast<float>(v) * scale;
        }
    }

    if (out_idx < n) {
        uint32_t chunk = 0;
        int remaining_bytes = std::min<size_t>(3, (n - out_idx + 7) / 8);
        for (int j = 0; j < remaining_bytes; ++j) {
            chunk = (chunk << 8) | packed[byte_idx++];
        }
        for (int k = 7; k >= 0 && out_idx < n; --k) {
            int8_t v = static_cast<int8_t>((chunk >> (k * 3)) & 0x07) - 4;
            out[out_idx++] = static_cast<float>(v) * scale;
        }
    }
}

void dequantize_polar_simd(
    const uint8_t* packed_bytes,
    float* out,
    size_t num_pairs,
    const float* r_table,
    const float* theta_table,
    uint32_t r_bits,
    uint32_t theta_bits
) {
    // Unpack each pair from bitstream
    size_t bit_offset = 0;
    for (size_t i = 0; i < num_pairs; ++i) {
        // Read r_bits
        uint32_t r_idx = 0;
        for (uint32_t b = 0; b < r_bits; ++b) {
            size_t curr_bit = bit_offset + b;
            uint8_t byte = packed_bytes[curr_bit / 8];
            uint8_t bit = (byte >> (7 - (curr_bit % 8))) & 1;
            r_idx = (r_idx << 1) | bit;
        }
        bit_offset += r_bits;

        // Read theta_bits
        uint32_t theta_idx = 0;
        for (uint32_t b = 0; b < theta_bits; ++b) {
            size_t curr_bit = bit_offset + b;
            uint8_t byte = packed_bytes[curr_bit / 8];
            uint8_t bit = (byte >> (7 - (curr_bit % 8))) & 1;
            theta_idx = (theta_idx << 1) | bit;
        }
        bit_offset += theta_bits;

        float r = r_table[r_idx];
        float theta = theta_table[theta_idx];

        out[i * 2] = r * std::cos(theta);
        out[i * 2 + 1] = r * std::sin(theta);
    }
}

void fused_gemv_int4(
    const float* x,
    const uint8_t* packed_w,
    float* y,
    size_t M,
    size_t K,
    float scale
) {
    size_t bytes_per_row = (K + 1) / 2;

    for (size_t m = 0; m < M; ++m) {
        const uint8_t* row_packed = packed_w + m * bytes_per_row;
        float acc = 0.0f;
        size_t k = 0;

        for (size_t b = 0; b < bytes_per_row && k < K; ++b) {
            uint8_t byte = row_packed[b];
            int8_t v0 = static_cast<int8_t>((byte >> 4) & 0x0F) - 8;
            int8_t v1 = static_cast<int8_t>(byte & 0x0F) - 8;

            acc += static_cast<float>(v0) * x[k++];
            if (k < K) {
                acc += static_cast<float>(v1) * x[k++];
            }
        }
        y[m] = acc * scale;
    }
}

void fused_gemv_int2(
    const float* x,
    const uint8_t* packed_w,
    float* y,
    size_t M,
    size_t K,
    float scale
) {
    size_t bytes_per_row = (K + 3) / 4;

    for (size_t m = 0; m < M; ++m) {
        const uint8_t* row_packed = packed_w + m * bytes_per_row;
        float acc = 0.0f;
        size_t k = 0;

        for (size_t b = 0; b < bytes_per_row && k < K; ++b) {
            uint8_t byte = row_packed[b];
            int8_t v0 = static_cast<int8_t>((byte >> 6) & 0x03) - 2;
            int8_t v1 = static_cast<int8_t>((byte >> 4) & 0x03) - 2;
            int8_t v2 = static_cast<int8_t>((byte >> 2) & 0x03) - 2;
            int8_t v3 = static_cast<int8_t>(byte & 0x03) - 2;

            acc += static_cast<float>(v0) * x[k++];
            if (k < K) acc += static_cast<float>(v1) * x[k++];
            if (k < K) acc += static_cast<float>(v2) * x[k++];
            if (k < K) acc += static_cast<float>(v3) * x[k++];
        }
        y[m] = acc * scale;
    }
}

} // namespace polarflash::rt

