#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

typedef _Float16 my_half4 __attribute__((ext_vector_type(4)));
typedef float my_float4 __attribute__((ext_vector_type(4)));

constexpr int kQueryTile = 16;
constexpr int kKeyTile = 16;
constexpr int kHeadDim = 32;
constexpr int kBlockThreads = 256;
constexpr int kMaxTokens = 144;

constexpr std::size_t kHalfTileElements = kQueryTile * kHeadDim;
constexpr std::size_t kScoreTileElements = kQueryTile * kKeyTile;

constexpr std::size_t kQOffset = 0;
constexpr std::size_t kKOffset =
    kQOffset + kHalfTileElements * sizeof(half);
constexpr std::size_t kVOffset =
    kKOffset + kHalfTileElements * sizeof(half);
constexpr std::size_t kScoreOffset =
    kVOffset + kHalfTileElements * sizeof(half);
constexpr std::size_t kMaxOffset =
    kScoreOffset + kScoreTileElements * sizeof(float);
constexpr std::size_t kSumOffset =
    kMaxOffset + kQueryTile * sizeof(float);
constexpr std::size_t kAlphaOffset =
    kSumOffset + kQueryTile * sizeof(float);
constexpr std::size_t kDynamicSharedBytes =
    kAlphaOffset + kQueryTile * sizeof(float);

constexpr std::size_t kParityQOffset = 0;
constexpr std::size_t kParityKVOffset =
    kParityQOffset + kHalfTileElements * sizeof(half);
constexpr std::size_t kParityScoreOffset =
    kParityKVOffset + kHalfTileElements * sizeof(half);
constexpr std::size_t kParityScoreElements = kQueryTile * kMaxTokens;
constexpr std::size_t kParityDynamicSharedBytes =
    kParityScoreOffset + kParityScoreElements * sizeof(half);

static_assert(kHeadDim % 2 == 0, "half2 dot products require an even head dimension");
static_assert(
    kBlockThreads == kQueryTile * kKeyTile,
    "one thread is required for every query/key pair in a tile");
static_assert(
    alignof(float) >= alignof(half2),
    "float-aligned dynamic LDS must also satisfy half2 alignment");
static_assert(kQOffset % alignof(half2) == 0, "Q LDS must be half2 aligned");
static_assert(kKOffset % alignof(half2) == 0, "K LDS must be half2 aligned");
static_assert(kVOffset % alignof(half2) == 0, "V LDS must be half2 aligned");
static_assert(kScoreOffset % alignof(float) == 0, "score LDS must be float aligned");
static_assert(
    kParityKVOffset % alignof(float) == 0,
    "parity K/V scratch LDS must be float aligned");
static_assert(
    kParityScoreOffset % alignof(half2) == 0,
    "parity score LDS must be half2 aligned");
static_assert(
    kParityDynamicSharedBytes == 6656,
    "full-row parity LDS must remain below the 8 KiB occupancy boundary");

__device__ __forceinline__ std::int64_t qkv_offset(
    int width_index,
    int pressure_height_index,
    int token_index,
    int qkv_index,
    int head_index,
    int dim_index,
    int pressure_height,
    int tokens,
    int heads)
{
    return (((((static_cast<std::int64_t>(width_index) * pressure_height +
                pressure_height_index) *
                   tokens +
               token_index) *
                  3 +
              qkv_index) *
                 heads +
             head_index) *
                kHeadDim +
            dim_index);
}

__device__ __forceinline__ std::int64_t output_offset(
    int width_index,
    int pressure_height_index,
    int token_index,
    int head_index,
    int dim_index,
    int pressure_height,
    int tokens,
    int heads)
{
    return ((((static_cast<std::int64_t>(width_index) * pressure_height +
               pressure_height_index) *
                  tokens +
              token_index) *
                 heads +
             head_index) *
                kHeadDim +
            dim_index);
}

__device__ __forceinline__ std::int64_t diagnostic_offset(
    int width_index,
    int head_index,
    int pressure_height_index,
    int query_index,
    int key_index,
    int heads,
    int pressure_height,
    int tokens)
{
    return (((((static_cast<std::int64_t>(width_index) * heads + head_index) *
                pressure_height +
                pressure_height_index) *
               tokens +
               query_index) *
              tokens) +
            key_index);
}

__device__ __forceinline__ float tiled_exp(float value)
{
#if defined(PANGU_TILED_STANDARD_EXP)
    return expf(value);
#else
    return __expf(value);
#endif
}

template <bool kStandardExp>
__device__ __forceinline__ float parity_exp(float value)
{
    if constexpr (kStandardExp)
    {
        return expf(value);
    }
    return __expf(value);
}

template <bool kShifted>
__global__ void earth_attention_tiled_fwd_fp16_kernel(
    const half* qkv,
    const half* bias_table,
    const std::int16_t* position_index,
    const std::uint8_t* region_id,
    half* output,
    int width,
    int pressure_height,
    int heads,
    int tokens,
    int bias_rows,
    int mask_width,
    int width_offset,
    float scale)
{
    extern __shared__ float shared_words[];
    unsigned char* shared_raw = reinterpret_cast<unsigned char*>(shared_words);
    half* q_shared = reinterpret_cast<half*>(shared_raw + kQOffset);
    half* k_shared = reinterpret_cast<half*>(shared_raw + kKOffset);
    half* v_shared = reinterpret_cast<half*>(shared_raw + kVOffset);
    float* score_shared = reinterpret_cast<float*>(shared_raw + kScoreOffset);
    float* running_max = reinterpret_cast<float*>(shared_raw + kMaxOffset);
    float* running_sum = reinterpret_cast<float*>(shared_raw + kSumOffset);
    float* rescale = reinterpret_cast<float*>(shared_raw + kAlphaOffset);

    const int thread_index = static_cast<int>(threadIdx.x);
    const int query_local = thread_index / kKeyTile;

    const int query_start = static_cast<int>(blockIdx.x) * kQueryTile;
    const int query_index = query_start + query_local;
    const int pressure_head_index = static_cast<int>(blockIdx.y);
    const int head_index = pressure_head_index / pressure_height;
    const int pressure_height_index = pressure_head_index % pressure_height;
    const int width_index = static_cast<int>(blockIdx.z);
    const std::int64_t head_stride =
        static_cast<std::int64_t>(heads) * kHeadDim;
    const std::int64_t token_stride = 3 * head_stride;
    const std::int64_t plane_base =
        ((static_cast<std::int64_t>(width_index) * pressure_height +
          pressure_height_index) * tokens * token_stride) +
        static_cast<std::int64_t>(head_index) * kHeadDim;

    for(int item = thread_index * 4;
        item < static_cast<int>(kHalfTileElements);
        item += kBlockThreads * 4)
    {
        const int query_row = item / kHeadDim;
        const int dim_index = item % kHeadDim;
        const int global_query = query_start + query_row;
        const std::int64_t offset =
            plane_base + static_cast<std::int64_t>(global_query) * token_stride +
            dim_index;
        const my_half4 value = *reinterpret_cast<const my_half4*>(qkv + offset);
        q_shared[item + 0] = __float2half_rn(__half2float(value[0]) * scale);
        q_shared[item + 1] = __float2half_rn(__half2float(value[1]) * scale);
        q_shared[item + 2] = __float2half_rn(__half2float(value[2]) * scale);
        q_shared[item + 3] = __float2half_rn(__half2float(value[3]) * scale);
    }

    if(thread_index < kQueryTile)
    {
        running_max[thread_index] = -INFINITY;
        running_sum[thread_index] = 0.0f;
        rescale[thread_index] = 0.0f;
    }
    __syncthreads();

    const int masked_width_index = kShifted
        ? (width_offset + width_index) % mask_width
        : 0;

    const int lane_id = thread_index % 64;
    const int wave_id = thread_index / 64;
    float pv_acc_0_x = 0.0f;
    float pv_acc_0_y = 0.0f;
    float pv_acc_0_z = 0.0f;
    float pv_acc_0_w = 0.0f;
    float pv_acc_1_x = 0.0f;
    float pv_acc_1_y = 0.0f;
    float pv_acc_1_z = 0.0f;
    float pv_acc_1_w = 0.0f;

    for(int key_start = 0; key_start < tokens; key_start += kKeyTile)
    {
        for(int item = thread_index * 4;
            item < static_cast<int>(kHalfTileElements);
            item += kBlockThreads * 4)
        {
            const int key_row = item / kHeadDim;
            const int dim_index = item % kHeadDim;
            const int key_index = key_start + key_row;
            const std::int64_t offset =
                plane_base + static_cast<std::int64_t>(key_index) * token_stride +
                dim_index;
            const my_half4 key_value = *reinterpret_cast<const my_half4*>(
                qkv + offset + head_stride);
            const my_half4 value_value = *reinterpret_cast<const my_half4*>(
                qkv + offset + 2 * head_stride);
            *reinterpret_cast<my_half4*>(k_shared + item) = key_value;
            *reinterpret_cast<my_half4*>(v_shared + item) = value_value;
        }
        __syncthreads();

        my_half4 a_0 = {0.0f, 0.0f, 0.0f, 0.0f};
        my_half4 a_1 = {0.0f, 0.0f, 0.0f, 0.0f};
        my_half4 b_0 = {0.0f, 0.0f, 0.0f, 0.0f};
        my_half4 b_1 = {0.0f, 0.0f, 0.0f, 0.0f};
        my_float4 d_0 = {0.0f, 0.0f, 0.0f, 0.0f};
        my_float4 d_1 = {0.0f, 0.0f, 0.0f, 0.0f};

        if (wave_id == 0)
        {
            int r_A = lane_id % 16;
            int c_A_base = (lane_id / 16) * 4;
            const half* q_ptr = q_shared + r_A * kHeadDim + c_A_base;
            a_0 = *reinterpret_cast<const my_half4*>(q_ptr);

            int c_B = ((lane_id & 3) << 2) | ((lane_id & 12) >> 2);
            int r_B_base = (lane_id / 16) * 4;
            const half* k_ptr = k_shared + c_B * kHeadDim + r_B_base;
            b_0 = *reinterpret_cast<const my_half4*>(k_ptr);

#if defined(__HIP_DEVICE_COMPILE__)
            d_0 = __builtin_amdgcn_mmac_f32_16x16x16f16(a_0, b_0, d_0);
#endif
        }

        if (wave_id == 1)
        {
            int r_A = lane_id % 16;
            int c_A_base = (lane_id / 16) * 4 + 16;
            const half* q_ptr = q_shared + r_A * kHeadDim + c_A_base;
            a_1 = *reinterpret_cast<const my_half4*>(q_ptr);

            int c_B = ((lane_id & 3) << 2) | ((lane_id & 12) >> 2);
            int r_B_base = (lane_id / 16) * 4 + 16;
            const half* k_ptr = k_shared + c_B * kHeadDim + r_B_base;
            b_1 = *reinterpret_cast<const my_half4*>(k_ptr);

#if defined(__HIP_DEVICE_COMPILE__)
            d_1 = __builtin_amdgcn_mmac_f32_16x16x16f16(a_1, b_1, d_1);
#endif
        }
        __syncthreads();

        if (wave_id == 0)
        {
            int row = lane_id % 16;
            int col_base = (lane_id / 16) * 4;

            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                int col = col_base + i;
                score_shared[row * 16 + col] = d_0[i];
            }
        }
        __syncthreads();

        if (wave_id == 1)
        {
            int row = lane_id % 16;
            int col_base = (lane_id / 16) * 4;

#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                int col = col_base + i;
                score_shared[row * 16 + col] += d_1[i];
            }
        }
        __syncthreads();

        {
            int row = thread_index / 16;
            int col = thread_index % 16;
            int query_index = query_start + row;
            int key_index = key_start + col;

            if (query_index < tokens && key_index < tokens)
            {
                float val = score_shared[thread_index];
                const std::int16_t bias_index = position_index[
                    static_cast<std::int64_t>(query_index) * tokens + key_index];
                const std::int64_t bias_offset =
                    (static_cast<std::int64_t>(head_index) * pressure_height +
                     pressure_height_index) *
                        bias_rows +
                    static_cast<int>(bias_index);
                val += __half2float(bias_table[bias_offset]);

                if constexpr (kShifted)
                {
                    const std::int64_t region_base =
                        (static_cast<std::int64_t>(masked_width_index) *
                             pressure_height +
                         pressure_height_index) *
                        tokens;
                    if(region_id[region_base + query_index] !=
                       region_id[region_base + key_index])
                    {
                        val -= 100.0f;
                    }
                }
                score_shared[thread_index] = val;
            }
            else
            {
                score_shared[thread_index] = -INFINITY;
            }
        }
        __syncthreads();

        if(thread_index < kQueryTile)
        {
            const int row = thread_index;
            const int global_query = query_start + row;
            float tile_max = -INFINITY;
            if(global_query < tokens)
            {
#pragma unroll
                for(int column = 0; column < kKeyTile; ++column)
                {
                    tile_max = fmaxf(
                        tile_max,
                        score_shared[row * kKeyTile + column]);
                }
            }

            const float old_max = running_max[row];
            const float new_max = fmaxf(old_max, tile_max);
            const float alpha = isfinite(old_max)
                ? tiled_exp(old_max - new_max)
                : 0.0f;
            float tile_sum = 0.0f;
#pragma unroll
            for(int column = 0; column < kKeyTile; ++column)
            {
                const float row_score =
                    score_shared[row * kKeyTile + column];
                const float row_probability = tiled_exp(row_score - new_max);
                tile_sum += row_probability;
                score_shared[row * kKeyTile + column] = row_probability;
            }

            running_max[row] = new_max;
            running_sum[row] = running_sum[row] * alpha + tile_sum;
            rescale[row] = alpha;
        }
        __syncthreads();

        if (thread_index < 256)
        {
            k_shared[thread_index] = __float2half_rn(score_shared[thread_index]);
        }
        __syncthreads();

        my_half4 p_0 = {0.0f, 0.0f, 0.0f, 0.0f};
        my_half4 p_1 = {0.0f, 0.0f, 0.0f, 0.0f};
        my_half4 val_0 = {0.0f, 0.0f, 0.0f, 0.0f};
        my_half4 val_1 = {0.0f, 0.0f, 0.0f, 0.0f};
        my_float4 d_out_0 = {0.0f, 0.0f, 0.0f, 0.0f};
        my_float4 d_out_1 = {0.0f, 0.0f, 0.0f, 0.0f};

        if (wave_id == 2)
        {
            int r_A = lane_id % 16;
            int c_A_base = (lane_id / 16) * 4;
            const half* p_ptr = k_shared + r_A * 16 + c_A_base;
            p_0 = *reinterpret_cast<const my_half4*>(p_ptr);

            const int c_B = ((lane_id & 3) << 2) | ((lane_id & 12) >> 2);
            const int r_B_base = (lane_id / 16) * 4;
            val_0.x = v_shared[(r_B_base + 0) * kHeadDim + c_B + 0];
            val_0.y = v_shared[(r_B_base + 1) * kHeadDim + c_B + 0];
            val_0.z = v_shared[(r_B_base + 2) * kHeadDim + c_B + 0];
            val_0.w = v_shared[(r_B_base + 3) * kHeadDim + c_B + 0];

            int row_D = lane_id % 16;
            d_out_0.x = pv_acc_0_x * rescale[row_D];
            d_out_0.y = pv_acc_0_y * rescale[row_D];
            d_out_0.z = pv_acc_0_z * rescale[row_D];
            d_out_0.w = pv_acc_0_w * rescale[row_D];

#if defined(__HIP_DEVICE_COMPILE__)
            d_out_0 = __builtin_amdgcn_mmac_f32_16x16x16f16(p_0, val_0, d_out_0);
#endif

            pv_acc_0_x = d_out_0.x;
            pv_acc_0_y = d_out_0.y;
            pv_acc_0_z = d_out_0.z;
            pv_acc_0_w = d_out_0.w;
        }

        if (wave_id == 3)
        {
            int r_A = lane_id % 16;
            int c_A_base = (lane_id / 16) * 4;
            const half* p_ptr = k_shared + r_A * 16 + c_A_base;
            p_1 = *reinterpret_cast<const my_half4*>(p_ptr);

            const int c_B = ((lane_id & 3) << 2) | ((lane_id & 12) >> 2);
            const int r_B_base = (lane_id / 16) * 4;
            val_1.x = v_shared[(r_B_base + 0) * kHeadDim + c_B + 16];
            val_1.y = v_shared[(r_B_base + 1) * kHeadDim + c_B + 16];
            val_1.z = v_shared[(r_B_base + 2) * kHeadDim + c_B + 16];
            val_1.w = v_shared[(r_B_base + 3) * kHeadDim + c_B + 16];

            int row_D = lane_id % 16;
            d_out_1.x = pv_acc_1_x * rescale[row_D];
            d_out_1.y = pv_acc_1_y * rescale[row_D];
            d_out_1.z = pv_acc_1_z * rescale[row_D];
            d_out_1.w = pv_acc_1_w * rescale[row_D];

#if defined(__HIP_DEVICE_COMPILE__)
            d_out_1 = __builtin_amdgcn_mmac_f32_16x16x16f16(p_1, val_1, d_out_1);
#endif

            pv_acc_1_x = d_out_1.x;
            pv_acc_1_y = d_out_1.y;
            pv_acc_1_z = d_out_1.z;
            pv_acc_1_w = d_out_1.w;
        }
        __syncthreads();
    }

    if(wave_id == 2 || wave_id == 3)
    {
        const int row_D = lane_id % kQueryTile;
        const int col_D_base = (lane_id / kQueryTile) * 4 +
            (wave_id == 3 ? kHeadDim / 2 : 0);
        const int final_query_index = query_start + row_D;
        const float denominator = running_sum[row_D];
        const float inv_sum = denominator > 0.0f ? 1.0f / denominator : 0.0f;
        if(final_query_index < tokens)
        {
            const std::int64_t output_base = output_offset(
                width_index,
                pressure_height_index,
                final_query_index,
                head_index,
                col_D_base,
                pressure_height,
                tokens,
                heads);
            if(wave_id == 2)
            {
                output[output_base + 0] = __float2half_rn(pv_acc_0_x * inv_sum);
                output[output_base + 1] = __float2half_rn(pv_acc_0_y * inv_sum);
                output[output_base + 2] = __float2half_rn(pv_acc_0_z * inv_sum);
                output[output_base + 3] = __float2half_rn(pv_acc_0_w * inv_sum);
            }
            else
            {
                output[output_base + 0] = __float2half_rn(pv_acc_1_x * inv_sum);
                output[output_base + 1] = __float2half_rn(pv_acc_1_y * inv_sum);
                output[output_base + 2] = __float2half_rn(pv_acc_1_z * inv_sum);
                output[output_base + 3] = __float2half_rn(pv_acc_1_w * inv_sum);
            }
        }
    }
    return;
}

template <bool kStandardExp, int kLogicalWidth, int kIterations>
__device__ __forceinline__ void full_row_softmax(
    half* row_scores,
    int tokens,
    int lane)
{
    float elements[kIterations];
    float max_value = -INFINITY;
#pragma unroll
    for(int iteration = 0; iteration < kIterations; ++iteration)
    {
        const int element_index = lane + iteration * kLogicalWidth;
        elements[iteration] = element_index < tokens
            ? __half2float(row_scores[element_index])
            : -INFINITY;
        max_value = max_value > elements[iteration]
            ? max_value
            : elements[iteration];
    }
#pragma unroll
    for(int offset = kLogicalWidth / 2; offset > 0; offset /= 2)
    {
        const float other = __shfl_xor(max_value, offset, kLogicalWidth);
        max_value = max_value < other ? other : max_value;
    }

    float sum = 0.0f;
#pragma unroll
    for(int iteration = 0; iteration < kIterations; ++iteration)
    {
        elements[iteration] = parity_exp<kStandardExp>(
            elements[iteration] - max_value);
        sum += elements[iteration];
    }
#pragma unroll
    for(int offset = kLogicalWidth / 2; offset > 0; offset /= 2)
    {
        sum += __shfl_xor(sum, offset, kLogicalWidth);
    }

#pragma unroll
    for(int iteration = 0; iteration < kIterations; ++iteration)
    {
        const int element_index = lane + iteration * kLogicalWidth;
        if(element_index < tokens)
        {
            row_scores[element_index] = __float2half_rn(
                elements[iteration] / sum);
        }
    }
    return;
}

template <bool kShifted, bool kStandardExp, bool kDiagnostics>
__global__ void earth_attention_tiled_full_row_fwd_fp16_kernel(
    const half* qkv,
    const half* bias_table,
    const std::int16_t* position_index,
    const std::uint8_t* region_id,
    half* output,
    half* diagnostic_qk,
    half* diagnostic_biased,
    half* diagnostic_probability,
    int width,
    int pressure_height,
    int heads,
    int tokens,
    int bias_rows,
    int mask_width,
    int width_offset,
    float scale)
{
    extern __shared__ float shared_words[];
    unsigned char* shared_raw = reinterpret_cast<unsigned char*>(shared_words);
    half* q_shared = reinterpret_cast<half*>(shared_raw + kParityQOffset);
    half* kv_shared = reinterpret_cast<half*>(shared_raw + kParityKVOffset);
    half* full_scores = reinterpret_cast<half*>(shared_raw + kParityScoreOffset);
    float* qk_scratch = reinterpret_cast<float*>(kv_shared);

    const int thread_index = static_cast<int>(threadIdx.x);
    const int lane_id = thread_index % 64;
    const int wave_id = thread_index / 64;
    const int query_start = static_cast<int>(blockIdx.x) * kQueryTile;
    const int pressure_head_index = static_cast<int>(blockIdx.y);
    const int head_index = pressure_head_index / pressure_height;
    const int pressure_height_index = pressure_head_index % pressure_height;
    const int width_index = static_cast<int>(blockIdx.z);
    const std::int64_t head_stride =
        static_cast<std::int64_t>(heads) * kHeadDim;
    const std::int64_t token_stride = 3 * head_stride;
    const std::int64_t plane_base =
        ((static_cast<std::int64_t>(width_index) * pressure_height +
          pressure_height_index) * tokens * token_stride) +
        static_cast<std::int64_t>(head_index) * kHeadDim;
    const int masked_width_index = kShifted
        ? (width_offset + width_index) % mask_width
        : 0;

    for(int item = thread_index * 4;
        item < static_cast<int>(kHalfTileElements);
        item += kBlockThreads * 4)
    {
        const int query_row = item / kHeadDim;
        const int dim_index = item % kHeadDim;
        const int global_query = query_start + query_row;
        const std::int64_t offset =
            plane_base + static_cast<std::int64_t>(global_query) * token_stride +
            dim_index;
        const my_half4 value = *reinterpret_cast<const my_half4*>(qkv + offset);
        q_shared[item + 0] = __float2half_rn(__half2float(value[0]) * scale);
        q_shared[item + 1] = __float2half_rn(__half2float(value[1]) * scale);
        q_shared[item + 2] = __float2half_rn(__half2float(value[2]) * scale);
        q_shared[item + 3] = __float2half_rn(__half2float(value[3]) * scale);
    }
    __syncthreads();

    for(int key_start = 0; key_start < tokens; key_start += kKeyTile)
    {
        for(int item = thread_index * 4;
            item < static_cast<int>(kHalfTileElements);
            item += kBlockThreads * 4)
        {
            const int key_row = item / kHeadDim;
            const int dim_index = item % kHeadDim;
            const int key_index = key_start + key_row;
            const std::int64_t offset =
                plane_base + static_cast<std::int64_t>(key_index) * token_stride +
                head_stride + dim_index;
            *reinterpret_cast<my_half4*>(kv_shared + item) =
                *reinterpret_cast<const my_half4*>(qkv + offset);
        }
        __syncthreads();

        my_half4 a_low = {0.0f, 0.0f, 0.0f, 0.0f};
        my_half4 a_high = {0.0f, 0.0f, 0.0f, 0.0f};
        my_half4 b_low = {0.0f, 0.0f, 0.0f, 0.0f};
        my_half4 b_high = {0.0f, 0.0f, 0.0f, 0.0f};
        my_float4 d = {0.0f, 0.0f, 0.0f, 0.0f};
        if(wave_id == 0)
        {
            const int r_A = lane_id % 16;
            const int c_B = ((lane_id & 3) << 2) | ((lane_id & 12) >> 2);
            const int k_base = (lane_id / 16) * 4;
            a_low = *reinterpret_cast<const my_half4*>(
                q_shared + r_A * kHeadDim + k_base);
            b_low = *reinterpret_cast<const my_half4*>(
                kv_shared + c_B * kHeadDim + k_base);
            a_high = *reinterpret_cast<const my_half4*>(
                q_shared + r_A * kHeadDim + k_base + 16);
            b_high = *reinterpret_cast<const my_half4*>(
                kv_shared + c_B * kHeadDim + k_base + 16);
#if defined(__HIP_DEVICE_COMPILE__)
            d = __builtin_amdgcn_mmac_f32_16x16x16f16(a_low, b_low, d);
            d = __builtin_amdgcn_mmac_f32_16x16x16f16(a_high, b_high, d);
#endif
        }
        __syncthreads();

        if(wave_id == 0)
        {
            const int row = lane_id % 16;
            const int col_base = (lane_id / 16) * 4;
#pragma unroll
            for(int item = 0; item < 4; ++item)
            {
                qk_scratch[row * 16 + col_base + item] = d[item];
            }
        }
        __syncthreads();

        {
            const int row = thread_index / 16;
            const int column = thread_index % 16;
            const int query_index = query_start + row;
            const int key_index = key_start + column;
            half rounded_score = __float2half_rn(qk_scratch[thread_index]);
            const std::int64_t stage_offset = diagnostic_offset(
                width_index,
                head_index,
                pressure_height_index,
                query_index,
                key_index,
                heads,
                pressure_height,
                tokens);
            if constexpr (kDiagnostics)
            {
                diagnostic_qk[stage_offset] = rounded_score;
            }
            const std::int16_t bias_index = position_index[
                static_cast<std::int64_t>(query_index) * tokens + key_index];
            const std::int64_t bias_offset =
                (static_cast<std::int64_t>(head_index) * pressure_height +
                 pressure_height_index) * bias_rows +
                static_cast<int>(bias_index);
            rounded_score = __float2half_rn(
                __half2float(rounded_score) +
                __half2float(bias_table[bias_offset]));
            if constexpr (kShifted)
            {
                const std::int64_t region_base =
                    (static_cast<std::int64_t>(masked_width_index) *
                         pressure_height +
                     pressure_height_index) * tokens;
                if(region_id[region_base + query_index] !=
                   region_id[region_base + key_index])
                {
                    rounded_score = __float2half_rn(
                        __half2float(rounded_score) - 100.0f);
                }
            }
            full_scores[row * kMaxTokens + key_index] = rounded_score;
            if constexpr (kDiagnostics)
            {
                diagnostic_biased[stage_offset] = rounded_score;
            }
        }
        __syncthreads();
    }

    if(tokens == 144)
    {
        for(int row_group = 0; row_group < 4; ++row_group)
        {
            const int row = wave_id * 4 + row_group;
            full_row_softmax<kStandardExp, 64, 4>(
                full_scores + row * kMaxTokens,
                tokens,
                lane_id);
        }
    }
    else
    {
        const int logical_lane = thread_index % 32;
        const int logical_group = thread_index / 32;
#pragma unroll
        for(int batch = 0; batch < 2; ++batch)
        {
            const int row = logical_group * 2 + batch;
            full_row_softmax<kStandardExp, 32, 1>(
                full_scores + row * kMaxTokens,
                tokens,
                logical_lane);
        }
    }
    __syncthreads();

    if constexpr (kDiagnostics)
    {
        for(int item = thread_index;
            item < kQueryTile * tokens;
            item += kBlockThreads)
        {
            const int row = item / tokens;
            const int key_index = item % tokens;
            const int query_index = query_start + row;
            const std::int64_t stage_offset = diagnostic_offset(
                width_index,
                head_index,
                pressure_height_index,
                query_index,
                key_index,
                heads,
                pressure_height,
                tokens);
            diagnostic_probability[stage_offset] =
                full_scores[row * kMaxTokens + key_index];
        }
    }
    __syncthreads();

    float pv_acc_0_x = 0.0f;
    float pv_acc_0_y = 0.0f;
    float pv_acc_0_z = 0.0f;
    float pv_acc_0_w = 0.0f;
    float pv_acc_1_x = 0.0f;
    float pv_acc_1_y = 0.0f;
    float pv_acc_1_z = 0.0f;
    float pv_acc_1_w = 0.0f;
    for(int key_start = 0; key_start < tokens; key_start += kKeyTile)
    {
        for(int item = thread_index * 4;
            item < static_cast<int>(kHalfTileElements);
            item += kBlockThreads * 4)
        {
            const int key_row = item / kHeadDim;
            const int dim_index = item % kHeadDim;
            const int key_index = key_start + key_row;
            const std::int64_t offset =
                plane_base + static_cast<std::int64_t>(key_index) * token_stride +
                2 * head_stride + dim_index;
            *reinterpret_cast<my_half4*>(kv_shared + item) =
                *reinterpret_cast<const my_half4*>(qkv + offset);
        }
        __syncthreads();

        my_half4 probability = {0.0f, 0.0f, 0.0f, 0.0f};
        my_half4 value = {0.0f, 0.0f, 0.0f, 0.0f};
        my_float4 accumulator = {0.0f, 0.0f, 0.0f, 0.0f};
        if(wave_id == 2 || wave_id == 3)
        {
            const int r_A = lane_id % 16;
            const int c_A_base = (lane_id / 16) * 4;
            probability = *reinterpret_cast<const my_half4*>(
                full_scores + r_A * kMaxTokens + key_start + c_A_base);

            const int c_B = ((lane_id & 3) << 2) | ((lane_id & 12) >> 2);
            const int r_B_base = (lane_id / 16) * 4;
            const int dim_offset = wave_id == 3 ? 16 : 0;
            value.x = kv_shared[(r_B_base + 0) * kHeadDim + c_B + dim_offset];
            value.y = kv_shared[(r_B_base + 1) * kHeadDim + c_B + dim_offset];
            value.z = kv_shared[(r_B_base + 2) * kHeadDim + c_B + dim_offset];
            value.w = kv_shared[(r_B_base + 3) * kHeadDim + c_B + dim_offset];
            if(wave_id == 2)
            {
                accumulator.x = pv_acc_0_x;
                accumulator.y = pv_acc_0_y;
                accumulator.z = pv_acc_0_z;
                accumulator.w = pv_acc_0_w;
            }
            else
            {
                accumulator.x = pv_acc_1_x;
                accumulator.y = pv_acc_1_y;
                accumulator.z = pv_acc_1_z;
                accumulator.w = pv_acc_1_w;
            }
#if defined(__HIP_DEVICE_COMPILE__)
            accumulator = __builtin_amdgcn_mmac_f32_16x16x16f16(
                probability,
                value,
                accumulator);
#endif
            if(wave_id == 2)
            {
                pv_acc_0_x = accumulator.x;
                pv_acc_0_y = accumulator.y;
                pv_acc_0_z = accumulator.z;
                pv_acc_0_w = accumulator.w;
            }
            else
            {
                pv_acc_1_x = accumulator.x;
                pv_acc_1_y = accumulator.y;
                pv_acc_1_z = accumulator.z;
                pv_acc_1_w = accumulator.w;
            }
        }
        __syncthreads();
    }

    if(wave_id == 2 || wave_id == 3)
    {
        const int row = lane_id % kQueryTile;
        const int col_base = (lane_id / kQueryTile) * 4 +
            (wave_id == 3 ? kHeadDim / 2 : 0);
        const int query_index = query_start + row;
        const std::int64_t output_base = output_offset(
            width_index,
            pressure_height_index,
            query_index,
            head_index,
            col_base,
            pressure_height,
            tokens,
            heads);
        if(wave_id == 2)
        {
            output[output_base + 0] = __float2half_rn(pv_acc_0_x);
            output[output_base + 1] = __float2half_rn(pv_acc_0_y);
            output[output_base + 2] = __float2half_rn(pv_acc_0_z);
            output[output_base + 3] = __float2half_rn(pv_acc_0_w);
        }
        else
        {
            output[output_base + 0] = __float2half_rn(pv_acc_1_x);
            output[output_base + 1] = __float2half_rn(pv_acc_1_y);
            output[output_base + 2] = __float2half_rn(pv_acc_1_z);
            output[output_base + 3] = __float2half_rn(pv_acc_1_w);
        }
    }
    return;
}

} // namespace

extern "C" int pangu_earth_attention_tiled_fwd_fp16(
    const void* qkv,
    const void* bias_table,
    const void* position_index,
    const void* region_id,
    void* output,
    int width,
    int pressure_height,
    int heads,
    int tokens,
    int bias_rows,
    int mask_width,
    int width_offset,
    float scale,
    hipStream_t stream)
{
    if(qkv == nullptr || bias_table == nullptr || position_index == nullptr ||
       output == nullptr || width <= 0 || pressure_height <= 0 || heads <= 0 ||
       (tokens != 32 && tokens != 144) || bias_rows <= 0 || bias_rows > 32768 ||
       width_offset < 0 || !std::isfinite(scale) || !(scale > 0.0f) ||
       (region_id != nullptr && mask_width <= 0))
    {
        return static_cast<int>(hipErrorInvalidValue);
    }

    const dim3 grid(
        static_cast<unsigned int>((tokens + kQueryTile - 1) / kQueryTile),
        static_cast<unsigned int>(heads * pressure_height),
        static_cast<unsigned int>(width));
    const dim3 block(static_cast<unsigned int>(kBlockThreads));
    if(region_id == nullptr)
    {
        hipLaunchKernelGGL(
            earth_attention_tiled_fwd_fp16_kernel<false>,
            grid,
            block,
            kDynamicSharedBytes,
            stream,
            static_cast<const half*>(qkv),
            static_cast<const half*>(bias_table),
            static_cast<const std::int16_t*>(position_index),
            static_cast<const std::uint8_t*>(region_id),
            static_cast<half*>(output),
            width,
            pressure_height,
            heads,
            tokens,
            bias_rows,
            mask_width,
            width_offset,
            scale);
    }
    else
    {
        hipLaunchKernelGGL(
            earth_attention_tiled_fwd_fp16_kernel<true>,
            grid,
            block,
            kDynamicSharedBytes,
            stream,
            static_cast<const half*>(qkv),
            static_cast<const half*>(bias_table),
            static_cast<const std::int16_t*>(position_index),
            static_cast<const std::uint8_t*>(region_id),
            static_cast<half*>(output),
            width,
            pressure_height,
            heads,
            tokens,
            bias_rows,
            mask_width,
            width_offset,
            scale);
    }
    return static_cast<int>(hipGetLastError());
}

extern "C" int pangu_earth_attention_tiled_full_row_fwd_fp16(
    const void* qkv,
    const void* bias_table,
    const void* position_index,
    const void* region_id,
    void* output,
    int width,
    int pressure_height,
    int heads,
    int tokens,
    int bias_rows,
    int mask_width,
    int width_offset,
    float scale,
    int standard_exp,
    hipStream_t stream)
{
    if(qkv == nullptr || bias_table == nullptr || position_index == nullptr ||
       output == nullptr || width <= 0 || pressure_height <= 0 || heads <= 0 ||
       (tokens != 32 && tokens != 144) || bias_rows <= 0 || bias_rows > 32768 ||
       width_offset < 0 || !std::isfinite(scale) || !(scale > 0.0f) ||
       (standard_exp != 0 && standard_exp != 1) ||
       (region_id != nullptr && mask_width <= 0))
    {
        return static_cast<int>(hipErrorInvalidValue);
    }

    const dim3 grid(
        static_cast<unsigned int>((tokens + kQueryTile - 1) / kQueryTile),
        static_cast<unsigned int>(heads * pressure_height),
        static_cast<unsigned int>(width));
    const dim3 block(static_cast<unsigned int>(kBlockThreads));
    if(region_id == nullptr && standard_exp == 0)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                earth_attention_tiled_full_row_fwd_fp16_kernel<
                    false, false, false>),
            grid,
            block,
            kParityDynamicSharedBytes,
            stream,
            static_cast<const half*>(qkv),
            static_cast<const half*>(bias_table),
            static_cast<const std::int16_t*>(position_index),
            static_cast<const std::uint8_t*>(region_id),
            static_cast<half*>(output),
            static_cast<half*>(nullptr),
            static_cast<half*>(nullptr),
            static_cast<half*>(nullptr),
            width,
            pressure_height,
            heads,
            tokens,
            bias_rows,
            mask_width,
            width_offset,
            scale);
    }
    else if(region_id == nullptr)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                earth_attention_tiled_full_row_fwd_fp16_kernel<
                    false, true, false>),
            grid,
            block,
            kParityDynamicSharedBytes,
            stream,
            static_cast<const half*>(qkv),
            static_cast<const half*>(bias_table),
            static_cast<const std::int16_t*>(position_index),
            static_cast<const std::uint8_t*>(region_id),
            static_cast<half*>(output),
            static_cast<half*>(nullptr),
            static_cast<half*>(nullptr),
            static_cast<half*>(nullptr),
            width,
            pressure_height,
            heads,
            tokens,
            bias_rows,
            mask_width,
            width_offset,
            scale);
    }
    else if(standard_exp == 0)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                earth_attention_tiled_full_row_fwd_fp16_kernel<
                    true, false, false>),
            grid,
            block,
            kParityDynamicSharedBytes,
            stream,
            static_cast<const half*>(qkv),
            static_cast<const half*>(bias_table),
            static_cast<const std::int16_t*>(position_index),
            static_cast<const std::uint8_t*>(region_id),
            static_cast<half*>(output),
            static_cast<half*>(nullptr),
            static_cast<half*>(nullptr),
            static_cast<half*>(nullptr),
            width,
            pressure_height,
            heads,
            tokens,
            bias_rows,
            mask_width,
            width_offset,
            scale);
    }
    else
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                earth_attention_tiled_full_row_fwd_fp16_kernel<
                    true, true, false>),
            grid,
            block,
            kParityDynamicSharedBytes,
            stream,
            static_cast<const half*>(qkv),
            static_cast<const half*>(bias_table),
            static_cast<const std::int16_t*>(position_index),
            static_cast<const std::uint8_t*>(region_id),
            static_cast<half*>(output),
            static_cast<half*>(nullptr),
            static_cast<half*>(nullptr),
            static_cast<half*>(nullptr),
            width,
            pressure_height,
            heads,
            tokens,
            bias_rows,
            mask_width,
            width_offset,
            scale);
    }
    return static_cast<int>(hipGetLastError());
}

extern "C" int pangu_earth_attention_tiled_full_row_diagnostic_fp16(
    const void* qkv,
    const void* bias_table,
    const void* position_index,
    const void* region_id,
    void* output,
    void* diagnostic_qk,
    void* diagnostic_biased,
    void* diagnostic_probability,
    int width,
    int pressure_height,
    int heads,
    int tokens,
    int bias_rows,
    int mask_width,
    int width_offset,
    float scale,
    int standard_exp,
    hipStream_t stream)
{
    if(qkv == nullptr || bias_table == nullptr || position_index == nullptr ||
       output == nullptr || diagnostic_qk == nullptr ||
       diagnostic_biased == nullptr || diagnostic_probability == nullptr ||
       width <= 0 || pressure_height <= 0 || heads <= 0 ||
       (tokens != 32 && tokens != 144) || bias_rows <= 0 || bias_rows > 32768 ||
       width_offset < 0 || !std::isfinite(scale) || !(scale > 0.0f) ||
       (standard_exp != 0 && standard_exp != 1) ||
       (region_id != nullptr && mask_width <= 0))
    {
        return static_cast<int>(hipErrorInvalidValue);
    }

    const dim3 grid(
        static_cast<unsigned int>((tokens + kQueryTile - 1) / kQueryTile),
        static_cast<unsigned int>(heads * pressure_height),
        static_cast<unsigned int>(width));
    const dim3 block(static_cast<unsigned int>(kBlockThreads));
#define PANGU_LAUNCH_FULL_ROW_DIAGNOSTIC(SHIFTED, STANDARD)                    \
    hipLaunchKernelGGL(                                                       \
        HIP_KERNEL_NAME(earth_attention_tiled_full_row_fwd_fp16_kernel<      \
            SHIFTED, STANDARD, true>),                                        \
        grid,                                                                 \
        block,                                                                \
        kParityDynamicSharedBytes,                                            \
        stream,                                                               \
        static_cast<const half*>(qkv),                                        \
        static_cast<const half*>(bias_table),                                 \
        static_cast<const std::int16_t*>(position_index),                     \
        static_cast<const std::uint8_t*>(region_id),                          \
        static_cast<half*>(output),                                           \
        static_cast<half*>(diagnostic_qk),                                    \
        static_cast<half*>(diagnostic_biased),                                \
        static_cast<half*>(diagnostic_probability),                           \
        width,                                                                \
        pressure_height,                                                      \
        heads,                                                                \
        tokens,                                                               \
        bias_rows,                                                            \
        mask_width,                                                           \
        width_offset,                                                         \
        scale)

    if(region_id == nullptr && standard_exp == 0)
    {
        PANGU_LAUNCH_FULL_ROW_DIAGNOSTIC(false, false);
    }
    else if(region_id == nullptr)
    {
        PANGU_LAUNCH_FULL_ROW_DIAGNOSTIC(false, true);
    }
    else if(standard_exp == 0)
    {
        PANGU_LAUNCH_FULL_ROW_DIAGNOSTIC(true, false);
    }
    else
    {
        PANGU_LAUNCH_FULL_ROW_DIAGNOSTIC(true, true);
    }
#undef PANGU_LAUNCH_FULL_ROW_DIAGNOSTIC
    return static_cast<int>(hipGetLastError());
}

extern "C" int pangu_earth_attention_tiled_get_config(
    int* query_tile,
    int* key_tile,
    int* head_dim,
    int* block_threads,
    std::size_t* dynamic_shared_bytes)
{
    if(query_tile == nullptr || key_tile == nullptr || head_dim == nullptr ||
       block_threads == nullptr || dynamic_shared_bytes == nullptr)
    {
        return static_cast<int>(hipErrorInvalidValue);
    }
    *query_tile = kQueryTile;
    *key_tile = kKeyTile;
    *head_dim = kHeadDim;
    *block_threads = kBlockThreads;
    *dynamic_shared_bytes = kDynamicSharedBytes;
    return static_cast<int>(hipSuccess);
}

extern "C" int pangu_earth_attention_tiled_get_occupancy(
    int* active_blocks_per_multiprocessor)
{
    if(active_blocks_per_multiprocessor == nullptr)
    {
        return static_cast<int>(hipErrorInvalidValue);
    }
    const hipError_t status = hipOccupancyMaxActiveBlocksPerMultiprocessor(
        active_blocks_per_multiprocessor,
        earth_attention_tiled_fwd_fp16_kernel<false>,
        kBlockThreads,
        kDynamicSharedBytes);
    return static_cast<int>(status);
}

extern "C" int pangu_earth_attention_tiled_full_row_get_config(
    int* query_tile,
    int* key_tile,
    int* head_dim,
    int* block_threads,
    std::size_t* dynamic_shared_bytes)
{
    if(query_tile == nullptr || key_tile == nullptr || head_dim == nullptr ||
       block_threads == nullptr || dynamic_shared_bytes == nullptr)
    {
        return static_cast<int>(hipErrorInvalidValue);
    }
    *query_tile = kQueryTile;
    *key_tile = kKeyTile;
    *head_dim = kHeadDim;
    *block_threads = kBlockThreads;
    *dynamic_shared_bytes = kParityDynamicSharedBytes;
    return static_cast<int>(hipSuccess);
}

extern "C" int pangu_earth_attention_tiled_full_row_get_occupancy(
    int standard_exp,
    int* active_blocks_per_multiprocessor)
{
    if((standard_exp != 0 && standard_exp != 1) ||
       active_blocks_per_multiprocessor == nullptr)
    {
        return static_cast<int>(hipErrorInvalidValue);
    }
    hipError_t status;
    if(standard_exp == 0)
    {
        status = hipOccupancyMaxActiveBlocksPerMultiprocessor(
            active_blocks_per_multiprocessor,
            earth_attention_tiled_full_row_fwd_fp16_kernel<false, false, false>,
            kBlockThreads,
            kParityDynamicSharedBytes);
    }
    else
    {
        status = hipOccupancyMaxActiveBlocksPerMultiprocessor(
            active_blocks_per_multiprocessor,
            earth_attention_tiled_full_row_fwd_fp16_kernel<false, true, false>,
            kBlockThreads,
            kParityDynamicSharedBytes);
    }
    return static_cast<int>(status);
}

extern "C" const char* pangu_earth_attention_tiled_implementation_kind()
{
    return "matrix_tiled_mmac_online_fp32_gfx936";
}

extern "C" const char* pangu_earth_attention_tiled_full_row_implementation_kind(
    int standard_exp)
{
    return standard_exp == 0
        ? "matrix_tiled_mmac_full_row_fp16_fast_exp_gfx936"
        : "matrix_tiled_mmac_full_row_fp16_standard_exp_gfx936";
}

extern "C" const char* pangu_earth_attention_tiled_error_string(int status)
{
    return hipGetErrorString(static_cast<hipError_t>(status));
}
