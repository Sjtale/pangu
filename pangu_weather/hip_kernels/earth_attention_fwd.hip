#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kHeadDim = 32;

struct Strides5
{
    int64_t value[5];
};

struct Strides4
{
    int64_t value[4];
};

__global__ void earth_attention_fwd_fp16_kernel(const half* q,
                                                const half* k,
                                                const half* v,
                                                const half* earth_bias,
                                                const half* shifted_mask,
                                                half* output,
                                                Strides5 q_stride,
                                                Strides5 k_stride,
                                                Strides5 v_stride,
                                                Strides4 earth_stride,
                                                Strides4 mask_stride,
                                                Strides5 output_stride,
                                                int heads,
                                                int pressure_height,
                                                int tokens,
                                                float scale)
{
    const int tid         = threadIdx.x;
    const int query_index = blockIdx.x % tokens;
    const int attention   = blockIdx.x / tokens;
    const int ph_index    = attention % pressure_height;
    const int width_head  = attention / pressure_height;
    const int head_index  = width_head % heads;
    const int width_index = width_head / heads;

    __shared__ half q_shared[kHeadDim];
    __shared__ float scratch[kThreads];

    if(tid < kHeadDim)
    {
        const int64_t q_offset = width_index * q_stride.value[0] +
                                 head_index * q_stride.value[1] +
                                 ph_index * q_stride.value[2] +
                                 query_index * q_stride.value[3] +
                                 tid * q_stride.value[4];
        q_shared[tid] = q[q_offset];
    }
    __syncthreads();

    float score = -INFINITY;
    if(tid < tokens)
    {
        const int key_index = tid;
        float qk            = 0.0f;
        const int64_t k_base = width_index * k_stride.value[0] +
                               head_index * k_stride.value[1] +
                               ph_index * k_stride.value[2] +
                               key_index * k_stride.value[3];
        for(int dim = 0; dim < kHeadDim; ++dim)
        {
            qk += __half2float(q_shared[dim]) *
                  __half2float(k[k_base + dim * k_stride.value[4]]);
        }

        const int64_t earth_offset = head_index * earth_stride.value[0] +
                                     ph_index * earth_stride.value[1] +
                                     query_index * earth_stride.value[2] +
                                     key_index * earth_stride.value[3];
        const int64_t mask_offset = width_index * mask_stride.value[0] +
                                    ph_index * mask_stride.value[1] +
                                    query_index * mask_stride.value[2] +
                                    key_index * mask_stride.value[3];
        score = qk * scale + __half2float(earth_bias[earth_offset]) +
                __half2float(shifted_mask[mask_offset]);
    }

    scratch[tid] = score;
    __syncthreads();
    for(int offset = kThreads / 2; offset > 0; offset >>= 1)
    {
        if(tid < offset)
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + offset]);
        __syncthreads();
    }

    const float row_max = scratch[0];
    const float weight  = tid < tokens ? expf(score - row_max) : 0.0f;
    scratch[tid]        = weight;
    __syncthreads();
    for(int offset = kThreads / 2; offset > 0; offset >>= 1)
    {
        if(tid < offset)
            scratch[tid] += scratch[tid + offset];
        __syncthreads();
    }

    const float inv_sum = scratch[0] > 0.0f ? 1.0f / scratch[0] : 0.0f;
    if(tid < tokens)
        scratch[tid] = weight * inv_sum;
    __syncthreads();

    if(tid < kHeadDim)
    {
        float accumulator = 0.0f;
        for(int key_index = 0; key_index < tokens; ++key_index)
        {
            const int64_t v_offset = width_index * v_stride.value[0] +
                                     head_index * v_stride.value[1] +
                                     ph_index * v_stride.value[2] +
                                     key_index * v_stride.value[3] +
                                     tid * v_stride.value[4];
            accumulator += scratch[key_index] * __half2float(v[v_offset]);
        }
        const int64_t output_offset = width_index * output_stride.value[0] +
                                      head_index * output_stride.value[1] +
                                      ph_index * output_stride.value[2] +
                                      query_index * output_stride.value[3] +
                                      tid * output_stride.value[4];
        output[output_offset] = __float2half_rn(accumulator);
    }
}

template <int N>
void copy_strides(int64_t (&destination)[N], const int64_t* source)
{
    for(int i = 0; i < N; ++i)
        destination[i] = source[i];
}

} // namespace

extern "C" int pangu_earth_attention_fwd_fp16(const void* q,
                                               const void* k,
                                               const void* v,
                                               const void* earth_bias,
                                               const void* shifted_mask,
                                               void* output,
                                               const int64_t* q_stride,
                                               const int64_t* k_stride,
                                               const int64_t* v_stride,
                                               const int64_t* earth_stride,
                                               const int64_t* mask_stride,
                                               const int64_t* output_stride,
                                               int width,
                                               int heads,
                                               int pressure_height,
                                               int tokens,
                                               int head_dim,
                                               float scale,
                                               hipStream_t stream)
{
    if(q == nullptr || k == nullptr || v == nullptr || earth_bias == nullptr ||
       shifted_mask == nullptr || output == nullptr || q_stride == nullptr ||
       k_stride == nullptr || v_stride == nullptr || earth_stride == nullptr ||
       mask_stride == nullptr || output_stride == nullptr || width <= 0 ||
       heads <= 0 || pressure_height <= 0 || tokens <= 0 || tokens > kThreads ||
       head_dim != kHeadDim)
        return static_cast<int>(hipErrorInvalidValue);

    Strides5 q_s{}, k_s{}, v_s{}, output_s{};
    Strides4 earth_s{}, mask_s{};
    copy_strides(q_s.value, q_stride);
    copy_strides(k_s.value, k_stride);
    copy_strides(v_s.value, v_stride);
    copy_strides(earth_s.value, earth_stride);
    copy_strides(mask_s.value, mask_stride);
    copy_strides(output_s.value, output_stride);

    const unsigned int blocks =
        static_cast<unsigned int>(width * heads * pressure_height * tokens);
    hipLaunchKernelGGL(earth_attention_fwd_fp16_kernel,
                       dim3(blocks),
                       dim3(kThreads),
                       0,
                       stream,
                       static_cast<const half*>(q),
                       static_cast<const half*>(k),
                       static_cast<const half*>(v),
                       static_cast<const half*>(earth_bias),
                       static_cast<const half*>(shifted_mask),
                       static_cast<half*>(output),
                       q_s,
                       k_s,
                       v_s,
                       earth_s,
                       mask_s,
                       output_s,
                       heads,
                       pressure_height,
                       tokens,
                       scale);
    return static_cast<int>(hipGetLastError());
}

extern "C" const char* pangu_hip_error_string(int status)
{
    return hipGetErrorString(static_cast<hipError_t>(status));
}
