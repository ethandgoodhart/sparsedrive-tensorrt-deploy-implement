#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

#define DIVUP(m, n) ((m + n - 1) / n)

// =========================================================================
// 🎯 原子加万能补丁 (支持 float 和 __half)
// =========================================================================
__device__ __forceinline__ void dfa_atomicAdd(float* addr, float val) { atomicAdd(addr, val); }

__device__ __forceinline__ void dfa_atomicAdd(__half* addr, __half val) {
#if __CUDA_ARCH__ >= 700
    atomicAdd(addr, val);
#else
    unsigned int *addr_as_ui = (unsigned int *)((char *)addr - ((size_t)addr & 2));
    unsigned int old = *addr_as_ui;
    unsigned int assumed;
    do {
        assumed = old;
        unsigned short h_raw = (size_t)addr & 2 ? (old >> 16) : (old & 0xFFFF);
        __half h_val = *reinterpret_cast<__half*>(&h_raw);
        __half h_sum = __hadd(h_val, val);
        unsigned short res_raw = *reinterpret_cast<unsigned short*>(&h_sum);
        unsigned int new_val = (size_t)addr & 2 ? (old & 0xFFFF) | (res_raw << 16) : (old & 0xFFFF0000) | res_raw;
        old = atomicCAS(addr_as_ui, assumed, new_val);
    } while (assumed != old);
#endif
}

// =========================================================================
// 1. Map-head kernel: block-per-(anchor, vec_group), warp-shuffle reduction.
//    Replaces the atomic-add version. One warp (32 threads) per output cell;
//    each thread iterates over a stride of (cam, scale, point) and accumulates
//    into 8 partial sums; final reduction via __shfl_xor_sync.
//    No atomics, no shared memory, fully coalesced output writes.
// =========================================================================
template<typename T>
__global__ void dfa_block_reduce_kernel(
    const T* __restrict__ mc_ms_feat, const int* __restrict__ spatial_shape,
    const int* __restrict__ scale_start_index, const T* __restrict__ sample_location,
    const T* __restrict__ weights, T* __restrict__ output,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_anchors, int num_pts, int num_groups, bool is_cam_shared) {

    // Grid: (num_anchors * batch_size, num_vec). Block: 32 threads (one warp).
    const int real_a_idx = blockIdx.x;
    const int v_idx      = blockIdx.y;
    const int lane       = threadIdx.x;  // 0..31

    const int b_idx   = real_a_idx / num_anchors;
    const int a_idx   = real_a_idx % num_anchors;
    const int c_start = v_idx * 8;
    const int g_idx   = c_start / (num_embeds / num_groups);

    const int total_work = num_pts * num_cams * num_scale;

    float fres[8] = {0.f};

    // Stride loop over (point, cam, scale) — 32-wide.
    for (int w_idx = lane; w_idx < total_work; w_idx += 32) {
        int s   = w_idx % num_scale;
        int cam = (w_idx / num_scale) % num_cams;
        int p   = w_idx / (num_scale * num_cams);

        int loc_ptr = ((real_a_idx * num_pts + p) * num_cams + cam) << 1;
        float lw = (float)sample_location[loc_ptr];
        float lh = (float)sample_location[loc_ptr + 1];
        if (lw <= -0.5f || lw >= 1.5f || lh <= -0.5f || lh >= 1.5f) continue;

        int s_idx = is_cam_shared ? s : (cam * num_scale + s);
        int h = spatial_shape[s_idx << 1];
        int W = spatial_shape[(s_idx << 1) + 1];
        int cam_off = is_cam_shared ? (cam * (num_feat / num_cams)) : 0;
        int w_base = (((real_a_idx * num_pts + p) * num_cams + cam) * num_scale + s) * num_groups;
        float wt = (float)weights[w_base + g_idx];

        const T* f_ptr = mc_ms_feat
            + (static_cast<size_t>(b_idx * num_feat + cam_off + scale_start_index[s_idx]) * num_embeds)
            + c_start;

        float py = lh * h - 0.5f, px = lw * W - 0.5f;
        int yl = floorf(py), xl = floorf(px);
        float dy = py - yl, dx = px - xl;
        float w00 = (1.f - dy) * (1.f - dx) * wt;
        float w01 = (1.f - dy) * dx        * wt;
        float w10 = dy        * (1.f - dx) * wt;
        float w11 = dy        * dx         * wt;

        bool v00_ok = (yl   >= 0 && yl   < h && xl   >= 0 && xl   < W);
        bool v01_ok = (yl   >= 0 && yl   < h && xl+1 >= 0 && xl+1 < W);
        bool v10_ok = (yl+1 >= 0 && yl+1 < h && xl   >= 0 && xl   < W);
        bool v11_ok = (yl+1 >= 0 && yl+1 < h && xl+1 >= 0 && xl+1 < W);

        const T* p00 = f_ptr + (yl  ) * W * num_embeds + (xl  ) * num_embeds;
        const T* p01 = f_ptr + (yl  ) * W * num_embeds + (xl+1) * num_embeds;
        const T* p10 = f_ptr + (yl+1) * W * num_embeds + (xl  ) * num_embeds;
        const T* p11 = f_ptr + (yl+1) * W * num_embeds + (xl+1) * num_embeds;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float v00 = v00_ok ? (float)p00[i] : 0.f;
            float v01 = v01_ok ? (float)p01[i] : 0.f;
            float v10 = v10_ok ? (float)p10[i] : 0.f;
            float v11 = v11_ok ? (float)p11[i] : 0.f;
            fres[i] += w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
        }
    }

    // Warp-level reduction across 32 lanes.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            fres[i] += __shfl_xor_sync(0xffffffff, fres[i], offset);
        }
    }

    if (lane == 0) {
        T* out_ptr = output + (static_cast<size_t>(b_idx * num_anchors + a_idx) * num_embeds) + c_start;
        #pragma unroll
        for (int i = 0; i < 8; ++i) out_ptr[i] = (T)fres[i];
    }
}

// =========================================================================
// 2. Det 专用：高带宽向量化读取 (针对 13 Pts 场景)
// =========================================================================
template<typename T>
__global__ void dfa_vector_kernel(
    const T* __restrict__ mc_ms_feat, const int* __restrict__ spatial_shape,
    const int* __restrict__ scale_start_index, const T* __restrict__ sample_location,
    const T* __restrict__ weights, T* __restrict__ output,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_anchors, int num_pts, int num_groups, bool is_cam_shared) {
    
    int num_vec = num_embeds / 8;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * num_anchors * num_vec) return;

    int v_idx = tid % num_vec;
    int a_idx = (tid / num_vec) % num_anchors;
    int b_idx = tid / (num_vec * num_anchors);
    int real_a_idx = b_idx * num_anchors + a_idx;
    int c_start = v_idx * 8;
    int g_idx = c_start / (num_embeds / num_groups);

    float fres[8] = {0.0f};

    for (int p = 0; p < num_pts; ++p) {
        for (int cam = 0; cam < num_cams; ++cam) {
            int loc_ptr = ((real_a_idx * num_pts + p) * num_cams + cam) << 1;
            float lw = (float)sample_location[loc_ptr], lh = (float)sample_location[loc_ptr+1];
            if (lw <= -0.5f || lw >= 1.5f || lh <= -0.5f || lh >= 1.5f) continue;
            
            int cam_off = is_cam_shared ? (cam * (num_feat / num_cams)) : 0;
            int w_base = (((real_a_idx * num_pts + p) * num_cams + cam) * num_scale) * num_groups;

            for (int s = 0; s < num_scale; ++s) {
                int s_idx = is_cam_shared ? s : (cam * num_scale + s);
                int h = spatial_shape[s_idx << 1], w = spatial_shape[(s_idx << 1) + 1];
                float weight = (float)weights[w_base + s * num_groups + g_idx];
                const T* f_ptr = mc_ms_feat + (static_cast<size_t>(b_idx * num_feat + cam_off + scale_start_index[s_idx]) * num_embeds) + c_start;

                float py = lh * h - 0.5f, px = lw * w - 0.5f;
                int yl = floorf(py), xl = floorf(px);
                float dy = py - yl, dx = px - xl;

                #pragma unroll
                for(int i=0; i<8; ++i) {
                    float v00=0, v01=0, v10=0, v11=0;
                    if (yl>=0 && xl>=0 && yl<h && xl<w) v00 = (float)f_ptr[yl*w*num_embeds + xl*num_embeds + i];
                    if (yl>=0 && xl+1<w && yl<h && xl+1>=0) v01 = (float)f_ptr[yl*w*num_embeds + (xl+1)*num_embeds + i];
                    if (yl+1<h && xl>=0 && yl+1>=0 && xl<w) v10 = (float)f_ptr[(yl+1)*w*num_embeds + xl*num_embeds + i];
                    if (yl+1<h && xl+1<w && yl+1>=0 && xl+1>=0) v11 = (float)f_ptr[(yl+1)*w*num_embeds + (xl+1)*num_embeds + i];
                    fres[i] += ((1.f-dy)*(1.f-dx)*v00 + (1.f-dy)*dx*v01 + dy*(1.f-dx)*v10 + dy*dx*v11) * weight;
                }
            }
        }
    }
    T* out_ptr = output + (static_cast<size_t>(tid) * 8);
    #pragma unroll
    for(int i=0; i<8; ++i) { out_ptr[i] = (T)fres[i]; }
}

// =========================================================================
// int64 -> int32 cast (for TRT 10 ONNX parser which emits INT64 by default)
// =========================================================================
__global__ void cast_i64_to_i32_kernel(const int64_t* __restrict__ in, int* __restrict__ out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) out[tid] = static_cast<int>(in[tid]);
}

extern "C" void CastInt64ToInt32(const void* in, void* out, int n, cudaStream_t stream) {
    if (n <= 0) return;
    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    cast_i64_to_i32_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const int64_t*>(in), static_cast<int*>(out), n);
}

// =========================================================================
// 3. Launcher: 智能选择 Kernel
// =========================================================================
extern "C" void DeformableAggregationLauncher(
    const void* mc_ms_feat, const int* spatial_shape, const int* scale_start_index,
    const void* sample_location, const void* weights, void* output,
    int batch_size, int num_cams, int num_feat, int num_embeds,
    int num_scale, int num_anchors, int num_pts, int num_groups,
    bool is_cam_shared, bool is_fp16, cudaStream_t stream) 
{
    // Dispatch: map head (pts ~ 300) → block-reduce; det head (pts ~ 13) → vector kernel.
    bool use_block_reduce = (num_pts > 64);
    int num_vec = num_embeds / 8;

    if (is_fp16) {
        if (use_block_reduce) {
            dim3 grid(batch_size * num_anchors, num_vec);
            dfa_block_reduce_kernel<__half><<<grid, 32, 0, stream>>>(
                (const __half*)mc_ms_feat, spatial_shape, scale_start_index,
                (const __half*)sample_location, (const __half*)weights,
                (__half*)output, batch_size, num_cams, num_feat, num_embeds,
                num_scale, num_anchors, num_pts, num_groups, is_cam_shared);
        } else {
            int threads = batch_size * num_anchors * num_vec;
            dfa_vector_kernel<__half><<<DIVUP(threads, 256), 256, 0, stream>>>(
                (const __half*)mc_ms_feat, spatial_shape, scale_start_index,
                (const __half*)sample_location, (const __half*)weights,
                (__half*)output, batch_size, num_cams, num_feat, num_embeds,
                num_scale, num_anchors, num_pts, num_groups, is_cam_shared);
        }
    } else {
        if (use_block_reduce) {
            dim3 grid(batch_size * num_anchors, num_vec);
            dfa_block_reduce_kernel<float><<<grid, 32, 0, stream>>>(
                (const float*)mc_ms_feat, spatial_shape, scale_start_index,
                (const float*)sample_location, (const float*)weights,
                (float*)output, batch_size, num_cams, num_feat, num_embeds,
                num_scale, num_anchors, num_pts, num_groups, is_cam_shared);
        } else {
            int threads = batch_size * num_anchors * num_vec;
            dfa_vector_kernel<float><<<DIVUP(threads, 256), 256, 0, stream>>>(
                (const float*)mc_ms_feat, spatial_shape, scale_start_index,
                (const float*)sample_location, (const float*)weights,
                (float*)output, batch_size, num_cams, num_feat, num_embeds,
                num_scale, num_anchors, num_pts, num_groups, is_cam_shared);
        }
    }
}