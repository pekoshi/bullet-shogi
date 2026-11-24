#ifndef STUFF
#define DECL_CIN 192
#define DECL_COUT 192
#define DECL_H 9
#define DECL_W 9
#define DECL_HOUT 9
#define DECL_WOUT 9
#define DECL_K 3
#define DECL_PAD 1
#define DECL_STRIDE 1
#endif

constexpr int Cin = DECL_CIN;
constexpr int Cout = DECL_COUT;
constexpr int H = DECL_H;
constexpr int W = DECL_W;
constexpr int Hout = DECL_HOUT;
constexpr int Wout = DECL_WOUT;
constexpr int K = DECL_K;
constexpr int Pad = DECL_PAD;
constexpr int Stride = DECL_STRIDE;

extern "C" __global__ void kernel(
    const int N,
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y)
{
    const int wo = blockIdx.x * blockDim.x + threadIdx.x;
    const int ho = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc = blockIdx.z;
    if (wo >= Wout || ho >= Hout || nc >= N * Cout) return;

    const int n  = nc / Cout;
    const int co = nc % Cout;

    float acc = 0.0f;
    #pragma unroll
    for (int ci = 0; ci < Cin; ++ci) {
        #pragma unroll
        for (int kh = 0; kh < K; ++kh) {
            const int hi = ho * Stride + kh - Pad;
            if (hi < 0 || hi >= H) continue;
            #pragma unroll
            for (int kw = 0; kw < K; ++kw) {
                const int wi = wo * Stride + kw - Pad;
                if (wi < 0 || wi >= W) continue;
                const float xv = x[((n * Cin + ci) * H + hi) * W + wi];
                const float wv = w[((co * Cin + ci) * K + kh) * K + kw];
                acc += xv * wv;
            }
        }
    }
    y[((n * Cout + co) * Hout + ho) * Wout + wo] = acc;
}
