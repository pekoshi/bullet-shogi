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

// dw[co,ci,kh,kw] += sum_{n,ho,wo} x[n,ci,hi,wi] * dy[n,co,ho,wo]
// hi = ho*Stride + kh - Pad
extern "C" __global__ void kernel(
    const int N,
    const float* __restrict__ x,
    const float* __restrict__ dy,
    float* __restrict__ dw)
{
    const int kw = blockIdx.x * blockDim.x + threadIdx.x;
    const int kh = blockIdx.y * blockDim.y + threadIdx.y;
    const int co_ci = blockIdx.z;
    if (kw >= K || kh >= K || co_ci >= Cout * Cin) return;

    const int co = co_ci / Cin;
    const int ci = co_ci % Cin;

    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        #pragma unroll
        for (int ho = 0; ho < Hout; ++ho) {
            const int hi = ho * Stride + kh - Pad;
            if (hi < 0 || hi >= H) continue;
            #pragma unroll
            for (int wo = 0; wo < Wout; ++wo) {
                const int wi = wo * Stride + kw - Pad;
                if (wi < 0 || wi >= W) continue;
                const float xv  = x[((n * Cin + ci) * H + hi) * W + wi];
                const float dyv = dy[((n * Cout + co) * Hout + ho) * Wout + wo];
                acc += xv * dyv;
            }
        }
    }
    // weight grad は accumulate (multiple step の training で重ねる)
    dw[((co * Cin + ci) * K + kh) * K + kw] += acc;
}
