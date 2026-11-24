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

// dx[n,ci,hi,wi] = sum_{co,kh,kw} dy[n,co,ho,wo] * w[co,ci,kh,kw]
// ho = (hi + Pad - kh) / Stride、 wo 同様。 Stride で割り切れ ho ∈ [0, Hout) のとき有効
extern "C" __global__ void kernel(
    const int N,
    const float* __restrict__ dy,
    const float* __restrict__ w,
    float* __restrict__ dx)
{
    const int wi = blockIdx.x * blockDim.x + threadIdx.x;
    const int hi = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc = blockIdx.z;
    if (wi >= W || hi >= H || nc >= N * Cin) return;

    const int n  = nc / Cin;
    const int ci = nc % Cin;

    float acc = 0.0f;
    #pragma unroll
    for (int co = 0; co < Cout; ++co) {
        #pragma unroll
        for (int kh = 0; kh < K; ++kh) {
            const int ho_num = hi + Pad - kh;
            if (ho_num < 0 || (ho_num % Stride) != 0) continue;
            const int ho = ho_num / Stride;
            if (ho >= Hout) continue;
            #pragma unroll
            for (int kw = 0; kw < K; ++kw) {
                const int wo_num = wi + Pad - kw;
                if (wo_num < 0 || (wo_num % Stride) != 0) continue;
                const int wo = wo_num / Stride;
                if (wo >= Wout) continue;
                const float dyv = dy[((n * Cout + co) * Hout + ho) * Wout + wo];
                const float wv  = w[((co * Cin + ci) * K + kh) * K + kw];
                acc += dyv * wv;
            }
        }
    }
    // bullet の backward は grad を累積する慣行 (other op が同じ buffer に書き込む可能性)
    dx[((n * Cin + ci) * H + hi) * W + wi] += acc;
}
