/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "kernel_operator.h"
#ifndef FLA_NPU_REGBASE_HPP_INCLUDED
#define FLA_NPU_REGBASE_HPP_INCLUDED
#include "kernel_utils/vector/regbase.hpp"
#endif

namespace KdaPreciseState {
using namespace AscendC;

template <typename T, typename BetaT, typename Tiling>
class Kernel {
public:
    __aicore__ inline void Run(GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR gk,
                               GM_ADDR initial, GM_ADDR cuSeqlens, GM_ADDR finalState,
                               const Tiling &tiling)
    {
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(k));
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(v));
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ BetaT *>(beta));
        gk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gk));
        initial_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initial));
        final_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(finalState));
        const uint64_t sub = GetSubBlockIdx();
        const uint64_t core = GetBlockIdx() / GetSubBlockNum();
        const uint64_t cores = tiling.outputUsedCoreNum;
        const uint64_t dimK = tiling.kHeadDim;
        const uint64_t dimV = tiling.vHeadDim;
        const uint64_t countV = dimV / 2;
        const uint64_t col = sub * countV;
        const uint64_t stride = ((countV + 63) / 64) * 64;
        pipe_.InitBuffer(buf_, 144 * 1024);
        auto state = buf_.template Get<float>();
        auto previous = state[32768];
        auto current = state[33024];
        auto decay = state[33280];
        auto key = state[33536];
        auto value = state[33792];
        auto betaFloat = state[33920];
        auto keyTyped = buf_.template Get<T>()[33984 * sizeof(float) / sizeof(T)];
        auto valueTyped = keyTyped[256];
        auto betaTyped = buf_.template Get<BetaT>()[(33984 * sizeof(float) + 768) / sizeof(BetaT)];
        auto m2v = pipe_.template AllocEventID<HardEvent::MTE2_V>();
        auto v2m2 = pipe_.template AllocEventID<HardEvent::V_MTE2>();
        auto v2m3 = pipe_.template AllocEventID<HardEvent::V_MTE3>();
        auto m32v = pipe_.template AllocEventID<HardEvent::MTE3_V>();
        __gm__ int64_t *cu = reinterpret_cast<__gm__ int64_t *>(cuSeqlens);
        for (uint64_t work = core; work < tiling.seqNum * tiling.vHeadNum; work += cores) {
            const uint64_t seq = work / tiling.vHeadNum;
            const uint64_t hv = work % tiling.vHeadNum;
            const uint64_t hk = hv / (tiling.vHeadNum / tiling.qHeadNum);
            const uint64_t b = tiling.isVarLen ? 0 : seq;
            const uint64_t begin = tiling.isVarLen ? static_cast<uint64_t>(cu[seq]) : 0;
            const uint64_t end = tiling.isVarLen ? static_cast<uint64_t>(cu[seq + 1]) : tiling.seqlen;
            const uint64_t stateBase = work * dimK * dimV + col;
            if (tiling.hasInitialState) {
                DataCopyExtParams params{static_cast<uint16_t>(dimK),
                    static_cast<uint32_t>(countV * sizeof(float)),
                    static_cast<uint32_t>((dimV - countV) * sizeof(float)),
                    static_cast<uint32_t>((stride - countV) * sizeof(float) / 32), 0};
                DataCopyPad(state, initial_[stateBase], params, DataCopyPadExtParams<float>{false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>(m2v);
                WaitFlag<HardEvent::MTE2_V>(m2v);
            } else {
                Duplicate(state, 0.0f, static_cast<uint32_t>(dimK * stride));
                PipeBarrier<PIPE_V>();
            }
            for (uint64_t token = begin; token < end; ++token) {
                if ((token - begin) % tiling.chunkSize == 0) {
                    Duplicate(previous, 0.0f, static_cast<uint32_t>(dimK));
                    PipeBarrier<PIPE_V>();
                }
                const uint64_t kRow = tiling.inputSequenceMajor
                    ? (b * tiling.seqlen + token) * tiling.qHeadNum + hk
                    : (b * tiling.qHeadNum + hk) * tiling.seqlen + token;
                const uint64_t vRow = tiling.inputSequenceMajor
                    ? (b * tiling.seqlen + token) * tiling.vHeadNum + hv
                    : (b * tiling.vHeadNum + hv) * tiling.seqlen + token;
                const uint64_t gRow = (b * tiling.vHeadNum + hv) * tiling.seqlen + token;
                DataCopy(keyTyped, k_[kRow * dimK], static_cast<uint32_t>(dimK));
                DataCopy(current, gk_[gRow * dimK], static_cast<uint32_t>(dimK));
                DataCopyPad(valueTyped, v_[vRow * dimV + col],
                    DataCopyExtParams{1, static_cast<uint32_t>(countV * sizeof(T)), 0, 0, 0},
                    DataCopyPadExtParams<T>{false, 0, 0, 0});
                DataCopyPad(betaTyped, beta_[gRow],
                    DataCopyExtParams{1, sizeof(BetaT), 0, 0, 0},
                    DataCopyPadExtParams<BetaT>{false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>(m2v);
                WaitFlag<HardEvent::MTE2_V>(m2v);
                Cast(key, keyTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(dimK));
                Cast(value, valueTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(countV));
                if constexpr (IsSameType<BetaT, float>::value) {
                    Adds(betaFloat, betaTyped, 0.0f, 1);
                } else {
                    Cast(betaFloat, betaTyped, RoundMode::CAST_NONE, 1);
                }
                Sub(decay, current, previous, static_cast<uint32_t>(dimK));
                PipeBarrier<PIPE_V>();
                Muls(decay, decay, 0.6931471805599453f, static_cast<uint32_t>(dimK));
                Adds(previous, current, 0.0f, static_cast<uint32_t>(dimK));
                PipeBarrier<PIPE_V>();
                Exp(decay, decay, static_cast<uint32_t>(dimK));
                PipeBarrier<PIPE_V>();
                Update(state, key, value, betaFloat, decay, dimK, countV, stride);
                SetFlag<HardEvent::V_MTE2>(v2m2);
                WaitFlag<HardEvent::V_MTE2>(v2m2);
            }
            SetFlag<HardEvent::V_MTE3>(v2m3);
            WaitFlag<HardEvent::V_MTE3>(v2m3);
            DataCopyExtParams outParams{static_cast<uint16_t>(dimK),
                static_cast<uint32_t>(countV * sizeof(float)),
                static_cast<uint32_t>((stride - countV) * sizeof(float) / 32),
                static_cast<uint32_t>((dimV - countV) * sizeof(float)), 0};
            DataCopyPad(final_[stateBase], state, outParams);
            SetFlag<HardEvent::MTE3_V>(m32v);
            WaitFlag<HardEvent::MTE3_V>(m32v);
        }
        pipe_.template ReleaseEventID<HardEvent::MTE2_V>(m2v);
        pipe_.template ReleaseEventID<HardEvent::V_MTE2>(v2m2);
        pipe_.template ReleaseEventID<HardEvent::V_MTE3>(v2m3);
        pipe_.template ReleaseEventID<HardEvent::MTE3_V>(m32v);
    }

private:
    __aicore__ inline void Update(LocalTensor<float> state, LocalTensor<float> key,
                                   LocalTensor<float> value, LocalTensor<float> beta,
                                   LocalTensor<float> decay, uint16_t dimK,
                                   uint16_t countV, uint16_t stride)
    {
        __ubuf__ float *h = (__ubuf__ float *)state.GetPhyAddr();
        __ubuf__ float *k = (__ubuf__ float *)key.GetPhyAddr();
        __ubuf__ float *v = (__ubuf__ float *)value.GetPhyAddr();
        __ubuf__ float *b = (__ubuf__ float *)beta.GetPhyAddr();
        __ubuf__ float *g = (__ubuf__ float *)decay.GetPhyAddr();
        const uint16_t tileCount = (countV + 63) / 64;
        __VEC_SCOPE__
        {
            using namespace AscendC::MicroAPI;
            RegTensor<float> hr, kr, gr, vr, br, dot, delta, product;
            uint32_t remaining = countV;
            for (uint16_t tile = 0; tile < tileCount; ++tile) {
                MaskReg mask = UpdateMask<float>(remaining);
                Duplicate(dot, 0.0f);
                for (uint16_t row = 0; row < dimK; ++row) {
                    DataCopy(hr, h + row * stride + tile * 64);
                    DataCopy<float, LoadDist::DIST_BRC_B32>(kr, k + row);
                    DataCopy<float, LoadDist::DIST_BRC_B32>(gr, g + row);
                    Mul(product, hr, gr, mask);
                    Mul(product, product, kr, mask);
                    Add(dot, dot, product, mask);
                }
                DataCopy(vr, v + tile * 64);
                DataCopy<float, LoadDist::DIST_BRC_B32>(br, b);
                Sub(delta, vr, dot, mask);
                Mul(delta, delta, br, mask);
                for (uint16_t row = 0; row < dimK; ++row) {
                    DataCopy(hr, h + row * stride + tile * 64);
                    DataCopy<float, LoadDist::DIST_BRC_B32>(kr, k + row);
                    DataCopy<float, LoadDist::DIST_BRC_B32>(gr, g + row);
                    Mul(hr, hr, gr, mask);
                    Mul(product, kr, delta, mask);
                    Add(hr, hr, product, mask);
                    DataCopy(h + row * stride + tile * 64, hr, mask);
                }
            }
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> buf_;
    GlobalTensor<T> k_, v_;
    GlobalTensor<BetaT> beta_;
    GlobalTensor<float> gk_, initial_, final_;
};

template <typename T, typename BetaT, typename Tiling>
__aicore__ inline void Run(GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR initial,
                          GM_ADDR cuSeqlens, GM_ADDR gk, GM_ADDR finalState,
                          const Tiling &tiling)
{
    if ASCEND_IS_AIV {
        Kernel<T, BetaT, Tiling> kernel;
        kernel.Run(k, v, beta, gk, initial, cuSeqlens, finalState, tiling);
    }
}
} // namespace KdaPreciseState
