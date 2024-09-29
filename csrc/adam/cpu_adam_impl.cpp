// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>
#include <cassert>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include "cpu_adam.h"

#include <map>
#include <functional>

#if defined(__ENABLE_CUDA__)
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include "cuda.h"
#include "curand.h"
#include "custom_cuda_layers.h"
#endif

using namespace std::string_literals;
static std::unordered_map<int, std::shared_ptr<void>> s_optimizers;
static std::unordered_map<int, std::shared_ptr<void>> sn_optimizers;


// C++ interface

void Adam_Optimizer::Step_1(float* _params,
                            float* grads,
                            float* _exp_avg,
                            float* _exp_avg_sq,
                            size_t _param_size,
                            ds_half_precision_t* dev_params,
                            bool half_precision)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<1>(&rounded_size,
                _params,
                grads,
                _exp_avg,
                _exp_avg_sq,
                _param_size,
                dev_params,
                half_precision);
#endif
    if (_param_size > rounded_size) {
        float betta1_minus1 = 1 - _betta1;
        float betta2_minus1 = 1 - _betta2;

        float step_size = -1 * _alpha / _bias_correction1;
        float w_decay = -1 * _alpha * _weight_decay;
        ds_half_precision_t* grads_cast_h;
        ds_half_precision_t* params_cast_h;
        if (half_precision) {
            grads_cast_h = reinterpret_cast<ds_half_precision_t*>(grads);
            params_cast_h = reinterpret_cast<ds_half_precision_t*>(_params);
        }

        for (size_t t = rounded_size; t < _param_size; t += TILE) {
            size_t copy_size = TILE;
            if ((t + TILE) > _param_size) copy_size = _param_size - t;
            size_t offset = copy_size + t;
#if defined(__ENABLE_CUDA__)
            if ((t / TILE) >= 2) { cudaStreamSynchronize(_streams[_buf_index]); }
#elif defined(__ENABLE_CANN__)
            if ((t / TILE) >= 2) { aclrtSynchronizeStream(_streams[_buf_index].stream()); }
#endif
#pragma omp parallel for
            for (size_t k = t; k < offset; k++) {
                float grad = half_precision ? (float)grads_cast_h[k] : grads[k];
                float param = half_precision ? (float)params_cast_h[k] : _params[k];
                float momentum = _exp_avg[k];
                float variance = _exp_avg_sq[k];
                if (_weight_decay > 0 && !_adamw_mode) { grad = param * _weight_decay + grad; }
                momentum = momentum * _betta1;
                momentum = grad * betta1_minus1 + momentum;

                variance = variance * _betta2;
                grad = grad * grad;
                variance = grad * betta2_minus1 + variance;

                grad = sqrt(variance);
                grad = grad * _bias_correction2 + _eps;
                grad = momentum / grad;
                if (_weight_decay > 0 && _adamw_mode) { param += w_decay * param; }
                param = grad * step_size + param;
#if defined(__ENABLE_CUDA__) or defined(__ENABLE_CANN__)
                if (dev_params) _doubled_buffer[_buf_index][k - t] = param;
#endif
                if (half_precision)
                    params_cast_h[k] = (ds_half_precision_t)param;
                else
                    _params[k] = param;
                _exp_avg[k] = momentum;
                _exp_avg_sq[k] = variance;
            }
#if defined(__ENABLE_CUDA__)
            if (dev_params) {
                launch_param_update(
                    _doubled_buffer[_buf_index], dev_params + t, (copy_size), _streams[_buf_index]);

                _buf_index = !_buf_index;
            }
#elif defined(__ENABLE_CANN__)
            if (dev_params) {
                size_t memcpy_size = copy_size * sizeof(_doubled_buffer[_buf_index][0]);
                aclrtMemcpy(dev_params + t,
                            memcpy_size,
                            _doubled_buffer[_buf_index],
                            memcpy_size,
                            aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE);

                _buf_index = !_buf_index;
            }
#endif
        }
    }
}

void Adam_Optimizer::Step_4(float* _params,
                            float* grads,
                            float* _exp_avg,
                            float* _exp_avg_sq,
                            size_t _param_size,
                            ds_half_precision_t* dev_params,
                            bool half_precision)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<4>(&rounded_size,
                _params,
                grads,
                _exp_avg,
                _exp_avg_sq,
                _param_size,
                dev_params,
                half_precision);
#endif
    if (_param_size > rounded_size)
        Step_1((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size),
               (dev_params != nullptr ? (dev_params + rounded_size) : dev_params),
               half_precision);
}

int create_adam_optimizer(int optimizer_id,
                          float alpha,
                          float betta1,
                          float betta2,
                          float eps,
                          float weight_decay,
                          bool adamw_mode,
                          bool should_log)
{
    auto opt =
        std::make_shared<Adam_Optimizer>(alpha, betta1, betta2, eps, weight_decay, adamw_mode);

    s_optimizers[optimizer_id] = opt;

    if (should_log) {
        std::string avx_type = "";
#if defined(__AVX512__)
        avx_type = "AVX512";
#else
#if defined(__AVX256__)
        avx_type = "AVX2";
#else
        avx_type = "scalar";
#endif
#endif

        printf("Adam Optimizer #%d is created with %s arithmetic capability.\n",
               optimizer_id,
               avx_type.c_str());
        printf("Config: alpha=%f, betas=(%f, %f), weight_decay=%f, adam_w=%d\n",
               alpha,
               betta1,
               betta2,
               weight_decay,
               (int)adamw_mode);
    }

    return 0;
}

void Adam_Optimizer::Step_8(float* _params,
                            float* grads,
                            float* _exp_avg,
                            float* _exp_avg_sq,
                            size_t _param_size,
                            ds_half_precision_t* dev_params,
                            bool half_precision)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<8>(&rounded_size,
                _params,
                grads,
                _exp_avg,
                _exp_avg_sq,
                _param_size,
                dev_params,
                half_precision);
#endif
    if (_param_size > rounded_size)
        Step_4((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size),
               (dev_params != nullptr ? (dev_params + rounded_size) : dev_params),
               half_precision);
}

int ds_adam_step(int optimizer_id,
                 size_t step,
                 float lr,
                 float beta1,
                 float beta2,
                 float epsilon,
                 float weight_decay,
                 bool bias_correction,
                 torch::Tensor& params,
                 torch::Tensor& grads,
                 torch::Tensor& exp_avg,
                 torch::Tensor& exp_avg_sq)
{
    auto params_c = params.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();

    // assert(params.options().dtype() == grads.options().dtype());

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    std::shared_ptr<Adam_Optimizer> opt =
        std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, epsilon, weight_decay, bias_correction);

    opt->Step_8(params_ptr,
                grads_ptr,
                exp_avg_ptr,
                exp_avg_sq_ptr,
                params_c.numel(),
                nullptr,
                (params.options().dtype() == at::kHalf));

#if defined(__ENABLE_CUDA__) or defined(__ENABLE_CANN__)
    opt->SynchronizeStreams();
#endif
    return 0;
}

int ds_adam_step_plus_copy(int optimizer_id,
                           size_t step,
                           float lr,
                           float beta1,
                           float beta2,
                           float epsilon,
                           float weight_decay,
                           bool bias_correction,
                           torch::Tensor& params,
                           torch::Tensor& grads,
                           torch::Tensor& exp_avg,
                           torch::Tensor& exp_avg_sq,
                           torch::Tensor& device_params)
{
#if defined(__ENABLE_CUDA__) or defined(__ENABLE_CANN__)
    auto params_c = params.contiguous();
    auto device_params_c = device_params.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();
    auto grads_c = grads.contiguous();

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    ds_half_precision_t* device_params_ptr = (ds_half_precision_t*)device_params_c.data_ptr();
    float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    std::shared_ptr<Adam_Optimizer> opt =
        std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, epsilon, weight_decay, bias_correction);
    opt->Step_8(params_ptr,
                grads_ptr,
                exp_avg_ptr,
                exp_avg_sq_ptr,
                params_c.numel(),
                device_params_ptr,
                (params.options().dtype() == at::kHalf));

    opt->SynchronizeStreams();
#else
    assert(false);
#endif
    return 0;
}

int destroy_adam_optimizer(int optimizer_id)
{
    s_optimizers.erase(optimizer_id);

    return 0;
}

/*Sonnet Optimizer*/
//TODO ************************Sonnet Code*************************

// VERSION1.0 First Sonnet optimizer still keep some params we dont need
// I will fix it latter

//TODO Sonnet Step part
template <typename ds_params_percision_t, typename ds_state_precision_t>
void Sonnet_Optimizer::SNStep_1(ds_params_percision_t* grads,
                                ds_state_precision_t* _exp_avg,
                                ds_state_precision_t* _exp_avg_sq,
                                size_t _param_size)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    SNStep_AVX<1>(&rounded_size, grads, _exp_avg, _exp_avg_sq, _param_size);
#endif
    if (_param_size > rounded_size) {
        float betta1_minus1 = 1 - _betta1;
        float betta2_minus1 = 1 - _betta2;

        // float step_size = -1 * _alpha / _bias_correction1;
        // float w_decay = -1 * _alpha * _weight_decay;

        for (size_t t = rounded_size; t < _param_size; t += TILE) {
            size_t copy_size = TILE;
            if ((t + TILE) > _param_size) copy_size = _param_size - t;
            size_t offset = copy_size + t;
#pragma omp parallel for
            for (size_t k = t; k < offset; k++) {
                float grad = (float)grads[k];
                // float param = (float)_params[k];
                float momentum = _exp_avg[k];
                float variance = _exp_avg_sq[k];

                // if (_weight_decay > 0 && !_adamw_mode) { grad = param * _weight_decay + grad; }
                momentum = momentum * _betta1;
                momentum = grad * betta1_minus1 + momentum;

                variance = variance * _betta2;
                grad = grad * grad;
                variance = grad * betta2_minus1 + variance;

                // grad = sqrt(variance);
                // grad = grad * _bias_correction2 + _eps;
                // grad = momentum / grad;
                // if (_weight_decay > 0 && _adamw_mode) { param += w_decay * param; }
                // param = grad * step_size + param;
                // _params[k] = param;
                _exp_avg[k] = momentum;
                _exp_avg_sq[k] = variance;
            }
        }
    }
}

template <typename ds_params_percision_t, typename ds_state_precision_t>
void Sonnet_Optimizer::SNStep_4(ds_params_percision_t* grads,
                            ds_state_precision_t* _exp_avg,
                            ds_state_precision_t* _exp_avg_sq,
                            size_t _param_size)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    SNStep_AVX<4>(&rounded_size, grads, _exp_avg, _exp_avg_sq, _param_size);
#endif
    if (_param_size > rounded_size)
        SNStep_1((grads + rounded_size),
               (_exp_avg + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size));
}

int create_sonnet_adam_optimizer(int optimizer_id,
                                 float alpha,
                                 float betta1,
                                 float betta2,
                                 float eps,
                                 float weight_decay,
                                 bool adamw_mode,
                                 bool should_log)
{
    auto SNopt =
        std::make_shared<Sonnet_Optimizer>(alpha, betta1, betta2, eps, weight_decay, adamw_mode);

    sn_optimizers[optimizer_id] = SNopt;

    if (should_log) {
        std::string avx_type = "";
#if defined(__AVX512__)
        avx_type = "AVX512";
#else
#if defined(__AVX256__)
        avx_type = "AVX2";
#else
        avx_type = "scalar";
#endif
#endif

        printf("Adam Optimizer #%d is created with %s arithmetic capability.\n",
               optimizer_id,
               avx_type.c_str());
        printf("Config: alpha=%f, betas=(%f, %f), weight_decay=%f, adam_w=%d\n",
               alpha,
               betta1,
               betta2,
               weight_decay,
               (int)adamw_mode);
    }

    return 0;
}

template <typename ds_params_percision_t, typename ds_state_precision_t>
void Sonnet_Optimizer::SNStep_8(ds_params_percision_t* grads,
                                ds_state_precision_t* _exp_avg,
                                ds_state_precision_t* _exp_avg_sq,
                                size_t _param_size)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    SNStep_AVX<8>(&rounded_size, grads, _exp_avg, _exp_avg_sq, _param_size);
#endif
    if (_param_size > rounded_size)
        SNStep_4((grads + rounded_size),
                 (_exp_avg + rounded_size),
                 (_exp_avg_sq + rounded_size),
                 (_param_size - rounded_size));
}

//TODO Sonnet invoker
template <typename ds_params_percision_t, typename ds_state_precision_t>
void SNstep_invoker(std::shared_ptr<Sonnet_Optimizer> SNopt,
                  void* grads,
                  void* _exp_avg,
                  void* _exp_avg_sq,
                  size_t _param_size)
{
    SNopt->SNStep_8((ds_params_percision_t*)(grads),
                (ds_state_precision_t*)(_exp_avg),
                (ds_state_precision_t*)(_exp_avg_sq),
                _param_size);
}

std::map<std::tuple<c10::ScalarType, c10::ScalarType>,
         std::function<void(std::shared_ptr<Sonnet_Optimizer>, void*, void*, void*, size_t)>>
    SNinvokers;

// Fill map with template functions for each type
template <class ds_params_percision_t, class ds_state_precision_t>
void create_SNinvoker()
{
    SNinvokers[std::tuple(c10::CppTypeToScalarType<ds_params_percision_t>(),
                        c10::CppTypeToScalarType<ds_state_precision_t>())] =
        SNstep_invoker<ds_params_percision_t, ds_state_precision_t>;
}
struct SNInvokerInitializer {
    SNInvokerInitializer()
    {
        create_SNinvoker<c10::Half, float>();
        create_SNinvoker<c10::Half, c10::Half>();
        create_SNinvoker<c10::BFloat16, float>();
        create_SNinvoker<c10::BFloat16, c10::BFloat16>();
        create_SNinvoker<float, float>();
    }
} _SNinvoker_initializer;

void SNinvoke(std::shared_ptr<Sonnet_Optimizer> SNopt,
            torch::Tensor& grads,
            torch::Tensor& exp_avg,
            torch::Tensor& exp_avg_sq,
            size_t param_size)
{
    c10::ScalarType params_type = at::typeMetaToScalarType(grads.options().dtype());
    c10::ScalarType state_type = at::typeMetaToScalarType(exp_avg.options().dtype());

    auto it = SNinvokers.find(std::tuple(params_type, state_type));
    if (it == SNinvokers.end()) {
        throw std::runtime_error("Adam optimizer with param type "s + c10::toString(params_type) +
                                 " and state type "s + c10::toString(state_type) +
                                 " is not supported on current hardware"s);
    }

    it->second(SNopt,
               grads.data_ptr(),
               exp_avg.data_ptr(),
               exp_avg_sq.data_ptr(),
               param_size);
}

//TODO function part

int sonnet_adam_step(int optimizer_id,
                 size_t step,
                 float lr,
                 float beta1,
                 float beta2,
                 float epsilon,
                 float weight_decay,
                 bool bias_correction,
                 torch::Tensor& grads,
                 torch::Tensor& exp_avg,
                 torch::Tensor& exp_avg_sq)
{
    auto params_c = grads.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();

    std::shared_ptr<Sonnet_Optimizer> SNopt =
        std::static_pointer_cast<Sonnet_Optimizer>(sn_optimizers[optimizer_id]);
    SNopt->SNIncrementStep(step, beta1, beta2);
    SNopt->SNupdate_state(lr, epsilon, weight_decay, bias_correction);

    SNinvoke(SNopt, grads_c, exp_avg_c, exp_avg_sq_c, params_c.numel());

    return 0;
}

int destroy_sonnet_adam_optimizer(int optimizer_id)
{
    sn_optimizers.erase(optimizer_id);

    return 0;
}