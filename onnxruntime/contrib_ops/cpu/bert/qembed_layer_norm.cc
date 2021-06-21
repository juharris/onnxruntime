// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qembed_layer_norm.h"

#include <cmath>

#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

namespace {

// TODO(kreeger): Find a global home for these helper methods.
template <typename T>
inline T GetQuantizedInputTensorValue(OpKernelContext* context, int index) {
  const Tensor* tensor = context->Input<Tensor>(index);
  return *(tensor->template Data<T>());
}

template <typename T>
inline float Dequantize(T value, float scale, T zero_point) {
  return static_cast<float>(static_cast<int32_t>(value) - zero_point) * scale;
}

}  // namespace

// This op is internal-only, so register outside of onnx:
#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      QEmbedLayerNormalization,                                    \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      QEmbedLayerNorm<T>);

REGISTER_KERNEL_TYPED(float)


template <typename T>
QEmbedLayerNorm<T>::QEmbedLayerNorm(const OpKernelInfo& op_kernel_info)
    : EmbedLayerNorm(op_kernel_info) {
}

template <typename T>
QEmbedLayerNorm<T>::QInputs::QInputs(const Tensor* input_ids,
                                     const Tensor* segment_ids,
                                     const Tensor* word_embedding,
                                     const Tensor* position_embedding,
                                     const Tensor* segment_embedding,
                                     const Tensor* gamma,
                                     const Tensor* beta,
                                     const Tensor* word_embedding_scale,
                                     const Tensor* position_embedding_scale,
                                     const Tensor* segment_embedding_scale,
                                     const Tensor* gamma_scale,
                                     const Tensor* beta_scale,
                                     const Tensor* word_embedding_zero_point,
                                     const Tensor* position_embedding_zero_point,
                                     const Tensor* segment_embedding_zero_point,
                                     const Tensor* gamma_zero_point,
                                     const Tensor* beta_zero_point,
                                     const Tensor* mask)
    : Inputs(input_ids,
             segment_ids,
             word_embedding,
             position_embedding,
             segment_embedding,
             gamma,
             beta,
             mask)
    , word_embedding_scale(word_embedding_scale)
    , position_embedding_scale(position_embedding_scale)
    , segment_embedding_scale(segment_embedding_scale)
    , gamma_scale(gamma_scale)
    , beta_scale(beta_scale)
    , word_embedding_zero_point(word_embedding_zero_point)
    , position_embedding_zero_point(position_embedding_zero_point)
    , segment_embedding_zero_point(segment_embedding_zero_point)
    , gamma_zero_point(gamma_zero_point)
    , beta_zero_point(beta_zero_point) {}

template <typename T>
Status QEmbedLayerNorm<T>::Compute(OpKernelContext* context) const {
  /*
  Input Tensors List:
  [0] input_ids (int32)
  [1] segment_ids (int32) (optional)
  [2] word_embedding_quant (uint8)
  [3] position_embedding_quant (uint8)
  [4] segment_embedding_quant (uint8)
  [5] gamma_quant (uint8)
  [6] beta_quant (uint8)
  [7] word_embedding_scale (float)
  [8] position_embedding_scale (float)
  [9] segment_embedding_scale (float) (optional)
  [10] gamma_scale (float)
  [11] beta_scale (float)
  [12] word_embedding_zero_point (uint8)
  [13] position_embedding_zero_point (uint8)
  [14] segment_embedding_zero_point (uint8) (optional)
  [15] gamma_zero_point (uint8)
  [16] beta_zero_point (uint8)
  [17] mask (int32) (optional)
  */
  QInputs inputs(/*input_ids=*/context->Input<Tensor>(0),
                 /*segment_ids=*/context->Input<Tensor>(1),
                 /*word_embedding=*/context->Input<Tensor>(2),
                 /*position_embedding=*/context->Input<Tensor>(3),
                 /*segment_embedding=*/context->Input<Tensor>(4),
                 /*gamma=*/context->Input<Tensor>(5),
                 /*beta=*/context->Input<Tensor>(6),
                 /*word_embedding_scale=*/context->Input<Tensor>(7),
                 /*position_embedding_scaled=*/context->Input<Tensor>(8),
                 /*segment_embedding_scale=*/context->Input<Tensor>(9),
                 /*gamma_scale=*/context->Input<Tensor>(10),
                 /*beta_scale=*/context->Input<Tensor>(11),
                 /*word_embedding_zero_point=*/context->Input<Tensor>(12),
                 /*position_embedding_zero_point=*/context->Input<Tensor>(13),
                 /*segment_embedding_zero_point=*/context->Input<Tensor>(14),
                 /*gamma_zero_point=*/context->Input<Tensor>(15),
                 /*beta_zero_point=*/context->Input<Tensor>(16),
                 /*mask=*/context->Input<Tensor>(17));

  ORT_RETURN_IF_ERROR(CheckInputs(inputs));
  //
  // TODO(kreeger): Need to make everything run refactored here!
  //
  return Status::OK();
}

template <typename T>
Status QEmbedLayerNorm<T>::CheckInputs(const QInputs& inputs) const {
  ORT_RETURN_IF_ERROR(EmbedLayerNorm<T>::CheckInputs(inputs));

  /*
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(context->Input<Tensor>(7)),
      "Word embedding scale must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(context->Input<Tensor>(8)),
      "Position embedding scale must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(context->Input<Tensor>(9)),
      "Segment embedding scale must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(context->Input<Tensor>(10)),
      "Layer norm weights scale must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(context->Input<Tensor>(11)),
      "Layer norm bias must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(context->Input<Tensor>(12)),
      "Word embedding zero point must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(context->Input<Tensor>(13)),
      "Position embedding zero point must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(context->Input<Tensor>(14)),
      "Segment embedding zero point must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(context->Input<Tensor>(15)),
      "Layer norm weights zero point must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(context->Input<Tensor>(16)),
      "Layer norm bias zero point must be a scalar or 1D tensor of size 1");
      */

  return Status::OK();
}

template <typename T>
Status QEmbedLayerNorm<T>::ComputeInternal(OpKernelContext* context, const QInputs& inputs) const {
  //
  // TODO(kreeger): write me!
  //
  // Determine shapes
  const auto& input_dims = input_ids->Shape().GetDims();
  int64_t hidden_size = word_embedding->Shape()[1];

  int batch_size = static_cast<int>(input_dims[0]);
  int sequence_length = static_cast<int>(input_dims[1]);

  int word_embedding_length = static_cast<int>(word_embedding->Shape()[0]);
  int position_embedding_length = static_cast<int>(position_embedding->Shape()[0]);
  int segment_embedding_length =
      (nullptr == segment_embedding) ? 0 : static_cast<int>(segment_embedding->Shape()[0]);

  // Grab quantization values:
  float word_embedding_scale = GetQuantizedInputTensorValue<float>(context, 7);
  uint8_t word_embedding_zero_point = GetQuantizedInputTensorValue<uint8_t>(context, 12); 

  float position_embedding_scale = GetQuantizedInputTensorValue<float>(context, 8);
  uint8_t position_embedding_zero_point = GetQuantizedInputTensorValue<uint8_t>(context, 13);

  // TODO - this can be optional as well!
  float segment_embedding_scale = GetQuantizedInputTensorValue<float>(context, 9);
  uint8_t segment_embedding_zero_point = GetQuantizedInputTensorValue<uint8_t>(context, 14);

  float layer_norm_weights_scale = GetQuantizedInputTensorValue<float>(context, 10);
  uint8_t layer_norm_weights_zero_point = GetQuantizedInputTensorValue<uint8_t>(context, 15);

  float layer_norm_bias_scale = GetQuantizedInputTensorValue<float>(context, 11);
  uint8_t layer_norm_bias_zero_point = GetQuantizedInputTensorValue<uint8_t>(context, 16);
  
  /*
  Output Tensors List:
  [0] layernorm_out (T)
  [1] mask_index_out (int32)
  */
  TensorShape output_shape({input_dims[0], input_dims[1], hidden_size});
  Tensor* output = context->Output(0, output_shape);

  TensorShape mask_index_shape({input_dims[0]});
  Tensor* mask_index = context->Output(1, mask_index_shape);

  // Grab pointers to buffers each Tensor represents:
  const int32_t* input_ids_data = input_ids->template Data<int32_t>();
  // TODO(kreeger): Handle missing segment_ids with the quantization params too?
  const int32_t* segment_ids_data =
      (nullptr == segment_ids) ? nullptr : segment_ids->template Data<int32_t>();
  const uint8_t* word_embedding_data = word_embedding->template Data<uint8_t>();
  const uint8_t* position_embedding_data = position_embedding->template Data<uint8_t>();
  // TODO(kreeger): Handle missing segment_embedding_data with the quantization params too?
  const uint8_t* segment_embedding_data =
      (nullptr == segment_embedding) ? nullptr : segment_embedding->template Data<uint8_t>();
  const uint8_t* gamma_data = gamma->template Data<uint8_t>();
  const uint8_t* beta_data = beta->template Data<uint8_t>();

  T* output_data = output->template MutableData<T>();

  // TODO(kreeger): consider using std::function<> here to reuse this code w/ the floating
  //                point version. See qlinear_binary_op_test.cc:~141
  // Perform the Op:
  {
    std::atomic_bool failed{false};

    int n = batch_size * sequence_length;
    concurrency::ThreadPool::TryBatchParallelFor(
        context->GetOperatorThreadPool(), n, [=, &failed](ptrdiff_t index) {
      int word_col_index = input_ids_data[index];
      if (word_col_index < 0 || word_col_index >= word_embedding_length) {
        failed.store(true, std::memory_order_release);
        return;
      }
      int position_col_index = index % sequence_length;
      if (position_col_index >= position_embedding_length) {
        failed.store(true, std::memory_order_release);
        return;
      }
      int segment_col_index = 0;
      if (nullptr != segment_ids_data) {
        segment_col_index = segment_ids_data[index];
        if (segment_col_index < 0 || segment_col_index >= segment_embedding_length) {
          failed.store(true, std::memory_order_release);
          return;
        }
      }

      // Grab inputs for the embeddings for the current batch index:
      const uint8_t* input_word_embedding = word_embedding_data + (word_col_index * hidden_size);
      const uint8_t* input_position_embedding =
          position_embedding_data + (position_col_index * hidden_size);
      const uint8_t* input_segment_embedding = nullptr;
      if (segment_embedding_data != nullptr) {
        input_segment_embedding = segment_embedding_data + (segment_col_index * hidden_size);
      }

      T* output = output_data + (index * hidden_size);

      T sum = static_cast<T>(0);
      for (int i = 0; i < hidden_size; ++i) {
        // pass a lambda for these dequantize calls.
        T subtotal = Dequantize(input_word_embedding[i],
                                word_embedding_scale,
                                word_embedding_zero_point) +
                     Dequantize(input_position_embedding[i],
                                position_embedding_scale,
                                position_embedding_zero_point);
        if (segment_embedding_data != nullptr) {
          subtotal += Dequantize(input_segment_embedding[i],
                                 segment_embedding_scale,
                                 segment_embedding_zero_point);
        }
        output[i] = subtotal;
        sum += subtotal;
      }

      T mean = sum / hidden_size;
      sum = 0;

      for (int i = 0; i < hidden_size; i++) {
        T a = output[i] - mean;
        output[i] = a;
        sum += a * a;
      }

      T e = sqrt(sum / hidden_size + static_cast<T>(epsilon_));
      for (int i = 0; i < hidden_size; i++) {
        T cur_gamma = Dequantize(gamma_data[i],
                                  layer_norm_weights_scale,
                                  layer_norm_weights_zero_point);
        T cur_beta = Dequantize(beta_data[i],
                                layer_norm_bias_scale,
                                layer_norm_bias_zero_point);
        output[i] = output[i] / e * cur_gamma + cur_beta;
      }
    }, 0);

    if (failed.load(std::memory_order_acquire)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "input index out of range");
    }
  }

  // Calculate mask
  if (nullptr != mask) {
    const int32_t* mask_data = mask->template Data<int32_t>();
    for (int b = 0; b < batch_size; b++) {
      // TODO(kreeger): Fix static cast warning here:
      mask_index->template MutableData<int32_t>()[b] =
          static_cast<int32_t>(std::count_if(mask_data + (b * sequence_length),
                                             mask_data + (b * sequence_length) + sequence_length,
                                             [](int v) { return v == 1; }));
    }
  } else {
    memset(mask_index->template MutableData<int32_t>(), 0, batch_size * sizeof(int32_t));
  }
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
