// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class EmbedLayerNorm : public OpKernel {
 public:
  explicit EmbedLayerNorm(const OpKernelInfo& op_kernel_info);
  Status Compute(OpKernelContext* context) const override;

 protected:
  Status ComputeInternal(OpKernelContext* context,
                         const Tensor* input_ids,
                         const Tensor* segment_ids,
                         const Tensor* word_embedding,
                         const Tensor* position_embedding,
                         const Tensor* segment_embedding,
                         const Tensor* gamma,
                         const Tensor* beta,
                         const Tensor* mask) const;

// TODO - this should be private.
  float epsilon_;
};

}  // namespace contrib
}  // namespace onnxruntime
