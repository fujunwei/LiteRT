// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/vendors/intel_openvino/dispatch/remote_tensor_buffer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/intel_openvino/dispatch/openvino_shared_core.h"
#include "litert/vendors/intel_openvino/utils.h"

RemoteTensorBuffer::RemoteTensorBuffer()
    : level_zero_buffer_({}), allocated_(false) {}

RemoteTensorBuffer::~RemoteTensorBuffer() = default;

litert::Expected<void> RemoteTensorBuffer::Alloc(
    const LiteRtRankedTensorType& tensor_type, size_t size) {
  if (allocated_) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "The remote tensor has been allocated.");
  }

  ov::element::Type ov_element_type =
      litert::openvino::MapLiteTypeToOV(tensor_type.element_type);
  std::vector<int32_t> ov_shape_vec(tensor_type.layout.rank);
  for (int i = 0; i < ov_shape_vec.size(); i++)
    ov_shape_vec[i] = tensor_type.layout.dimensions[i];
#ifdef LITERT_CPU_DEVICE
  auto host_tensor = ov::Tensor(
          ov_element_type, ov::Shape{ov_shape_vec.begin(), ov_shape_vec.end()});
  host_tensor_ = host_tensor;
#else
  // TODO:: Release the shared OpenVINO Core.
  std::shared_ptr<ov::Core> core = OpenVINOSharedCore::GetInstance()->getCore();
  auto context = core->get_default_context("NPU")
                     .as<ov::intel_npu::level_zero::ZeroContext>();
  auto level_zero_buffer = context.create_l0_host_tensor(
      ov_element_type, ov::Shape{ov_shape_vec.begin(), ov_shape_vec.end()});
  level_zero_buffer_ = level_zero_buffer;
#endif
  allocated_ = true;

  return {};
}

litert::Expected<void*> RemoteTensorBuffer::GetZeroBufferPtr() {
  if (!allocated_) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "The remote tensor didn't allocate.");
  }

#ifdef LITERT_CPU_DEVICE
  return host_tensor_.data();
#else
  return level_zero_buffer_.get();
#endif
}

#ifdef LITERT_CPU_DEVICE
litert::Expected<ov::Tensor> 
#else
litert::Expected<ov::intel_npu::level_zero::ZeroBufferTensor>
#endif
RemoteTensorBuffer::GetZeroBufferTensor() {
  if (!allocated_) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "Failed to get zero buffer remote tensor.");
  }
#ifdef LITERT_CPU_DEVICE
  return host_tensor_;
#else
  return level_zero_buffer_;
#endif
}
