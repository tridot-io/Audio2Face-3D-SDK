// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
#include "audio2x/internal/audio2x.h"
#include "audio2x/internal/logger.h"
#include "audio2x/internal/macros.h"
#include "audio2x/error.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstdlib>

namespace {

// -1 = not set: follow the A2X_CUDA_GRAPH environment variable.
std::atomic<int> gCudaGraphsEnabled{-1};

bool CudaGraphsEnabledByEnvironment() {
  static const bool enabled = [] {
    const char* env = std::getenv("A2X_CUDA_GRAPH");
    return env != nullptr && env[0] == '1';
  }();
  return enabled;
}

} // namespace

void nva2x::internal::SetCudaGraphsEnabled(bool enabled) {
  gCudaGraphsEnabled.store(enabled ? 1 : 0);
}

bool nva2x::internal::AreCudaGraphsEnabled() {
  const int value = gCudaGraphsEnabled.load();
  return value < 0 ? CudaGraphsEnabledByEnvironment() : value == 1;
}

std::error_code nva2x::internal::SetCudaDeviceIfNeeded(int device) {
  int current_device;
  A2X_CUDA_CHECK_ERROR(cudaGetDevice(&current_device), ErrorCode::eCudaDeviceGetError);
  if (current_device != device) {
    A2X_CUDA_CHECK_ERROR(cudaSetDevice(device), ErrorCode::eCudaDeviceSetError);
  }
  return ErrorCode::eSuccess;
}
