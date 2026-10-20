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
#pragma once

#include "audio2x/export.h"

#include <system_error>

namespace nva2x {

// Set the CUDA device if needed by first checking if the current device is the one requested
// to minimize the overhead of setting the device when it's already set.
AUDIO2X_SDK_EXPORT std::error_code SetCudaDeviceIfNeeded(int device);

// CUDA graph replay of the TensorRT inference engines and of the GPU blendshape solve. Each is recorded once per set of
// bindings and replayed afterwards; an engine that cannot be recorded (loops, data-dependent shapes) keeps running
// directly. Off by default; the environment variable A2X_CUDA_GRAPH=1 also turns it on. Applies to work enqueued after
// the call.
AUDIO2X_SDK_EXPORT void SetCudaGraphsEnabled(bool enabled);
AUDIO2X_SDK_EXPORT bool AreCudaGraphsEnabled();

} // namespace nva2x
