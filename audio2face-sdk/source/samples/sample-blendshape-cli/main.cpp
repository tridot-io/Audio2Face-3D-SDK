// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Sample CLI tool for blendshape inference using Diffusion model.
// Outputs JSON with ARKit 52 blendshape weights.

#include "audio2face/audio2face.h"
#include "audio2x/cuda_utils.h"

#include <AudioFile.h>

#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <cstring>
#include <sstream>
#include <iomanip>

//
// Utility macros and types
//

#define CHECK_RESULT(func)                                                     \
  {                                                                            \
    std::error_code error = (func);                                            \
    if (error) {                                                               \
      std::cerr << "Error (" << __LINE__ << "): Failed to execute: " << #func; \
      std::cerr << ", Reason: "<< error.message() << std::endl;                \
      exit(1);                                                                 \
    }                                                                          \
  }

#define CHECK_ERROR(expression)                                                \
  {                                                                            \
    if (!(expression)) {                                                       \
      std::cerr << "Error (" << __LINE__ << "): " << #expression;              \
      std::cerr << " is NULL or failed" << std::endl;                          \
      exit(1);                                                                 \
    }                                                                          \
  }

struct Destroyer {
  template <typename T> void operator()(T *obj) const {
    obj->Destroy();
  }
};
template <typename T> using UniquePtr = std::unique_ptr<T, Destroyer>;
template <typename T> UniquePtr<T> ToUniquePtr(T* ptr) { return UniquePtr<T>(ptr); }

//
// Blendshape frame data structure
//

struct BlendshapeFrame {
    double timeCode;
    std::vector<float> weights;
};

//
// Global storage for results (used by callback)
//

struct CallbackData {
    std::vector<BlendshapeFrame> frames;
    std::size_t weightCount;
};

//
// Audio loading with resampling to 16kHz
//

std::vector<float> downsample(const std::vector<float>& input, int targetSampleRate, int originalSampleRate) {
    std::vector<float> output;
    float ratio = static_cast<float>(originalSampleRate) / targetSampleRate;
    for (size_t i = 0; i < input.size(); i += static_cast<size_t>(ratio)) {
        output.push_back(input[i]);
    }
    return output;
}

std::vector<float> upsample(const std::vector<float>& input, int targetSampleRate, int originalSampleRate) {
    std::vector<float> output;
    float ratio = static_cast<float>(targetSampleRate) / originalSampleRate;
    for (size_t i = 0; i < input.size(); ++i) {
        output.push_back(input[i]);
        if (i < input.size() - 1) {
            float nextSample = input[i + 1];
            for (float t = 1.0f; t < ratio; t += 1.0f) {
                float interpolatedSample = input[i] + (nextSample - input[i]) * (t / ratio);
                output.push_back(interpolatedSample);
            }
        }
    }
    return output;
}

std::vector<float> readAudioFile(const std::string& filename) {
    AudioFile<float> audio(filename);
    if (audio.getNumChannels() == 0 || audio.getLengthInSeconds() == 0) {
        std::cerr << "Failed to load audio file: " << filename << std::endl;
        return {};
    }

    const auto sr = audio.getSampleRate();
    std::cerr << "Loaded audio: " << filename << " (" << sr << " Hz, "
              << audio.getLengthInSeconds() << " seconds)" << std::endl;

    // Resample to 16kHz if needed
    if (sr == 16000) {
        return audio.samples[0];
    }

    const auto original = audio.samples[0];

    if (sr < 16000) {
        std::cerr << "Unsupported sample rate " << sr << " (must be >= 16000)" << std::endl;
        return {};
    }

    // Simple resampling for common sample rates
    const int multiple = sr / 16000;
    if (multiple * 16000 == sr) {
        return downsample(original, 16000, sr);
    }
    if (sr == 24000) {
        const int lcm = 48000;
        return downsample(upsample(original, lcm, sr), 16000, lcm);
    }
    if (sr == 44100 || sr == 88200) {
        const int lcm = 7056000;
        return downsample(upsample(original, lcm, sr), 16000, lcm);
    }

    std::cerr << "Unsupported sample rate " << sr << std::endl;
    return {};
}

//
// JSON output
//

std::string escapeJsonString(const std::string& str) {
    std::ostringstream oss;
    for (char c : str) {
        switch (c) {
            case '"':  oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:   oss << c; break;
        }
    }
    return oss.str();
}

void outputJson(
    const std::vector<BlendshapeFrame>& frames,
    const std::vector<std::string>& blendshapeNames,
    int fps
) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "{\n";
    std::cout << "  \"fps\": " << fps << ",\n";
    std::cout << "  \"frame_count\": " << frames.size() << ",\n";

    // Output blendshape names
    std::cout << "  \"blendshape_names\": [";
    for (size_t i = 0; i < blendshapeNames.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << "\"" << escapeJsonString(blendshapeNames[i]) << "\"";
    }
    std::cout << "],\n";

    // Output frames
    std::cout << "  \"frames\": [\n";
    for (size_t i = 0; i < frames.size(); ++i) {
        const auto& frame = frames[i];
        std::cout << "    {\"time_code\": " << frame.timeCode << ", \"weights\": [";
        for (size_t j = 0; j < frame.weights.size(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << frame.weights[j];
        }
        std::cout << "]}";
        if (i < frames.size() - 1) std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "  ]\n";
    std::cout << "}\n";
}

//
// Usage
//

void printUsage(const char* progName) {
    std::cerr << "Usage: " << progName << " --model <model.json> --audio <audio.wav> [--identity <index>]\n";
    std::cerr << "\n";
    std::cerr << "Options:\n";
    std::cerr << "  --model     Path to model.json (e.g., multi-diffusion/model.json)\n";
    std::cerr << "  --audio     Path to audio file (WAV format, will be resampled to 16kHz)\n";
    std::cerr << "  --identity  Identity index for multi-identity models (default: 0)\n";
    std::cerr << "\n";
    std::cerr << "Output: JSON with blendshape weights is written to stdout.\n";
    std::cerr << "        Diagnostic messages are written to stderr.\n";
}

//
// Main
//

int main(int argc, char* argv[]) {
    // Parse arguments
    std::string modelPath;
    std::string audioPath;
    std::size_t identityIndex = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            modelPath = argv[++i];
        } else if (strcmp(argv[i], "--audio") == 0 && i + 1 < argc) {
            audioPath = argv[++i];
        } else if (strcmp(argv[i], "--identity") == 0 && i + 1 < argc) {
            identityIndex = std::stoul(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        }
    }

    if (modelPath.empty() || audioPath.empty()) {
        printUsage(argv[0]);
        return 1;
    }

    std::cerr << "Model: " << modelPath << std::endl;
    std::cerr << "Audio: " << audioPath << std::endl;
    std::cerr << "Identity: " << identityIndex << std::endl;

    // Initialize CUDA
    constexpr int deviceID = 0;
    CHECK_RESULT(nva2x::SetCudaDeviceIfNeeded(deviceID));

    // Load audio
    auto audioBuffer = readAudioFile(audioPath);
    CHECK_ERROR(!audioBuffer.empty());
    std::cerr << "Audio samples: " << audioBuffer.size() << " (16kHz)" << std::endl;

    // Create blendshape executor bundle
    // Using SkinTongue execution option for full blendshape output
    // GPU solver for better performance
    nva2f::IDiffusionModel::IGeometryModelInfo* rawModelInfoPtr = nullptr;
    nva2f::IDiffusionModel::IBlendshapeSolveModelInfo* rawBlendshapeSolveModelInfoPtr = nullptr;

    auto bundle = ToUniquePtr(
        nva2f::ReadDiffusionBlendshapeSolveExecutorBundle(
            1,  // nbTracks
            modelPath.c_str(),
            nva2f::IGeometryExecutor::ExecutionOption::SkinTongue,
            false,  // useGpuSolver (use CPU for host results)
            identityIndex,
            true,   // constantNoise
            &rawModelInfoPtr,
            &rawBlendshapeSolveModelInfoPtr
        )
    );
    CHECK_ERROR(bundle != nullptr);

    auto modelInfo = ToUniquePtr(rawModelInfoPtr);
    auto blendshapeSolveModelInfo = ToUniquePtr(rawBlendshapeSolveModelInfoPtr);

    std::cerr << "Executor bundle created successfully" << std::endl;

    auto& executor = bundle->GetExecutor();
    const std::size_t weightCount = executor.GetWeightCount();
    std::cerr << "Weight count: " << weightCount << std::endl;

    // Get blendshape names from the skin solver
    std::vector<std::string> blendshapeNames;
    {
        nva2f::IBlendshapeSolver* skinSolver = nullptr;
        auto error = nva2f::GetExecutorSkinSolver(executor, 0, &skinSolver);
        if (!error && skinSolver) {
            int numPoses = skinSolver->NumBlendshapePoses();
            std::cerr << "Skin blendshape count: " << numPoses << std::endl;
            for (int i = 0; i < numPoses; ++i) {
                const char* name = skinSolver->GetPoseName(i);
                blendshapeNames.push_back(name ? name : "unknown");
            }
        }

        // Also get tongue blendshapes if available
        nva2f::IBlendshapeSolver* tongueSolver = nullptr;
        error = nva2f::GetExecutorTongueSolver(executor, 0, &tongueSolver);
        if (!error && tongueSolver) {
            int numPoses = tongueSolver->NumBlendshapePoses();
            std::cerr << "Tongue blendshape count: " << numPoses << std::endl;
            for (int i = 0; i < numPoses; ++i) {
                const char* name = tongueSolver->GetPoseName(i);
                blendshapeNames.push_back(name ? name : "unknown");
            }
        }
    }

    // If no names found, use generic names
    if (blendshapeNames.empty()) {
        for (std::size_t i = 0; i < weightCount; ++i) {
            blendshapeNames.push_back("blendshape_" + std::to_string(i));
        }
    }

    // Prepare callback data
    CallbackData callbackData;
    callbackData.weightCount = weightCount;

    // Set results callback (for HOST results since useGpuSolver=false)
    auto callback = [](void* userdata, const nva2f::IBlendshapeExecutor::HostResults& results, std::error_code errorCode) {
        if (errorCode) {
            std::cerr << "Error in callback: " << errorCode.message() << std::endl;
            return;
        }
        auto& data = *static_cast<CallbackData*>(userdata);
        BlendshapeFrame frame;
        frame.timeCode = static_cast<double>(results.timeStampCurrentFrame) / 16000.0;  // Convert sample to seconds
        const float* weightsData = results.weights.Data();
        frame.weights.resize(results.weights.Size());
        for (std::size_t i = 0; i < results.weights.Size(); ++i) {
            frame.weights[i] = weightsData[i];
        }
        data.frames.push_back(std::move(frame));
    };
    CHECK_RESULT(executor.SetResultsCallback(callback, &callbackData));

    // Add default (neutral) emotion
    {
        auto& emotionAccumulator = bundle->GetEmotionAccumulator(0);
        std::vector<float> emptyEmotion(emotionAccumulator.GetEmotionSize(), 0.0f);
        CHECK_RESULT(emotionAccumulator.Accumulate(
            0,
            nva2x::HostTensorFloatConstView{emptyEmotion.data(), emptyEmotion.size()},
            bundle->GetCudaStream().Data()
        ));
        CHECK_RESULT(emotionAccumulator.Close());
    }

    // Accumulate audio
    CHECK_RESULT(
        bundle->GetAudioAccumulator(0).Accumulate(
            nva2x::HostTensorFloatConstView{audioBuffer.data(), audioBuffer.size()},
            bundle->GetCudaStream().Data()
        )
    );
    CHECK_RESULT(bundle->GetAudioAccumulator(0).Close());

    std::cerr << "Running inference..." << std::endl;

    // Execute until all frames are processed
    while (nva2x::GetNbReadyTracks(executor) > 0) {
        CHECK_RESULT(executor.Execute(nullptr));
    }

    // Wait for all async work to complete
    CHECK_RESULT(executor.Wait(0));

    std::cerr << "Generated " << callbackData.frames.size() << " frames" << std::endl;

    // Output JSON
    outputJson(callbackData.frames, blendshapeNames, 30);

    std::cerr << "Done." << std::endl;

    return 0;
}
