/**
 * @file a2f_wrapper.cpp
 * @brief C API wrapper implementation for Audio2Face-3D-SDK
 */

#include "a2f_wrapper.h"
#include "audio2face/audio2face.h"
#include "audio2x/cuda_utils.h"

#include <mutex>
#include <atomic>
#include <vector>
#include <string>
#include <memory>
#include <cstring>
#include <thread>

//
// Thread-local error message storage
//

static thread_local std::string g_lastErrorMessage;
static const char* A2F_VERSION = "1.0.0";

static void SetLastError(const std::string& msg) {
    g_lastErrorMessage = msg;
}

static void SetLastError(const std::error_code& ec) {
    g_lastErrorMessage = ec.message();
}

//
// Utility: Custom deleter for SDK objects
//

struct Destroyer {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) obj->Destroy();
    }
};

template <typename T>
using UniquePtr = std::unique_ptr<T, Destroyer>;

template <typename T>
UniquePtr<T> ToUniquePtr(T* ptr) {
    return UniquePtr<T>(ptr);
}

//
// Internal structures
//

struct A2FSession;  // Forward declaration

struct A2FContext {
    // SDK components
    UniquePtr<nva2f::IBlendshapeExecutorBundle> bundle;
    UniquePtr<nva2f::IDiffusionModel::IGeometryModelInfo> modelInfo;
    UniquePtr<nva2f::IDiffusionModel::IBlendshapeSolveModelInfo> blendshapeSolveModelInfo;

    // Cached blendshape names
    std::vector<std::string> blendshapeNames;
    std::vector<const char*> blendshapeNamePtrs;  // For C API
    std::size_t weightCount = 0;

    // Configuration
    std::string modelPath;
    int deviceId = 0;
    std::size_t identityIndex = 0;
    bool useConstantNoise = true;

    // Thread safety
    std::mutex mutex;
    std::atomic<int> sessionCount{0};

    // Active session for callback routing
    A2FSession* activeSession = nullptr;

    // State
    bool initialized = false;
};

struct A2FSession {
    A2FContext* context = nullptr;
    std::size_t trackIndex = 0;

    // Callback state
    A2FFrameCallback callback = nullptr;
    void* userData = nullptr;

    // Frame buffer for results
    std::vector<float> frameWeights;

    // State
    std::atomic<bool> audioFinalized{false};
    std::atomic<bool> emotionInitialized{false};

    // Thread safety
    std::mutex mutex;
};

//
// SDK callback adapter
//

static void SdkHostResultsCallback(
    void* userdata,
    const nva2f::IBlendshapeExecutor::HostResults& results,
    std::error_code errorCode
) {
    // userdata is now A2FContext*, not A2FSession*
    auto* context = static_cast<A2FContext*>(userdata);
    if (!context || !context->activeSession || !context->activeSession->callback) {
        return;
    }

    auto* session = context->activeSession;

    if (errorCode) {
        SetLastError(errorCode);
        return;
    }

    // Build frame structure
    A2FBlendshapeFrame frame;
    frame.time_code = static_cast<double>(results.timeStampCurrentFrame) / 16000.0;
    frame.weight_count = results.weights.Size();

    // Copy weights to session buffer (so callback receives stable pointer)
    session->frameWeights.resize(frame.weight_count);
    const float* srcData = results.weights.Data();
    std::copy(srcData, srcData + frame.weight_count, session->frameWeights.begin());
    frame.weights = session->frameWeights.data();

    // Invoke user callback
    session->callback(session->userData, &frame);
}

//
// Context Management Functions
//

extern "C" {

A2F_API A2FErrorCode a2f_context_create(
    const A2FContextConfig* config,
    A2FContext** out_context
) {
    if (!config || !out_context) {
        SetLastError("Invalid argument: config or out_context is NULL");
        return A2F_ERROR_INVALID_ARGUMENT;
    }

    if (!config->model_path) {
        SetLastError("Invalid argument: model_path is NULL");
        return A2F_ERROR_INVALID_ARGUMENT;
    }

    try {
        auto ctx = std::make_unique<A2FContext>();
        ctx->modelPath = config->model_path;
        ctx->deviceId = config->device_id;
        ctx->identityIndex = config->identity_index;
        ctx->useConstantNoise = (config->use_constant_noise != 0);

        // Initialize CUDA
        auto err = nva2x::SetCudaDeviceIfNeeded(ctx->deviceId);
        if (err) {
            SetLastError(err);
            return A2F_ERROR_CUDA_ERROR;
        }

        // Create blendshape executor bundle
        nva2f::IDiffusionModel::IGeometryModelInfo* rawModelInfoPtr = nullptr;
        nva2f::IDiffusionModel::IBlendshapeSolveModelInfo* rawBlendshapeSolveModelInfoPtr = nullptr;

        auto* rawBundle = nva2f::ReadDiffusionBlendshapeSolveExecutorBundle(
            1,  // nbTracks
            ctx->modelPath.c_str(),
            nva2f::IGeometryExecutor::ExecutionOption::SkinTongue,
            false,  // useGpuSolver (use CPU for host results callback)
            ctx->identityIndex,
            ctx->useConstantNoise,
            &rawModelInfoPtr,
            &rawBlendshapeSolveModelInfoPtr
        );

        if (!rawBundle) {
            SetLastError("Failed to create executor bundle from model: " + ctx->modelPath);
            return A2F_ERROR_MODEL_LOAD_FAILED;
        }

        ctx->bundle = ToUniquePtr(rawBundle);
        ctx->modelInfo = ToUniquePtr(rawModelInfoPtr);
        ctx->blendshapeSolveModelInfo = ToUniquePtr(rawBlendshapeSolveModelInfoPtr);

        // Get weight count
        auto& executor = ctx->bundle->GetExecutor();
        ctx->weightCount = executor.GetWeightCount();

        // Get blendshape names from solvers
        {
            nva2f::IBlendshapeSolver* skinSolver = nullptr;
            auto error = nva2f::GetExecutorSkinSolver(executor, 0, &skinSolver);
            if (!error && skinSolver) {
                int numPoses = skinSolver->NumBlendshapePoses();
                for (int i = 0; i < numPoses; ++i) {
                    const char* name = skinSolver->GetPoseName(i);
                    ctx->blendshapeNames.push_back(name ? name : "unknown");
                }
            }

            nva2f::IBlendshapeSolver* tongueSolver = nullptr;
            error = nva2f::GetExecutorTongueSolver(executor, 0, &tongueSolver);
            if (!error && tongueSolver) {
                int numPoses = tongueSolver->NumBlendshapePoses();
                for (int i = 0; i < numPoses; ++i) {
                    const char* name = tongueSolver->GetPoseName(i);
                    ctx->blendshapeNames.push_back(name ? name : "unknown");
                }
            }
        }

        // Fallback to generic names if no names found
        if (ctx->blendshapeNames.empty()) {
            for (std::size_t i = 0; i < ctx->weightCount; ++i) {
                ctx->blendshapeNames.push_back("blendshape_" + std::to_string(i));
            }
        }

        // Build C-compatible name array
        ctx->blendshapeNamePtrs.clear();
        for (const auto& name : ctx->blendshapeNames) {
            ctx->blendshapeNamePtrs.push_back(name.c_str());
        }

        // Set up the results callback ONCE at context creation time
        // This must be done BEFORE any Execute() calls, and cannot be changed afterwards
        {
            auto& executor = ctx->bundle->GetExecutor();
            auto callbackErr = executor.SetResultsCallback(SdkHostResultsCallback, ctx.get());
            if (callbackErr) {
                SetLastError(callbackErr);
                return A2F_ERROR_EXECUTION_FAILED;
            }
        }

        ctx->initialized = true;
        *out_context = ctx.release();
        return A2F_OK;

    } catch (const std::exception& e) {
        SetLastError(std::string("Exception: ") + e.what());
        return A2F_ERROR_UNKNOWN;
    }
}

A2F_API A2FErrorCode a2f_context_destroy(A2FContext* context) {
    if (!context) {
        SetLastError("Invalid argument: context is NULL");
        return A2F_ERROR_INVALID_ARGUMENT;
    }

    if (context->sessionCount > 0) {
        SetLastError("Cannot destroy context: sessions still active");
        return A2F_ERROR_INVALID_ARGUMENT;
    }

    delete context;
    return A2F_OK;
}

A2F_API A2FErrorCode a2f_get_blendshape_count(
    const A2FContext* context,
    size_t* out_count
) {
    if (!context || !out_count) {
        SetLastError("Invalid argument: context or out_count is NULL");
        return A2F_ERROR_INVALID_ARGUMENT;
    }

    *out_count = context->weightCount;
    return A2F_OK;
}

A2F_API A2FErrorCode a2f_get_blendshape_name(
    const A2FContext* context,
    size_t index,
    const char** out_name
) {
    if (!context || !out_name) {
        SetLastError("Invalid argument: context or out_name is NULL");
        return A2F_ERROR_INVALID_ARGUMENT;
    }

    if (index >= context->blendshapeNames.size()) {
        SetLastError("Index out of bounds");
        return A2F_ERROR_INVALID_ARGUMENT;
    }

    *out_name = context->blendshapeNamePtrs[index];
    return A2F_OK;
}

A2F_API A2FErrorCode a2f_get_blendshape_names(
    const A2FContext* context,
    const char*** out_names,
    size_t* out_count
) {
    if (!context || !out_names || !out_count) {
        SetLastError("Invalid argument: NULL pointer");
        return A2F_ERROR_INVALID_ARGUMENT;
    }

    *out_names = const_cast<const char**>(context->blendshapeNamePtrs.data());
    *out_count = context->blendshapeNamePtrs.size();
    return A2F_OK;
}

//
// Session Management Functions
//

A2F_API A2FErrorCode a2f_session_create(
    A2FContext* context,
    const A2FSessionConfig* config,
    A2FSession** out_session
) {
    if (!context || !config || !out_session) {
        SetLastError("Invalid argument: NULL pointer");
        return A2F_ERROR_INVALID_ARGUMENT;
    }

    if (!context->initialized) {
        SetLastError("Context not initialized");
        return A2F_ERROR_NOT_INITIALIZED;
    }

    try {
        auto session = std::make_unique<A2FSession>();
        session->context = context;
        session->trackIndex = config->track_index;
        session->frameWeights.reserve(context->weightCount);

        // Set this session as the active session for callback routing
        // Note: SetResultsCallback is already done once in a2f_context_create()
        context->activeSession = session.get();

        context->sessionCount++;
        *out_session = session.release();
        return A2F_OK;

    } catch (const std::exception& e) {
        SetLastError(std::string("Exception: ") + e.what());
        return A2F_ERROR_UNKNOWN;
    }
}

A2F_API A2FErrorCode a2f_session_set_callback(
    A2FSession* session,
    A2FFrameCallback callback,
    void* user_data
) {
    if (!session) {
        SetLastError("Invalid argument: session is NULL");
        return A2F_ERROR_INVALID_ARGUMENT;
    }

    std::lock_guard<std::mutex> lock(session->mutex);
    session->callback = callback;
    session->userData = user_data;
    return A2F_OK;
}

A2F_API A2FErrorCode a2f_session_push_audio(
    A2FSession* session,
    const float* samples,
    size_t sample_count
) {
    if (!session || !samples) {
        SetLastError("Invalid argument: NULL pointer");
        return A2F_ERROR_INVALID_ARGUMENT;
    }

    if (session->audioFinalized) {
        SetLastError("Session already finalized");
        return A2F_ERROR_SESSION_CLOSED;
    }

    auto* context = session->context;
    if (!context || !context->bundle) {
        SetLastError("Context not available");
        return A2F_ERROR_NOT_INITIALIZED;
    }

    try {
        // Initialize emotion accumulator on first audio push (neutral emotion)
        if (!session->emotionInitialized.exchange(true)) {
            auto& emotionAccumulator = context->bundle->GetEmotionAccumulator(session->trackIndex);
            std::vector<float> emptyEmotion(emotionAccumulator.GetEmotionSize(), 0.0f);
            auto err = emotionAccumulator.Accumulate(
                0,
                nva2x::HostTensorFloatConstView{emptyEmotion.data(), emptyEmotion.size()},
                context->bundle->GetCudaStream().Data()
            );
            if (err) {
                SetLastError(err);
                return A2F_ERROR_EXECUTION_FAILED;
            }
            err = emotionAccumulator.Close();
            if (err) {
                SetLastError(err);
                return A2F_ERROR_EXECUTION_FAILED;
            }
        }

        // Accumulate audio
        auto& audioAccumulator = context->bundle->GetAudioAccumulator(session->trackIndex);
        auto err = audioAccumulator.Accumulate(
            nva2x::HostTensorFloatConstView{samples, sample_count},
            context->bundle->GetCudaStream().Data()
        );
        if (err) {
            SetLastError(err);
            return A2F_ERROR_EXECUTION_FAILED;
        }

        return A2F_OK;

    } catch (const std::exception& e) {
        SetLastError(std::string("Exception: ") + e.what());
        return A2F_ERROR_UNKNOWN;
    }
}

A2F_API A2FErrorCode a2f_session_push_audio_int16(
    A2FSession* session,
    const int16_t* samples,
    size_t sample_count
) {
    if (!session || !samples) {
        SetLastError("Invalid argument: NULL pointer");
        return A2F_ERROR_INVALID_ARGUMENT;
    }

    // Convert int16 to float32
    std::vector<float> floatSamples(sample_count);
    for (size_t i = 0; i < sample_count; ++i) {
        floatSamples[i] = static_cast<float>(samples[i]) / 32768.0f;
    }

    return a2f_session_push_audio(session, floatSamples.data(), sample_count);
}

A2F_API A2FErrorCode a2f_session_execute(
    A2FSession* session,
    size_t* out_pending_frames
) {
    if (!session) {
        SetLastError("Invalid argument: session is NULL");
        return A2F_ERROR_INVALID_ARGUMENT;
    }

    auto* context = session->context;
    if (!context || !context->bundle) {
        SetLastError("Context not available");
        return A2F_ERROR_NOT_INITIALIZED;
    }

    try {
        auto& executor = context->bundle->GetExecutor();

        // Execute for ready tracks
        size_t readyTracks = nva2x::GetNbReadyTracks(executor);
        if (out_pending_frames) {
            *out_pending_frames = readyTracks;
        }

        if (readyTracks > 0) {
            auto err = executor.Execute(nullptr);
            if (err) {
                SetLastError(err);
                return A2F_ERROR_EXECUTION_FAILED;
            }
        }

        return A2F_OK;

    } catch (const std::exception& e) {
        SetLastError(std::string("Exception: ") + e.what());
        return A2F_ERROR_UNKNOWN;
    }
}

A2F_API A2FErrorCode a2f_session_finalize(A2FSession* session) {
    if (!session) {
        SetLastError("Invalid argument: session is NULL");
        return A2F_ERROR_INVALID_ARGUMENT;
    }

    if (session->audioFinalized) {
        return A2F_OK;  // Already finalized
    }

    auto* context = session->context;
    if (!context || !context->bundle) {
        SetLastError("Context not available");
        return A2F_ERROR_NOT_INITIALIZED;
    }

    try {
        // Close audio accumulator
        auto& audioAccumulator = context->bundle->GetAudioAccumulator(session->trackIndex);
        auto err = audioAccumulator.Close();
        if (err) {
            SetLastError(err);
            return A2F_ERROR_EXECUTION_FAILED;
        }

        session->audioFinalized = true;

        auto& executor = context->bundle->GetExecutor();

        // Execute until all frames are processed
        while (nva2x::GetNbReadyTracks(executor) > 0) {
            err = executor.Execute(nullptr);
            if (err) {
                SetLastError(err);
                return A2F_ERROR_EXECUTION_FAILED;
            }
        }

        // Wait for async work to complete
        err = executor.Wait(session->trackIndex);
        if (err) {
            SetLastError(err);
            return A2F_ERROR_EXECUTION_FAILED;
        }

        return A2F_OK;

    } catch (const std::exception& e) {
        SetLastError(std::string("Exception: ") + e.what());
        return A2F_ERROR_UNKNOWN;
    }
}

A2F_API A2FErrorCode a2f_session_reset(A2FSession* session) {
    if (!session) {
        SetLastError("Invalid argument: session is NULL");
        return A2F_ERROR_INVALID_ARGUMENT;
    }

    auto* context = session->context;
    if (!context || !context->bundle) {
        SetLastError("Context not available");
        return A2F_ERROR_NOT_INITIALIZED;
    }

    try {
        // Reset audio accumulator
        auto& audioAccumulator = context->bundle->GetAudioAccumulator(session->trackIndex);
        auto err = audioAccumulator.Reset();
        if (err) {
            SetLastError(err);
            return A2F_ERROR_EXECUTION_FAILED;
        }

        // Reset emotion accumulator
        auto& emotionAccumulator = context->bundle->GetEmotionAccumulator(session->trackIndex);
        err = emotionAccumulator.Reset();
        if (err) {
            SetLastError(err);
            return A2F_ERROR_EXECUTION_FAILED;
        }

        // Reset executor track
        auto& executor = context->bundle->GetExecutor();
        err = executor.Reset(session->trackIndex);
        if (err) {
            SetLastError(err);
            return A2F_ERROR_EXECUTION_FAILED;
        }

        session->audioFinalized = false;
        session->emotionInitialized = false;
        session->frameWeights.clear();

        return A2F_OK;

    } catch (const std::exception& e) {
        SetLastError(std::string("Exception: ") + e.what());
        return A2F_ERROR_UNKNOWN;
    }
}

A2F_API A2FErrorCode a2f_session_destroy(A2FSession* session) {
    if (!session) {
        SetLastError("Invalid argument: session is NULL");
        return A2F_ERROR_INVALID_ARGUMENT;
    }

    auto* context = session->context;
    if (context) {
        // Reset accumulators and executor track for the next session
        if (context->bundle) {
            try {
                // Reset audio accumulator
                auto& audioAccumulator = context->bundle->GetAudioAccumulator(session->trackIndex);
                audioAccumulator.Reset();

                // Reset emotion accumulator
                auto& emotionAccumulator = context->bundle->GetEmotionAccumulator(session->trackIndex);
                emotionAccumulator.Reset();

                // Reset executor track
                auto& executor = context->bundle->GetExecutor();
                executor.Reset(session->trackIndex);
            } catch (...) {
                // Ignore errors during cleanup
            }
        }

        // Clear activeSession if this session is the active one
        if (context->activeSession == session) {
            context->activeSession = nullptr;
        }
        context->sessionCount--;
    }

    delete session;
    return A2F_OK;
}

//
// Utility Functions
//

A2F_API const char* a2f_get_last_error_message(void) {
    return g_lastErrorMessage.c_str();
}

A2F_API const char* a2f_get_version(void) {
    return A2F_VERSION;
}

} // extern "C"
