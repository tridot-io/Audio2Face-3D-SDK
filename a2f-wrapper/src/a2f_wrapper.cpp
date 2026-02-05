/**
 * @file a2f_wrapper.cpp
 * @brief C API wrapper implementation for Audio2Face-3D-SDK
 */

#include "a2f_wrapper.h"
#include "audio2face/audio2face.h"
#include "audio2emotion/audio2emotion.h"
#include "audio2x/cuda_utils.h"

#include <mutex>
#include <atomic>
#include <vector>
#include <string>
#include <memory>
#include <cstring>
#include <thread>
#include <chrono>
#include <cstdio>

// Simple logging macro for timing (to stderr)
#define A2F_LOG_TIMING(fmt, ...) fprintf(stderr, "[A2F TIMING] " fmt "\n", ##__VA_ARGS__)

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

    // A2E (Audio2Emotion) components
    UniquePtr<nva2e::IClassifierModel::IEmotionModelInfo> a2eModelInfo;
    UniquePtr<nva2e::IEmotionExecutor> emotionExecutor;
    UniquePtr<nva2e::IEmotionBinder> emotionBinder;
    bool a2eEnabled = false;
    std::string a2eModelPath;

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

    // Skin parameters (cached for apply during execute)
    nva2f::AnimatorSkinParams skinParams{};
    bool skinParamsSet = false;

    // Audio tracking for emotion timestamp sync
    std::atomic<size_t> totalSamplesPushed{0};
    std::atomic<bool> neutralEmotionAccumulated{false};  // For A2E disabled mode

    // State
    std::atomic<bool> audioFinalized{false};
    std::atomic<bool> emotionInitialized{false};
    std::atomic<bool> emotionAccumulatorClosed{false};

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

        // Initialize A2E (Audio2Emotion) if enabled
        if (config->enable_audio2emotion && config->a2e_model_path) {
            ctx->a2eEnabled = true;
            ctx->a2eModelPath = config->a2e_model_path;

            // 1. Load A2E classifier model info
            auto* rawA2eModelInfo = nva2e::ReadClassifierModelInfo(ctx->a2eModelPath.c_str());
            if (!rawA2eModelInfo) {
                SetLastError("Failed to load A2E model from: " + ctx->a2eModelPath);
                return A2F_ERROR_A2E_INIT_FAILED;
            }
            ctx->a2eModelInfo = ToUniquePtr(rawA2eModelInfo);

            // 2. Create EmotionExecutor creation parameters
            nva2e::EmotionExecutorCreationParameters emotionParams;
            emotionParams.cudaStream = ctx->bundle->GetCudaStream().Data();
            emotionParams.nbTracks = 1;
            const auto sharedAudioAccumulator = &ctx->bundle->GetAudioAccumulator(0);
            emotionParams.sharedAudioAccumulators = &sharedAudioAccumulator;

            // 3. Get classifier-specific parameters (30 FPS output, skip 30 inferences)
            auto classifierParams = ctx->a2eModelInfo->GetExecutorCreationParameters(
                60000,  // bufferLength (samples)
                30,     // frameRateNumerator
                1,      // frameRateDenominator
                30      // inferencesToSkip
            );

            // 4. Create EmotionExecutor
            auto* rawEmotionExecutor = nva2e::CreateClassifierEmotionExecutor(emotionParams, classifierParams);
            if (!rawEmotionExecutor) {
                SetLastError("Failed to create A2E emotion executor");
                return A2F_ERROR_A2E_INIT_FAILED;
            }
            ctx->emotionExecutor = ToUniquePtr(rawEmotionExecutor);

            // 5. Create EmotionBinder (connects executor output to emotionAccumulator)
            auto* emotionAcc = &ctx->bundle->GetEmotionAccumulator(0);
            auto* rawEmotionBinder = nva2e::CreateEmotionBinder(*ctx->emotionExecutor, &emotionAcc, 1);
            if (!rawEmotionBinder) {
                SetLastError("Failed to create A2E emotion binder");
                return A2F_ERROR_A2E_INIT_FAILED;
            }
            ctx->emotionBinder = ToUniquePtr(rawEmotionBinder);
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

A2F_API A2FErrorCode a2f_session_set_skin_params(
    A2FSession* session,
    const A2FSkinParams* params
) {
    if (!session || !params) {
        SetLastError("Invalid argument: session or params is NULL");
        return A2F_ERROR_INVALID_ARGUMENT;
    }

    std::lock_guard<std::mutex> lock(session->mutex);

    // Copy params to SDK structure (field order matches)
    session->skinParams.lowerFaceSmoothing = params->lower_face_smoothing;
    session->skinParams.upperFaceSmoothing = params->upper_face_smoothing;
    session->skinParams.lowerFaceStrength = params->lower_face_strength;
    session->skinParams.upperFaceStrength = params->upper_face_strength;
    session->skinParams.faceMaskLevel = params->face_mask_level;
    session->skinParams.faceMaskSoftness = params->face_mask_softness;
    session->skinParams.skinStrength = params->skin_strength;
    session->skinParams.blinkStrength = params->blink_strength;
    session->skinParams.eyelidOpenOffset = params->eyelid_open_offset;
    session->skinParams.lipOpenOffset = params->lip_open_offset;
    session->skinParams.blinkOffset = params->blink_offset;

    session->skinParamsSet = true;
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
        // Note: Emotion accumulator initialization moved to a2f_session_finalize()
        // A2E executor will populate emotions from audio, or neutral fallback will be used

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

        // Track total samples for emotion timestamp sync
        session->totalSamplesPushed += sample_count;

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
        using Clock = std::chrono::high_resolution_clock;

        // Execute A2E (Audio2Emotion) first to populate emotion accumulator
        if (context->a2eEnabled && context->emotionExecutor) {
            auto a2eStart = Clock::now();
            int a2eExecCount = 0;
            while (nva2x::GetNbReadyTracks(*context->emotionExecutor) > 0) {
                auto err = context->emotionExecutor->Execute(nullptr);
                if (err) {
                    SetLastError(err);
                    return A2F_ERROR_A2E_EXECUTION_FAILED;
                }
                a2eExecCount++;
            }
            if (a2eExecCount > 0) {
                auto a2eEnd = Clock::now();
                auto a2eMs = std::chrono::duration<double, std::milli>(a2eEnd - a2eStart).count();
                A2F_LOG_TIMING("A2E execute: %d calls, %.2fms", a2eExecCount, a2eMs);
            }
        } else if (!context->a2eEnabled) {
            // A2E disabled: continuously feed neutral emotion during streaming
            // This allows A2F executor to have ready tracks during streaming
            // Use totalSamplesPushed as timestamp to ensure increasing order
            auto& emotionAccumulator = context->bundle->GetEmotionAccumulator(session->trackIndex);
            std::vector<float> neutralEmotion(emotionAccumulator.GetEmotionSize(), 0.0f);
            auto err = emotionAccumulator.Accumulate(
                static_cast<int64_t>(session->totalSamplesPushed.load()),
                nva2x::HostTensorFloatConstView{neutralEmotion.data(), neutralEmotion.size()},
                context->bundle->GetCudaStream().Data()
            );
            if (err) {
                SetLastError(err);
                return A2F_ERROR_EXECUTION_FAILED;
            }
            session->neutralEmotionAccumulated = true;
        }

        auto& executor = context->bundle->GetExecutor();

        // Apply skin params if set (only once, before first execution)
        if (session->skinParamsSet) {
            auto err = nva2f::SetExecutorSkinParameters(executor, session->trackIndex, session->skinParams);
            if (err) {
                SetLastError(err);
                return A2F_ERROR_EXECUTION_FAILED;
            }
            session->skinParamsSet = false;  // Only apply once
        }

        // Execute for ready tracks
        size_t readyTracks = nva2x::GetNbReadyTracks(executor);
        if (out_pending_frames) {
            *out_pending_frames = readyTracks;
        }

        if (readyTracks > 0) {
            auto a2fStart = Clock::now();
            auto err = executor.Execute(nullptr);
            auto a2fEnd = Clock::now();
            auto a2fMs = std::chrono::duration<double, std::milli>(a2fEnd - a2fStart).count();
            A2F_LOG_TIMING("A2F execute: %.2fms (ready tracks: %zu)", a2fMs, readyTracks);

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
        using Clock = std::chrono::high_resolution_clock;
        auto finalizeStart = Clock::now();
        double totalA2eMs = 0.0;
        double totalA2fMs = 0.0;
        int a2eExecCount = 0;
        int a2fExecCount = 0;

        // 1. Close audio accumulator
        auto& audioAccumulator = context->bundle->GetAudioAccumulator(session->trackIndex);
        auto err = audioAccumulator.Close();
        if (err) {
            SetLastError(err);
            return A2F_ERROR_EXECUTION_FAILED;
        }

        session->audioFinalized = true;

        auto& executor = context->bundle->GetExecutor();
        auto& emotionAccumulator = context->bundle->GetEmotionAccumulator(session->trackIndex);

        // 2. Process loop: A2E → close emotion accumulator when done → A2F
        while (true) {
            // Process A2E (Audio2Emotion)
            if (context->a2eEnabled && context->emotionExecutor) {
                while (nva2x::GetNbReadyTracks(*context->emotionExecutor) > 0) {
                    auto a2eStart = Clock::now();
                    err = context->emotionExecutor->Execute(nullptr);
                    auto a2eEnd = Clock::now();
                    totalA2eMs += std::chrono::duration<double, std::milli>(a2eEnd - a2eStart).count();
                    a2eExecCount++;
                    if (err) {
                        SetLastError(err);
                        return A2F_ERROR_A2E_EXECUTION_FAILED;
                    }
                }

                // Check if A2E is complete - close emotion accumulator
                if (!session->emotionAccumulatorClosed &&
                    context->emotionExecutor->GetNbAvailableExecutions(session->trackIndex) == 0) {
                    err = emotionAccumulator.Close();
                    if (err) {
                        SetLastError(err);
                        return A2F_ERROR_EXECUTION_FAILED;
                    }
                    session->emotionAccumulatorClosed = true;
                }
            } else if (!session->emotionAccumulatorClosed) {
                // A2E disabled: handle emotion accumulator
                if (!session->neutralEmotionAccumulated) {
                    // Batch mode: no execute() was called, need to add neutral emotion
                    std::vector<float> neutralEmotion(emotionAccumulator.GetEmotionSize(), 0.0f);
                    err = emotionAccumulator.Accumulate(
                        static_cast<int64_t>(session->totalSamplesPushed.load()),
                        nva2x::HostTensorFloatConstView{neutralEmotion.data(), neutralEmotion.size()},
                        context->bundle->GetCudaStream().Data()
                    );
                    if (err) {
                        SetLastError(err);
                        return A2F_ERROR_EXECUTION_FAILED;
                    }
                }
                // Close the accumulator (streaming mode already has emotions from execute())
                err = emotionAccumulator.Close();
                if (err) {
                    SetLastError(err);
                    return A2F_ERROR_EXECUTION_FAILED;
                }
                session->emotionAccumulatorClosed = true;
            }

            // Process A2F (Audio2Face)
            if (nva2x::GetNbReadyTracks(executor) > 0) {
                auto a2fStart = Clock::now();
                err = executor.Execute(nullptr);
                auto a2fEnd = Clock::now();
                totalA2fMs += std::chrono::duration<double, std::milli>(a2fEnd - a2fStart).count();
                a2fExecCount++;
                if (err) {
                    SetLastError(err);
                    return A2F_ERROR_EXECUTION_FAILED;
                }
                continue;
            }

            break;
        }

        // Wait for async work to complete
        auto waitStart = Clock::now();
        err = executor.Wait(session->trackIndex);
        auto waitEnd = Clock::now();
        auto waitMs = std::chrono::duration<double, std::milli>(waitEnd - waitStart).count();

        if (err) {
            SetLastError(err);
            return A2F_ERROR_EXECUTION_FAILED;
        }

        auto finalizeEnd = Clock::now();
        auto totalMs = std::chrono::duration<double, std::milli>(finalizeEnd - finalizeStart).count();

        A2F_LOG_TIMING("Finalize summary: A2E %.2fms (%d calls), A2F %.2fms (%d calls), Wait %.2fms, Total %.2fms",
            totalA2eMs, a2eExecCount, totalA2fMs, a2fExecCount, waitMs, totalMs);

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

        // Reset emotion executor if A2E is enabled
        if (context->a2eEnabled && context->emotionExecutor) {
            err = context->emotionExecutor->Reset(session->trackIndex);
            if (err) {
                SetLastError(err);
                return A2F_ERROR_EXECUTION_FAILED;
            }
        }

        session->audioFinalized = false;
        session->emotionInitialized = false;
        session->emotionAccumulatorClosed = false;
        session->skinParamsSet = false;
        session->totalSamplesPushed = 0;
        session->neutralEmotionAccumulated = false;
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

                // Reset emotion executor if A2E is enabled
                if (context->a2eEnabled && context->emotionExecutor) {
                    context->emotionExecutor->Reset(session->trackIndex);
                }
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
