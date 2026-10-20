/**
 * @file a2f_wrapper.h
 * @brief C API wrapper for Audio2Face-3D-SDK
 *
 * This header provides a pure C interface to the Audio2Face SDK,
 * enabling Python ctypes integration for real-time audio streaming.
 */

#ifndef A2F_WRAPPER_H
#define A2F_WRAPPER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Export macro for shared library */
#ifdef _WIN32
    #ifdef A2F_WRAPPER_EXPORTS
        #define A2F_API __declspec(dllexport)
    #else
        #define A2F_API __declspec(dllimport)
    #endif
#else
    #define A2F_API __attribute__((visibility("default")))
#endif

/* Opaque handle types */
typedef struct A2FContext A2FContext;
typedef struct A2FSession A2FSession;

/**
 * Error codes returned by API functions
 */
typedef enum {
    A2F_OK = 0,
    A2F_ERROR_INVALID_ARGUMENT = 1,
    A2F_ERROR_NOT_INITIALIZED = 2,
    A2F_ERROR_CUDA_ERROR = 3,
    A2F_ERROR_MODEL_LOAD_FAILED = 4,
    A2F_ERROR_OUT_OF_MEMORY = 5,
    A2F_ERROR_SESSION_CLOSED = 6,
    A2F_ERROR_CALLBACK_ERROR = 7,
    A2F_ERROR_EXECUTION_FAILED = 8,
    A2F_ERROR_A2E_INIT_FAILED = 9,
    A2F_ERROR_A2E_EXECUTION_FAILED = 10,
    A2F_ERROR_UNKNOWN = 99
} A2FErrorCode;

/**
 * Configuration for context initialization
 */
typedef struct {
    const char* model_path;        /**< Path to model.json file */
    const char* a2e_model_path;    /**< Path to A2E model.json file (NULL to disable A2E) */
    int device_id;                 /**< CUDA device ID (default: 0) */
    size_t identity_index;         /**< Identity index for multi-identity models */
    int use_constant_noise;        /**< 1 = use constant noise for deterministic output */
    int enable_audio2emotion;      /**< 1 = enable Audio2Emotion for eyebrow/expression inference */
    size_t a2e_buffer_length;       /**< A2E buffer length in samples (0 = use default 10000) */
    size_t a2e_inferences_to_skip;  /**< A2E inferences to skip (0 = every frame) */
    size_t a2e_frame_rate_num;      /**< A2E output frame rate numerator (0 = default 30) */
    size_t a2e_frame_rate_den;      /**< A2E output frame rate denominator (0 = default 1) */
} A2FContextConfig;

/**
 * Configuration for session creation
 */
typedef struct {
    size_t track_index;            /**< Track index (usually 0 for single-track) */
} A2FSessionConfig;

/**
 * Single blendshape frame result
 */
typedef struct {
    double time_code;              /**< Frame timestamp in seconds */
    size_t weight_count;           /**< Number of blendshape weights */
    const float* weights;          /**< Pointer to weight array (valid until next callback) */
} A2FBlendshapeFrame;

/**
 * Skin parameters for controlling face animation behavior.
 * These map to SDK's AnimatorSkinParams structure.
 * Field order matches SDK's nva2f::AnimatorSkinParams.
 */
typedef struct {
    float lower_face_smoothing;    /**< Lower face smoothing (default: 0.001) */
    float upper_face_smoothing;    /**< Upper face smoothing (default: 0.001) */
    float lower_face_strength;     /**< Lower face animation strength (default: 1.0) */
    float upper_face_strength;     /**< Upper face animation strength (default: 1.0) */
    float face_mask_level;         /**< Face mask level (default: 0.6) */
    float face_mask_softness;      /**< Face mask softness (default: 0.0085) */
    float skin_strength;           /**< Overall skin strength (default: 1.0) */
    float blink_strength;          /**< Blink animation strength (default: 0.75) */
    float eyelid_open_offset;      /**< Eyelid open offset (default: 0.0) */
    float lip_open_offset;         /**< Lip open offset (default: 0.0) */
    float blink_offset;            /**< Blink offset (default: 0.0) */
} A2FSkinParams;

/**
 * Audio2Emotion post-processing parameters (map to nva2e::PostProcessParams).
 * preferred_emotion order: amazement, anger, cheekiness, disgust, fear, grief, joy, outofbreath, pain, sadness.
 */
typedef struct {
    float emotion_contrast;            /**< default 1.0 */
    size_t max_emotions;               /**< default 6 */
    float live_blend_coef;             /**< smoothing of detected emotion, default 0.7 */
    int enable_preferred_emotion;      /**< 1 = blend preferred_emotion into the detected emotion */
    float preferred_emotion_strength;  /**< 0 = detected only, 1 = preferred only */
    float emotion_strength;            /**< overall strength, default 0.6 */
    float preferred_emotion[10];
} A2FEmotionParams;

/**
 * Callback function type for receiving blendshape frames.
 * Called from SDK thread - implementation must be thread-safe. Frames of one session arrive in order, one at a time;
 * a2f_session_wait / a2f_session_finalize / a2f_session_destroy return after the last pending callback.
 * The session functions are not synchronized against each other: call them for one session from one thread at a time.
 *
 * @param user_data User-provided data pointer
 * @param frame Pointer to the blendshape frame data
 * @return 0 to continue processing, non-zero to abort
 */
typedef int (*A2FFrameCallback)(void* user_data, const A2FBlendshapeFrame* frame);


/* ============================================================================
 * Context Management Functions
 * ============================================================================ */

/**
 * Create a new A2F context.
 * Loads the model and initializes CUDA resources.
 * Call once at application startup.
 *
 * @param config Configuration parameters
 * @param out_context Output pointer to receive the context handle
 * @return A2F_OK on success, error code on failure
 */
A2F_API A2FErrorCode a2f_context_create(
    const A2FContextConfig* config,
    A2FContext** out_context
);

/**
 * Choose where the blendshape (skin/tongue) solve runs for contexts created after this call.
 * 0 (default): CPU, bounded least squares (the SDK sample behaviour).
 * 1: GPU, ADMM approximation (SDK default iterations). About 35% less processing time per audio second and no
 *    measurable extra GPU memory; the curves differ from the CPU solve (less upper-lip raise, more upper-lip roll).
 * Existing contexts keep their solver. Added in wrapper 1.0.1 (ondevice branch); check the export, not the version.
 *
 * @param enable 1 for GPU, 0 for CPU
 * @return A2F_OK
 */
A2F_API A2FErrorCode a2f_set_gpu_blendshape_solver(int enable);

/**
 * Record the TensorRT inferences and the GPU blendshape solve once as CUDA graphs and replay them (process-wide).
 * The output is identical. Each engine is checked once: one with data-dependent layers (NonZero, e.g. the original A2F
 * diffusion engine) or without a layer list keeps running directly; the A2F diffusion engine rebuilt from
 * tools/fold_a2f_diffusion_nonzero.py is recorded.
 * Requirements while on: one context in the process (creating a second context turns graphs off, and turning them on
 * with more than one context fails), and audio push and execute on the same thread (a push from another thread during
 * a recording would be captured instead of run).
 * 0 (default): off. Added in wrapper 1.0.1 (ondevice branch); check the export, not the version.
 *
 * @param enable 1 to turn CUDA graphs on, 0 to turn them off
 * @return A2F_OK, or A2F_ERROR_INVALID_ARGUMENT when turning on with more than one live context
 */
A2F_API A2FErrorCode a2f_set_cuda_graphs(int enable);

/**
 * Destroy a context and release all resources.
 * All sessions must be destroyed before calling this.
 *
 * @param context Context handle to destroy
 * @return A2F_OK on success, error code on failure
 */
A2F_API A2FErrorCode a2f_context_destroy(A2FContext* context);

/**
 * Get the number of blendshape weights output by the model.
 *
 * @param context Context handle
 * @param out_count Output pointer to receive the count
 * @return A2F_OK on success, error code on failure
 */
A2F_API A2FErrorCode a2f_get_blendshape_count(
    const A2FContext* context,
    size_t* out_count
);

/**
 * Get the name of a specific blendshape.
 *
 * @param context Context handle
 * @param index Blendshape index (0 to count-1)
 * @param out_name Output pointer to receive the name (do not free)
 * @return A2F_OK on success, error code on failure
 */
A2F_API A2FErrorCode a2f_get_blendshape_name(
    const A2FContext* context,
    size_t index,
    const char** out_name
);

/**
 * Get all blendshape names as an array.
 *
 * @param context Context handle
 * @param out_names Output pointer to receive array of name pointers (do not free)
 * @param out_count Output pointer to receive the count
 * @return A2F_OK on success, error code on failure
 */
A2F_API A2FErrorCode a2f_get_blendshape_names(
    const A2FContext* context,
    const char*** out_names,
    size_t* out_count
);


/* ============================================================================
 * Session Management Functions
 * ============================================================================ */

/**
 * Create a new streaming session.
 * Each session maintains its own audio buffer and GRU state.
 *
 * @param context Parent context handle
 * @param config Session configuration
 * @param out_session Output pointer to receive the session handle
 * @return A2F_OK on success, error code on failure
 */
A2F_API A2FErrorCode a2f_session_create(
    A2FContext* context,
    const A2FSessionConfig* config,
    A2FSession** out_session
);

/**
 * Set the callback function for receiving blendshape frames.
 * Must be called before pushing audio.
 *
 * @param session Session handle
 * @param callback Callback function pointer
 * @param user_data User data passed to callback
 * @return A2F_OK on success, error code on failure
 */
A2F_API A2FErrorCode a2f_session_set_callback(
    A2FSession* session,
    A2FFrameCallback callback,
    void* user_data
);

/**
 * Set skin parameters for animation control.
 * Parameters are applied during subsequent execute calls.
 * Call before pushing audio or before execute for best results.
 *
 * @param session Session handle
 * @param params Skin parameters to apply
 * @return A2F_OK on success, error code on failure
 */
A2F_API A2FErrorCode a2f_session_set_skin_params(
    A2FSession* session,
    const A2FSkinParams* params
);

/**
 * Set tongue parameters (map to nva2f::AnimatorTongueParams). Applied before the next execute, like skin params.
 * tongue_strength 0 = no tongue motion (model default 1.3).
 */
A2F_API A2FErrorCode a2f_session_set_tongue_params(
    A2FSession* session,
    float tongue_strength,
    float tongue_height_offset,
    float tongue_depth_offset
);

/**
 * Set Audio2Emotion post-processing parameters for this session's track (emotion strength, preferred emotion, ...).
 * Applies to subsequent A2E executions. No-op (A2F_OK) when A2E is disabled.
 */
A2F_API A2FErrorCode a2f_session_set_emotion_params(
    A2FSession* session,
    const A2FEmotionParams* params
);

/**
 * Push audio samples to the session (float32 format).
 * Samples should be 16kHz mono.
 * Thread-safe: can be called from any thread.
 *
 * @param session Session handle
 * @param samples Pointer to float32 audio samples
 * @param sample_count Number of samples
 * @return A2F_OK on success, error code on failure
 */
A2F_API A2FErrorCode a2f_session_push_audio(
    A2FSession* session,
    const float* samples,
    size_t sample_count
);

/**
 * Push audio samples to the session (int16 PCM format).
 * Samples should be 16kHz mono.
 * Automatically converts to float32.
 *
 * @param session Session handle
 * @param samples Pointer to int16 PCM audio samples
 * @param sample_count Number of samples
 * @return A2F_OK on success, error code on failure
 */
A2F_API A2FErrorCode a2f_session_push_audio_int16(
    A2FSession* session,
    const int16_t* samples,
    size_t sample_count
);

/**
 * Execute inference for currently accumulated audio.
 * Frames are delivered asynchronously via callback.
 *
 * @param session Session handle
 * @param out_pending_frames Optional output for pending frame count (can be NULL)
 * @return A2F_OK on success, error code on failure
 */
A2F_API A2FErrorCode a2f_session_execute(
    A2FSession* session,
    size_t* out_pending_frames
);

/**
 * Wait until every frame of the work started so far has been delivered through the callback.
 * execute() returns before the last CPU blendshape solve of each batch: that frame's callback runs on an SDK
 * worker thread. Call this after execute() when the callback must not run outside the caller's own locking.
 *
 * @param session Session handle
 * @return A2F_OK on success, error code on failure
 */
A2F_API A2FErrorCode a2f_session_wait(A2FSession* session);

/**
 * Signal end of audio and process all remaining data.
 * Blocks until all frames are delivered via callback.
 *
 * @param session Session handle
 * @return A2F_OK on success, error code on failure
 */
A2F_API A2FErrorCode a2f_session_finalize(A2FSession* session);

/**
 * Reset session for reuse.
 * Clears audio buffer and resets GRU state.
 *
 * @param session Session handle
 * @return A2F_OK on success, error code on failure
 */
A2F_API A2FErrorCode a2f_session_reset(A2FSession* session);

/**
 * Destroy a session and release resources.
 *
 * @param session Session handle to destroy
 * @return A2F_OK on success, error code on failure
 */
A2F_API A2FErrorCode a2f_session_destroy(A2FSession* session);


/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * Get the last error message (thread-local).
 *
 * @return Error message string (do not free)
 */
A2F_API const char* a2f_get_last_error_message(void);

/**
 * Get the library version string.
 *
 * @return Version string (do not free)
 */
A2F_API const char* a2f_get_version(void);

#ifdef __cplusplus
}
#endif

#endif /* A2F_WRAPPER_H */
