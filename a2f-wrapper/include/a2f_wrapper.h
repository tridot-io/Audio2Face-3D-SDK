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
    A2F_ERROR_UNKNOWN = 99
} A2FErrorCode;

/**
 * Configuration for context initialization
 */
typedef struct {
    const char* model_path;        /**< Path to model.json file */
    int device_id;                 /**< CUDA device ID (default: 0) */
    size_t identity_index;         /**< Identity index for multi-identity models */
    int use_constant_noise;        /**< 1 = use constant noise for deterministic output */
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
 * Callback function type for receiving blendshape frames.
 * Called from SDK thread - implementation must be thread-safe.
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
