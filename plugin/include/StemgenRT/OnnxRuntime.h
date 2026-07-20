#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <array>
#include <juce_core/juce_core.h>
#include "Constants.h"

// Forward declarations to avoid exposing ORT headers
struct OrtEnv;
struct OrtSession;
struct OrtMemoryInfo;
struct OrtApi;
struct OrtApiBase;

namespace audio_plugin {

// RAII wrapper for the qualified CPU ONNX Runtime session and stateful graph.
class OnnxRuntime {
public:
    OnnxRuntime();
    ~OnnxRuntime();

    // Non-copyable, non-movable (owns ORT resources)
    OnnxRuntime(const OnnxRuntime&) = delete;
    OnnxRuntime& operator=(const OnnxRuntime&) = delete;
    OnnxRuntime(OnnxRuntime&&) = delete;
    OnnxRuntime& operator=(OnnxRuntime&&) = delete;

    // Initialize the ORT environment (called in constructor, but can fail)
    // Returns true if environment is ready
    bool isInitialized() const { return ortInitialized_; }

    // Load a model from file path
    // Returns true on success, sets errorMessage on failure
    bool loadModel(const juce::String& modelPath, juce::String& errorMessage);

    // Check if model is loaded and ready for inference
    bool isModelLoaded() const { return modelLoaded_; }

    // Get the active execution provider name (the qualified build uses "CPU").
    const std::string& getExecutionProvider() const { return executionProvider_; }

    // Get the ORT runtime version string
    const std::string& getRuntimeVersion() const { return runtimeVersion_; }

    // Prepare for inference (allocate scratch/state buffers and memory info)
    // Must be called before runInference
    void prepareForInference();

    // Reset every persistent model state to zero. The inference queue calls this
    // from its worker thread on transport epochs and sequence gaps. It is also
    // safe to call from a stopped/non-real-time control path.
    void resetStreamingState();

    // Run one stateful graph hop. The returned samples and alignedInput belong
    // to the previous input hop. outputValid is false for the pre-roll result
    // immediately after reset; callers must discard that result.
    bool runInference(
        const std::array<std::vector<float>, kNumChannels>& inputChunk,
        std::array<std::array<std::vector<float>, kNumChannels>, kNumStems>& outputChunks,
        std::array<std::vector<float>, kNumChannels>& alignedInput,
        bool& outputValid);

    // Get a status string suitable for display
    juce::String getStatusString() const;

private:
    // Custom deleters for ORT handles
    struct OrtEnvDeleter {
        void operator()(OrtEnv* p) const noexcept;
    };
    struct OrtSessionDeleter {
        void operator()(OrtSession* p) const noexcept;
    };

    // Helper to safely get ORT API
    static const OrtApiBase* getSafeOrtApiBase() noexcept;
    static const OrtApi* getSafeOrtApi() noexcept;

    bool validateModelContract(juce::String& errorMessage) const;
    void resetStreamingStateUnlocked();

#ifdef _WIN32
    // Load the exact DLL beside this module and return its native handle.
    // Kept opaque here so windows.h does not leak into the public header.
    static void* ensureOrtDllLoaded() noexcept;
#endif

    // ORT handles
    std::unique_ptr<OrtEnv, OrtEnvDeleter> ortEnv_;
    std::unique_ptr<OrtSession, OrtSessionDeleter> ortSession_;
    OrtMemoryInfo* ortMemoryInfo_{nullptr};

    // Pre-allocated graph inputs/state. Only the inference worker mutates these
    // during normal operation; the mutex protects non-RT control-path resets.
    std::vector<float> audioChunkBuffer_;
    std::vector<float> pastAudio_;
    std::vector<float> overlapAddBuffer_;
    std::vector<float> fusionHidden_;
    bool hasPastAudio_{false};
    std::mutex streamingStateMutex_;

    // State
    bool ortInitialized_{false};
    bool modelLoaded_{false};
    std::string runtimeVersion_;
    std::string executionProvider_;
    juce::String modelLoadError_;
};

}  // namespace audio_plugin
