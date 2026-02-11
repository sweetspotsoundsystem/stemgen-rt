#pragma once

#include <memory>
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

namespace audio_plugin {

// RAII wrapper for ONNX Runtime environment, session, and inference.
// Handles initialization, GPU provider selection, model loading, and inference execution.
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

    // Check if GPU acceleration is active
    bool isUsingGPU() const { return usingGPU_; }

    // Get the active execution provider name ("TensorRT", "CUDA", or "CPU")
    const std::string& getExecutionProvider() const { return executionProvider_; }

    // Get the ORT runtime version string
    const std::string& getRuntimeVersion() const { return runtimeVersion_; }

    // Prepare for inference (allocate scratch buffer, memory info)
    // Must be called before runInference
    void prepareForInference();

    // Run inference on a prepared input chunk
    // inputChunk: [channel][kInternalChunkSize] - padded input
    // outputChunks: [stem][channel] - vectors to receive kOutputChunkSize samples each
    // normalizationGain: gain that was applied to input (inverse is applied to output)
    // lowFreqChunk: [channel][kOutputChunkSize] - LP component carried with the request
    // (reinjected on the audio thread after boundary crossfade)
    //
    // Returns true on success
    bool runInference(
        const std::array<std::vector<float>, kNumChannels>& contextSnapshot,
        const std::array<std::vector<float>, kNumChannels>& inputChunk,
        const std::array<std::vector<float>, kNumChannels>& lowFreqChunk,
        float normalizationGain,
        std::array<std::array<std::vector<float>, kNumChannels>, kNumStems>& outputChunks,
        std::array<std::array<std::vector<float>, kNumChannels>, kNumStems>& overlapTail);

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
    static const OrtApi* getSafeOrtApi() noexcept;

#ifdef _WIN32
    // Windows-specific DLL loading
    static void ensureOrtDllLoaded() noexcept;
#endif

    // ORT handles
    std::unique_ptr<OrtEnv, OrtEnvDeleter> ortEnv_;
    std::unique_ptr<OrtSession, OrtSessionDeleter> ortSession_;
    OrtMemoryInfo* ortMemoryInfo_{nullptr};

    // Pre-allocated scratch buffer for inference input
    std::vector<float> scratchBuffer_;

    // State
    bool ortInitialized_{false};
    bool modelLoaded_{false};
    bool usingGPU_{false};
    std::string runtimeVersion_;
    std::string executionProvider_;
    juce::String modelLoadError_;
};

}  // namespace audio_plugin
