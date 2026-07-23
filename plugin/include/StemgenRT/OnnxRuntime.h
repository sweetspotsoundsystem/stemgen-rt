#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include <array>
#include <juce_core/juce_core.h>
#include "Constants.h"

// Forward declarations to avoid exposing ORT headers
struct OrtEnv;
struct OrtSession;
struct OrtMemoryInfo;
struct OrtValue;
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
  bool isInitialized() const noexcept {
    return ortInitialized_.load(std::memory_order_acquire);
  }

  // Load a model from file path. An omitted intra-op override preserves the
  // qualified production heuristic; explicit values are intended for
  // controlled CPU qualification and must be positive. ORT thread-pool size
  // is immutable after session creation, so each benchmark candidate needs a
  // fresh runtime/session.
  // Returns true on success, sets errorMessage on failure.
  bool loadModel(const juce::String& modelPath,
                 juce::String& errorMessage,
                 std::optional<int> intraOpThreadCount = std::nullopt);

  // Check if model is loaded and ready for inference
  bool isModelLoaded() const noexcept {
    return modelLoaded_.load(std::memory_order_acquire);
  }

  // Get the active execution provider name (the qualified build uses "CPU").
  std::string getExecutionProvider() const;

  // Get the ORT runtime version string
  std::string getRuntimeVersion() const;

  // Prepare fixed scratch/state buffers and tensor bindings. Model loading and
  // streaming readiness are separate fail-closed states; runInference remains
  // unavailable until this succeeds.
  bool prepareForInference(juce::String& errorMessage);
  bool prepareForInference() {
    juce::String ignoredError;
    return prepareForInference(ignoredError);
  }

  bool isReadyForInference() const noexcept {
    return inferenceReady_.load(std::memory_order_acquire);
  }

  // Reset every persistent model state to zero. The inference queue calls this
  // from its worker thread on transport epochs and sequence gaps. It is also
  // safe to call from a stopped/non-real-time control path.
  void resetStreamingState();

  // Run one stateful graph hop. The returned samples and alignedInput belong
  // to the current input hop. Every successful run is valid; there is no
  // pre-roll output or zero-hop flush in the c126 boundary contract.
  bool runInference(
      const std::array<std::vector<float>, kNumChannels>& inputChunk,
      std::array<std::array<std::vector<float>, kNumChannels>, kNumStems>&
          outputChunks,
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
  bool createPreallocatedTensorValues(juce::String& errorMessage);
  void releasePreallocatedTensorValues() noexcept;
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

  // Pre-allocated graph inputs/state. Audio enters the graph at its native
  // floating-point level: the c126 quality evaluation used this exact path,
  // and hop-varying deployment gain was measured to damage bass and drum
  // quality. Only the inference worker mutates these during normal operation;
  // the mutex protects non-RT control-path resets.
  std::vector<float> audioChunkBuffer_;
  std::vector<float> pastAudio_;
  std::vector<float> fusionHidden_;
  std::vector<float> separatedOutputBuffer_;
  std::vector<float> nextPastAudioBuffer_;
  std::vector<float> nextFusionHiddenBuffer_;
  std::array<OrtValue*, 3> inputTensorValues_{};
  std::array<OrtValue*, 3> outputTensorValues_{};
  std::mutex streamingStateMutex_;

  // State
  std::atomic<bool> ortInitialized_{false};
  std::atomic<bool> modelLoaded_{false};
  std::atomic<bool> inferenceReady_{false};
  mutable std::mutex statusMutex_;
  std::string runtimeVersion_;
  std::string executionProvider_;
  juce::String modelLoadError_;
  juce::String inferencePreparationError_;
};

}  // namespace audio_plugin
