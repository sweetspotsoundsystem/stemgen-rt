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
// Execution has one owner: load/prepare/destruction require a stopped worker;
// run/reset belong exclusively to the worker while it is running. The audio
// callback invalidates queue epochs and never calls this mutable API.
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
  // single-thread production policy; explicit values are intended for
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

  // Zero until a model has successfully loaded; includes the calling worker.
  int getConfiguredIntraOpThreadCount() const noexcept {
    return configuredIntraOpThreads_.load(std::memory_order_acquire);
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
  // safe to call from a non-real-time control path after the worker is joined.
  void resetStreamingState();

  // Run one stateful graph hop. The returned samples and alignedInput belong
  // to the previous input hop. The first successful run after reset is
  // pre-roll (outputValid=false); one final zero hop flushes the last input.
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
  void clearStreamingState();

#ifdef _WIN32
  // Load the exact DLL beside this module and return its native handle.
  // Kept opaque here so windows.h does not leak into the public header.
  static void* ensureOrtDllLoaded() noexcept;
#endif

  // ORT handles
  const OrtApi* api_{nullptr};  // Resolved once, before worker startup.
  std::unique_ptr<OrtEnv, OrtEnvDeleter> ortEnv_;
  std::unique_ptr<OrtSession, OrtSessionDeleter> ortSession_;
  OrtMemoryInfo* ortMemoryInfo_{nullptr};

  // Pre-allocated graph inputs/state. Audio enters the graph at its exact
  // native floating-point level. previousAlignedInput_ remains in that raw
  // domain so Main/residual alignment never depends on provider copies of
  // recurrent state. Only the inference worker mutates these during normal
  // operation; control paths must join the worker before touching them.
  std::vector<float> audioChunkBuffer_;
  std::vector<float> audioHistory_;
  std::vector<float> spectralNumeratorTail_;
  std::vector<float> waveformTail_;
  std::vector<float> fusionHidden_;
  std::vector<float> previousAlignedInput_;
  std::vector<float> separatedOutputBuffer_;
  std::vector<float> nextAudioHistoryBuffer_;
  std::vector<float> nextSpectralNumeratorTail_;
  std::vector<float> nextWaveformTail_;
  std::vector<float> nextFusionHiddenBuffer_;
  std::array<OrtValue*, 5> inputTensorValues_{};
  std::array<OrtValue*, 5> outputTensorValues_{};
  bool hasPreviousAlignedInput_{false};

  // State
  std::atomic<bool> ortInitialized_{false};
  std::atomic<bool> modelLoaded_{false};
  std::atomic<bool> inferenceReady_{false};
  std::atomic<int> configuredIntraOpThreads_{0};
  mutable std::mutex statusMutex_;
  std::string runtimeVersion_;
  std::string executionProvider_;
  juce::String modelLoadError_;
  juce::String inferencePreparationError_;
};

}  // namespace audio_plugin
