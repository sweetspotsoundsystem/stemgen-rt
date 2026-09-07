#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>
#include "Constants.h"
#include "StreamingSampleRateAdapter.h"
#include "WorkerTimingTrace.h"

namespace audio_plugin {

// Forward declaration
class OnnxRuntime;

// A single inference request with input/output buffers and one atomic ownership
// state. The epoch is packed into the same control word as the state so stale
// cleanup can never race a newer publication and clear the newer request.
struct InferenceRequest {
  enum class SlotState : uint8_t {
    Empty,
    Writing,
    Ready,
    Processing,
    Processed,
    Reading,
  };

  // Input data (filled by audio thread)
  std::array<std::vector<float>, kNumChannels> inputChunk;

  // Output data (filled by inference thread)
  std::array<std::array<std::vector<float>, kNumChannels>, kNumStems>
      outputChunk;
  // At host rates above 44.1 kHz, the inference worker converts the three
  // retained sources onto the absolute host clock before publication. Other
  // remains a host-domain residual and is never sample-rate converted.
  std::array<std::array<std::vector<float>, kNumChannels>, kNumStems>
      hostOutputChunk;
  std::array<std::vector<float>, kNumChannels> alignedInput;
  // True only when the graph result aligned to the preceding input sequence is
  // safe to publish. The first successful run after reset is invalid pre-roll.
  bool outputValid{false};
  // Successful invalid pre-roll is distinct from an inference failure.
  bool inferenceSucceeded{false};
  bool hostOutputValid{false};
  uint64_t hostOutputStartSample{0};
  size_t hostOutputSampleCount{0};

  // Monotonic input chunk index assigned by the audio thread. The worker uses
  // it for recurrent-state gap detection; the scheduler subtracts the graph's
  // one-hop delay to recover the exact previous-hop output timestamp.
  uint64_t chunkSequence{0};

  // Allocate buffers to expected sizes
  void allocate(size_t hostOutputCapacity = 0U);

  SlotState getState() const;
  uint32_t getEpoch() const;
  bool isProcessed() const { return getState() == SlotState::Processed; }

private:
  friend class InferenceQueue;

  static constexpr uint64_t makeControl(uint32_t epoch, SlotState state) {
    return (static_cast<uint64_t>(epoch) << 32U) | static_cast<uint64_t>(state);
  }
  static constexpr SlotState stateFromControl(uint64_t control) {
    return static_cast<SlotState>(control & 0xffU);
  }
  static constexpr uint32_t epochFromControl(uint64_t control) {
    return static_cast<uint32_t>(control >> 32U);
  }

  std::atomic<uint64_t> control_{makeControl(0, SlotState::Empty)};
};

static_assert(
    std::atomic<uint64_t>::is_always_lock_free,
    "Inference queue control must remain lock-free on the audio thread");

// Lock-free producer-consumer queue for inference requests. The audio-thread
// API publishes and resets with atomics only; it never notifies a condition
// variable or enters an OS wait/wake primitive. The worker performs its own
// bounded polling backoff when no request is ready.
// Audio thread produces requests, inference thread consumes them.
// Uses epoch tracking to handle resets without cross-thread buffer clearing.
class InferenceQueue {
public:
  enum class WorkerPriorityStatus : uint8_t {
    NotAttempted,
    Applied,
    Failed,
    Unsupported,
  };

  static_assert(
      std::atomic<WorkerPriorityStatus>::is_always_lock_free,
      "Inference worker priority status must remain lock-free to observe");

  InferenceQueue();
  ~InferenceQueue();

  // Non-copyable
  InferenceQueue(const InferenceQueue&) = delete;
  InferenceQueue& operator=(const InferenceQueue&) = delete;

  // Allocate all request buffers
  void allocate();

  // Configure worker-side conversion of retained model stems. Call only while
  // the worker is stopped. The optional fractional host-sample delay is used
  // to make the paired input/output FIR delay exactly integral for PDC.
  bool prepareOutputSampleRate(double hostSampleRate,
                               double additionalHostDelaySamples = 0.0);
  void disableOutputSampleRateConversion();
  bool isOutputSampleRateConversionEnabled() const noexcept {
    return outputSampleRateConversionEnabled_;
  }
  double getOutputSampleRateConversionDelaySeconds() const noexcept {
    return outputSampleRateAdapter_.filterGroupDelaySeconds();
  }
  double getOutputSampleRateConversionDelaySamples() const noexcept {
    return outputSampleRateAdapter_.filterGroupDelayOutputSamples();
  }
  size_t getMaximumHostOutputSamplesPerHop() const noexcept {
    return outputSampleRateConversionEnabled_
               ? hostOutputCapacity_
               : static_cast<size_t>(kOutputChunkSize);
  }

  // Start/stop the background inference thread
  bool startThread(OnnxRuntime* runtime);
  void stopThread();
  bool isThreadRunning() const {
    return threadRunning_.load(std::memory_order_acquire);
  }
  WorkerPriorityStatus getWorkerPriorityStatus() const noexcept {
    return workerPriorityStatus_.load(std::memory_order_acquire);
  }

  // Optional diagnostic. Configure only from the lifecycle thread while the
  // worker is stopped; concurrent start/stop/configuration is unsupported.
  bool setWorkerTimingTrace(WorkerTimingTrace* trace) noexcept;

  // Check if a write slot is available (called from audio thread)
  // Returns pointer to the request if available, nullptr if queue is full
  InferenceRequest* getWriteSlot();

  // Submit the current write slot for processing (called from audio thread)
  // Must call getWriteSlot() first and fill in the data
  void submitWriteSlot(uint32_t epoch);

  // Non-real-time compatibility/test helper. Production callbacks consume
  // only results already published at their boundary and never call this.
  // The helper performs lock-free state reads and cooperative yields; it does
  // not enter an OS wait/wake primitive. A false result leaves the request and
  // recurrent stream intact so exact-timeline consumption can discard a late
  // completion instead of shifting it.
  bool waitUntilProcessed(
      const InferenceRequest* request,
      uint32_t epoch,
      std::chrono::steady_clock::time_point deadline) const noexcept;

  // Submit a warmup request without advancing write index
  // Used during prepareToPlay for ORT lazy initialization
  void submitForWarmup();

  // Release the warmup slot without advancing the real-time consume index.
  // Only used by prepareToPlay after isProcessed() becomes true.
  void releaseWarmupSlot(InferenceRequest* request);

  // Get the next processed output slot (called from audio thread)
  // Returns nullptr if no output is ready or if the slot is stale
  // (auto-discarded)
  InferenceRequest* getOutputSlot(uint32_t currentEpoch);

  // Get the current output slot without validation (called from audio thread)
  // Use this when you already validated with getOutputSlot and need to access
  // the same slot again (e.g., during amortized chunk copying)
  InferenceRequest* getCurrentOutputSlot();

  // Release the output slot after consuming (called from audio thread)
  void releaseOutputSlot();

  // RT-safe reset: increment epoch to invalidate in-flight requests. This is
  // an audio-thread operation and must remain serialized with output copying.
  // Returns the new epoch value.
  uint32_t reset();

  // Full reset: clears all slot ownership and resets indices
  // NOT RT-safe (should only be called from non-audio thread)
  void fullReset();

  // Get current epoch
  uint32_t getEpoch() const;

private:
  struct WorkerCallbacks {
    void* context{nullptr};
    bool (*run)(void*, InferenceRequest&){nullptr};
    void (*reset)(void*){nullptr};
  };

  friend class InferenceQueueTestPeer;
  friend class AudioPluginProcessorTestPeer;

  static constexpr uint64_t makeEpochControl(uint32_t epoch,
                                             size_t startIndex) {
    return (static_cast<uint64_t>(epoch) << 32U) |
           static_cast<uint64_t>(startIndex);
  }
  static constexpr uint32_t epochFromControl(uint64_t control) {
    return static_cast<uint32_t>(control >> 32U);
  }
  static constexpr size_t startIndexFromControl(uint64_t control) {
    return static_cast<size_t>(static_cast<uint32_t>(control));
  }

  bool startThreadWithCallbacks(WorkerCallbacks callbacks);
  void inferenceThreadFunc(WorkerCallbacks callbacks);
  bool reclaimStaleSlots(uint64_t observedEpochControl);
  bool convertOutputToHost(InferenceRequest& request) noexcept;
  bool resetOutputConversionAtModelSample(uint64_t modelSample) noexcept;

  std::array<std::unique_ptr<InferenceRequest>, kNumInferenceBuffers> queue_;
  StreamingSampleRateAdapter outputSampleRateAdapter_;
  size_t hostOutputCapacity_{0};
  bool outputSampleRateConversionEnabled_{false};

  // Indices (atomics for lock-free operation)
  std::atomic<size_t> writeIdx_{0};  // Next slot for audio thread to write
  std::atomic<size_t> readIdx_{0};  // Next slot for inference thread to process
  std::atomic<size_t> consumeIdx_{
      0};  // Next slot for audio thread to consume output

  // Epoch and its starting ring index are published as one atomic snapshot.
  // This prevents the worker from observing a new epoch with an old start
  // index (or vice versa) during an audio-thread reset.
  std::atomic<uint64_t> epochControl_{makeEpochControl(0, 0)};

  // Thread management
  std::unique_ptr<std::thread> thread_;
  std::atomic<bool> shouldStop_{false};
  std::atomic<bool> threadRunning_{false};
  std::atomic<WorkerPriorityStatus> workerPriorityStatus_{
      WorkerPriorityStatus::NotAttempted};
  WorkerTimingTrace* workerTimingTrace_{nullptr};
};

}  // namespace audio_plugin
