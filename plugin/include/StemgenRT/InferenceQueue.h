#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "Constants.h"

namespace audio_plugin {

// Forward declaration
class OnnxRuntime;

// A single inference request with input/output buffers and atomic state flags.
struct InferenceRequest {
    // Input data (filled by audio thread)
    std::array<std::vector<float>, kNumChannels> inputChunk;     // kOutputChunkSize HP-filtered samples
    std::array<std::vector<float>, kNumChannels> contextSnapshot; // kContextSize HP-filtered samples
    std::array<std::vector<float>, kNumChannels> originalInput;   // kOutputChunkSize fullband samples
    std::array<std::vector<float>, kNumChannels> lowFreqChunk;    // kOutputChunkSize LP-filtered samples
    float normalizationGain{1.0f};

    // Output data (filled by inference thread)
    std::array<std::array<std::vector<float>, kNumChannels>, kNumStems> outputChunk;

    // State flags
    std::atomic<bool> ready{false};      // True when input data is ready for inference
    std::atomic<bool> processed{false};  // True when inference is complete

    // Epoch when request was created (for stale detection after reset)
    uint32_t epoch{0};

    // Allocate buffers to expected sizes
    void allocate();
};

// Lock-free producer-consumer queue for inference requests.
// Audio thread produces requests, inference thread consumes them.
// Uses epoch tracking to handle resets without cross-thread buffer clearing.
class InferenceQueue {
public:
    InferenceQueue();
    ~InferenceQueue();

    // Non-copyable
    InferenceQueue(const InferenceQueue&) = delete;
    InferenceQueue& operator=(const InferenceQueue&) = delete;

    // Allocate all request buffers
    void allocate();

    // Start/stop the background inference thread
    void startThread(OnnxRuntime* runtime);
    void stopThread();

    // Check if a write slot is available (called from audio thread)
    // Returns pointer to the request if available, nullptr if queue is full
    InferenceRequest* getWriteSlot();

    // Submit the current write slot for processing (called from audio thread)
    // Must call getWriteSlot() first and fill in the data
    void submitWriteSlot(uint32_t epoch);

    // Submit a warmup request without advancing write index
    // Used during prepareToPlay for ORT lazy initialization
    void submitForWarmup();

    // Get the next processed output slot (called from audio thread)
    // Returns nullptr if no output is ready or if the slot is stale (auto-discarded)
    InferenceRequest* getOutputSlot(uint32_t currentEpoch);

    // Get the current output slot without validation (called from audio thread)
    // Use this when you already validated with getOutputSlot and need to access
    // the same slot again (e.g., during amortized chunk copying)
    InferenceRequest* getCurrentOutputSlot();

    // Release the output slot after consuming (called from audio thread)
    void releaseOutputSlot();

    // RT-safe reset: increment epoch to invalidate in-flight requests
    // Returns the new epoch value
    uint32_t reset();

    // Full reset: clears all slot flags and resets indices
    // NOT RT-safe (should only be called from non-audio thread)
    void fullReset();

    // Get current epoch
    uint32_t getEpoch() const { return resetEpoch_.load(std::memory_order_acquire); }

    // Notify the inference thread that work is available
    void notifyThread();

private:
    void inferenceThreadFunc(OnnxRuntime* runtime);

    std::array<std::unique_ptr<InferenceRequest>, kNumInferenceBuffers> queue_;

    // Indices (atomics for lock-free operation)
    std::atomic<size_t> writeIdx_{0};      // Next slot for audio thread to write
    std::atomic<size_t> readIdx_{0};       // Next slot for inference thread to process
    std::atomic<size_t> consumeIdx_{0};    // Next slot for audio thread to consume output

    // Epoch counter for stale detection
    std::atomic<uint32_t> resetEpoch_{0};

    // Thread management
    std::unique_ptr<std::thread> thread_;
    std::atomic<bool> shouldStop_{false};
    std::mutex mutex_;
    std::condition_variable cv_;
};

}  // namespace audio_plugin
