#include "StemgenRT/InferenceQueue.h"
#include "StemgenRT/OnnxRuntime.h"
#include <juce_core/juce_core.h>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace audio_plugin {

void InferenceRequest::allocate() {
    for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
        inputChunk[ch].resize(static_cast<size_t>(kOutputChunkSize), 0.0f);
        contextSnapshot[ch].resize(static_cast<size_t>(kContextSize), 0.0f);
        originalInput[ch].resize(static_cast<size_t>(kOutputChunkSize), 0.0f);
        lowFreqChunk[ch].resize(static_cast<size_t>(kOutputChunkSize), 0.0f);
    }
    for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
        for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
            outputChunk[stem][ch].resize(static_cast<size_t>(kOutputChunkSize), 0.0f);
            overlapTail[stem][ch].resize(static_cast<size_t>(kCrossfadeSamples), 0.0f);
        }
    }
}

InferenceQueue::InferenceQueue() {
    for (auto& slot : queue_) {
        slot = std::make_unique<InferenceRequest>();
    }
}

InferenceQueue::~InferenceQueue() {
    stopThread();
}

void InferenceQueue::allocate() {
    for (auto& slot : queue_) {
        if (slot) {
            slot->allocate();
        }
    }
}

void InferenceQueue::startThread(OnnxRuntime* runtime) {
    if (thread_ && thread_->joinable()) {
        return;  // Already running
    }

    shouldStop_.store(false, std::memory_order_release);
    thread_ = std::make_unique<std::thread>(&InferenceQueue::inferenceThreadFunc, this, runtime);

    // Set thread priority (platform-specific)
#if defined(__APPLE__)
    pthread_t nativeHandle = thread_->native_handle();
    struct sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_FIFO) - 1;
    pthread_setschedparam(nativeHandle, SCHED_FIFO, &param);
#elif defined(_WIN32)
    SetThreadPriority(thread_->native_handle(), THREAD_PRIORITY_HIGHEST);
#elif defined(__linux__)
    pthread_t nativeHandle = thread_->native_handle();
    struct sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_RR) / 2;
    pthread_setschedparam(nativeHandle, SCHED_RR, &param);
#endif
}

void InferenceQueue::stopThread() {
    shouldStop_.store(true, std::memory_order_release);
    cv_.notify_all();

    if (thread_ && thread_->joinable()) {
        thread_->join();
    }
    thread_.reset();
}

InferenceRequest* InferenceQueue::getWriteSlot() {
    size_t idx = writeIdx_.load(std::memory_order_acquire);
    auto& slot = queue_[idx];

    if (!slot) {
        return nullptr;
    }

    // Reclaim stale processed slots from previous epochs when safe.
    // Do not touch stale ready slots here: they may still be in-flight on inference thread.
    const uint32_t currentEpoch = resetEpoch_.load(std::memory_order_acquire);
    if (slot->epoch != currentEpoch &&
        !slot->ready.load(std::memory_order_acquire)) {
        slot->processed.store(false, std::memory_order_release);
    }

    if (slot->ready.load(std::memory_order_acquire) ||
        slot->processed.load(std::memory_order_acquire)) {
        return nullptr;  // Slot not available
    }

    return slot.get();
}

void InferenceQueue::submitWriteSlot(uint32_t epoch) {
    size_t idx = writeIdx_.load(std::memory_order_acquire);
    auto& slot = queue_[idx];

    if (slot) {
        slot->epoch = epoch;
        slot->ready.store(true, std::memory_order_release);

        // Advance write index
        writeIdx_.store((idx + 1) % kNumInferenceBuffers, std::memory_order_release);

        // Notify inference thread
        cv_.notify_one();
    }
}

void InferenceQueue::submitForWarmup() {
    // For warmup, submit slot 0 without advancing indices
    // This ensures indices remain at 0 after warmup completes
    // Use current epoch so the request isn't discarded as stale
    auto& slot = queue_[0];
    if (slot) {
        slot->epoch = resetEpoch_.load(std::memory_order_acquire);
        slot->ready.store(true, std::memory_order_release);
        cv_.notify_one();
    }
}

InferenceRequest* InferenceQueue::getOutputSlot(uint32_t currentEpoch) {
    while (true) {
        size_t idx = consumeIdx_.load(std::memory_order_acquire);
        auto& slot = queue_[idx];

        if (!slot || !slot->processed.load(std::memory_order_acquire)) {
            return nullptr;  // No output ready
        }

        // Skip stale results from previous epochs and continue scanning.
        if (slot->epoch != currentEpoch) {
            slot->processed.store(false, std::memory_order_release);
            consumeIdx_.store((idx + 1) % kNumInferenceBuffers,
                              std::memory_order_release);
            continue;
        }

        return slot.get();
    }
}

InferenceRequest* InferenceQueue::getCurrentOutputSlot() {
    size_t idx = consumeIdx_.load(std::memory_order_acquire);
    return queue_[idx].get();
}

void InferenceQueue::releaseOutputSlot() {
    size_t idx = consumeIdx_.load(std::memory_order_acquire);
    auto& slot = queue_[idx];

    if (slot) {
        slot->processed.store(false, std::memory_order_release);
        consumeIdx_.store((idx + 1) % kNumInferenceBuffers, std::memory_order_release);
    }
}

uint32_t InferenceQueue::reset() {
    // Capture where new-epoch writes will begin.
    const size_t startIdx = writeIdx_.load(std::memory_order_acquire);
    epochStartIdx_.store(startIdx, std::memory_order_release);

    // Skip old-epoch output consumption immediately.
    consumeIdx_.store(startIdx, std::memory_order_release);

    return resetEpoch_.fetch_add(1, std::memory_order_acq_rel) + 1;
}

void InferenceQueue::fullReset() {
    // Clear all slot flags
    for (auto& slot : queue_) {
        if (slot) {
            slot->ready.store(false, std::memory_order_release);
            slot->processed.store(false, std::memory_order_release);
        }
    }

    // Reset all indices
    writeIdx_.store(0, std::memory_order_release);
    readIdx_.store(0, std::memory_order_release);
    consumeIdx_.store(0, std::memory_order_release);
    epochStartIdx_.store(0, std::memory_order_release);
}

void InferenceQueue::notifyThread() {
    cv_.notify_one();
}

void InferenceQueue::inferenceThreadFunc(OnnxRuntime* runtime) {
    DBG("[InferenceQueue] Thread running");

    uint32_t lastSeenEpoch = resetEpoch_.load(std::memory_order_acquire);

    while (!shouldStop_.load(std::memory_order_acquire)) {
        // Check for epoch change (reset occurred)
        uint32_t currentEpoch = resetEpoch_.load(std::memory_order_acquire);
        if (currentEpoch != lastSeenEpoch) {
            // Jump to the slot where new-epoch writes started to avoid
            // waiting on old non-ready holes.
            const size_t startIdx =
                epochStartIdx_.load(std::memory_order_acquire);
            readIdx_.store(startIdx, std::memory_order_release);

            // Clear stale flags from previous epochs so they can't block ring reuse.
            for (auto& slot : queue_) {
                if (!slot) {
                    continue;
                }
                if (slot->epoch != currentEpoch) {
                    slot->ready.store(false, std::memory_order_release);
                    slot->processed.store(false, std::memory_order_release);
                }
            }
            lastSeenEpoch = currentEpoch;
        }

        // Wait for work
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_for(lock, std::chrono::milliseconds(5), [this] {
                size_t idx = readIdx_.load(std::memory_order_acquire);
                return shouldStop_.load(std::memory_order_acquire) ||
                       (queue_[idx] && queue_[idx]->ready.load(std::memory_order_acquire));
            });
        }

        if (shouldStop_.load(std::memory_order_acquire)) break;

        // Process all ready requests
        while (true) {
            size_t idx = readIdx_.load(std::memory_order_acquire);
            auto& request = queue_[idx];

            if (!request || !request->ready.load(std::memory_order_acquire)) {
                break;
            }

            uint32_t requestEpoch = request->epoch;
            currentEpoch = resetEpoch_.load(std::memory_order_acquire);

            if (requestEpoch != currentEpoch) {
                // Stale request - discard
                request->ready.store(false, std::memory_order_release);
                readIdx_.store((idx + 1) % kNumInferenceBuffers, std::memory_order_release);
                // Keep lastSeenEpoch unchanged so outer loop executes epoch jump logic.
                continue;
            }

            // Run inference
            bool inferenceOk = false;
            if (runtime) {
                inferenceOk = runtime->runInference(
                    request->contextSnapshot,
                    request->inputChunk,
                    request->lowFreqChunk,
                    request->normalizationGain,
                    request->outputChunk,
                    request->overlapTail);
            }

            // Treat inference errors as dropped chunks (fallback path will crossfade to dry).
            if (!inferenceOk) {
                request->ready.store(false, std::memory_order_release);
                request->processed.store(false, std::memory_order_release);
                readIdx_.store((idx + 1) % kNumInferenceBuffers,
                               std::memory_order_release);
                continue;
            }

            // Check if reset occurred during inference
            currentEpoch = resetEpoch_.load(std::memory_order_acquire);
            if (requestEpoch != currentEpoch) {
                request->ready.store(false, std::memory_order_release);
                request->processed.store(false, std::memory_order_release);
                readIdx_.store((idx + 1) % kNumInferenceBuffers, std::memory_order_release);
                // Keep lastSeenEpoch unchanged so outer loop executes epoch jump logic.
                continue;
            }

            // Mark as processed
            request->ready.store(false, std::memory_order_release);
            request->processed.store(true, std::memory_order_release);
            readIdx_.store((idx + 1) % kNumInferenceBuffers, std::memory_order_release);
        }
    }

    DBG("[InferenceQueue] Thread exiting");
}

}  // namespace audio_plugin
