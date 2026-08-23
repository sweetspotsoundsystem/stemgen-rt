#include "StemgenRT/InferenceQueue.h"
#include "StemgenRT/OnnxRuntime.h"

#include <chrono>
#include <cmath>
#include <limits>

#include <juce_core/juce_core.h>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__) || defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace audio_plugin {

namespace {

InferenceQueue::WorkerPriorityStatus configureCurrentThreadPriority() noexcept {
#if defined(__APPLE__)
  const int result =
      pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  if (result != 0) {
    return InferenceQueue::WorkerPriorityStatus::Failed;
  }
  qos_class_t observedClass{};
  int relativePriority = 0;
  const int getResult = pthread_get_qos_class_np(
      pthread_self(), &observedClass, &relativePriority);
  return getResult == 0 && observedClass == QOS_CLASS_USER_INTERACTIVE
             ? InferenceQueue::WorkerPriorityStatus::Applied
             : InferenceQueue::WorkerPriorityStatus::Failed;
#elif defined(_WIN32)
  const BOOL result =
      SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
  return result != 0 ? InferenceQueue::WorkerPriorityStatus::Applied
                     : InferenceQueue::WorkerPriorityStatus::Failed;
#elif defined(__linux__)
  const int maxPriority = sched_get_priority_max(SCHED_RR);
  if (maxPriority < 1) {
    return InferenceQueue::WorkerPriorityStatus::Failed;
  }

  sched_param param{};
  param.sched_priority = maxPriority > 1 ? maxPriority / 2 : maxPriority;
  const int result = pthread_setschedparam(pthread_self(), SCHED_RR, &param);
  return result == 0 ? InferenceQueue::WorkerPriorityStatus::Applied
                     : InferenceQueue::WorkerPriorityStatus::Failed;
#else
  return InferenceQueue::WorkerPriorityStatus::Unsupported;
#endif
}

}  // namespace

void InferenceRequest::allocate(size_t hostOutputCapacity) {
  for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
    inputChunk[ch].resize(static_cast<size_t>(kOutputChunkSize), 0.0f);
    alignedInput[ch].resize(static_cast<size_t>(kOutputChunkSize), 0.0f);
  }
  for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
    for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
      outputChunk[stem][ch].resize(static_cast<size_t>(kOutputChunkSize), 0.0f);
      hostOutputChunk[stem][ch].resize(hostOutputCapacity, 0.0f);
    }
  }
  outputValid = false;
  hostOutputValid = false;
  hostOutputStartSample = 0U;
  hostOutputSampleCount = 0U;
  control_.store(makeControl(0, SlotState::Empty), std::memory_order_release);
}

InferenceRequest::SlotState InferenceRequest::getState() const {
  return stateFromControl(control_.load(std::memory_order_acquire));
}

uint32_t InferenceRequest::getEpoch() const {
  return epochFromControl(control_.load(std::memory_order_acquire));
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
      slot->allocate(hostOutputCapacity_);
    }
  }
}

bool InferenceQueue::prepareOutputSampleRate(
    double hostSampleRate,
    double additionalHostDelaySamples) {
  if (isThreadRunning() ||
      !outputSampleRateAdapter_.prepare(
          static_cast<double>(kModelSampleRate), hostSampleRate,
          static_cast<size_t>(3 * kNumChannels), additionalHostDelaySamples)) {
    outputSampleRateConversionEnabled_ = false;
    hostOutputCapacity_ = 0U;
    return false;
  }

  const size_t nominalCapacity = outputSampleRateAdapter_.maxOutputForInput(
      static_cast<size_t>(kOutputChunkSize));
  if (nominalCapacity == std::numeric_limits<size_t>::max() ||
      nominalCapacity == std::numeric_limits<size_t>::max() - 1U) {
    outputSampleRateConversionEnabled_ = false;
    hostOutputCapacity_ = 0U;
    return false;
  }
  // A different absolute rational phase can add one output sample to a hop.
  hostOutputCapacity_ = nominalCapacity + 1U;
  outputSampleRateConversionEnabled_ = true;
  return true;
}

void InferenceQueue::disableOutputSampleRateConversion() {
  if (isThreadRunning()) {
    return;
  }
  outputSampleRateConversionEnabled_ = false;
  hostOutputCapacity_ = 0U;
}

bool InferenceQueue::resetOutputConversionAtModelSample(
    uint64_t modelSample) noexcept {
  return !outputSampleRateConversionEnabled_ ||
         outputSampleRateAdapter_.resetAtInputSample(modelSample);
}

bool InferenceQueue::convertOutputToHost(InferenceRequest& request) noexcept {
  request.hostOutputValid = false;
  request.hostOutputSampleCount = 0U;
  if (!outputSampleRateConversionEnabled_ || !request.outputValid) {
    return true;
  }

  constexpr uint64_t kChunkSize = static_cast<uint64_t>(kOutputChunkSize);
  if (request.chunkSequence <
      static_cast<uint64_t>(kModelOutputDelayChunks)) {
    return false;
  }
  const uint64_t alignedSequence =
      request.chunkSequence -
      static_cast<uint64_t>(kModelOutputDelayChunks);
  if (alignedSequence > std::numeric_limits<uint64_t>::max() / kChunkSize) {
    return false;
  }
  const uint64_t expectedModelStart = alignedSequence * kChunkSize;
  if (outputSampleRateAdapter_.nextInputSampleIndex() != expectedModelStart &&
      !outputSampleRateAdapter_.resetAtInputSample(expectedModelStart)) {
    return false;
  }

  const size_t outputCount = outputSampleRateAdapter_.maxOutputForInput(
      static_cast<size_t>(kOutputChunkSize));
  if (outputCount > hostOutputCapacity_) {
    return false;
  }

  std::array<const float*, 3 * kNumChannels> inputPointers{};
  std::array<float*, 3 * kNumChannels> outputPointers{};
  constexpr std::array<int, 3> kRetainedStems = {kStemDrums, kStemBass,
                                                 kStemVocals};
  for (size_t retained = 0U; retained < kRetainedStems.size(); ++retained) {
    const size_t stem = static_cast<size_t>(kRetainedStems[retained]);
    for (size_t channel = 0U; channel < static_cast<size_t>(kNumChannels);
         ++channel) {
      const size_t flattened =
          retained * static_cast<size_t>(kNumChannels) + channel;
      inputPointers[flattened] = request.outputChunk[stem][channel].data();
      outputPointers[flattened] = request.hostOutputChunk[stem][channel].data();
    }
  }

  request.hostOutputStartSample =
      outputSampleRateAdapter_.nextOutputSampleIndex();
  const auto conversion = outputSampleRateAdapter_.process(
      inputPointers.data(), static_cast<size_t>(kOutputChunkSize),
      outputPointers.data(), hostOutputCapacity_);
  if (!conversion.ok ||
      conversion.inputConsumed != static_cast<size_t>(kOutputChunkSize) ||
      conversion.outputProduced != outputCount) {
    request.hostOutputSampleCount = 0U;
    return false;
  }

  request.hostOutputSampleCount = conversion.outputProduced;
  request.hostOutputValid = true;
  return true;
}

void InferenceQueue::startThread(OnnxRuntime* runtime) {
  WorkerCallbacks callbacks;
  callbacks.context = runtime;
  if (runtime != nullptr) {
    callbacks.run = [](void* context, InferenceRequest& request) {
      auto* onnxRuntime = static_cast<OnnxRuntime*>(context);
      return onnxRuntime->runInference(request.inputChunk, request.outputChunk,
                                       request.alignedInput,
                                       request.outputValid);
    };
    callbacks.reset = [](void* context) {
      static_cast<OnnxRuntime*>(context)->resetStreamingState();
    };
  }
  startThreadWithCallbacks(callbacks);
}

void InferenceQueue::startThreadWithCallbacks(WorkerCallbacks callbacks) {
  if (thread_ && thread_->joinable()) {
    return;  // Already running
  }

  shouldStop_.store(false, std::memory_order_release);
  workerPriorityStatus_.store(WorkerPriorityStatus::NotAttempted,
                              std::memory_order_release);
  thread_ = std::make_unique<std::thread>(&InferenceQueue::inferenceThreadFunc,
                                          this, callbacks);
  threadRunning_.store(true, std::memory_order_release);
}

void InferenceQueue::stopThread() {
  shouldStop_.store(true, std::memory_order_release);

  if (thread_ && thread_->joinable()) {
    thread_->join();
  }
  thread_.reset();
  threadRunning_.store(false, std::memory_order_release);
}

InferenceRequest* InferenceQueue::getWriteSlot() {
  const size_t idx = writeIdx_.load(std::memory_order_acquire);
  auto& slot = queue_[idx];

  if (!slot) {
    return nullptr;
  }

  const uint32_t currentEpoch = getEpoch();
  uint64_t observed = slot->control_.load(std::memory_order_acquire);
  while (true) {
    const auto state = InferenceRequest::stateFromControl(observed);
    const uint32_t slotEpoch = InferenceRequest::epochFromControl(observed);

    // Empty slots and stale queued/completed slots can be claimed in one
    // CAS. If the worker or consumer acquired the stale slot first, the
    // CAS fails and its new owner remains untouched.
    const bool claimable = state == InferenceRequest::SlotState::Empty ||
                           ((state == InferenceRequest::SlotState::Ready ||
                             state == InferenceRequest::SlotState::Processed) &&
                            slotEpoch != currentEpoch);
    if (!claimable) {
      return nullptr;
    }

    const uint64_t desired = InferenceRequest::makeControl(
        currentEpoch, InferenceRequest::SlotState::Writing);
    if (slot->control_.compare_exchange_weak(observed, desired,
                                             std::memory_order_acquire,
                                             std::memory_order_acquire)) {
      return slot.get();
    }
  }
}

void InferenceQueue::submitWriteSlot(uint32_t epoch) {
  const size_t idx = writeIdx_.load(std::memory_order_acquire);
  auto& slot = queue_[idx];

  if (slot) {
    uint64_t expected = InferenceRequest::makeControl(
        epoch, InferenceRequest::SlotState::Writing);

    // A reset cannot normally interleave here because reset and submit are
    // audio-thread operations. Fail closed if an API caller supplies an old
    // epoch instead of publishing a request with incoherent ownership.
    if (epoch != getEpoch()) {
      slot->control_.compare_exchange_strong(
          expected,
          InferenceRequest::makeControl(epoch,
                                        InferenceRequest::SlotState::Empty),
          std::memory_order_release, std::memory_order_relaxed);
      return;
    }

    const bool published = slot->control_.compare_exchange_strong(
        expected,
        InferenceRequest::makeControl(epoch,
                                      InferenceRequest::SlotState::Ready),
        std::memory_order_release, std::memory_order_relaxed);
    if (!published) {
      return;
    }

    // Advance write index
    writeIdx_.store((idx + 1) % kNumInferenceBuffers,
                    std::memory_order_release);
  }
}

bool InferenceQueue::waitUntilProcessed(
    const InferenceRequest* request,
    uint32_t epoch,
    std::chrono::steady_clock::time_point deadline) const noexcept {
  if (request == nullptr) {
    return false;
  }

  while (true) {
    if (getEpoch() != epoch || request->getEpoch() != epoch) {
      return false;
    }
    if (request->isProcessed()) {
      // Admission is based on when the audio thread can observe completion,
      // not merely when the worker may have published it. A callback that was
      // preempted past its deadline must fall back instead of scheduling a
      // result and returning late.
      return std::chrono::steady_clock::now() <= deadline;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::yield();
  }
}

void InferenceQueue::submitForWarmup() {
  // For warmup, publish the currently reserved write slot without advancing
  // the index. This leaves real-time indices unchanged after warmup/reset.
  const size_t idx = writeIdx_.load(std::memory_order_acquire);
  auto& slot = queue_[idx];
  if (slot) {
    const uint32_t epoch = getEpoch();
    uint64_t expected = InferenceRequest::makeControl(
        epoch, InferenceRequest::SlotState::Writing);
    static_cast<void>(slot->control_.compare_exchange_strong(
        expected,
        InferenceRequest::makeControl(epoch,
                                      InferenceRequest::SlotState::Ready),
        std::memory_order_release, std::memory_order_relaxed));
  }
}

void InferenceQueue::releaseWarmupSlot(InferenceRequest* request) {
  if (request == nullptr) {
    return;
  }

  uint64_t observed = request->control_.load(std::memory_order_acquire);
  while (InferenceRequest::stateFromControl(observed) ==
         InferenceRequest::SlotState::Processed) {
    const uint64_t desired = InferenceRequest::makeControl(
        InferenceRequest::epochFromControl(observed),
        InferenceRequest::SlotState::Empty);
    if (request->control_.compare_exchange_weak(observed, desired,
                                                std::memory_order_release,
                                                std::memory_order_acquire)) {
      return;
    }
  }
}

InferenceRequest* InferenceQueue::getOutputSlot(uint32_t currentEpoch) {
  if (currentEpoch != getEpoch()) {
    return nullptr;
  }

  while (true) {
    size_t idx = consumeIdx_.load(std::memory_order_acquire);
    auto& slot = queue_[idx];

    if (!slot) {
      return nullptr;  // No output ready
    }

    uint64_t observed = slot->control_.load(std::memory_order_acquire);
    if (InferenceRequest::stateFromControl(observed) !=
        InferenceRequest::SlotState::Processed) {
      return nullptr;
    }

    const uint32_t slotEpoch = InferenceRequest::epochFromControl(observed);
    if (slotEpoch != currentEpoch) {
      const uint64_t desired = InferenceRequest::makeControl(
          slotEpoch, InferenceRequest::SlotState::Empty);
      if (!slot->control_.compare_exchange_strong(observed, desired,
                                                  std::memory_order_release,
                                                  std::memory_order_acquire)) {
        continue;
      }
      consumeIdx_.store((idx + 1) % kNumInferenceBuffers,
                        std::memory_order_release);
      continue;
    }

    const uint64_t desired = InferenceRequest::makeControl(
        slotEpoch, InferenceRequest::SlotState::Reading);
    if (slot->control_.compare_exchange_strong(observed, desired,
                                               std::memory_order_acquire,
                                               std::memory_order_acquire)) {
      return slot.get();
    }
  }
}

InferenceRequest* InferenceQueue::getCurrentOutputSlot() {
  const size_t idx = consumeIdx_.load(std::memory_order_acquire);
  auto& slot = queue_[idx];
  if (slot && slot->getState() == InferenceRequest::SlotState::Reading) {
    return slot.get();
  }
  return nullptr;
}

void InferenceQueue::releaseOutputSlot() {
  size_t idx = consumeIdx_.load(std::memory_order_acquire);
  auto& slot = queue_[idx];

  if (slot) {
    uint64_t observed = slot->control_.load(std::memory_order_acquire);
    if (InferenceRequest::stateFromControl(observed) !=
        InferenceRequest::SlotState::Reading) {
      return;
    }
    const uint64_t desired = InferenceRequest::makeControl(
        InferenceRequest::epochFromControl(observed),
        InferenceRequest::SlotState::Empty);
    if (slot->control_.compare_exchange_strong(observed, desired,
                                               std::memory_order_release,
                                               std::memory_order_acquire)) {
      consumeIdx_.store((idx + 1) % kNumInferenceBuffers,
                        std::memory_order_release);
    }
  }
}

uint32_t InferenceQueue::reset() {
  // The audio thread owns any Reading slot. Relinquish it before abandoning
  // a partially copied output during a transport discontinuity.
  const size_t consumeIdx = consumeIdx_.load(std::memory_order_acquire);
  auto& consumedSlot = queue_[consumeIdx];
  if (consumedSlot) {
    uint64_t observed = consumedSlot->control_.load(std::memory_order_acquire);
    if (InferenceRequest::stateFromControl(observed) ==
        InferenceRequest::SlotState::Reading) {
      consumedSlot->control_.compare_exchange_strong(
          observed,
          InferenceRequest::makeControl(
              InferenceRequest::epochFromControl(observed),
              InferenceRequest::SlotState::Empty),
          std::memory_order_release, std::memory_order_relaxed);
    }
  }

  // A producer reservation cannot legitimately span reset(), since both are
  // audio-thread operations. Release one defensively so a cancelled write
  // cannot stall the new epoch.
  const size_t startIdx = writeIdx_.load(std::memory_order_acquire);
  auto& writeSlot = queue_[startIdx];
  if (writeSlot) {
    uint64_t observed = writeSlot->control_.load(std::memory_order_acquire);
    if (InferenceRequest::stateFromControl(observed) ==
        InferenceRequest::SlotState::Writing) {
      writeSlot->control_.compare_exchange_strong(
          observed,
          InferenceRequest::makeControl(
              InferenceRequest::epochFromControl(observed),
              InferenceRequest::SlotState::Empty),
          std::memory_order_release, std::memory_order_relaxed);
    }
  }

  // Skip old-epoch output consumption immediately.
  consumeIdx_.store(startIdx, std::memory_order_release);

  uint64_t observed = epochControl_.load(std::memory_order_acquire);
  uint32_t newEpoch = 0;
  while (true) {
    newEpoch = epochFromControl(observed) + 1U;
    const uint64_t desired = makeEpochControl(newEpoch, startIdx);
    if (epochControl_.compare_exchange_weak(observed, desired,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
      break;
    }
  }

  return newEpoch;
}

void InferenceQueue::fullReset() {
  const uint32_t epoch = getEpoch();

  // The worker must be stopped before fullReset(), so direct stores cannot
  // race any slot owner.
  for (auto& slot : queue_) {
    if (slot) {
      slot->control_.store(InferenceRequest::makeControl(
                               epoch, InferenceRequest::SlotState::Empty),
                           std::memory_order_release);
      slot->outputValid = false;
      slot->hostOutputValid = false;
      slot->hostOutputSampleCount = 0U;
    }
  }

  // Reset all indices
  writeIdx_.store(0, std::memory_order_release);
  readIdx_.store(0, std::memory_order_release);
  consumeIdx_.store(0, std::memory_order_release);
  epochControl_.store(makeEpochControl(epoch, 0), std::memory_order_release);
}

uint32_t InferenceQueue::getEpoch() const {
  return epochFromControl(epochControl_.load(std::memory_order_acquire));
}

bool InferenceQueue::reclaimStaleSlots(uint64_t observedEpochControl) {
  const uint32_t currentEpoch = epochFromControl(observedEpochControl);

  for (auto& slot : queue_) {
    if (!slot) {
      continue;
    }

    uint64_t slotControl = slot->control_.load(std::memory_order_acquire);
    const auto state = InferenceRequest::stateFromControl(slotControl);
    const uint32_t slotEpoch = InferenceRequest::epochFromControl(slotControl);
    if (slotEpoch == currentEpoch ||
        (state != InferenceRequest::SlotState::Ready &&
         state != InferenceRequest::SlotState::Processed)) {
      continue;
    }

    // Re-check the packed epoch/start snapshot before each CAS. If a newer
    // reset was published while scanning, restart synchronization instead
    // of treating a newer-epoch slot as stale. The exact packed slot CAS
    // also protects a slot that was reclaimed and republished meanwhile.
    if (epochControl_.load(std::memory_order_acquire) != observedEpochControl) {
      return false;
    }

    slot->control_.compare_exchange_strong(
        slotControl,
        InferenceRequest::makeControl(slotEpoch,
                                      InferenceRequest::SlotState::Empty),
        std::memory_order_acq_rel, std::memory_order_acquire);
  }

  return epochControl_.load(std::memory_order_acquire) == observedEpochControl;
}

void InferenceQueue::inferenceThreadFunc(WorkerCallbacks callbacks) {
  const WorkerPriorityStatus priorityStatus = configureCurrentThreadPriority();
  workerPriorityStatus_.store(priorityStatus, std::memory_order_release);
  if (priorityStatus == WorkerPriorityStatus::Failed) {
    DBG("[InferenceQueue] Failed to raise inference worker priority");
  }
  DBG("[InferenceQueue] Thread running");

  uint64_t lastSeenEpochControl = epochControl_.load(std::memory_order_acquire);
  readIdx_.store(startIndexFromControl(lastSeenEpochControl),
                 std::memory_order_release);
  bool hasPreviousInputSequence = false;
  uint64_t previousInputSequence = 0;
  const auto modelSampleForSequence = [](uint64_t sequence,
                                         uint64_t& modelSample) {
    constexpr uint64_t kChunkSize = static_cast<uint64_t>(kOutputChunkSize);
    if (sequence > std::numeric_limits<uint64_t>::max() / kChunkSize) {
      return false;
    }
    modelSample = sequence * kChunkSize;
    return true;
  };
  const auto resetWorkerAtModelSample = [&](uint64_t modelSample) {
    if (callbacks.reset != nullptr) {
      callbacks.reset(callbacks.context);
    }
    return resetOutputConversionAtModelSample(modelSample);
  };
  if (callbacks.reset != nullptr) {
    static_cast<void>(resetWorkerAtModelSample(0U));
  } else {
    static_cast<void>(resetOutputConversionAtModelSample(0U));
  }
  reclaimStaleSlots(lastSeenEpochControl);

  const auto synchronizeEpoch = [&]() {
    while (true) {
      const uint64_t currentEpochControl =
          epochControl_.load(std::memory_order_acquire);
      if (currentEpochControl == lastSeenEpochControl) {
        return currentEpochControl;
      }

      static_cast<void>(resetWorkerAtModelSample(0U));
      hasPreviousInputSequence = false;

      readIdx_.store(startIndexFromControl(currentEpochControl),
                     std::memory_order_release);
      lastSeenEpochControl = currentEpochControl;

      if (reclaimStaleSlots(currentEpochControl)) {
        return currentEpochControl;
      }
      // Another reset arrived during cleanup. Loop immediately and reset
      // runtime state again before inspecting or running any request.
    }
  };

  while (!shouldStop_.load(std::memory_order_acquire)) {
    synchronizeEpoch();

    // Audio-thread publication is atomic-only. When idle, the worker owns the
    // scheduling tradeoff and sleeps for a short bounded interval instead of
    // requiring the callback to enter an OS condition-variable wake path.
    const size_t pendingIndex = readIdx_.load(std::memory_order_acquire);
    const bool requestReady =
        queue_[pendingIndex] &&
        queue_[pendingIndex]->getState() == InferenceRequest::SlotState::Ready;
    const bool epochChanged =
        epochControl_.load(std::memory_order_acquire) != lastSeenEpochControl;
    if (!requestReady && !epochChanged &&
        !shouldStop_.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    if (shouldStop_.load(std::memory_order_acquire)) {
      break;
    }

    // A reset may have been published instead of a request. Synchronize
    // before looking at a slot, and repeat this check before every run.
    synchronizeEpoch();

    // Process all ready requests
    while (true) {
      if (shouldStop_.load(std::memory_order_acquire)) {
        break;
      }

      const uint64_t currentEpochControl = synchronizeEpoch();
      const uint32_t currentEpoch = epochFromControl(currentEpochControl);
      const size_t idx = readIdx_.load(std::memory_order_acquire);
      auto& request = queue_[idx];

      if (!request) {
        break;
      }

      uint64_t requestControl =
          request->control_.load(std::memory_order_acquire);
      if (InferenceRequest::stateFromControl(requestControl) !=
          InferenceRequest::SlotState::Ready) {
        break;
      }

      const uint32_t requestEpoch =
          InferenceRequest::epochFromControl(requestControl);
      const uint64_t processingControl = InferenceRequest::makeControl(
          requestEpoch, InferenceRequest::SlotState::Processing);
      if (!request->control_.compare_exchange_strong(
              requestControl, processingControl, std::memory_order_acquire,
              std::memory_order_acquire)) {
        continue;
      }

      // The epoch may have changed after synchronizeEpoch() but before
      // ownership was acquired. Discard without running; the next loop
      // synchronizes and jumps to the new epoch's exact start slot.
      if (requestEpoch != currentEpoch ||
          epochControl_.load(std::memory_order_acquire) !=
              currentEpochControl) {
        uint64_t expected = processingControl;
        request->control_.compare_exchange_strong(
            expected,
            InferenceRequest::makeControl(requestEpoch,
                                          InferenceRequest::SlotState::Empty),
            std::memory_order_release, std::memory_order_relaxed);
        break;
      }

      const uint64_t inputSequence = request->chunkSequence;
      if (callbacks.reset != nullptr && hasPreviousInputSequence &&
          inputSequence != previousInputSequence + 1) {
        DBG("[InferenceQueue] Input sequence gap; resetting model state");
        uint64_t modelSample = 0U;
        if (!modelSampleForSequence(inputSequence, modelSample) ||
            !resetWorkerAtModelSample(modelSample)) {
          request->outputValid = false;
          request->hostOutputValid = false;
        }
        hasPreviousInputSequence = false;
      }

      // Run inference. c91 emits the preceding input hop, so a successful run
      // immediately after reset publishes outputValid=false as normal pre-roll.
      bool inferenceOk = false;
      request->outputValid = false;
      request->hostOutputValid = false;
      request->hostOutputSampleCount = 0U;
      // This is the final epoch check immediately before invoking the
      // graph. An in-flight old-epoch run is discarded below if reset()
      // arrives concurrently.
      if (epochControl_.load(std::memory_order_acquire) ==
              currentEpochControl &&
          callbacks.run != nullptr) {
        inferenceOk = callbacks.run(callbacks.context, *request);
      }

      if (inferenceOk && request->outputValid &&
          !convertOutputToHost(*request)) {
        inferenceOk = false;
        request->outputValid = false;
        request->hostOutputValid = false;
      }

      // Check if reset occurred during inference
      if (epochControl_.load(std::memory_order_acquire) !=
          currentEpochControl) {
        uint64_t expected = processingControl;
        request->control_.compare_exchange_strong(
            expected,
            InferenceRequest::makeControl(requestEpoch,
                                          InferenceRequest::SlotState::Empty),
            std::memory_order_release, std::memory_order_relaxed);
        break;
      }

      // Publish an invalid marker so the consumer can advance past a
      // failed slot instead of deadlocking behind it. Reset runtime state
      // first so the next successful run starts with exactly one pre-roll.
      if (!inferenceOk) {
        uint64_t nextModelSample = 0U;
        const bool haveNextModelSample =
            inputSequence != std::numeric_limits<uint64_t>::max() &&
            modelSampleForSequence(inputSequence + 1U, nextModelSample);
        if (!haveNextModelSample ||
            !resetWorkerAtModelSample(haveNextModelSample ? nextModelSample
                                                          : 0U)) {
          request->hostOutputValid = false;
        }
        hasPreviousInputSequence = false;
      } else {
        if (!request->outputValid && outputSampleRateConversionEnabled_) {
          uint64_t nextAlignedModelSample = 0U;
          if (!modelSampleForSequence(inputSequence, nextAlignedModelSample) ||
              !resetOutputConversionAtModelSample(nextAlignedModelSample)) {
            request->hostOutputValid = false;
          }
        }
        previousInputSequence = inputSequence;
        hasPreviousInputSequence = true;
      }

      uint64_t expected = processingControl;
      if (request->control_.compare_exchange_strong(
              expected,
              InferenceRequest::makeControl(
                  requestEpoch, InferenceRequest::SlotState::Processed),
              std::memory_order_release, std::memory_order_relaxed)) {
        readIdx_.store((idx + 1) % kNumInferenceBuffers,
                       std::memory_order_release);
      }
    }
  }

  DBG("[InferenceQueue] Thread exiting");
}

}  // namespace audio_plugin
