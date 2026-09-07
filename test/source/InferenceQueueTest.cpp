#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

#include <gtest/gtest.h>

#include "StemgenRT/InferenceQueue.h"

namespace audio_plugin {

class InferenceQueueTestPeer {
public:
  using RunCallback = bool (*)(void*, InferenceRequest&);
  using ResetCallback = void (*)(void*);

  static void startThread(InferenceQueue& queue,
                          void* context,
                          RunCallback run,
                          ResetCallback reset) {
    InferenceQueue::WorkerCallbacks callbacks;
    callbacks.context = context;
    callbacks.run = run;
    callbacks.reset = reset;
    queue.startThreadWithCallbacks(callbacks);
  }
};

}  // namespace audio_plugin

namespace {

using audio_plugin::InferenceQueue;
using audio_plugin::InferenceQueueTestPeer;
using audio_plugin::InferenceRequest;

constexpr auto kTestTimeout = std::chrono::seconds(3);

template <typename Predicate>
bool waitUntil(Predicate&& predicate) {
  const auto deadline = std::chrono::steady_clock::now() + kTestTimeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::yield();
  }
  return predicate();
}

InferenceRequest* waitForWriteSlot(InferenceQueue& queue) {
  InferenceRequest* request = nullptr;
  const bool acquired = waitUntil([&] {
    request = queue.getWriteSlot();
    return request != nullptr;
  });
  EXPECT_TRUE(acquired);
  return request;
}

void submit(InferenceQueue& queue, uint32_t epoch, uint64_t sequence) {
  InferenceRequest* request = waitForWriteSlot(queue);
  ASSERT_NE(request, nullptr);
  request->chunkSequence = sequence;
  queue.submitWriteSlot(epoch);
}

InferenceRequest* waitForOutput(InferenceQueue& queue, uint32_t epoch) {
  InferenceRequest* request = nullptr;
  const bool acquired = waitUntil([&] {
    request = queue.getOutputSlot(epoch);
    return request != nullptr;
  });
  EXPECT_TRUE(acquired);
  return request;
}

class FakeRuntime {
public:
  static bool runCallback(void* context, InferenceRequest& request) {
    return static_cast<FakeRuntime*>(context)->process(request);
  }

  static void resetCallback(void* context) {
    static_cast<FakeRuntime*>(context)->resetState();
  }

  bool process(InferenceRequest& request) {
    runCalls.fetch_add(1, std::memory_order_relaxed);
    if (blockFirstRun.exchange(false, std::memory_order_acq_rel)) {
      firstRunEntered.store(true, std::memory_order_release);
      while (!releaseFirstRun.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
    }

    const uint32_t runIndex =
        runsSinceReset.fetch_add(1, std::memory_order_relaxed);
    request.outputValid = runIndex > 0U;
    return true;
  }

  void resetState() {
    runsSinceReset.store(0, std::memory_order_relaxed);
    resetCalls.fetch_add(1, std::memory_order_release);
  }

  std::atomic<uint32_t> resetCalls{0};
  std::atomic<uint32_t> runCalls{0};
  std::atomic<uint32_t> runsSinceReset{0};
  std::atomic<bool> blockFirstRun{false};
  std::atomic<bool> firstRunEntered{false};
  std::atomic<bool> releaseFirstRun{false};
};

TEST(InferenceQueueTest, ReservationIsExclusiveAndResetReclaimsStaleReadyRing) {
  InferenceQueue queue;
  queue.allocate();

  const uint32_t oldEpoch = queue.getEpoch();
  for (int index = 0; index < audio_plugin::kNumInferenceBuffers; ++index) {
    InferenceRequest* request = queue.getWriteSlot();
    ASSERT_NE(request, nullptr);
    EXPECT_EQ(request->getState(), InferenceRequest::SlotState::Writing);
    EXPECT_EQ(request->getEpoch(), oldEpoch);
    EXPECT_EQ(queue.getWriteSlot(), nullptr);
    request->chunkSequence = static_cast<uint64_t>(index);
    queue.submitWriteSlot(oldEpoch);
    EXPECT_EQ(request->getState(), InferenceRequest::SlotState::Ready);
  }
  EXPECT_EQ(queue.getWriteSlot(), nullptr);

  const uint32_t newEpoch = queue.reset();
  EXPECT_EQ(newEpoch, oldEpoch + 1U);

  // writeIdx wrapped to the first old Ready slot. Claiming it atomically for
  // the new epoch proves a full stale ring cannot permanently stall reset.
  InferenceRequest* request = queue.getWriteSlot();
  ASSERT_NE(request, nullptr);
  EXPECT_EQ(request->getState(), InferenceRequest::SlotState::Writing);
  EXPECT_EQ(request->getEpoch(), newEpoch);
  request->chunkSequence = 0;
  queue.submitWriteSlot(newEpoch);
  EXPECT_EQ(request->getState(), InferenceRequest::SlotState::Ready);

  queue.fullReset();
  InferenceRequest* abandonedWrite = queue.getWriteSlot();
  ASSERT_NE(abandonedWrite, nullptr);
  EXPECT_EQ(abandonedWrite->getState(), InferenceRequest::SlotState::Writing);
  const uint32_t epochAfterAbandonedWrite = queue.reset();
  EXPECT_EQ(abandonedWrite->getState(), InferenceRequest::SlotState::Empty);
  InferenceRequest* replacement = queue.getWriteSlot();
  ASSERT_EQ(replacement, abandonedWrite);
  EXPECT_EQ(replacement->getEpoch(), epochAfterAbandonedWrite);
}

TEST(InferenceQueueTest, WorkerPublishesPriorityConfigurationResult) {
  FakeRuntime runtime;
  InferenceQueue queue;
  queue.allocate();

  EXPECT_EQ(queue.getWorkerPriorityStatus(),
            InferenceQueue::WorkerPriorityStatus::NotAttempted);
  InferenceQueueTestPeer::startThread(
      queue, &runtime, &FakeRuntime::runCallback, &FakeRuntime::resetCallback);

  ASSERT_TRUE(waitUntil([&] {
    return queue.getWorkerPriorityStatus() !=
           InferenceQueue::WorkerPriorityStatus::NotAttempted;
  }));

  const auto status = queue.getWorkerPriorityStatus();
#if defined(__APPLE__) || defined(_WIN32) || defined(__linux__)
  EXPECT_NE(status, InferenceQueue::WorkerPriorityStatus::Unsupported);
#else
  EXPECT_EQ(status, InferenceQueue::WorkerPriorityStatus::Unsupported);
#endif
  EXPECT_TRUE(status == InferenceQueue::WorkerPriorityStatus::Applied ||
              status == InferenceQueue::WorkerPriorityStatus::Failed ||
              status == InferenceQueue::WorkerPriorityStatus::Unsupported);

  queue.stopThread();
  EXPECT_EQ(queue.getWorkerPriorityStatus(), status);
}

TEST(InferenceQueueTest, CompatibilityWaitObservesProcessedPreroll) {
  FakeRuntime runtime;
  InferenceQueue queue;
  queue.allocate();
  InferenceQueueTestPeer::startThread(
      queue, &runtime, &FakeRuntime::runCallback, &FakeRuntime::resetCallback);

  const uint32_t epoch = queue.getEpoch();
  InferenceRequest* request = queue.getWriteSlot();
  ASSERT_NE(request, nullptr);
  request->chunkSequence = 0U;
  queue.submitWriteSlot(epoch);

  EXPECT_TRUE(queue.waitUntilProcessed(
      request, epoch, std::chrono::steady_clock::now() + kTestTimeout));
  EXPECT_FALSE(queue.waitUntilProcessed(
      request, epoch,
      std::chrono::steady_clock::now() - std::chrono::microseconds(1)))
      << "An already processed request must not be admitted after the "
         "callback deadline";
  InferenceRequest* output = queue.getOutputSlot(epoch);
  ASSERT_EQ(output, request);
  EXPECT_FALSE(output->outputValid);
  queue.releaseOutputSlot();
  queue.stopThread();
}

TEST(InferenceQueueTest,
     CompatibilityTimeoutLeavesRequestLiveForExactTimelineDiscard) {
  FakeRuntime runtime;
  InferenceQueue queue;
  queue.allocate();
  struct FirstRunReleaseGuard {
    ~FirstRunReleaseGuard() {
      runtime.releaseFirstRun.store(true, std::memory_order_release);
    }
    FakeRuntime& runtime;
  } releaseGuard{runtime};

  runtime.blockFirstRun.store(true, std::memory_order_release);
  InferenceQueueTestPeer::startThread(
      queue, &runtime, &FakeRuntime::runCallback, &FakeRuntime::resetCallback);

  const uint32_t epoch = queue.getEpoch();
  InferenceRequest* request = queue.getWriteSlot();
  ASSERT_NE(request, nullptr);
  request->chunkSequence = 0U;
  queue.submitWriteSlot(epoch);
  ASSERT_TRUE(waitUntil(
      [&] { return runtime.firstRunEntered.load(std::memory_order_acquire); }));

  EXPECT_FALSE(queue.waitUntilProcessed(
      request, epoch,
      std::chrono::steady_clock::now() + std::chrono::milliseconds(1)));
  EXPECT_EQ(request->getEpoch(), epoch);
  EXPECT_EQ(request->getState(), InferenceRequest::SlotState::Processing);

  runtime.releaseFirstRun.store(true, std::memory_order_release);
  EXPECT_TRUE(queue.waitUntilProcessed(
      request, epoch, std::chrono::steady_clock::now() + kTestTimeout));
  InferenceRequest* lateOutput = queue.getOutputSlot(epoch);
  ASSERT_EQ(lateOutput, request);
  EXPECT_FALSE(lateOutput->outputValid);
  queue.releaseOutputSlot();
  queue.stopThread();
}

TEST(InferenceQueueTest,
     ResetDuringInFlightRunDiscardsStaleOutputAndKeepsOnePreroll) {
  FakeRuntime runtime;
  InferenceQueue queue;
  queue.allocate();
  struct FirstRunReleaseGuard {
    ~FirstRunReleaseGuard() {
      runtime.releaseFirstRun.store(true, std::memory_order_release);
    }
    FakeRuntime& runtime;
  } releaseGuard{runtime};

  runtime.blockFirstRun.store(true, std::memory_order_release);
  InferenceQueueTestPeer::startThread(
      queue, &runtime, &FakeRuntime::runCallback, &FakeRuntime::resetCallback);
  ASSERT_TRUE(waitUntil([&] {
    return runtime.resetCalls.load(std::memory_order_acquire) == 1U;
  }));

  submit(queue, queue.getEpoch(), 42);
  ASSERT_TRUE(waitUntil(
      [&] { return runtime.firstRunEntered.load(std::memory_order_acquire); }));

  const uint32_t currentEpoch = queue.reset();
  submit(queue, currentEpoch, 0);
  submit(queue, currentEpoch, 1);
  runtime.releaseFirstRun.store(true, std::memory_order_release);

  InferenceRequest* first = waitForOutput(queue, currentEpoch);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first->chunkSequence, 0U);
  EXPECT_FALSE(first->outputValid);
  EXPECT_EQ(queue.getCurrentOutputSlot(), first);
  queue.releaseOutputSlot();

  InferenceRequest* output = waitForOutput(queue, currentEpoch);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->chunkSequence, 1U);
  EXPECT_TRUE(output->outputValid);
  queue.releaseOutputSlot();

  EXPECT_EQ(queue.getOutputSlot(currentEpoch), nullptr);
  ASSERT_TRUE(waitUntil([&] {
    return runtime.resetCalls.load(std::memory_order_acquire) == 2U;
  }));
  EXPECT_EQ(runtime.runCalls.load(std::memory_order_acquire), 3U);
  queue.stopThread();
}

TEST(InferenceQueueTest,
     ResetOfFullRingDoesNotLoseFirstHopBehindOldInFlightRun) {
  FakeRuntime runtime;
  InferenceQueue queue;
  queue.allocate();
  struct ReleaseGuard {
    ~ReleaseGuard() {
      runtime.releaseFirstRun.store(true, std::memory_order_release);
    }
    FakeRuntime& runtime;
  } release{runtime};
  runtime.blockFirstRun.store(true, std::memory_order_release);
  InferenceQueueTestPeer::startThread(
      queue, &runtime, &FakeRuntime::runCallback, &FakeRuntime::resetCallback);
  submit(queue, queue.getEpoch(), 0);
  ASSERT_TRUE(waitUntil(
      [&] { return runtime.firstRunEntered.load(std::memory_order_acquire); }));
  for (uint64_t sequence = 1; sequence < audio_plugin::kNumInferenceBuffers;
       ++sequence)
    submit(queue, queue.getEpoch(), sequence);
  ASSERT_EQ(queue.getWriteSlot(), nullptr);

  const uint32_t epoch = queue.reset();
  // Claim synchronously while the old graph is still blocked: no wait/retry
  // may hide a lost first request or overwrite the old Processing lease.
  auto* firstInput = queue.getWriteSlot();
  ASSERT_NE(firstInput, nullptr);
  firstInput->chunkSequence = 0;
  queue.submitWriteSlot(epoch);
  submit(queue, epoch, 1);
  runtime.releaseFirstRun.store(true, std::memory_order_release);
  auto* preroll = waitForOutput(queue, epoch);
  ASSERT_NE(preroll, nullptr);
  EXPECT_EQ(preroll->chunkSequence, 0U);
  EXPECT_FALSE(preroll->outputValid);
  queue.releaseOutputSlot();
  auto* firstValid = waitForOutput(queue, epoch);
  ASSERT_NE(firstValid, nullptr);
  EXPECT_EQ(firstValid->chunkSequence, 1U);
  EXPECT_TRUE(firstValid->outputValid);
  queue.releaseOutputSlot();
  EXPECT_EQ(runtime.runCalls.load(std::memory_order_acquire), 3U);
  queue.stopThread();
}

TEST(InferenceQueueTest,
     RepeatedResetPublicationRacesDoNotLoseCurrentEpochOrStall) {
  FakeRuntime runtime;
  InferenceQueue queue;
  queue.allocate();
  InferenceQueueTestPeer::startThread(
      queue, &runtime, &FakeRuntime::runCallback, &FakeRuntime::resetCallback);
  ASSERT_TRUE(waitUntil([&] {
    return runtime.resetCalls.load(std::memory_order_acquire) == 1U;
  }));

  constexpr uint32_t kResetCount = 200;
  uint32_t previousEpoch = queue.getEpoch();
  for (uint32_t iteration = 0; iteration < kResetCount; ++iteration) {
    // This request deliberately races the following reset. It may still be
    // Ready, may be Processing, or may have just become Processed.
    submit(queue, previousEpoch, 2);

    const uint32_t currentEpoch = queue.reset();
    ASSERT_EQ(currentEpoch, previousEpoch + 1U);
    submit(queue, currentEpoch, 0);
    submit(queue, currentEpoch, 1);

    InferenceRequest* first = waitForOutput(queue, currentEpoch);
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->getEpoch(), currentEpoch);
    EXPECT_EQ(first->chunkSequence, 0U);
    EXPECT_FALSE(first->outputValid);
    queue.releaseOutputSlot();

    InferenceRequest* output = waitForOutput(queue, currentEpoch);
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->getEpoch(), currentEpoch);
    EXPECT_EQ(output->chunkSequence, 1U);
    EXPECT_TRUE(output->outputValid);
    queue.releaseOutputSlot();

    previousEpoch = currentEpoch;
  }

  ASSERT_TRUE(waitUntil([&] {
    return runtime.resetCalls.load(std::memory_order_acquire) ==
           kResetCount + 1U;
  }));
  EXPECT_EQ(queue.getOutputSlot(previousEpoch), nullptr);
  queue.stopThread();
}

TEST(InferenceQueueTest,
     WorkerResamplesRetainedStemsOnAbsoluteHostTimelineAcrossSequenceGap) {
  FakeRuntime runtime;
  InferenceQueue queue;
  ASSERT_TRUE(queue.prepareOutputSampleRate(48000.0));
  queue.allocate();
  InferenceQueueTestPeer::startThread(
      queue, &runtime, &FakeRuntime::runCallback, &FakeRuntime::resetCallback);

  const uint32_t epoch = queue.getEpoch();
  submit(queue, epoch, 0U);
  submit(queue, epoch, 1U);
  submit(queue, epoch, 2U);

  InferenceRequest* first = waitForOutput(queue, epoch);
  ASSERT_NE(first, nullptr);
  EXPECT_FALSE(first->outputValid);
  EXPECT_FALSE(first->hostOutputValid);
  EXPECT_EQ(first->hostOutputSampleCount, 0U);
  queue.releaseOutputSlot();

  InferenceRequest* second = waitForOutput(queue, epoch);
  ASSERT_NE(second, nullptr);
  EXPECT_TRUE(second->outputValid);
  EXPECT_TRUE(second->hostOutputValid);
  EXPECT_EQ(second->hostOutputStartSample, 0U);
  EXPECT_EQ(second->hostOutputSampleCount, 279U);
  queue.releaseOutputSlot();

  InferenceRequest* third = waitForOutput(queue, epoch);
  ASSERT_NE(third, nullptr);
  EXPECT_TRUE(third->hostOutputValid);
  EXPECT_EQ(third->hostOutputStartSample, 279U);
  queue.releaseOutputSlot();

  // A sequence gap resets state and converter phase before processing the new
  // sequence. Sequence five is the new invalid pre-roll; sequence six emits
  // frame five at absolute model sample 5 * 256 instead of reusing the prior
  // local converter phase.
  submit(queue, epoch, 5U);
  submit(queue, epoch, 6U);
  InferenceRequest* afterGap = waitForOutput(queue, epoch);
  ASSERT_NE(afterGap, nullptr);
  EXPECT_EQ(afterGap->chunkSequence, 5U);
  EXPECT_FALSE(afterGap->outputValid);
  EXPECT_FALSE(afterGap->hostOutputValid);
  queue.releaseOutputSlot();

  InferenceRequest* next = waitForOutput(queue, epoch);
  ASSERT_NE(next, nullptr);
  EXPECT_EQ(next->chunkSequence, 6U);
  EXPECT_TRUE(next->outputValid);
  EXPECT_TRUE(next->hostOutputValid);
  EXPECT_EQ(next->hostOutputStartSample, 1394U);
  queue.releaseOutputSlot();
  queue.stopThread();
}

}  // namespace
