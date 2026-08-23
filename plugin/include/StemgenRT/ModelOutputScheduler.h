#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "Constants.h"

namespace audio_plugin {

enum class ModelOutputScheduleAction {
  kSchedule,
  kWaitForHorizon,
  kDiscardInvalidRange,
  kDiscardTimelineOverflow,
  kDiscardFullyLate,
};

// Callback-boundary admission for the fixed-hop asynchronous c91 path.  The
// audio thread asks for exactly one sequence: the request submitted by the
// preceding callback.  Older results have permanently missed their physical
// range, while a newer result must retain queue ownership until its own due
// boundary rather than being replayed early.
enum class AsyncDueResultAction {
  kDiscardLate,
  kHoldFuture,
  kConsumeInvalid,
  kConsumeValid,
};

struct AsyncDueResultPlan {
  AsyncDueResultAction action{AsyncDueResultAction::kDiscardLate};
  uint64_t dueSequence{0};
  uint64_t resultSequence{0};
};

// A play-to-stop transition must preserve enough zero-input callbacks to
// drain both c91's graph delay and the asynchronous queue delay. Only an exact
// qualified stopped callback consumes that budget; a mismatched callback is
// fail-closed and cannot stand in for a physical 512-sample hop.
struct StoppedFlushCallbackPlan {
  uint32_t callbacksRemaining{0};
  bool resetAfterCallback{false};
};

constexpr StoppedFlushCallbackPlan planStoppedFlushCallback(
    uint32_t callbacksRemaining,
    bool playbackStopped,
    bool qualifiedStoppedCallback,
    uint32_t requiredCallbacks =
        static_cast<uint32_t>(kPluginLatencyChunks)) {
  StoppedFlushCallbackPlan plan;
  plan.callbacksRemaining =
      playbackStopped ? requiredCallbacks : callbacksRemaining;
  if (qualifiedStoppedCallback && plan.callbacksRemaining > 0U) {
    --plan.callbacksRemaining;
    plan.resetAfterCallback = plan.callbacksRemaining == 0U;
  }
  return plan;
}

constexpr AsyncDueResultPlan planAsyncDueResult(
    uint64_t dueSequence,
    uint64_t resultSequence,
    bool outputValid,
    uint64_t modelOutputDelayChunks =
        static_cast<uint64_t>(kModelOutputDelayChunks)) {
  AsyncDueResultPlan plan;
  plan.dueSequence = dueSequence;
  plan.resultSequence = resultSequence;
  if (resultSequence < dueSequence) {
    plan.action = AsyncDueResultAction::kDiscardLate;
  } else if (resultSequence > dueSequence) {
    plan.action = AsyncDueResultAction::kHoldFuture;
  } else if (!outputValid || resultSequence < modelOutputDelayChunks) {
    // c91 sequence zero is the successful-but-invalid graph pre-roll.  Failed
    // runs also publish invalid markers so the timeline can advance entirely
    // on latency-aligned fallback.
    plan.action = AsyncDueResultAction::kConsumeInvalid;
  } else {
    plan.action = AsyncDueResultAction::kConsumeValid;
  }
  return plan;
}

// Pure timeline mapping for one stateful graph result. The returned source
// offset always refers to the original 512-sample result; elapsed samples are
// never shifted onto a newer output timeline.
struct ModelOutputSchedulePlan {
  ModelOutputScheduleAction action{
      ModelOutputScheduleAction::kDiscardInvalidRange};
  uint64_t firstTimelineSample{0};
  uint64_t scheduleTimelineSample{0};
  size_t sourceOffset{0};
  size_t sampleCount{0};
};

// The qualified real-time path admits a result only when its complete source
// hop begins at the callback's current physical boundary. A partially elapsed
// range is useful to the explicit offline renderer, but never to real time.
constexpr bool isCompleteModelOutputHopAtBoundary(
    const ModelOutputSchedulePlan& plan,
    uint64_t outputTimelineSample,
    size_t completeHopSampleCount) {
  return plan.action == ModelOutputScheduleAction::kSchedule &&
         plan.firstTimelineSample == outputTimelineSample &&
         plan.scheduleTimelineSample == outputTimelineSample &&
         plan.sourceOffset == 0U &&
         plan.sampleCount == completeHopSampleCount;
}

// Plan publication for an already timestamped host-domain range. Sample-rate
// conversion can produce a different number of host samples for each 512-
// sample model hop, so the generic range form is the source of truth. The
// chunk-sequence overload below preserves the exact 44.1 kHz mapping.
constexpr ModelOutputSchedulePlan planModelOutputRange(
    uint64_t firstTimelineSample,
    size_t sampleCount,
    uint64_t outputTimelineSample,
    size_t schedulingCapacity) {
  ModelOutputSchedulePlan plan;
  plan.firstTimelineSample = firstTimelineSample;
  if (sampleCount == 0 || schedulingCapacity == 0) {
    return plan;
  }

  constexpr uint64_t kMaximumTimelineSample =
      std::numeric_limits<uint64_t>::max();
  const uint64_t sampleCount64 = static_cast<uint64_t>(sampleCount);
  if (firstTimelineSample > kMaximumTimelineSample - sampleCount64) {
    plan.action = ModelOutputScheduleAction::kDiscardTimelineOverflow;
    return plan;
  }

  const uint64_t endTimelineSample = firstTimelineSample + sampleCount64;
  if (endTimelineSample <= outputTimelineSample) {
    plan.action = ModelOutputScheduleAction::kDiscardFullyLate;
    return plan;
  }

  if (firstTimelineSample < outputTimelineSample) {
    plan.sourceOffset = outputTimelineSample - firstTimelineSample;
  }
  plan.scheduleTimelineSample = firstTimelineSample < outputTimelineSample
                                    ? outputTimelineSample
                                    : firstTimelineSample;
  plan.sampleCount = sampleCount - plan.sourceOffset;

  const uint64_t scheduleDistance =
      plan.scheduleTimelineSample - outputTimelineSample;
  if (scheduleDistance >= static_cast<uint64_t>(schedulingCapacity)) {
    plan.action = ModelOutputScheduleAction::kWaitForHorizon;
    return plan;
  }
  const size_t availableCapacity =
      schedulingCapacity - static_cast<size_t>(scheduleDistance);
  if (plan.sampleCount > availableCapacity) {
    plan.action = ModelOutputScheduleAction::kWaitForHorizon;
    return plan;
  }

  plan.action = ModelOutputScheduleAction::kSchedule;
  return plan;
}

constexpr ModelOutputSchedulePlan planModelOutputSchedule(
    uint64_t chunkSequence,
    uint64_t latencySamples,
    uint64_t outputTimelineSample,
    size_t schedulingCapacity,
    size_t chunkSize = static_cast<size_t>(kOutputChunkSize),
    uint64_t modelOutputDelayChunks =
        static_cast<uint64_t>(kModelOutputDelayChunks)) {
  ModelOutputSchedulePlan plan;
  if (chunkSize == 0 || schedulingCapacity == 0 ||
      chunkSequence < modelOutputDelayChunks) {
    return plan;
  }

  constexpr uint64_t kMaximumTimelineSample =
      std::numeric_limits<uint64_t>::max();
  const uint64_t chunkSize64 = static_cast<uint64_t>(chunkSize);
  const uint64_t alignedSequence = chunkSequence - modelOutputDelayChunks;
  if (latencySamples > kMaximumTimelineSample - chunkSize64 ||
      alignedSequence >
          (kMaximumTimelineSample - latencySamples) / chunkSize64) {
    plan.action = ModelOutputScheduleAction::kDiscardTimelineOverflow;
    return plan;
  }

  const uint64_t firstTimelineSample =
      latencySamples + alignedSequence * chunkSize64;
  return planModelOutputRange(firstTimelineSample, chunkSize,
                              outputTimelineSample, schedulingCapacity);
}

static_assert(planAsyncDueResult(0U, 0U, false).action ==
              AsyncDueResultAction::kConsumeInvalid);
static_assert(planAsyncDueResult(1U, 0U, false).action ==
              AsyncDueResultAction::kDiscardLate);
static_assert(planAsyncDueResult(1U, 2U, true).action ==
              AsyncDueResultAction::kHoldFuture);
static_assert(planAsyncDueResult(1U, 1U, true).action ==
              AsyncDueResultAction::kConsumeValid);
static_assert(
    planStoppedFlushCallback(0U, true, true).callbacksRemaining == 1U);
static_assert(
    !planStoppedFlushCallback(0U, true, true).resetAfterCallback);
static_assert(planStoppedFlushCallback(1U, false, true).resetAfterCallback);
static_assert(
    planStoppedFlushCallback(0U, true, false).callbacksRemaining == 2U);

}  // namespace audio_plugin
