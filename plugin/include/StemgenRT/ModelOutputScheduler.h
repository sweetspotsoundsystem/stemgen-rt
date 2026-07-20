#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "Constants.h"

namespace audio_plugin {

enum class ModelOutputScheduleAction {
  kSchedule,
  kWaitForHorizon,
  kDiscardInvalidSequence,
  kDiscardTimelineOverflow,
  kDiscardFullyLate,
};

// Pure timeline mapping for one stateful graph result. The returned source
// offset always refers to the original 512-sample result; elapsed samples are
// never shifted onto a newer output timeline.
struct ModelOutputSchedulePlan {
  ModelOutputScheduleAction action{
      ModelOutputScheduleAction::kDiscardInvalidSequence};
  uint64_t firstTimelineSample{0};
  uint64_t scheduleTimelineSample{0};
  size_t sourceOffset{0};
  size_t sampleCount{0};
};

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
    plan.sourceOffset =
        static_cast<size_t>(outputTimelineSample - firstTimelineSample);
  }
  plan.scheduleTimelineSample =
      firstTimelineSample + static_cast<uint64_t>(plan.sourceOffset);
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
    size_t chunkSize = static_cast<size_t>(kOutputChunkSize)) {
  ModelOutputSchedulePlan plan;
  if (chunkSequence == 0 || chunkSize == 0 || schedulingCapacity == 0) {
    return plan;
  }

  constexpr uint64_t kMaximumTimelineSample =
      std::numeric_limits<uint64_t>::max();
  const uint64_t chunkSize64 = static_cast<uint64_t>(chunkSize);
  const uint64_t alignedSequence = chunkSequence - 1U;
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

}  // namespace audio_plugin
