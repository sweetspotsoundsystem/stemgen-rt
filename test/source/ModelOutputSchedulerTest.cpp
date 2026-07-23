#include <StemgenRT/Constants.h>
#include <StemgenRT/ModelOutputScheduler.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

namespace audio_plugin_test {
namespace {

using audio_plugin::ModelOutputScheduleAction;
using audio_plugin::planModelOutputRange;
using audio_plugin::planModelOutputSchedule;

constexpr uint64_t kLatency =
    static_cast<uint64_t>(audio_plugin::kPluginLatencySamples);
constexpr size_t kChunkSize =
    static_cast<size_t>(audio_plugin::kOutputChunkSize);

TEST(ModelOutputSchedulerTest, FullyElapsedResultIsDiscarded) {
  const auto plan = planModelOutputSchedule(
      1, kLatency, kLatency + static_cast<uint64_t>(kChunkSize),
      4 * kChunkSize);

  EXPECT_EQ(plan.action, ModelOutputScheduleAction::kDiscardFullyLate);
  EXPECT_EQ(plan.firstTimelineSample, kLatency);
  EXPECT_EQ(plan.sampleCount, 0U);
}

TEST(ModelOutputSchedulerTest, PartiallyLateResultKeepsItsSourceOffset) {
  constexpr size_t kElapsedPrefix = 137;
  const uint64_t outputTimeline =
      kLatency + static_cast<uint64_t>(kElapsedPrefix);
  const auto plan =
      planModelOutputSchedule(1, kLatency, outputTimeline, 4 * kChunkSize);

  ASSERT_EQ(plan.action, ModelOutputScheduleAction::kSchedule);
  EXPECT_EQ(plan.firstTimelineSample, kLatency);
  EXPECT_EQ(plan.scheduleTimelineSample, outputTimeline);
  EXPECT_EQ(plan.sourceOffset, kElapsedPrefix);
  EXPECT_EQ(plan.sampleCount, kChunkSize - kElapsedPrefix);
}

TEST(ModelOutputSchedulerTest,
     FutureResultRetainsAbsoluteTimelineAndExpiresWithoutReplay) {
  constexpr uint64_t kChunkSequence = 6;
  constexpr uint64_t kFirstTimeline =
      kLatency +
      (kChunkSequence -
       static_cast<uint64_t>(audio_plugin::kModelOutputDelayChunks)) *
          static_cast<uint64_t>(kChunkSize);
  constexpr size_t kCapacity = 2 * kChunkSize;

  const auto beyondHorizon =
      planModelOutputSchedule(kChunkSequence, kLatency, kLatency, kCapacity);
  ASSERT_EQ(beyondHorizon.action, ModelOutputScheduleAction::kWaitForHorizon);
  EXPECT_EQ(beyondHorizon.firstTimelineSample, kFirstTimeline);
  EXPECT_EQ(beyondHorizon.scheduleTimelineSample, kFirstTimeline);
  EXPECT_EQ(beyondHorizon.sourceOffset, 0U);

  const uint64_t timelineInsideHorizon =
      kFirstTimeline - static_cast<uint64_t>(kChunkSize);
  const auto insideHorizon = planModelOutputSchedule(
      kChunkSequence, kLatency, timelineInsideHorizon, kCapacity);
  ASSERT_EQ(insideHorizon.action, ModelOutputScheduleAction::kSchedule);
  EXPECT_EQ(insideHorizon.scheduleTimelineSample, kFirstTimeline);
  EXPECT_NE(insideHorizon.scheduleTimelineSample, timelineInsideHorizon);
  EXPECT_EQ(insideHorizon.sourceOffset, 0U);
  EXPECT_EQ(insideHorizon.sampleCount, kChunkSize);

  const auto afterExpiry = planModelOutputSchedule(
      kChunkSequence, kLatency,
      kFirstTimeline + static_cast<uint64_t>(kChunkSize), kCapacity);
  EXPECT_EQ(afterExpiry.action, ModelOutputScheduleAction::kDiscardFullyLate);
  EXPECT_EQ(afterExpiry.sampleCount, 0U);
}

TEST(ModelOutputSchedulerTest, SequenceZeroIsPrerollAndSequenceOneMapsTo512) {
  const auto preroll =
      planModelOutputSchedule(0U, kLatency, 0U, 4U * kChunkSize);

  EXPECT_EQ(preroll.action,
            ModelOutputScheduleAction::kDiscardInvalidRange);

  const auto firstReal =
      planModelOutputSchedule(1U, kLatency, kLatency, 4U * kChunkSize);
  ASSERT_EQ(firstReal.action, ModelOutputScheduleAction::kSchedule);
  EXPECT_EQ(firstReal.firstTimelineSample, kLatency);
  EXPECT_EQ(firstReal.scheduleTimelineSample, kLatency);
  EXPECT_EQ(firstReal.sourceOffset, 0U);
  EXPECT_EQ(firstReal.sampleCount, kChunkSize);
}

TEST(ModelOutputSchedulerTest,
     RationalHostRangePreservesVariableLengthAndLatePrefix) {
  constexpr uint64_t kFirstTimeline = 2139U + 558U;
  constexpr size_t kConvertedHopSize = 557U;
  constexpr size_t kLatePrefix = 73U;
  const auto plan = planModelOutputRange(kFirstTimeline, kConvertedHopSize,
                                         kFirstTimeline + kLatePrefix, 4096U);

  ASSERT_EQ(plan.action, ModelOutputScheduleAction::kSchedule);
  EXPECT_EQ(plan.firstTimelineSample, kFirstTimeline);
  EXPECT_EQ(plan.scheduleTimelineSample, kFirstTimeline + kLatePrefix);
  EXPECT_EQ(plan.sourceOffset, kLatePrefix);
  EXPECT_EQ(plan.sampleCount, kConvertedHopSize - kLatePrefix);
}

}  // namespace
}  // namespace audio_plugin_test
