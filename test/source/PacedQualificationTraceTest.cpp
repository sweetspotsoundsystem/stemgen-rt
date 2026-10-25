#include "MacBenchmarkActivity.h"
#include "PacedHostTiming.h"
#include "PacedQualificationTrace.h"

#include <gtest/gtest.h>

namespace audio_plugin_test {

namespace {
PacedHostTiming::Clock::time_point atMicroseconds(int64_t value) {
  return PacedHostTiming::Clock::time_point(std::chrono::microseconds(value));
}
}  // namespace

TEST(PacedHostTimingTest, SeparatesSleepPreparationAndBookkeepingDelays) {
  PacedHostTiming timing(2902.494);
  timing.observe(1, atMicroseconds(3000), atMicroseconds(1000),
                 atMicroseconds(316000), atMicroseconds(316020),
                 atMicroseconds(900));
  timing.observe(2, atMicroseconds(320000), atMicroseconds(316100),
                 atMicroseconds(320010), atMicroseconds(620010),
                 atMicroseconds(316050));
  timing.observe(3, atMicroseconds(623000), atMicroseconds(920000),
                 atMicroseconds(920001), atMicroseconds(920011),
                 atMicroseconds(620020));

  ASSERT_EQ(timing.size(), 3U);
  EXPECT_DOUBLE_EQ(timing.events()[0].waitOverrunMicroseconds, 313000.0);
  EXPECT_DOUBLE_EQ(timing.events()[0].preparationMicroseconds, 20.0);
  EXPECT_DOUBLE_EQ(timing.events()[1].preparationMicroseconds, 300000.0);
  EXPECT_DOUBLE_EQ(timing.events()[2].bookkeepingMicroseconds, 299980.0);
  EXPECT_DOUBLE_EQ(timing.events()[2].waitOverrunMicroseconds, 1.0);
  EXPECT_EQ(timing.delayCount(), 3U);
}

TEST(PacedHostTimingTest, CatchupDoesNotInventAnotherSleepOverrun) {
  PacedHostTiming timing(2902.494);
  timing.observe(1, atMicroseconds(3000), atMicroseconds(316000),
                 atMicroseconds(316001), atMicroseconds(316021),
                 atMicroseconds(315950));
  EXPECT_EQ(timing.delayCount(), 0U);
  EXPECT_DOUBLE_EQ(timing.maximumWaitOverrun(), 1.0);
  EXPECT_DOUBLE_EQ(timing.maximumPreparation(), 20.0);
  EXPECT_DOUBLE_EQ(timing.maximumBookkeeping(), 50.0);
}

TEST(PacedHostTimingTest, CountsEveryDelayWhenStorageIsFull) {
  PacedHostTiming timing(2902.494);
  for (int index = 0; index < 100; ++index) {
    timing.observe(index, atMicroseconds(3000), atMicroseconds(1000),
                   atMicroseconds(7000), atMicroseconds(7020),
                   atMicroseconds(900));
  }
  EXPECT_EQ(timing.size(), PacedHostTiming::kCapacity);
  EXPECT_EQ(timing.delayCount(), 100U);
  EXPECT_EQ(timing.omittedEvents(), 36U);
}

#if defined(__APPLE__)
TEST(MacBenchmarkActivityTest, NestedActivitiesRetainTheirOwnLifetime) {
  MacBenchmarkActivity outer;
  ASSERT_TRUE(outer.isActive());
  {
    MacBenchmarkActivity inner;
    ASSERT_TRUE(inner.isActive());
  }
  EXPECT_TRUE(outer.isActive());
}
#endif

TEST(PacedQualificationTraceTest, SeparatesWarmupMissAndLaterMeasuredDiscard) {
  PacedQualificationTrace trace(100);
  trace.observe(98, {1, 128, 0}, 2200.0, 40.0, 740.0);
  trace.observe(99, {1, 128, 0}, 2900.0, 40.0, 40.0);
  trace.observe(100, {1, 128, 1}, 2900.0, 40.0, 40.0);
  trace.observe(105, {2, 256, 1}, 2250.0, 30.0, 680.0);

  ASSERT_EQ(trace.size(), 3U);
  EXPECT_TRUE(trace.events()[0].duringWarmup);
  EXPECT_FALSE(trace.events()[1].duringWarmup);
  EXPECT_EQ(trace.events()[1].delta.dueBoundaryMisses, 0U);
  EXPECT_EQ(trace.events()[1].delta.lateDiscardEvents, 1U);
  EXPECT_EQ(trace.events()[2].callbackIndex, 105);
  EXPECT_DOUBLE_EQ(trace.events()[2].callbackSpacingMicroseconds, 2250.0);
  EXPECT_EQ(trace.warmupCounters().dueBoundaryMisses, 1U);
  EXPECT_EQ(trace.measuredCounters().dueBoundaryMisses, 1U);
  EXPECT_EQ(trace.measuredCounters().underrunSamples, 128U);
  EXPECT_EQ(trace.measuredCounters().lateDiscardEvents, 1U);
  EXPECT_TRUE(trace.countersMonotonic());
}

TEST(PacedQualificationTraceTest, BoundedTracePreservesCountsWhenFull) {
  PacedQualificationTrace trace(0);
  for (int index = 0; index < 100; ++index) {
    const auto count = static_cast<uint64_t>(index + 1);
    trace.observe(index, {count, count * 128U, count}, 2900.0, 0.0, 0.0);
  }
  EXPECT_EQ(trace.size(), PacedQualificationTrace::kCapacity);
  EXPECT_EQ(trace.omittedEvents(), 36U);
  EXPECT_EQ(trace.warmupCounters().dueBoundaryMisses, 0U);
  EXPECT_EQ(trace.measuredCounters().dueBoundaryMisses, 100U);
  EXPECT_EQ(trace.measuredCounters().underrunSamples, 12800U);
}

TEST(PacedQualificationTraceTest, CounterRegressionIsReportedWithoutUnderflow) {
  PacedQualificationTrace trace(100);
  trace.observe(99, {2, 256, 1}, 2900.0, 0.0, 0.0);
  trace.observe(100, {1, 128, 0}, 2900.0, 0.0, 0.0);
  EXPECT_FALSE(trace.countersMonotonic());
  EXPECT_EQ(trace.measuredCounters().dueBoundaryMisses, 0U);
  EXPECT_EQ(trace.measuredCounters().underrunSamples, 0U);
}

}  // namespace audio_plugin_test
