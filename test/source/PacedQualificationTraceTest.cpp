#include "PacedQualificationTrace.h"

#include <gtest/gtest.h>

namespace audio_plugin_test {

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
