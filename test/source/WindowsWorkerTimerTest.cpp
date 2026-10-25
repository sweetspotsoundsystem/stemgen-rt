#include <StemgenRT/WindowsWorkerTimer.h>
#include <gtest/gtest.h>

#if defined(_WIN32)
#include <chrono>

namespace {
using audio_plugin::WindowsWorkerTimer;

TEST(WindowsWorkerTimerTest, PreservesDeadlinesAcrossRepeatedArming) {
  WindowsWorkerTimer timer;
  ASSERT_TRUE(timer.isValid());
  ASSERT_TRUE(timer.waitFor(std::chrono::nanoseconds::zero()));
  ASSERT_TRUE(timer.waitUntil(WindowsWorkerTimer::Clock::now() -
                              std::chrono::seconds(1)));
  for (int iteration = 0; iteration < 64; ++iteration) {
    const auto deadline =
        WindowsWorkerTimer::Clock::now() + std::chrono::microseconds(50);
    ASSERT_TRUE(timer.waitUntil(deadline));
    EXPECT_GE(WindowsWorkerTimer::Clock::now(), deadline);
  }
}

TEST(WindowsWorkerTimerTest, RepeatedWorkerLifetimesReleaseHandles) {
  // Warm up library/kernel paths before counting; this is a lifetime check,
  // not an assertion about scheduler latency on a shared CI runner.
  {
    WindowsWorkerTimer timer;
    ASSERT_TRUE(timer.isValid());
  }
  DWORD before{}, after{};
  ASSERT_NE(GetProcessHandleCount(GetCurrentProcess(), &before), 0);
  for (int iteration = 0; iteration < 64; ++iteration) {
    WindowsWorkerTimer timer;
    ASSERT_TRUE(timer.isValid());
    ASSERT_TRUE(timer.waitFor(std::chrono::microseconds(1)));
  }
  ASSERT_NE(GetProcessHandleCount(GetCurrentProcess(), &after), 0);
  EXPECT_EQ(after, before);
}
}  // namespace
#endif
