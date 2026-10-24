#include <StemgenRT/InferenceTiming.h>
#include <StemgenRT/OnnxRuntime.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>

namespace {
using namespace audio_plugin;

TEST(InferenceTimingTest, MissingAndRegressingCpuSamplesStayUnavailable) {
  EXPECT_EQ(cpuMicrosecondsBetween(10, -1), -1.0);
  EXPECT_EQ(cpuMicrosecondsBetween(-1, 10), -1.0);
  EXPECT_EQ(cpuMicrosecondsBetween(9, 10), -1.0);
  EXPECT_DOUBLE_EQ(cpuMicrosecondsBetween(3000, 1000), 2.0);
}

TEST(InferenceTimingTest, CpuClockExcludesWorkOnAnotherThread) {
  const auto before = currentThreadCpuNanoseconds();
  ASSERT_GE(before, 0);
  int64_t otherCpu = -1;
  std::atomic<uint64_t> work{};
  std::thread other([&] {
    const auto start = currentThreadCpuNanoseconds();
    const auto limit =
        InferenceRunTiming::Clock::now() + std::chrono::seconds(5);
    do {
      for (int i = 0; i < 10000; ++i)
        work.fetch_add(1, std::memory_order_relaxed);
      otherCpu = currentThreadCpuNanoseconds() - start;
    } while (otherCpu < 100000000 && InferenceRunTiming::Clock::now() < limit);
  });
  other.join();
  const auto thisCpu = currentThreadCpuNanoseconds() - before;
  EXPECT_GE(thisCpu, 0);
  EXPECT_GE(otherCpu, 100000000);
  EXPECT_LT(thisCpu, otherCpu);
  EXPECT_GT(work.load(), 0U);
}

TEST(InferenceTimingTest, RejectedCallClearsPreviousEvidence) {
  OnnxRuntime runtime;
  std::array<std::vector<float>, kNumChannels> input, aligned;
  std::array<decltype(input), kNumStems> output;
  InferenceRunTiming timing;
  timing.observed = true;
  timing.cpuStarted = 12;
  timing.cpuFinished = 34;
  bool valid = true;
  EXPECT_FALSE(runtime.runInference(input, output, aligned, valid, &timing));
  EXPECT_FALSE(valid);
  EXPECT_FALSE(timing.observed);
  EXPECT_EQ(timing.cpuStarted, -1);
  EXPECT_EQ(timing.cpuFinished, -1);
}

TEST(InferenceTimingTest, TracedAndUntracedStreamingAreByteIdentical) {
  OnnxRuntime normal, traced;
  juce::String error;
  const auto model =
      juce::File::getSpecialLocation(juce::File::currentExecutableFile)
          .getParentDirectory()
          .getParentDirectory()
          .getChildFile("Resources/model.onnx")
          .getFullPathName();
  ASSERT_TRUE(normal.loadModel(model, error, 1)) << error.toStdString();
  ASSERT_TRUE(traced.loadModel(model, error, 1)) << error.toStdString();
  ASSERT_TRUE(normal.prepareForInference(error));
  ASSERT_TRUE(traced.prepareForInference(error));
  using Audio = std::array<std::vector<float>, kNumChannels>;
  Audio input, normalAligned, tracedAligned;
  std::array<Audio, kNumStems> normalOutput, tracedOutput;
  for (auto* audio : {&input, &normalAligned, &tracedAligned})
    for (auto& channel : *audio)
      channel.resize(kOutputChunkSize);
  for (auto* stems : {&normalOutput, &tracedOutput})
    for (auto& stem : *stems)
      for (auto& channel : stem)
        channel.resize(kOutputChunkSize);
  const auto equalAudio = [](const Audio& a, const Audio& b) {
    for (size_t channel = 0; channel < a.size(); ++channel)
      EXPECT_EQ(std::memcmp(a[channel].data(), b[channel].data(),
                            a[channel].size() * sizeof(float)),
                0);
  };
  for (int hop = 0; hop <= 32; ++hop) {
    for (size_t channel = 0; channel < input.size(); ++channel)
      for (size_t i = 0; i < input[channel].size(); ++i)
        input[channel][i] =
            hop == 32
                ? 0.0f
                : 0.25f * std::sin(static_cast<float>(hop * kOutputChunkSize +
                                                      static_cast<int>(i)) *
                                   (channel == 0 ? 0.031f : 0.047f));
    if (hop == 16) {
      normal.resetStreamingState();
      traced.resetStreamingState();
    }
    bool normalValid = false, tracedValid = false;
    InferenceRunTiming timing;
    const auto start = InferenceRunTiming::Clock::now();
    ASSERT_TRUE(traced.runInference(input, tracedOutput, tracedAligned,
                                    tracedValid, &timing));
    const auto finish = InferenceRunTiming::Clock::now();
    ASSERT_TRUE(
        normal.runInference(input, normalOutput, normalAligned, normalValid));
    EXPECT_EQ(tracedValid, normalValid);
    EXPECT_EQ(tracedValid, hop != 0 && hop != 16);
    EXPECT_TRUE(timing.observed);
    EXPECT_LE(start, timing.started);
    EXPECT_LE(timing.started, timing.finished);
    EXPECT_LE(timing.finished, finish);
    EXPECT_GE(cpuMicrosecondsBetween(timing.cpuFinished, timing.cpuStarted),
              0.0);
    equalAudio(normalAligned, tracedAligned);
    for (size_t stem = 0; stem < normalOutput.size(); ++stem)
      equalAudio(normalOutput[stem], tracedOutput[stem]);
  }
  input[0][0] = std::numeric_limits<float>::quiet_NaN();
  InferenceRunTiming rejected;
  rejected.observed = true;
  bool valid = true;
  EXPECT_FALSE(traced.runInference(input, tracedOutput, tracedAligned, valid,
                                   &rejected));
  EXPECT_FALSE(rejected.observed);
}
}  // namespace
