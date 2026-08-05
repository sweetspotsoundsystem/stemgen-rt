#include <StemgenRT/PluginProcessor.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

namespace audio_plugin_test {
namespace {

constexpr double kSampleRate = 44100.0;
constexpr int kBlockSize = audio_plugin::kOutputChunkSize;
constexpr int kTotalChannels = 12;  // 2 input + 10 output (5 buses * 2ch)
constexpr int kWarmupBlocks = 100;
constexpr int kMinimumQualificationCallbacks = 10000;
constexpr std::string_view kQualificationCallbacksEnvironment =
    "STEMGENRT_QUALIFICATION_CALLBACKS";

constexpr float kPi = 3.14159265358979323846f;

float sineAtSample(int64_t sampleIndex, float freqHz, float amplitude) {
  const float t =
      static_cast<float>(sampleIndex) / static_cast<float>(kSampleRate);
  return amplitude * std::sin(2.0f * kPi * freqHz * t);
}

int qualificationCallbackCount() {
  const char* overrideValue =
      std::getenv(kQualificationCallbacksEnvironment.data());
  if (overrideValue == nullptr || overrideValue[0] == '\0') {
    return kMinimumQualificationCallbacks;
  }

  const std::string_view text(overrideValue);
  int callbackCount = 0;
  const auto parseResult =
      std::from_chars(text.data(), text.data() + text.size(), callbackCount);
  if (parseResult.ec != std::errc{} ||
      parseResult.ptr != text.data() + text.size() ||
      callbackCount < kMinimumQualificationCallbacks ||
      callbackCount > std::numeric_limits<int>::max() - kWarmupBlocks) {
    return -1;
  }
  return callbackCount;
}

double percentileFromSorted(const std::vector<double>& sortedValues,
                            double percentile) {
  if (sortedValues.empty()) {
    return 0.0;
  }

  const double position =
      percentile * static_cast<double>(sortedValues.size() - 1U);
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return sortedValues[lower] +
         fraction * (sortedValues[upper] - sortedValues[lower]);
}

std::string_view workerPriorityStatusName(
    audio_plugin::InferenceQueue::WorkerPriorityStatus status) {
  using Status = audio_plugin::InferenceQueue::WorkerPriorityStatus;
  switch (status) {
    case Status::NotAttempted:
      return "not_attempted";
    case Status::Applied:
      return "applied";
    case Status::Failed:
      return "failed";
    case Status::Unsupported:
      return "unsupported";
  }
  return "unknown";
}

}  // namespace

// Explicit production-style real-time qualification soak. This remains
// disabled by default because 10,000 paced callbacks take almost two minutes
// at 44.1 kHz. Complete fallback routes the mixture only to Other, so every
// retained model source must become observably nonzero.
TEST(RealtimeStemSanityTest,
     DISABLED_StemsAreNotAllIdenticalWhenRealtimePaced) {
  const int measureBlocks = qualificationCallbackCount();
  ASSERT_GE(measureBlocks, kMinimumQualificationCallbacks)
      << kQualificationCallbacksEnvironment
      << " must be an integer greater than or equal to "
      << kMinimumQualificationCallbacks;

  audio_plugin::AudioPluginAudioProcessor processor;
  processor.prepareToPlay(kSampleRate, kBlockSize);

  ASSERT_GT(processor.getLatencySamples(), 0)
      << "Qualified model/runtime failed to load: "
      << processor.getOrtStatusString().toStdString();
  ASSERT_EQ(processor.getLatencySamples(), kBlockSize)
      << "The candidate must expose exactly one 256-sample PDC hop";

  juce::MidiBuffer midiBuffer;
  juce::AudioBuffer<float> buffer(kTotalChannels, kBlockSize);

  int64_t sampleIndex = 0;
  std::array<float, 3> maxAbsRetainedStems{};
  std::array<float, 3> maxAbsRetainedPairDifferences{};
  float maxAbsReconstructionError = 0.0f;
  bool allOutputSamplesFinite = true;
  std::vector<double> completeCallbackMicroseconds;
  completeCallbackMicroseconds.reserve(static_cast<size_t>(measureBlocks));
  uint64_t completeCallbackDeadlineMisses = 0U;
  auto nextDeadline = std::chrono::steady_clock::now();
  const auto blockDuration = std::chrono::duration<double>(
      static_cast<double>(kBlockSize) / kSampleRate);
  const double callbackDeadlineMicroseconds =
      1.0e6 * static_cast<double>(kBlockSize) / kSampleRate;

  for (int b = 0; b < (kWarmupBlocks + measureBlocks); ++b) {
    buffer.clear();

    // Fill input bus with a continuous multitone to encourage non-trivial stem
    // output.
    auto inputBus = processor.getBusBuffer(buffer, true /* isInput */, 0);
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
      const int64_t si = sampleIndex + i;
      const float l = sineAtSample(si, 55.0f, 0.30f) +
                      sineAtSample(si, 220.0f, 0.25f) +
                      sineAtSample(si, 880.0f, 0.20f);
      const float r = sineAtSample(si, 65.0f, 0.30f) +
                      sineAtSample(si, 330.0f, 0.25f) +
                      sineAtSample(si, 1320.0f, 0.20f);
      inputBus.setSample(0, i, l);
      inputBus.setSample(1, i, r);
    }

    const auto processStarted = std::chrono::steady_clock::now();
    processor.processBlock(buffer, midiBuffer);
    const auto processFinished = std::chrono::steady_clock::now();

    if (b >= kWarmupBlocks) {
      const double processMicroseconds =
          std::chrono::duration<double, std::micro>(
              processFinished - processStarted)
              .count();
      completeCallbackMicroseconds.push_back(processMicroseconds);
      if (processFinished - processStarted > blockDuration) {
        ++completeCallbackDeadlineMisses;
      }

      const int numOutputBuses = processor.getBusCount(false /* isInput */);
      ASSERT_GE(numOutputBuses, 5)
          << "Expected 5 output buses (Main + 4 stems)";

      auto drumsBus = processor.getBusBuffer(buffer, false /* isInput */, 1);
      auto bassBus = processor.getBusBuffer(buffer, false /* isInput */, 2);
      auto otherBus = processor.getBusBuffer(buffer, false /* isInput */, 3);
      auto vocalsBus = processor.getBusBuffer(buffer, false /* isInput */, 4);
      auto mainBus = processor.getBusBuffer(buffer, false /* isInput */, 0);

      for (int i = 0; i < buffer.getNumSamples(); ++i) {
        for (int ch = 0; ch < 2; ++ch) {
          const float d = drumsBus.getSample(ch, i);
          const float b0 = bassBus.getSample(ch, i);
          const float o = otherBus.getSample(ch, i);
          const float v = vocalsBus.getSample(ch, i);
          const float main = mainBus.getSample(ch, i);

          allOutputSamplesFinite =
              allOutputSamplesFinite && std::isfinite(d) &&
              std::isfinite(b0) && std::isfinite(o) && std::isfinite(v) &&
              std::isfinite(main);

          maxAbsRetainedStems[0] =
              std::max(maxAbsRetainedStems[0], std::abs(d));
          maxAbsRetainedStems[1] =
              std::max(maxAbsRetainedStems[1], std::abs(b0));
          maxAbsRetainedStems[2] =
              std::max(maxAbsRetainedStems[2], std::abs(v));
          maxAbsRetainedPairDifferences[0] = std::max(
              maxAbsRetainedPairDifferences[0], std::abs(d - b0));
          maxAbsRetainedPairDifferences[1] =
              std::max(maxAbsRetainedPairDifferences[1], std::abs(d - v));
          maxAbsRetainedPairDifferences[2] =
              std::max(maxAbsRetainedPairDifferences[2], std::abs(b0 - v));
          maxAbsReconstructionError =
              std::max(maxAbsReconstructionError,
                       std::abs(main - (d + b0 + o + v)));
        }
      }
    }

    sampleIndex += buffer.getNumSamples();

    // Pace at the real 256-sample callback interval. sleep_until returns
    // immediately after a complete processBlock overrun; the explicit timing
    // above covers the complete callback in addition to worker inference.
    nextDeadline +=
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            blockDuration);
    std::this_thread::sleep_until(nextDeadline);
  }

  ASSERT_EQ(completeCallbackMicroseconds.size(),
            static_cast<size_t>(measureBlocks));
  std::sort(completeCallbackMicroseconds.begin(),
            completeCallbackMicroseconds.end());
  const double meanCompleteCallbackMicroseconds =
      std::accumulate(completeCallbackMicroseconds.begin(),
                      completeCallbackMicroseconds.end(), 0.0) /
      static_cast<double>(completeCallbackMicroseconds.size());
  const double p50CompleteCallbackMicroseconds =
      percentileFromSorted(completeCallbackMicroseconds, 0.50);
  const double p95CompleteCallbackMicroseconds =
      percentileFromSorted(completeCallbackMicroseconds, 0.95);
  const double p99CompleteCallbackMicroseconds =
      percentileFromSorted(completeCallbackMicroseconds, 0.99);
  const double p999CompleteCallbackMicroseconds =
      percentileFromSorted(completeCallbackMicroseconds, 0.999);
  const double maximumCompleteCallbackMicroseconds =
      completeCallbackMicroseconds.back();

  const uint64_t underrunSamples = processor.getUnderrunSampleCount();
  const uint64_t underrunBlocks = processor.getUnderrunBlockCount();
  const uint64_t queueFullDrops = processor.getQueueFullChunkDropCount();
  const uint64_t ringOverflowEvents = processor.getRingOverflowEventCount();
  const uint64_t ringOverflowSamples =
      processor.getRingOverflowSampleDropCount();
  const uint64_t unsafeRealtimeCallbacks =
      processor.getUnsafeRealtimeCallbackCount();
  const auto workerPriorityStatus =
      processor.getInferenceWorkerPriorityStatus();
  const bool workerPriorityApplied =
      workerPriorityStatus ==
      audio_plugin::InferenceQueue::WorkerPriorityStatus::Applied;
  const bool retainedSourcesPresent =
      std::all_of(maxAbsRetainedStems.begin(), maxAbsRetainedStems.end(),
                  [](float peak) { return peak > 1.0e-3f; });
  const bool retainedSourcesDistinct = std::all_of(
      maxAbsRetainedPairDifferences.begin(),
      maxAbsRetainedPairDifferences.end(),
      [](float difference) { return difference > 1.0e-5f; });
  const bool qualificationPassed =
      completeCallbackDeadlineMisses == 0U && underrunSamples == 0U &&
      underrunBlocks == 0U && queueFullDrops == 0U &&
      ringOverflowEvents == 0U && ringOverflowSamples == 0U &&
      unsafeRealtimeCallbacks == 0U && workerPriorityApplied &&
      allOutputSamplesFinite &&
      retainedSourcesPresent && retainedSourcesDistinct &&
      maxAbsReconstructionError <= 1.0e-6f &&
      p999CompleteCallbackMicroseconds < callbackDeadlineMicroseconds &&
      maximumCompleteCallbackMicroseconds < callbackDeadlineMicroseconds;

  std::cerr << std::fixed << std::setprecision(3)
            << "STEMGENRT_QUALIFICATION_SUMMARY status="
            << (qualificationPassed ? "pass" : "fail")
            << " warmup_callbacks=" << kWarmupBlocks
            << " measured_callbacks=" << measureBlocks
            << " callback_samples=" << kBlockSize
            << " sample_rate=" << static_cast<int>(kSampleRate)
            << " deadline_us=" << callbackDeadlineMicroseconds
            << " mean_us=" << meanCompleteCallbackMicroseconds
            << " p50_us=" << p50CompleteCallbackMicroseconds
            << " p95_us=" << p95CompleteCallbackMicroseconds
            << " p99_us=" << p99CompleteCallbackMicroseconds
            << " p99.9_us=" << p999CompleteCallbackMicroseconds
            << " max_us=" << maximumCompleteCallbackMicroseconds
            << " deadline_misses=" << completeCallbackDeadlineMisses
            << " underrun_samples=" << underrunSamples
            << " underrun_blocks=" << underrunBlocks
            << " queue_full_drops=" << queueFullDrops
            << " ring_overflow_events=" << ringOverflowEvents
            << " ring_overflow_samples=" << ringOverflowSamples
            << " unsafe_realtime_callbacks=" << unsafeRealtimeCallbacks
            << " worker_priority="
            << workerPriorityStatusName(workerPriorityStatus)
            << " finite_outputs=" << (allOutputSamplesFinite ? 1 : 0)
            << std::scientific << std::setprecision(9)
            << " reconstruction_max_abs=" << maxAbsReconstructionError
            << " drums_max_abs=" << maxAbsRetainedStems[0]
            << " bass_max_abs=" << maxAbsRetainedStems[1]
            << " vocals_max_abs=" << maxAbsRetainedStems[2]
            << " drums_bass_max_abs_diff="
            << maxAbsRetainedPairDifferences[0]
            << " drums_vocals_max_abs_diff="
            << maxAbsRetainedPairDifferences[1]
            << " bass_vocals_max_abs_diff="
            << maxAbsRetainedPairDifferences[2] << '\n';

  EXPECT_TRUE(allOutputSamplesFinite);
  EXPECT_GT(maxAbsRetainedStems[0], 1.0e-3f)
      << "Drums remained silent (complete fallback only)";
  EXPECT_GT(maxAbsRetainedStems[1], 1.0e-3f)
      << "Bass remained silent (complete fallback only)";
  EXPECT_GT(maxAbsRetainedStems[2], 1.0e-3f)
      << "Vocals remained silent (complete fallback only)";
  EXPECT_GT(maxAbsRetainedPairDifferences[0], 1.0e-5f)
      << "Drums and Bass were identical throughout the soak";
  EXPECT_GT(maxAbsRetainedPairDifferences[1], 1.0e-5f)
      << "Drums and Vocals were identical throughout the soak";
  EXPECT_GT(maxAbsRetainedPairDifferences[2], 1.0e-5f)
      << "Bass and Vocals were identical throughout the soak";
  EXPECT_LE(maxAbsReconstructionError, 1.0e-6f)
      << "Stem buses did not reconstruct latency-aligned Main";
  EXPECT_EQ(underrunSamples, 0U);
  EXPECT_EQ(underrunBlocks, 0U);
  EXPECT_EQ(queueFullDrops, 0U);
  EXPECT_EQ(ringOverflowEvents, 0U);
  EXPECT_EQ(ringOverflowSamples, 0U);
  EXPECT_EQ(unsafeRealtimeCallbacks, 0U);
  EXPECT_EQ(workerPriorityStatus,
            audio_plugin::InferenceQueue::WorkerPriorityStatus::Applied);
  EXPECT_EQ(completeCallbackDeadlineMisses, 0U)
      << "Complete processBlock deadline misses; maximum callback was "
      << maximumCompleteCallbackMicroseconds << " us";
  EXPECT_LT(p999CompleteCallbackMicroseconds, callbackDeadlineMicroseconds);
  EXPECT_LT(maximumCompleteCallbackMicroseconds,
            callbackDeadlineMicroseconds);

  processor.releaseResources();
}

}  // namespace audio_plugin_test
