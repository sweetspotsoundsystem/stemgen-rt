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
#include <memory>
#include <numeric>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <pthread.h>
#endif

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
#if defined(_WIN32)
  char* duplicatedOverrideValue = nullptr;
  size_t duplicatedOverrideSize = 0U;
  const int duplicateResult =
      _dupenv_s(&duplicatedOverrideValue, &duplicatedOverrideSize,
                kQualificationCallbacksEnvironment.data());
  const std::unique_ptr<char, decltype(&std::free)> ownedOverrideValue(
      duplicatedOverrideValue, &std::free);
  if (duplicateResult != 0) {
    return -1;
  }
  const char* overrideValue = ownedOverrideValue.get();
#else
  const char* overrideValue =
      std::getenv(kQualificationCallbacksEnvironment.data());
#endif
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

struct TimingDistribution {
  double minimum{0.0};
  double mean{0.0};
  double p50{0.0};
  double p95{0.0};
  double p99{0.0};
  double p999{0.0};
  double maximum{0.0};
};

TimingDistribution summarizeTiming(std::vector<double>& values) {
  TimingDistribution summary;
  if (values.empty()) {
    return summary;
  }

  std::sort(values.begin(), values.end());
  summary.minimum = values.front();
  summary.mean = std::accumulate(values.begin(), values.end(), 0.0) /
                 static_cast<double>(values.size());
  summary.p50 = percentileFromSorted(values, 0.50);
  summary.p95 = percentileFromSorted(values, 0.95);
  summary.p99 = percentileFromSorted(values, 0.99);
  summary.p999 = percentileFromSorted(values, 0.999);
  summary.maximum = values.back();
  return summary;
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

bool configureAndVerifyCallbackThreadPriority() noexcept {
#if defined(__APPLE__)
  const int setResult =
      pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  if (setResult != 0) {
    return false;
  }

  qos_class_t observedClass{};
  int relativePriority = 0;
  const int getResult = pthread_get_qos_class_np(
      pthread_self(), &observedClass, &relativePriority);
  return getResult == 0 && observedClass == QOS_CLASS_USER_INTERACTIVE;
#else
  return false;
#endif
}

}  // namespace

// Explicit production-style asynchronous qualification soak. This remains
// disabled by default because 10,000 paced callbacks take almost two minutes
// at 44.1 kHz. Complete fallback routes the mixture only to Other, so every
// retained model source must become observably nonzero and distinct.
TEST(RealtimeStemSanityTest,
     DISABLED_StemsAreNotAllIdenticalWhenAsyncRealtimePaced) {
  const int measureBlocks = qualificationCallbackCount();
  ASSERT_GE(measureBlocks, kMinimumQualificationCallbacks)
      << kQualificationCallbacksEnvironment
      << " must be an integer greater than or equal to "
      << kMinimumQualificationCallbacks;

  const bool callbackPriorityApplied =
      configureAndVerifyCallbackThreadPriority();

  audio_plugin::AudioPluginAudioProcessor processor;
  processor.prepareToPlay(kSampleRate, kBlockSize);

  ASSERT_GT(processor.getLatencySamples(), 0)
      << "Qualified model/runtime failed to load: "
      << processor.getOrtStatusString().toStdString();
  ASSERT_EQ(processor.getLatencySamples(), 1024)
      << "The c91 candidate must expose graph delay 1 + queue delay 1";
  ASSERT_EQ(processor.getLatencySamples(), audio_plugin::kPluginLatencySamples);

  juce::MidiBuffer midiBuffer;
  juce::AudioBuffer<float> buffer(kTotalChannels, kBlockSize);

  int64_t sampleIndex = 0;
  std::array<float, 3> maxAbsRetainedStems{};
  std::array<float, 3> maxAbsRetainedPairDifferences{};
  float maxAbsReconstructionError = 0.0f;
  bool allOutputSamplesFinite = true;
  std::vector<double> completeCallbackMicroseconds;
  completeCallbackMicroseconds.reserve(static_cast<size_t>(measureBlocks));
  std::vector<double> callbackStartInterarrivalMicroseconds;
  callbackStartInterarrivalMicroseconds.reserve(
      static_cast<size_t>(measureBlocks));
  std::vector<double> callbackStartLatenessMicroseconds;
  callbackStartLatenessMicroseconds.reserve(
      static_cast<size_t>(measureBlocks));
  uint64_t completeCallbackDeadlineMisses = 0U;
  uint64_t callbackStartDeadlineMisses = 0U;
  uint64_t callbackStartCatchupIntervals = 0U;
  const auto blockDuration = std::chrono::duration<double>(
      static_cast<double>(kBlockSize) / kSampleRate);
  const auto blockDurationTicks =
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          blockDuration);
  auto scheduledCallbackStart = std::chrono::steady_clock::now();
  auto previousCallbackStart = scheduledCallbackStart;
  bool havePreviousCallbackStart = false;
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
      const double startLatenessMicroseconds =
          std::chrono::duration<double, std::micro>(processStarted -
                                                   scheduledCallbackStart)
              .count();
      callbackStartLatenessMicroseconds.push_back(startLatenessMicroseconds);
      if (processStarted - scheduledCallbackStart > blockDuration) {
        ++callbackStartDeadlineMisses;
      }

      if (havePreviousCallbackStart) {
        const double interarrivalMicroseconds =
            std::chrono::duration<double, std::micro>(processStarted -
                                                     previousCallbackStart)
                .count();
        callbackStartInterarrivalMicroseconds.push_back(
            interarrivalMicroseconds);
        if (interarrivalMicroseconds < 0.5 * callbackDeadlineMicroseconds) {
          ++callbackStartCatchupIntervals;
        }
      }

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
    previousCallbackStart = processStarted;
    havePreviousCallbackStart = true;

    sampleIndex += buffer.getNumSamples();

    // Pace at the real 512-sample callback interval. The worker has one full
    // callback period to publish each request for the following boundary;
    // processBlock itself must never wait for inference.
    scheduledCallbackStart += blockDurationTicks;
    std::this_thread::sleep_until(scheduledCallbackStart);
  }

  ASSERT_EQ(completeCallbackMicroseconds.size(),
            static_cast<size_t>(measureBlocks));
  ASSERT_EQ(callbackStartInterarrivalMicroseconds.size(),
            static_cast<size_t>(measureBlocks));
  ASSERT_EQ(callbackStartLatenessMicroseconds.size(),
            static_cast<size_t>(measureBlocks));

  const TimingDistribution callbackTiming =
      summarizeTiming(completeCallbackMicroseconds);
  const TimingDistribution startInterarrivalTiming =
      summarizeTiming(callbackStartInterarrivalMicroseconds);
  const TimingDistribution startLatenessTiming =
      summarizeTiming(callbackStartLatenessMicroseconds);

  const size_t underrunSamplesInLastBlock =
      processor.getUnderrunSamplesInLastBlock();
  const uint64_t underrunSamples = processor.getUnderrunSampleCount();
  const uint64_t underrunBlocks = processor.getUnderrunBlockCount();
  const bool underrunActive = processor.isUnderrunActive();
  const uint64_t queueFullDrops = processor.getQueueFullChunkDropCount();
  const uint64_t ringOverflowEvents = processor.getRingOverflowEventCount();
  const uint64_t ringOverflowSamples =
      processor.getRingOverflowSampleDropCount();
  const bool unsafeRealtimeCallback =
      processor.isRealtimeCallbackTimingUnsafe();
  const uint64_t unsafeRealtimeCallbacks =
      processor.getUnsafeRealtimeCallbackCount();
  const uint64_t dueBoundaryMisses =
      processor.getSameCallbackTimeoutCount();
  const int lastWaitMicroseconds =
      processor.getLastSameCallbackWaitMicroseconds();
  const int maximumWaitMicroseconds =
      processor.getMaximumSameCallbackWaitMicroseconds();
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
      completeCallbackDeadlineMisses == 0U &&
      callbackStartDeadlineMisses == 0U &&
      callbackStartCatchupIntervals == 0U &&
      callbackTiming.p999 < callbackDeadlineMicroseconds &&
      callbackTiming.maximum < callbackDeadlineMicroseconds &&
      dueBoundaryMisses == 0U && lastWaitMicroseconds == 0 &&
      maximumWaitMicroseconds == 0 &&
      audio_plugin::kAudioThreadWaitBudgetMicroseconds == 0 &&
      underrunSamplesInLastBlock == 0U && underrunSamples == 0U &&
      underrunBlocks == 0U && !underrunActive && queueFullDrops == 0U &&
      ringOverflowEvents == 0U && ringOverflowSamples == 0U &&
      !unsafeRealtimeCallback && unsafeRealtimeCallbacks == 0U &&
      callbackPriorityApplied && workerPriorityApplied &&
      allOutputSamplesFinite && retainedSourcesPresent &&
      retainedSourcesDistinct &&
      maxAbsReconstructionError <= 1.0e-6f;

  std::cerr << std::fixed << std::setprecision(3)
            << "STEMGENRT_QUALIFICATION_SUMMARY status="
            << (qualificationPassed ? "pass" : "fail")
            << " warmup_callbacks=" << kWarmupBlocks
            << " measured_callbacks=" << measureBlocks
            << " callback_samples=" << kBlockSize
            << " sample_rate=" << static_cast<int>(kSampleRate)
            << " pdc_samples=" << processor.getLatencySamples()
            << " deadline_us=" << callbackDeadlineMicroseconds
            << " min_us=" << callbackTiming.minimum
            << " mean_us=" << callbackTiming.mean
            << " p50_us=" << callbackTiming.p50
            << " p95_us=" << callbackTiming.p95
            << " p99_us=" << callbackTiming.p99
            << " p99.9_us=" << callbackTiming.p999
            << " max_us=" << callbackTiming.maximum
            << " deadline_misses=" << completeCallbackDeadlineMisses
            << " start_interarrival_min_us="
            << startInterarrivalTiming.minimum
            << " start_interarrival_mean_us="
            << startInterarrivalTiming.mean
            << " start_interarrival_p50_us=" << startInterarrivalTiming.p50
            << " start_interarrival_p95_us=" << startInterarrivalTiming.p95
            << " start_interarrival_p99_us=" << startInterarrivalTiming.p99
            << " start_interarrival_p99.9_us="
            << startInterarrivalTiming.p999
            << " start_interarrival_max_us="
            << startInterarrivalTiming.maximum
            << " start_lateness_min_us=" << startLatenessTiming.minimum
            << " start_lateness_mean_us=" << startLatenessTiming.mean
            << " start_lateness_p50_us=" << startLatenessTiming.p50
            << " start_lateness_p95_us=" << startLatenessTiming.p95
            << " start_lateness_p99_us=" << startLatenessTiming.p99
            << " start_lateness_p99.9_us=" << startLatenessTiming.p999
            << " start_lateness_max_us=" << startLatenessTiming.maximum
            << " start_deadline_misses=" << callbackStartDeadlineMisses
            << " start_catchup_intervals="
            << callbackStartCatchupIntervals
            << " due_boundary_misses=" << dueBoundaryMisses
            << " last_wait_us=" << lastWaitMicroseconds
            << " max_wait_us=" << maximumWaitMicroseconds
            << " wait_budget_us="
            << audio_plugin::kAudioThreadWaitBudgetMicroseconds
            << " underrun_samples_last_block="
            << underrunSamplesInLastBlock
            << " underrun_samples=" << underrunSamples
            << " underrun_blocks=" << underrunBlocks
            << " underrun_active=" << (underrunActive ? 1 : 0)
            << " queue_full_drops=" << queueFullDrops
            << " ring_overflow_events=" << ringOverflowEvents
            << " ring_overflow_samples=" << ringOverflowSamples
            << " unsafe_realtime_current="
            << (unsafeRealtimeCallback ? 1 : 0)
            << " unsafe_realtime_callbacks=" << unsafeRealtimeCallbacks
            << " callback_priority="
            << (callbackPriorityApplied ? "applied" : "failed")
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
  EXPECT_EQ(dueBoundaryMisses, 0U)
      << "Exact due results missed their asynchronous callback boundary";
  EXPECT_EQ(lastWaitMicroseconds, 0);
  EXPECT_EQ(maximumWaitMicroseconds, 0);
  EXPECT_EQ(audio_plugin::kAudioThreadWaitBudgetMicroseconds, 0);
  EXPECT_EQ(underrunSamplesInLastBlock, 0U);
  EXPECT_EQ(underrunSamples, 0U);
  EXPECT_EQ(underrunBlocks, 0U);
  EXPECT_FALSE(underrunActive);
  EXPECT_EQ(queueFullDrops, 0U);
  EXPECT_EQ(ringOverflowEvents, 0U);
  EXPECT_EQ(ringOverflowSamples, 0U);
  EXPECT_FALSE(unsafeRealtimeCallback);
  EXPECT_EQ(unsafeRealtimeCallbacks, 0U);
  EXPECT_TRUE(callbackPriorityApplied)
      << "Callback/test thread QoS was not independently observed as "
         "QOS_CLASS_USER_INTERACTIVE";
  EXPECT_EQ(workerPriorityStatus,
            audio_plugin::InferenceQueue::WorkerPriorityStatus::Applied);
  EXPECT_EQ(completeCallbackDeadlineMisses, 0U)
      << "Complete processBlock deadline misses; maximum callback was "
      << callbackTiming.maximum << " us";
  EXPECT_EQ(callbackStartDeadlineMisses, 0U)
      << "The synthetic host callback started at least one full period late";
  EXPECT_EQ(callbackStartCatchupIntervals, 0U)
      << "The synthetic host pacing collapsed into catch-up callbacks";
  EXPECT_LT(callbackTiming.p999, callbackDeadlineMicroseconds);
  EXPECT_LT(callbackTiming.maximum, callbackDeadlineMicroseconds);
  EXPECT_TRUE(qualificationPassed);

  processor.releaseResources();
}

}  // namespace audio_plugin_test
