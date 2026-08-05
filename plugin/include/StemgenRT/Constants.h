#pragma once

#include <StemgenRT/QualifiedModelContract.h>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>

namespace audio_plugin {

// The c214 current-chunk listening candidate emits
// [drums, bass, vocals, other].
constexpr int kNumStems = qualified_model::kNumStems;
constexpr int kNumChannels = qualified_model::kNumChannels;
constexpr int kStemDrums = qualified_model::kDrumsSourceIndex;
constexpr int kStemBass = qualified_model::kBassSourceIndex;
constexpr int kStemVocals = qualified_model::kVocalsSourceIndex;
constexpr int kStemOther = qualified_model::kOtherSourceIndex;

// Fixed model contract. The graph consumes and emits the same 256-sample hop.
// Its three recurrent states are the 768-sample analysis history, fusion GRU
// hidden state, and emitted Drums/Bass history used by the c191 correction.
constexpr int kModelSampleRate = qualified_model::kSampleRate;
constexpr int kOutputChunkSize = qualified_model::kHopSamples;
constexpr int kAnalysisWindowSize = qualified_model::kAnalysisWindowSamples;
constexpr int kAnalysisHistorySamples =
    qualified_model::kAnalysisHistorySamples;
constexpr int kFusionHiddenLayers = qualified_model::kFusionHiddenLayers;
constexpr int kFusionHiddenSize = qualified_model::kFusionHiddenSize;
constexpr int kEmittedDbChannels = qualified_model::kEmittedDbChannels;
constexpr int kEmittedDbHistorySamples =
    qualified_model::kEmittedDbHistorySamples;

// The worker gets one complete hop for inference. c214 emits its current input
// hop, so the asynchronous collection/queue hop is the entire 256-sample PDC.
constexpr int kModelOutputDelayChunks =
    qualified_model::kModelOutputDelayChunks;
constexpr int kAsyncQueueDelayChunks = 1;
constexpr int kPluginLatencyChunks =
    kModelOutputDelayChunks + kAsyncQueueDelayChunks;
constexpr int kPluginLatencySamples = kPluginLatencyChunks * kOutputChunkSize;

// This unpromoted listening candidate starts with the exact host configuration
// that exposes the intended 256-sample PDC.
constexpr int kCurrentChunkQualifiedHostSampleRate = kModelSampleRate;
constexpr int kCurrentChunkQualifiedHostBlockSize = kOutputChunkSize;

constexpr bool isQualifiedCurrentChunkHostConfiguration(int sampleRate,
                                                        int blockSize) {
  return sampleRate == kCurrentChunkQualifiedHostSampleRate &&
         blockSize == kCurrentChunkQualifiedHostBlockSize;
}

// Host clocks explicitly covered by the native sample-rate bridge. The graph
// contract itself remains fixed at 44.1 kHz. Keep this list qualification-
// gated rather than accepting arbitrary rates that have not been exercised by
// the latency, reconstruction, and SRC quality tests.
constexpr std::array<int, 6> kQualifiedHostSampleRates = {
    44100, 48000, 88200, 96000, 176400, 192000};

constexpr bool isQualifiedHostSampleRate(int sampleRate) {
  for (const int qualifiedRate : kQualifiedHostSampleRates) {
    if (sampleRate == qualifiedRate) {
      return true;
    }
  }
  return false;
}

constexpr std::uint64_t ceilDivide(std::uint64_t numerator,
                                   std::uint64_t denominator) {
  return numerator / denominator +
         static_cast<std::uint64_t>(numerator % denominator != 0U);
}

// A result for hop N is aligned to input hop N. The worker gets one complete
// 256-sample interval after the callback that supplies that request.
constexpr int calculatePluginLatencySamples(int hostBlockSize) {
  const int safeBlockSize = hostBlockSize > 0 ? hostBlockSize : 1;
  const int callbacksPerModelHop = 1 + (kOutputChunkSize - 1) / safeBlockSize;
  return kPluginLatencySamples + callbacksPerModelHop * safeBlockSize -
         std::gcd(safeBlockSize, kOutputChunkSize);
}

// Rate-aware form of the same diagnostic reserve. This remains available for
// later requalification; the checked-export current-chunk candidate accepts
// only the exact configuration above.
constexpr int calculateModelSchedulingLatencySamples(int hostSampleRate,
                                                     int hostBlockSize) {
  const std::uint64_t safeSampleRate = static_cast<std::uint64_t>(
      hostSampleRate > 0 ? hostSampleRate : kModelSampleRate);
  const std::uint64_t safeBlockSize =
      static_cast<std::uint64_t>(hostBlockSize > 0 ? hostBlockSize : 1);
  const std::uint64_t hopNumerator =
      static_cast<std::uint64_t>(kOutputChunkSize) * safeSampleRate;
  const std::uint64_t modelRate = static_cast<std::uint64_t>(kModelSampleRate);

  std::uint64_t latency = 0U;
  if (hopNumerator % modelRate == 0U) {
    const std::uint64_t hostHop = hopNumerator / modelRate;
    const std::uint64_t callbacksPerHop = ceilDivide(hostHop, safeBlockSize);
    latency = static_cast<std::uint64_t>(kPluginLatencyChunks) * hostHop +
              callbacksPerHop * safeBlockSize -
              std::gcd(hostHop, safeBlockSize);
  } else {
    const std::uint64_t schedulingSamples = ceilDivide(
        static_cast<std::uint64_t>(kPluginLatencyChunks) * hopNumerator,
        modelRate);
    const std::uint64_t callbacksPerHop =
        ceilDivide(hopNumerator, modelRate * safeBlockSize);
    latency = schedulingSamples + callbacksPerHop * safeBlockSize;
  }

  return latency <= static_cast<std::uint64_t>(std::numeric_limits<int>::max())
             ? static_cast<int>(latency)
             : std::numeric_limits<int>::max();
}

constexpr int calculatePluginLatencySamples(int hostSampleRate,
                                            int hostBlockSize,
                                            int sampleRateConversionDelay) {
  const int schedulingLatency =
      calculateModelSchedulingLatencySamples(hostSampleRate, hostBlockSize);
  const int safeConversionDelay =
      sampleRateConversionDelay > 0 ? sampleRateConversionDelay : 0;
  if (schedulingLatency >
      std::numeric_limits<int>::max() - safeConversionDelay) {
    return std::numeric_limits<int>::max();
  }
  return schedulingLatency + safeConversionDelay;
}

static_assert(kAnalysisWindowSize ==
              kAnalysisHistorySamples + kOutputChunkSize);
static_assert(kPluginLatencySamples == 256);
static_assert(calculatePluginLatencySamples(32) == 480);
static_assert(calculatePluginLatencySamples(64) == 448);
static_assert(calculatePluginLatencySamples(128) == 384);
static_assert(calculatePluginLatencySamples(256) == 256);
static_assert(calculatePluginLatencySamples(512) == 512);
static_assert(calculatePluginLatencySamples(768) == 768);
static_assert(calculatePluginLatencySamples(1024) == 1024);
static_assert(kModelOutputDelayChunks == 0);
static_assert(kAsyncQueueDelayChunks == 1);
static_assert(isQualifiedCurrentChunkHostConfiguration(44100, 256));
static_assert(!isQualifiedCurrentChunkHostConfiguration(48000, 256));
static_assert(!isQualifiedCurrentChunkHostConfiguration(44100, 512));
static_assert(isQualifiedHostSampleRate(44100));
static_assert(isQualifiedHostSampleRate(48000));
static_assert(isQualifiedHostSampleRate(192000));
static_assert(!isQualifiedHostSampleRate(48001));
static_assert(calculateModelSchedulingLatencySamples(44100, 64) == 448);
static_assert(calculateModelSchedulingLatencySamples(44100, 256) == 256);
static_assert(calculateModelSchedulingLatencySamples(48000, 256) == 791);
static_assert(calculateModelSchedulingLatencySamples(88200, 1024) == 1024);

// The c212 target-Mac order-balanced qualification selected two ORT intra-op
// threads. Retain that safe starting point for listening, then rerun the sweep
// against this 256-hop graph before promotion.
#if defined(__APPLE__)
constexpr int kOrtAutomaticIntraOpThreadCap = 2;
#else
constexpr int kOrtAutomaticIntraOpThreadCap = 4;
#endif

constexpr int calculateAutomaticOrtIntraOpThreadCount(int hardwareThreads) {
  const int safeHardwareThreads = hardwareThreads > 0 ? hardwareThreads : 1;
  const int halfHardwareThreads = safeHardwareThreads / 2;
  const int atLeastTwoThreads =
      halfHardwareThreads > 2 ? halfHardwareThreads : 2;
  return atLeastTwoThreads < kOrtAutomaticIntraOpThreadCap
             ? atLeastTwoThreads
             : kOrtAutomaticIntraOpThreadCap;
}

static_assert(calculateAutomaticOrtIntraOpThreadCount(0) == 2);
static_assert(calculateAutomaticOrtIntraOpThreadCount(4) == 2);
static_assert(calculateAutomaticOrtIntraOpThreadCount(14) ==
              kOrtAutomaticIntraOpThreadCap);

// Allow bounded timing variation without blocking the real-time audio thread.
constexpr int kNumInferenceBuffers = 16;
constexpr int kOutputRingBufferSlackChunks = 8;
constexpr int kOutputRingBufferChunks =
    kNumInferenceBuffers + kOutputRingBufferSlackChunks;

// JUCE's maximumExpectedSamplesPerBlock is advisory. Reserve a bounded amount
// of additional dry-delay storage so ordinary offline/variable-block renders
// cannot overwrite unread latency history when a host exceeds its estimate.
constexpr int kMinimumHostBlockCapacity = 65536;

// Smooth transitions between model output and the latency-aligned dry fallback.
constexpr int kUnderrunCrossfadeSamples = 64;

// The deployed graph has an observed, approximately level-independent per-stem
// floor near silence. Fade only the unreliable model contribution below this
// range; the final residual remains in Other so the output stays
// mixture-lossless.
constexpr float kLowLevelSeparationClosed = 0.0000158489f;  // -96 dBFS peak
constexpr float kLowLevelSeparationOpen = 0.000251189f;     // -72 dBFS peak
constexpr float kLowLevelHoldSeconds = 0.05f;
constexpr float kLowLevelReleaseSeconds = 0.1f;  // Time for a 60 dB decay

static_assert(kLowLevelSeparationOpen > kLowLevelSeparationClosed);
static_assert(kLowLevelSeparationClosed > 0.0f);
static_assert(kLowLevelHoldSeconds > 0.0f);
static_assert(kLowLevelReleaseSeconds > 0.0f);

}  // namespace audio_plugin
