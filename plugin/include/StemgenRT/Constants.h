#pragma once

#include <StemgenRT/QualifiedModelContract.h>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>

namespace audio_plugin {

// The qualified c91 deployment emits [drums, bass, vocals, other].
constexpr int kNumStems = qualified_model::kNumStems;
constexpr int kNumChannels = qualified_model::kNumChannels;
constexpr int kStemDrums = qualified_model::kDrumsSourceIndex;
constexpr int kStemBass = qualified_model::kBassSourceIndex;
constexpr int kStemVocals = qualified_model::kVocalsSourceIndex;
constexpr int kStemOther = qualified_model::kOtherSourceIndex;

// Fixed model contract. The graph consumes one 512-sample hop and emits the
// preceding hop while carrying its analysis overlap and fusion-GRU state.
constexpr int kModelSampleRate = qualified_model::kSampleRate;
constexpr int kOutputChunkSize = qualified_model::kHopSamples;
constexpr int kAnalysisWindowSize = qualified_model::kAnalysisWindowSamples;
constexpr int kFusionHiddenLayers = qualified_model::kFusionHiddenLayers;
constexpr int kFusionHiddenSize = qualified_model::kFusionHiddenSize;

// The background design needs one hop to collect/queue audio and the graph has
// one hop of output delay. Report both to the host for honest PDC.
constexpr int kModelOutputDelayChunks =
    qualified_model::kModelOutputDelayChunks;
constexpr int kAsyncQueueDelayChunks = 1;
constexpr int kPluginLatencyChunks =
    kModelOutputDelayChunks + kAsyncQueueDelayChunks;
constexpr int kPluginLatencySamples = kPluginLatencyChunks * kOutputChunkSize;

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

// A result for hop N is produced only after hop N+1 has been submitted. The
// worker still needs one complete 512-sample hop interval after the callback
// that supplies the last samples of N+1. Results are consumed only at callback
// boundaries, so reserve enough whole callbacks for that compute interval plus
// the callback/model-hop phase offset. This is the minimum fixed latency for a
// stable host block size; without the callback-count term, blocks below 512
// leave only one short callback for inference and continuously fall back dry.
constexpr int calculatePluginLatencySamples(int hostBlockSize) {
  const int safeBlockSize = hostBlockSize > 0 ? hostBlockSize : 1;
  const int callbacksPerModelHop = 1 + (kOutputChunkSize - 1) / safeBlockSize;
  return kPluginLatencySamples + callbacksPerModelHop * safeBlockSize -
         std::gcd(safeBlockSize, kOutputChunkSize);
}

// Rate-aware form of the callback scheduling reserve. A source hop beginning
// at model frame n is emitted only after model hop n+1 has been supplied. The
// worker then receives one complete 512/44100 second compute interval before a
// callback is expected to consume the result. For integer host-hop durations,
// the gcd term is the exact callback/model phase bound. At 48/96/192 kHz the
// phase is rational; ceil(2 * hop) plus whole callback intervals is a safe
// bound and over-reserves by less than one rational phase quantum for the
// qualified power-of-two callback sizes.
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
    latency = 2U * hostHop + callbacksPerHop * safeBlockSize -
              std::gcd(hostHop, safeBlockSize);
  } else {
    const std::uint64_t twoHopSamples =
        ceilDivide(2U * hopNumerator, modelRate);
    const std::uint64_t callbacksPerHop =
        ceilDivide(hopNumerator, modelRate * safeBlockSize);
    latency = twoHopSamples + callbacksPerHop * safeBlockSize;
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

static_assert(kAnalysisWindowSize == 2 * kOutputChunkSize);
static_assert(kPluginLatencySamples == 1024);
static_assert(calculatePluginLatencySamples(32) == 1504);
static_assert(calculatePluginLatencySamples(64) == 1472);
static_assert(calculatePluginLatencySamples(128) == 1408);
static_assert(calculatePluginLatencySamples(256) == 1280);
static_assert(calculatePluginLatencySamples(512) == 1024);
static_assert(calculatePluginLatencySamples(768) == 1536);
static_assert(calculatePluginLatencySamples(1024) == 1536);
static_assert(isQualifiedHostSampleRate(44100));
static_assert(isQualifiedHostSampleRate(48000));
static_assert(isQualifiedHostSampleRate(192000));
static_assert(!isQualifiedHostSampleRate(48001));
static_assert(calculateModelSchedulingLatencySamples(44100, 64) == 1472);
static_assert(calculateModelSchedulingLatencySamples(44100, 512) == 1024);
static_assert(calculateModelSchedulingLatencySamples(48000, 512) == 2139);
static_assert(calculateModelSchedulingLatencySamples(88200, 1024) == 2048);

// A repeated, order-balanced Apple Silicon qualification found three ORT
// intra-op threads had lower mean/tail latency and lower aggregate CPU cost
// than four for this small recurrent hop. Keep the unmeasured Windows policy
// unchanged until the same sweep is run there.
#if defined(__APPLE__)
constexpr int kOrtAutomaticIntraOpThreadCap = 3;
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

// Keep quiet material near the nominal operating level established by the
// former deployment wrapper and listening tests. This is deliberately
// boost-only: input already at or above the target is passed to the graph
// unchanged. Gain is stereo-linked, capped, and undone on the separated output
// while Main remains at its original level.
constexpr float kModelInputTargetRms = 0.25118864f;  // -12 dBFS RMS
constexpr float kModelInputMaxBoost = 100.0f;        // +40 dB
// Boost must not push an otherwise in-range raw analysis-window peak above
// full scale. Input that is already hotter than this remains unchanged because
// normalization is deliberately boost-only rather than a limiter.
constexpr float kModelInputPeakCeiling = 1.0f;  // 0 dBFS peak

static_assert(kModelInputTargetRms > 0.0f);
static_assert(kModelInputTargetRms < 1.0f);
static_assert(kModelInputMaxBoost >= 1.0f);
static_assert(kModelInputPeakCeiling > 0.0f);

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
