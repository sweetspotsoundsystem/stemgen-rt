#include <StemgenRT/Constants.h>
#include <StemgenRT/OnnxRuntime.h>
#include <StemgenRT/QualifiedModelContract.h>
#include <StemgenRT/StreamingSampleRateAdapter.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <juce_core/juce_core.h>
#include <juce_cryptography/juce_cryptography.h>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace {

#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME

using AudioChunk = std::array<std::vector<float>, audio_plugin::kNumChannels>;
using SeparatedChunk =
    std::array<std::array<std::vector<float>, audio_plugin::kNumChannels>,
               audio_plugin::kNumStems>;

juce::File resolveModelPathForTestBinary() {
  // Keep in sync with PluginProcessor::prepareToPlay() model lookup contract.
  const juce::File executable =
      juce::File::getSpecialLocation(juce::File::currentExecutableFile);
  return executable.getParentDirectory().getParentDirectory().getChildFile(
      "Resources/model.onnx");
}

bool prepareRuntime(audio_plugin::OnnxRuntime& runtime,
                    std::string& failureMessage,
                    std::optional<int> intraOpThreadCount = std::nullopt) {
  if (!runtime.isInitialized()) {
    failureMessage = "ONNX Runtime did not initialize";
    return false;
  }

  const juce::File modelFile = resolveModelPathForTestBinary();
  if (!modelFile.existsAsFile()) {
    failureMessage =
        "Model file not found: " + modelFile.getFullPathName().toStdString();
    return false;
  }

  juce::String loadError;
  if (!runtime.loadModel(modelFile.getFullPathName(), loadError,
                         intraOpThreadCount)) {
    failureMessage = "Model load failed: " + loadError.toStdString();
    return false;
  }

  juce::String preparationError;
  if (!runtime.prepareForInference(preparationError)) {
    failureMessage =
        "Inference preparation failed: " + preparationError.toStdString();
    return false;
  }
  if (!runtime.isReadyForInference()) {
    failureMessage =
        "Inference preparation succeeded without publishing readiness";
    return false;
  }
  return true;
}

AudioChunk makeAudioChunk(std::int64_t firstSample) {
  AudioChunk chunk;
  for (auto& channel : chunk) {
    channel.resize(static_cast<size_t>(audio_plugin::kOutputChunkSize));
  }

  constexpr double kTwoPi = 6.28318530717958647692;
  constexpr double kSampleRate =
      static_cast<double>(audio_plugin::kModelSampleRate);
  for (int i = 0; i < audio_plugin::kOutputChunkSize; ++i) {
    const double sample = static_cast<double>(firstSample + i);
    const double time = sample / kSampleRate;
    chunk[0][static_cast<size_t>(i)] =
        static_cast<float>(0.17 * std::sin(kTwoPi * 173.0 * time) +
                           0.09 * std::cos(kTwoPi * 997.0 * time));
    chunk[1][static_cast<size_t>(i)] =
        static_cast<float>(-0.13 * std::cos(kTwoPi * 251.0 * time) +
                           0.07 * std::sin(kTwoPi * 1301.0 * time));
  }
  return chunk;
}

AudioChunk makeZeroAudioChunk() {
  AudioChunk chunk;
  for (auto& channel : chunk) {
    channel.assign(static_cast<size_t>(audio_plugin::kOutputChunkSize), 0.0f);
  }
  return chunk;
}

AudioChunk scaledAudioChunk(AudioChunk chunk, float gain) {
  for (auto& channel : chunk) {
    for (float& sample : channel) {
      sample *= gain;
    }
  }
  return chunk;
}

AudioChunk scaledToStereoRms(AudioChunk chunk, float targetRms) {
  double sumSquares = 0.0;
  size_t sampleCount = 0;
  for (const auto& channel : chunk) {
    for (const float sample : channel) {
      const double value = static_cast<double>(sample);
      sumSquares += value * value;
      ++sampleCount;
    }
  }
  const double rms = std::sqrt(sumSquares / static_cast<double>(sampleCount));
  return scaledAudioChunk(
      std::move(chunk),
      static_cast<float>(static_cast<double>(targetRms) / rms));
}

SeparatedChunk makeSeparatedChunk() {
  SeparatedChunk chunk;
  for (auto& stem : chunk) {
    for (auto& channel : stem) {
      channel.resize(static_cast<size_t>(audio_plugin::kOutputChunkSize));
    }
  }
  return chunk;
}

float maxAbsoluteValue(const AudioChunk& chunk) {
  float maximum = 0.0f;
  for (const auto& channel : chunk) {
    for (const float value : channel) {
      if (!std::isfinite(value)) {
        return std::numeric_limits<float>::infinity();
      }
      maximum = std::max(maximum, std::abs(value));
    }
  }
  return maximum;
}

float maxAbsoluteValue(const SeparatedChunk& chunk) {
  float maximum = 0.0f;
  for (const auto& stem : chunk) {
    for (const auto& channel : stem) {
      for (const float value : channel) {
        if (!std::isfinite(value)) {
          return std::numeric_limits<float>::infinity();
        }
        maximum = std::max(maximum, std::abs(value));
      }
    }
  }
  return maximum;
}

float maxAbsoluteStemValue(const SeparatedChunk& chunk, size_t stemIndex) {
  float maximum = 0.0f;
  for (const auto& channel : chunk[stemIndex]) {
    for (const float value : channel) {
      if (!std::isfinite(value)) {
        return std::numeric_limits<float>::infinity();
      }
      maximum = std::max(maximum, std::abs(value));
    }
  }
  return maximum;
}

float maxChunkDifference(const AudioChunk& lhs, const AudioChunk& rhs) {
  float maximum = 0.0f;
  for (size_t ch = 0; ch < lhs.size(); ++ch) {
    EXPECT_EQ(lhs[ch].size(), rhs[ch].size());
    const size_t count = std::min(lhs[ch].size(), rhs[ch].size());
    for (size_t i = 0; i < count; ++i) {
      const float difference = std::abs(lhs[ch][i] - rhs[ch][i]);
      if (!std::isfinite(difference)) {
        return std::numeric_limits<float>::infinity();
      }
      maximum = std::max(maximum, difference);
    }
  }
  return maximum;
}

float maxChunkDifference(const SeparatedChunk& lhs, const SeparatedChunk& rhs) {
  float maximum = 0.0f;
  for (size_t stem = 0; stem < lhs.size(); ++stem) {
    for (size_t ch = 0; ch < lhs[stem].size(); ++ch) {
      EXPECT_EQ(lhs[stem][ch].size(), rhs[stem][ch].size());
      const size_t count = std::min(lhs[stem][ch].size(), rhs[stem][ch].size());
      for (size_t i = 0; i < count; ++i) {
        const float difference = std::abs(lhs[stem][ch][i] - rhs[stem][ch][i]);
        if (!std::isfinite(difference)) {
          return std::numeric_limits<float>::infinity();
        }
        maximum = std::max(maximum, difference);
      }
    }
  }
  return maximum;
}

float maxMixtureReconstructionError(const SeparatedChunk& separated,
                                    const AudioChunk& alignedInput) {
  float maximum = 0.0f;
  for (size_t ch = 0; ch < alignedInput.size(); ++ch) {
    for (size_t i = 0; i < alignedInput[ch].size(); ++i) {
      float reconstructed = 0.0f;
      for (size_t stem = 0; stem < separated.size(); ++stem) {
        reconstructed += separated[stem][ch][i];
      }
      const float difference = std::abs(reconstructed - alignedInput[ch][i]);
      if (!std::isfinite(difference)) {
        return std::numeric_limits<float>::infinity();
      }
      maximum = std::max(maximum, difference);
    }
  }
  return maximum;
}

double percentileFromSorted(const std::vector<double>& sorted,
                            double percentile) {
  if (sorted.empty())
    return 0.0;
  const double position = percentile * static_cast<double>(sorted.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

struct CpuHopTimingSummary {
  double meanMilliseconds{0.0};
  double p50Milliseconds{0.0};
  double p95Milliseconds{0.0};
  double p99Milliseconds{0.0};
  double maximumMilliseconds{0.0};
  size_t deadlineMisses{0};
};

CpuHopTimingSummary summarizeCpuHopTimings(
    const std::vector<double>& runMilliseconds,
    double hopBudgetMilliseconds) {
  CpuHopTimingSummary summary;
  if (runMilliseconds.empty()) {
    return summary;
  }

  std::vector<double> sorted = runMilliseconds;
  std::sort(sorted.begin(), sorted.end());
  summary.meanMilliseconds =
      std::accumulate(sorted.begin(), sorted.end(), 0.0) /
      static_cast<double>(sorted.size());
  summary.p50Milliseconds = percentileFromSorted(sorted, 0.50);
  summary.p95Milliseconds = percentileFromSorted(sorted, 0.95);
  summary.p99Milliseconds = percentileFromSorted(sorted, 0.99);
  summary.maximumMilliseconds = sorted.back();
  summary.deadlineMisses = static_cast<size_t>(std::count_if(
      sorted.begin(), sorted.end(), [hopBudgetMilliseconds](double run) {
        return run > hopBudgetMilliseconds;
      }));
  return summary;
}

#endif

}  // namespace

TEST(QualifiedModelContractTest,
     GeneratedContractMatchesBundledArtifactAndRuntimeConstants) {
#if !(defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME)
  GTEST_SKIP() << "ONNX Runtime support not compiled";
#else
  namespace contract = audio_plugin::qualified_model;

  EXPECT_EQ(audio_plugin::kModelSampleRate, contract::kSampleRate);
  EXPECT_EQ(audio_plugin::kNumChannels, contract::kNumChannels);
  EXPECT_EQ(audio_plugin::kNumStems, contract::kNumStems);
  EXPECT_EQ(audio_plugin::kOutputChunkSize, contract::kHopSamples);
  EXPECT_EQ(audio_plugin::kAnalysisWindowSize,
            contract::kAnalysisWindowSamples);
  EXPECT_EQ(audio_plugin::kFusionHiddenLayers, contract::kFusionHiddenLayers);
  EXPECT_EQ(audio_plugin::kFusionHiddenSize, contract::kFusionHiddenSize);
  EXPECT_EQ(audio_plugin::kStemDrums, contract::kDrumsSourceIndex);
  EXPECT_EQ(audio_plugin::kStemBass, contract::kBassSourceIndex);
  EXPECT_EQ(audio_plugin::kStemVocals, contract::kVocalsSourceIndex);
  EXPECT_EQ(audio_plugin::kStemOther, contract::kOtherSourceIndex);
  EXPECT_EQ(contract::kOtherSourceIndex, contract::kResidualSourceIndex);
  EXPECT_EQ(audio_plugin::kModelOutputDelayChunks,
            contract::kModelOutputDelayChunks);

  ASSERT_EQ(contract::kInputNames.size(), 5U);
  ASSERT_EQ(contract::kOutputNames.size(), 5U);
  ASSERT_EQ(contract::kMetadata.size(), 46U);
  for (const std::string_view name : contract::kInputNames) {
    EXPECT_FALSE(name.empty());
  }
  for (const std::string_view name : contract::kOutputNames) {
    EXPECT_FALSE(name.empty());
  }
  for (const auto& [key, value] : contract::kMetadata) {
    EXPECT_FALSE(key.empty());
    EXPECT_FALSE(value.empty());
  }
  EXPECT_EQ(contract::kOutputAlignment, "previous_input_chunk");

  const juce::File modelFile = resolveModelPathForTestBinary();
  ASSERT_TRUE(modelFile.existsAsFile())
      << modelFile.getFullPathName().toStdString();
  EXPECT_EQ(modelFile.getSize(), contract::kModelByteSize);
  const juce::String actualSha = juce::SHA256(modelFile).toHexString();
  const juce::String expectedSha =
      juce::String::fromUTF8(contract::kModelSha256.data(),
                             static_cast<int>(contract::kModelSha256.size()));
  EXPECT_EQ(actualSha, expectedSha);
#endif
}

TEST(OrtStreamingRuntimeTest, ReadinessRequiresSuccessfulStreamingPreparation) {
#if !(defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME)
  GTEST_SKIP() << "ONNX Runtime support not compiled";
#else
  audio_plugin::OnnxRuntime runtime;
  ASSERT_TRUE(runtime.isInitialized());
  EXPECT_FALSE(runtime.isModelLoaded());
  EXPECT_FALSE(runtime.isReadyForInference());

  juce::String preparationError;
  EXPECT_FALSE(runtime.prepareForInference(preparationError));
  EXPECT_TRUE(preparationError.containsIgnoreCase("model is not loaded"));
  EXPECT_FALSE(runtime.isReadyForInference());
  EXPECT_TRUE(
      runtime.getStatusString().containsIgnoreCase("preparation error"));

  const juce::File modelFile = resolveModelPathForTestBinary();
  ASSERT_TRUE(modelFile.existsAsFile());
  juce::String loadError;
  ASSERT_TRUE(runtime.loadModel(modelFile.getFullPathName(), loadError))
      << loadError;
  ASSERT_TRUE(runtime.isModelLoaded());
  EXPECT_FALSE(runtime.isReadyForInference());
  EXPECT_TRUE(runtime.getStatusString().containsIgnoreCase("not prepared"));

  const AudioChunk input = makeAudioChunk(0);
  SeparatedChunk separated = makeSeparatedChunk();
  AudioChunk alignedInput = makeZeroAudioChunk();
  bool outputValid = true;
  EXPECT_FALSE(
      runtime.runInference(input, separated, alignedInput, outputValid));
  EXPECT_FALSE(outputValid);

  preparationError = "sentinel";
  ASSERT_TRUE(runtime.prepareForInference(preparationError))
      << preparationError;
  EXPECT_TRUE(preparationError.isEmpty());
  EXPECT_TRUE(runtime.isReadyForInference());
  EXPECT_TRUE(runtime.getStatusString().containsIgnoreCase("loaded"));

  ASSERT_TRUE(
      runtime.runInference(input, separated, alignedInput, outputValid));
  EXPECT_FALSE(outputValid);
  EXPECT_FLOAT_EQ(maxAbsoluteValue(alignedInput), 0.0f);
  EXPECT_FLOAT_EQ(maxAbsoluteValue(separated), 0.0f);
  EXPECT_TRUE(runtime.isReadyForInference());
#endif
}

TEST(OrtStreamingRuntimeTest, ExplicitIntraOpThreadOverrideMustBePositive) {
#if !(defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME)
  GTEST_SKIP() << "ONNX Runtime support not compiled";
#else
  audio_plugin::OnnxRuntime runtime;
  ASSERT_TRUE(runtime.isInitialized());
  const juce::File modelFile = resolveModelPathForTestBinary();
  ASSERT_TRUE(modelFile.existsAsFile());

  juce::String loadError;
  EXPECT_FALSE(runtime.loadModel(modelFile.getFullPathName(), loadError, 0));
  EXPECT_TRUE(loadError.containsIgnoreCase("must be positive"));
  EXPECT_FALSE(runtime.isModelLoaded());

  loadError.clear();
  EXPECT_FALSE(runtime.loadModel(modelFile.getFullPathName(), loadError, -1));
  EXPECT_TRUE(loadError.containsIgnoreCase("must be positive"));
  EXPECT_FALSE(runtime.isModelLoaded());

  loadError.clear();
  const int aboveProductionCap =
      audio_plugin::kOrtAutomaticIntraOpThreadCap + 1;
  ASSERT_TRUE(runtime.loadModel(modelFile.getFullPathName(), loadError,
                                aboveProductionCap))
      << loadError;
  ASSERT_TRUE(runtime.isModelLoaded());
  juce::String preparationError;
  ASSERT_TRUE(runtime.prepareForInference(preparationError))
      << preparationError;
  EXPECT_TRUE(runtime.isReadyForInference());
#endif
}

TEST(OrtStreamingRuntimeTest,
     StatefulSequenceHonorsPrerollFlushResetAndMixtureSum) {
#if !(defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME)
  GTEST_SKIP() << "ONNX Runtime support not compiled";
#else
  audio_plugin::OnnxRuntime runtime;
  std::string failureMessage;
  ASSERT_TRUE(prepareRuntime(runtime, failureMessage)) << failureMessage;

  const AudioChunk first = makeAudioChunk(0);
  const AudioChunk second = makeAudioChunk(audio_plugin::kOutputChunkSize);
  const AudioChunk zero = makeZeroAudioChunk();
  SeparatedChunk separated = makeSeparatedChunk();
  AudioChunk alignedInput = makeZeroAudioChunk();
  bool outputValid = true;

  ASSERT_TRUE(
      runtime.runInference(first, separated, alignedInput, outputValid));
  EXPECT_FALSE(outputValid);
  EXPECT_FLOAT_EQ(maxAbsoluteValue(alignedInput), 0.0f);
  EXPECT_FLOAT_EQ(maxAbsoluteValue(separated), 0.0f);

  ASSERT_TRUE(
      runtime.runInference(second, separated, alignedInput, outputValid));
  ASSERT_TRUE(outputValid);
  EXPECT_FLOAT_EQ(maxChunkDifference(alignedInput, first), 0.0f);
  EXPECT_LE(maxMixtureReconstructionError(separated, alignedInput), 1.0e-6f);
  const SeparatedChunk firstResult = separated;

  // A single zero hop flushes the second real input hop.
  ASSERT_TRUE(runtime.runInference(zero, separated, alignedInput, outputValid));
  ASSERT_TRUE(outputValid);
  EXPECT_FLOAT_EQ(maxChunkDifference(alignedInput, second), 0.0f);
  EXPECT_LE(maxMixtureReconstructionError(separated, alignedInput), 1.0e-6f);

  runtime.resetStreamingState();

  ASSERT_TRUE(
      runtime.runInference(first, separated, alignedInput, outputValid));
  EXPECT_FALSE(outputValid);
  EXPECT_FLOAT_EQ(maxAbsoluteValue(alignedInput), 0.0f);
  EXPECT_FLOAT_EQ(maxAbsoluteValue(separated), 0.0f);

  ASSERT_TRUE(
      runtime.runInference(second, separated, alignedInput, outputValid));
  ASSERT_TRUE(outputValid);
  EXPECT_FLOAT_EQ(maxChunkDifference(alignedInput, first), 0.0f);
  EXPECT_LE(maxMixtureReconstructionError(separated, alignedInput), 1.0e-6f);
  EXPECT_LE(maxChunkDifference(separated, firstResult), 2.0e-5f);
#endif
}

TEST(OrtStreamingRuntimeTest,
     RawInputLevelsRemainExactFiniteAndMixtureConsistent) {
#if !(defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME)
  GTEST_SKIP() << "ONNX Runtime support not compiled";
#else
  audio_plugin::OnnxRuntime referenceRuntime;
  audio_plugin::OnnxRuntime quietRuntime;
  audio_plugin::OnnxRuntime veryQuietRuntime;
  std::string failureMessage;
  ASSERT_TRUE(prepareRuntime(referenceRuntime, failureMessage))
      << failureMessage;
  failureMessage.clear();
  ASSERT_TRUE(prepareRuntime(quietRuntime, failureMessage)) << failureMessage;
  failureMessage.clear();
  ASSERT_TRUE(prepareRuntime(veryQuietRuntime, failureMessage))
      << failureMessage;

  constexpr float kQuietScale = 0.31622776f;  // -10 dB
  constexpr float kVeryQuietScale = 0.1f;     // -20 dB

  SeparatedChunk referenceSeparated = makeSeparatedChunk();
  SeparatedChunk quietSeparated = makeSeparatedChunk();
  SeparatedChunk veryQuietSeparated = makeSeparatedChunk();
  AudioChunk referenceAligned = makeZeroAudioChunk();
  AudioChunk quietAligned = makeZeroAudioChunk();
  AudioChunk veryQuietAligned = makeZeroAudioChunk();
  bool referenceValid = false;
  bool quietValid = false;
  bool veryQuietValid = false;
  AudioChunk previousReferenceInput = makeZeroAudioChunk();
  AudioChunk previousQuietInput = makeZeroAudioChunk();
  AudioChunk previousVeryQuietInput = makeZeroAudioChunk();
  std::array<float, 3> referenceStemPeaks{};

  constexpr size_t kHopCount = 8;
  for (size_t hop = 0; hop < kHopCount; ++hop) {
    const auto firstSample =
        static_cast<std::int64_t>(hop) * audio_plugin::kOutputChunkSize;
    // The deployment wrapper must preserve each distinct input level instead
    // of remapping quiet copies into a shared model domain.
    const AudioChunk referenceInput = makeAudioChunk(firstSample);
    const AudioChunk quietInput = scaledAudioChunk(referenceInput, kQuietScale);
    const AudioChunk veryQuietInput =
        scaledAudioChunk(referenceInput, kVeryQuietScale);

    ASSERT_TRUE(referenceRuntime.runInference(
        referenceInput, referenceSeparated, referenceAligned, referenceValid));
    ASSERT_TRUE(quietRuntime.runInference(quietInput, quietSeparated,
                                          quietAligned, quietValid));
    ASSERT_TRUE(veryQuietRuntime.runInference(
        veryQuietInput, veryQuietSeparated, veryQuietAligned, veryQuietValid));
    ASSERT_EQ(quietValid, referenceValid);
    ASSERT_EQ(veryQuietValid, referenceValid);
    if (!referenceValid) {
      EXPECT_EQ(hop, 0U);
      EXPECT_FLOAT_EQ(maxAbsoluteValue(referenceAligned), 0.0f);
      EXPECT_FLOAT_EQ(maxAbsoluteValue(referenceSeparated), 0.0f);
      previousReferenceInput = referenceInput;
      previousQuietInput = quietInput;
      previousVeryQuietInput = veryQuietInput;
      continue;
    }

    EXPECT_FLOAT_EQ(
        maxChunkDifference(referenceAligned, previousReferenceInput), 0.0f);
    EXPECT_FLOAT_EQ(maxChunkDifference(quietAligned, previousQuietInput),
                    0.0f);
    EXPECT_FLOAT_EQ(
        maxChunkDifference(veryQuietAligned, previousVeryQuietInput), 0.0f);
    EXPECT_TRUE(std::isfinite(maxAbsoluteValue(referenceSeparated)));
    EXPECT_TRUE(std::isfinite(maxAbsoluteValue(quietSeparated)));
    EXPECT_TRUE(std::isfinite(maxAbsoluteValue(veryQuietSeparated)));
    EXPECT_LE(
        maxMixtureReconstructionError(referenceSeparated, referenceAligned),
        1.0e-6f);
    EXPECT_LE(maxMixtureReconstructionError(quietSeparated, quietAligned),
              1.0e-6f);
    EXPECT_LE(
        maxMixtureReconstructionError(veryQuietSeparated, veryQuietAligned),
        1.0e-6f);
    for (size_t stem = 0; stem < referenceStemPeaks.size(); ++stem) {
      referenceStemPeaks[stem] =
          std::max(referenceStemPeaks[stem],
                   maxAbsoluteStemValue(referenceSeparated, stem));
    }
    previousReferenceInput = referenceInput;
    previousQuietInput = quietInput;
    previousVeryQuietInput = veryQuietInput;
  }

  // One zero hop flushes the last real input without changing its raw level.
  const AudioChunk zero = makeZeroAudioChunk();
  ASSERT_TRUE(referenceRuntime.runInference(zero, referenceSeparated,
                                            referenceAligned, referenceValid));
  ASSERT_TRUE(quietRuntime.runInference(zero, quietSeparated, quietAligned,
                                        quietValid));
  ASSERT_TRUE(veryQuietRuntime.runInference(zero, veryQuietSeparated,
                                            veryQuietAligned, veryQuietValid));
  ASSERT_TRUE(referenceValid);
  ASSERT_TRUE(quietValid);
  ASSERT_TRUE(veryQuietValid);
  EXPECT_FLOAT_EQ(maxChunkDifference(referenceAligned, previousReferenceInput),
                  0.0f);
  EXPECT_FLOAT_EQ(maxChunkDifference(quietAligned, previousQuietInput), 0.0f);
  EXPECT_FLOAT_EQ(
      maxChunkDifference(veryQuietAligned, previousVeryQuietInput), 0.0f);
  EXPECT_LE(maxMixtureReconstructionError(referenceSeparated, referenceAligned),
            1.0e-6f);
  EXPECT_LE(maxMixtureReconstructionError(quietSeparated, quietAligned),
            1.0e-6f);
  EXPECT_LE(
      maxMixtureReconstructionError(veryQuietSeparated, veryQuietAligned),
      1.0e-6f);
  for (size_t stem = 0; stem < referenceStemPeaks.size(); ++stem) {
    referenceStemPeaks[stem] =
        std::max(referenceStemPeaks[stem],
                 maxAbsoluteStemValue(referenceSeparated, stem));
    EXPECT_GT(referenceStemPeaks[stem], 1.0e-4f)
        << "Retained reference stem " << stem << " was degenerate";
  }
#endif
}

TEST(OrtStreamingRuntimeTest,
     LowFrequencyBassHopSeamIsMeasuredAndBoundedForListeningOnly) {
#if !(defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME)
  GTEST_SKIP() << "ONNX Runtime support not compiled";
#else
  audio_plugin::OnnxRuntime runtime;
  std::string failureMessage;
  ASSERT_TRUE(prepareRuntime(runtime, failureMessage)) << failureMessage;

  SeparatedChunk separated = makeSeparatedChunk();
  AudioChunk aligned = makeZeroAudioChunk();
  bool outputValid = false;
  constexpr size_t kWarmupHops = 12;
  constexpr size_t kMeasuredHops = 32;
  constexpr double kFrequencyHz = 50.0;
  constexpr float kInputPeak = 0.12589254f;  // -18 dBFS
  constexpr double kTwoPi = 6.28318530717958647692;

  constexpr std::array<size_t, 2> kMeasuredStemIndices = {0, 1};
  std::array<double, kMeasuredStemIndices.size()> boundarySumSquares{};
  std::array<double, kMeasuredStemIndices.size()> internalSumSquares{};
  std::array<size_t, kMeasuredStemIndices.size()> boundaryCounts{};
  std::array<size_t, kMeasuredStemIndices.size()> internalCounts{};
  std::array<std::array<float, audio_plugin::kNumChannels>,
             kMeasuredStemIndices.size()>
      previousLastSamples{};
  bool havePreviousStems = false;

  for (size_t hop = 0; hop < kWarmupHops + kMeasuredHops; ++hop) {
    AudioChunk input = makeZeroAudioChunk();
    for (size_t i = 0;
         i < static_cast<size_t>(audio_plugin::kOutputChunkSize); ++i) {
      const auto sampleIndex =
          hop * static_cast<size_t>(audio_plugin::kOutputChunkSize) + i;
      const double phase =
          kTwoPi * kFrequencyHz * static_cast<double>(sampleIndex) /
          static_cast<double>(audio_plugin::kModelSampleRate);
      const float sample = kInputPeak * static_cast<float>(std::sin(phase));
      input[0][i] = sample;
      input[1][i] = sample;
    }

    ASSERT_TRUE(
        runtime.runInference(input, separated, aligned, outputValid));
    if (hop == 0U) {
      ASSERT_FALSE(outputValid);
      continue;
    }
    ASSERT_TRUE(outputValid);
    if (hop >= kWarmupHops) {
      for (size_t measuredStem = 0;
           measuredStem < kMeasuredStemIndices.size(); ++measuredStem) {
        const size_t stem = kMeasuredStemIndices[measuredStem];
        for (size_t ch = 0;
             ch < static_cast<size_t>(audio_plugin::kNumChannels); ++ch) {
          const auto& channel = separated[stem][ch];
          ASSERT_EQ(channel.size(),
                    static_cast<size_t>(audio_plugin::kOutputChunkSize));
          if (havePreviousStems) {
            const double delta = static_cast<double>(
                channel[0] - previousLastSamples[measuredStem][ch]);
            boundarySumSquares[measuredStem] += delta * delta;
            ++boundaryCounts[measuredStem];
          }
          for (size_t i = 1; i < channel.size(); ++i) {
            const double delta =
                static_cast<double>(channel[i] - channel[i - 1]);
            internalSumSquares[measuredStem] += delta * delta;
            ++internalCounts[measuredStem];
          }
          previousLastSamples[measuredStem][ch] = channel.back();
        }
      }
      havePreviousStems = true;
    }
  }

  // c91 owns the synthesis overlap-add, so a low tone must not acquire the
  // large periodic boundary derivative observed in the no-OLA c126 graph.
  // These are listening-regression ceilings; target-hardware promotion also
  // compares p95/p99 target-relative boundary errors to the frozen c91 corpus.
  for (size_t measuredStem = 0;
       measuredStem < kMeasuredStemIndices.size(); ++measuredStem) {
    ASSERT_GT(boundaryCounts[measuredStem], 0U);
    ASSERT_GT(internalCounts[measuredStem], 0U);
    const double boundaryRms = std::sqrt(
        boundarySumSquares[measuredStem] /
        static_cast<double>(boundaryCounts[measuredStem]));
    const double internalRms =
        std::sqrt(internalSumSquares[measuredStem] /
                  static_cast<double>(internalCounts[measuredStem]));
    ASSERT_GT(internalRms, 0.0);
    const double seamRatio = boundaryRms / internalRms;
    EXPECT_LT(seamRatio, 4.0)
        << "Stem index " << kMeasuredStemIndices[measuredStem];
    EXPECT_LT(boundaryRms / static_cast<double>(kInputPeak), 0.05)
        << "Stem index " << kMeasuredStemIndices[measuredStem];
  }
#endif
}

TEST(OrtStreamingRuntimeTest,
     AlternatingRawLevelsAndSparseTransientsRemainFinite) {
#if !(defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME)
  GTEST_SKIP() << "ONNX Runtime support not compiled";
#else
  audio_plugin::OnnxRuntime referenceRuntime;
  audio_plugin::OnnxRuntime quietRuntime;
  std::string failureMessage;
  ASSERT_TRUE(prepareRuntime(referenceRuntime, failureMessage))
      << failureMessage;
  failureMessage.clear();
  ASSERT_TRUE(prepareRuntime(quietRuntime, failureMessage)) << failureMessage;

  constexpr float kQuietScale = 0.25f;
  std::vector<AudioChunk> referenceInputs;
  referenceInputs.reserve(8);
  referenceInputs.push_back(scaledToStereoRms(makeAudioChunk(0), 0.04f));
  referenceInputs.push_back(
      scaledToStereoRms(makeAudioChunk(audio_plugin::kOutputChunkSize), 0.12f));

  AudioChunk firstTransient = makeZeroAudioChunk();
  firstTransient[0][31] = 0.8f;
  firstTransient[1][233] = -0.7f;
  referenceInputs.push_back(firstTransient);

  referenceInputs.push_back(scaledToStereoRms(
      makeAudioChunk(3 * audio_plugin::kOutputChunkSize), 0.015f));
  referenceInputs.push_back(scaledToStereoRms(
      makeAudioChunk(4 * audio_plugin::kOutputChunkSize), 0.08f));

  AudioChunk secondTransient = makeZeroAudioChunk();
  secondTransient[0][73] = -0.6f;
  secondTransient[1].at(401U % audio_plugin::kOutputChunkSize) = 0.75f;
  referenceInputs.push_back(secondTransient);

  referenceInputs.push_back(scaledToStereoRms(
      makeAudioChunk(6 * audio_plugin::kOutputChunkSize), 0.025f));
  referenceInputs.push_back(scaledToStereoRms(
      makeAudioChunk(7 * audio_plugin::kOutputChunkSize), 0.14f));

  SeparatedChunk referenceSeparated = makeSeparatedChunk();
  SeparatedChunk quietSeparated = makeSeparatedChunk();
  AudioChunk referenceAligned = makeZeroAudioChunk();
  AudioChunk quietAligned = makeZeroAudioChunk();
  bool referenceValid = false;
  bool quietValid = false;
  AudioChunk previousReferenceInput = makeZeroAudioChunk();
  AudioChunk previousQuietInput = makeZeroAudioChunk();
  bool havePreviousInput = false;
  std::array<float, 3> retainedStemPeaks{};

  const auto verifyResult = [&](const AudioChunk& expectedReference,
                                const AudioChunk& expectedQuiet,
                                bool expectedValid) {
    ASSERT_EQ(referenceValid, expectedValid);
    ASSERT_EQ(quietValid, expectedValid);
    if (!expectedValid) {
      EXPECT_FLOAT_EQ(maxAbsoluteValue(referenceAligned), 0.0f);
      EXPECT_FLOAT_EQ(maxAbsoluteValue(referenceSeparated), 0.0f);
      return;
    }
    EXPECT_FLOAT_EQ(maxChunkDifference(referenceAligned, expectedReference),
                    0.0f);
    EXPECT_FLOAT_EQ(maxChunkDifference(quietAligned, expectedQuiet), 0.0f);
    EXPECT_LE(
        maxMixtureReconstructionError(referenceSeparated, referenceAligned),
        1.0e-6f);
    EXPECT_LE(maxMixtureReconstructionError(quietSeparated, quietAligned),
              1.0e-6f);
    EXPECT_TRUE(std::isfinite(maxAbsoluteValue(referenceSeparated)));
    EXPECT_TRUE(std::isfinite(maxAbsoluteValue(quietSeparated)));
    for (size_t stem = 0; stem < retainedStemPeaks.size(); ++stem) {
      retainedStemPeaks[stem] =
          std::max(retainedStemPeaks[stem],
                   maxAbsoluteStemValue(referenceSeparated, stem));
    }
  };

  for (const AudioChunk& referenceInput : referenceInputs) {
    const AudioChunk quietInput = scaledAudioChunk(referenceInput, kQuietScale);
    ASSERT_TRUE(referenceRuntime.runInference(
        referenceInput, referenceSeparated, referenceAligned, referenceValid));
    ASSERT_TRUE(quietRuntime.runInference(quietInput, quietSeparated,
                                          quietAligned, quietValid));
    verifyResult(previousReferenceInput, previousQuietInput,
                 havePreviousInput);
    previousReferenceInput = referenceInput;
    previousQuietInput = quietInput;
    havePreviousInput = true;
  }

  // Flush the final real hop with one zero input.
  const AudioChunk zero = makeZeroAudioChunk();
  ASSERT_TRUE(referenceRuntime.runInference(zero, referenceSeparated,
                                            referenceAligned, referenceValid));
  ASSERT_TRUE(quietRuntime.runInference(zero, quietSeparated, quietAligned,
                                        quietValid));
  ASSERT_TRUE(referenceValid);
  ASSERT_TRUE(quietValid);
  verifyResult(previousReferenceInput, previousQuietInput, true);

  for (size_t stem = 0; stem < retainedStemPeaks.size(); ++stem) {
    EXPECT_GT(retainedStemPeaks[stem], 1.0e-5f)
        << "Retained reference stem " << stem << " was degenerate";
  }
#endif
}

TEST(OrtStreamingRuntimeTest,
     NonFiniteInputFailsClosedAndRestartsWithDeterministicPreroll) {
#if !(defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME)
  GTEST_SKIP() << "ONNX Runtime support not compiled";
#else
  audio_plugin::OnnxRuntime runtime;
  audio_plugin::OnnxRuntime baselineRuntime;
  std::string failureMessage;
  ASSERT_TRUE(prepareRuntime(runtime, failureMessage)) << failureMessage;
  failureMessage.clear();
  ASSERT_TRUE(prepareRuntime(baselineRuntime, failureMessage))
      << failureMessage;

  const AudioChunk first = makeAudioChunk(0);
  const AudioChunk second = makeAudioChunk(audio_plugin::kOutputChunkSize);
  SeparatedChunk separated = makeSeparatedChunk();
  SeparatedChunk baselineSeparated = makeSeparatedChunk();
  AudioChunk alignedInput = makeZeroAudioChunk();
  AudioChunk baselineAligned = makeZeroAudioChunk();
  bool outputValid = true;
  bool baselineValid = true;

  ASSERT_TRUE(baselineRuntime.runInference(first, baselineSeparated,
                                           baselineAligned, baselineValid));
  ASSERT_FALSE(baselineValid);
  ASSERT_TRUE(baselineRuntime.runInference(second, baselineSeparated,
                                           baselineAligned, baselineValid));
  ASSERT_TRUE(baselineValid);
  ASSERT_FLOAT_EQ(maxChunkDifference(baselineAligned, first), 0.0f);
  const SeparatedChunk baselineFirstSeparated = baselineSeparated;
  const AudioChunk zero = makeZeroAudioChunk();
  ASSERT_TRUE(baselineRuntime.runInference(
      zero, baselineSeparated, baselineAligned, baselineValid));
  ASSERT_TRUE(baselineValid);
  ASSERT_FLOAT_EQ(maxChunkDifference(baselineAligned, second), 0.0f);
  const SeparatedChunk baselineSecondSeparated = baselineSeparated;

  for (const float invalidValue : {std::numeric_limits<float>::quiet_NaN(),
                                   std::numeric_limits<float>::infinity(),
                                   -std::numeric_limits<float>::infinity()}) {
    runtime.resetStreamingState();
    ASSERT_TRUE(
        runtime.runInference(first, separated, alignedInput, outputValid));
    ASSERT_FALSE(outputValid);
    EXPECT_FLOAT_EQ(maxAbsoluteValue(alignedInput), 0.0f);

    AudioChunk invalid = second;
    invalid[1][17] = invalidValue;
    outputValid = true;
    EXPECT_FALSE(
        runtime.runInference(invalid, separated, alignedInput, outputValid));
    EXPECT_FALSE(outputValid);

    ASSERT_TRUE(
        runtime.runInference(first, separated, alignedInput, outputValid));
    EXPECT_FALSE(outputValid);
    EXPECT_FLOAT_EQ(maxAbsoluteValue(alignedInput), 0.0f);
    ASSERT_TRUE(
        runtime.runInference(second, separated, alignedInput, outputValid));
    ASSERT_TRUE(outputValid);
    EXPECT_FLOAT_EQ(maxChunkDifference(alignedInput, first), 0.0f);
    EXPECT_LE(maxChunkDifference(separated, baselineFirstSeparated), 2.0e-5f);
    ASSERT_TRUE(
        runtime.runInference(zero, separated, alignedInput, outputValid));
    ASSERT_TRUE(outputValid);
    EXPECT_FLOAT_EQ(maxChunkDifference(alignedInput, second), 0.0f);
    EXPECT_LE(maxChunkDifference(separated, baselineSecondSeparated), 2.0e-5f);
  }
#endif
}

TEST(OrtStreamingRuntimeTest, UsesBundledRuntimeWhenASystemOrtIsAlreadyLoaded) {
#if !(defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME)
  GTEST_SKIP() << "ONNX Runtime support not compiled";
#elif !defined(_WIN32)
  GTEST_SKIP() << "Windows DLL-isolation test";
#else
  // Some Windows installations expose an older onnxruntime.dll in System32.
  // Preload it deliberately, then require StemgenRT to resolve the exact DLL
  // bundled beside this test binary instead of reusing the process-global one.
  wchar_t systemDirectory[MAX_PATH] = {};
  const UINT directoryLength = GetSystemDirectoryW(systemDirectory, MAX_PATH);
  HMODULE competingOrt = nullptr;
  if (directoryLength > 0 && directoryLength < MAX_PATH) {
    std::wstring systemOrtPath(systemDirectory, directoryLength);
    systemOrtPath += L"\\onnxruntime.dll";
    competingOrt = LoadLibraryW(systemOrtPath.c_str());
  }

  {
    audio_plugin::OnnxRuntime runtime;
    std::string failureMessage;
    ASSERT_TRUE(prepareRuntime(runtime, failureMessage)) << failureMessage;
    EXPECT_EQ(runtime.getRuntimeVersion(), "1.26.0");
    EXPECT_EQ(runtime.getExecutionProvider(), "CPU");
  }

  if (competingOrt != nullptr) {
    FreeLibrary(competingOrt);
  }
#endif
}

TEST(OrtStreamingRuntimeTest, DISABLED_BenchmarkStatefulCpuPerHop) {
#if !(defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME)
  GTEST_SKIP() << "ONNX Runtime support not compiled";
#else
  audio_plugin::OnnxRuntime runtime;
  std::string failureMessage;
  ASSERT_TRUE(prepareRuntime(runtime, failureMessage)) << failureMessage;
  if (runtime.getExecutionProvider() != "CPU") {
    GTEST_SKIP() << "CPU timing requested; active provider is "
                 << runtime.getExecutionProvider();
  }

  SeparatedChunk separated = makeSeparatedChunk();
  AudioChunk alignedInput = makeZeroAudioChunk();
  bool outputValid = false;

  constexpr int kWarmupIterations = 5;
  constexpr int kMeasureIterations = 500;
  const AudioChunk warmupSeed = makeAudioChunk(0);
  ASSERT_TRUE(
      runtime.runInference(warmupSeed, separated, alignedInput, outputValid));
  ASSERT_FALSE(outputValid);

  std::vector<double> runMilliseconds;
  runMilliseconds.reserve(static_cast<size_t>(kMeasureIterations));
  for (int iteration = 0; iteration < kWarmupIterations + kMeasureIterations;
       ++iteration) {
    const AudioChunk input =
        makeAudioChunk(static_cast<std::int64_t>(iteration + 1) *
                       audio_plugin::kOutputChunkSize);
    const auto begin = std::chrono::steady_clock::now();
    ASSERT_TRUE(
        runtime.runInference(input, separated, alignedInput, outputValid));
    const auto end = std::chrono::steady_clock::now();
    ASSERT_TRUE(outputValid);
    ASSERT_LE(maxMixtureReconstructionError(separated, alignedInput), 1.0e-6f);

    if (iteration >= kWarmupIterations) {
      runMilliseconds.push_back(
          std::chrono::duration<double, std::milli>(end - begin).count());
    }
  }

  ASSERT_EQ(runMilliseconds.size(), static_cast<size_t>(kMeasureIterations));
  std::sort(runMilliseconds.begin(), runMilliseconds.end());
  const double mean =
      std::accumulate(runMilliseconds.begin(), runMilliseconds.end(), 0.0) /
      static_cast<double>(runMilliseconds.size());
  const double hopBudgetMilliseconds =
      1000.0 * static_cast<double>(audio_plugin::kOutputChunkSize) /
      static_cast<double>(audio_plugin::kModelSampleRate);
  const auto deadlineMisses = static_cast<size_t>(
      std::count_if(runMilliseconds.begin(), runMilliseconds.end(),
                    [hopBudgetMilliseconds](double run) {
                      return run > hopBudgetMilliseconds;
                    }));

  std::cerr << "\nStateful CPU hop timing (ORT " << runtime.getRuntimeVersion()
            << "): mean=" << mean
            << " ms, p50=" << percentileFromSorted(runMilliseconds, 0.50)
            << " ms, p95=" << percentileFromSorted(runMilliseconds, 0.95)
            << " ms, p99=" << percentileFromSorted(runMilliseconds, 0.99)
            << " ms, max=" << runMilliseconds.back()
            << " ms, hop budget=" << hopBudgetMilliseconds
            << " ms, deadline misses=" << deadlineMisses << "/"
            << runMilliseconds.size() << "\n";

  EXPECT_GT(mean, 0.0);
  EXPECT_LT(percentileFromSorted(runMilliseconds, 0.95), hopBudgetMilliseconds);
#endif
}

TEST(OrtStreamingRuntimeTest,
     DISABLED_BenchmarkStatefulCpuWithPreservedSampleRateBridges) {
#if !(defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME)
  GTEST_SKIP() << "ONNX Runtime support not compiled";
#else
  audio_plugin::OnnxRuntime runtime;
  std::string failureMessage;
  ASSERT_TRUE(prepareRuntime(runtime, failureMessage)) << failureMessage;
  ASSERT_EQ(runtime.getExecutionProvider(), "CPU");

  constexpr int kWarmupIterations = 10;
  constexpr int kMeasureIterations = 500;
  constexpr std::array<int, 3> kHostRates = {48000, 96000, 192000};
  constexpr std::array<int, 3> kRetainedStems = {
      audio_plugin::kStemDrums,
      audio_plugin::kStemBass,
      audio_plugin::kStemVocals,
  };
  const double hopBudgetMilliseconds =
      1000.0 * static_cast<double>(audio_plugin::kOutputChunkSize) /
      static_cast<double>(audio_plugin::kModelSampleRate);

  for (const int hostRate : kHostRates) {
    SCOPED_TRACE("hostRate=" + std::to_string(hostRate));
    audio_plugin::StreamingSampleRateAdapter inputAdapter;
    audio_plugin::StreamingSampleRateAdapter outputAdapter;
    ASSERT_TRUE(inputAdapter.prepare(
        static_cast<double>(hostRate),
        static_cast<double>(audio_plugin::kModelSampleRate),
        static_cast<size_t>(audio_plugin::kNumChannels)));
    ASSERT_TRUE(outputAdapter.prepare(
        static_cast<double>(audio_plugin::kModelSampleRate),
        static_cast<double>(hostRate),
        static_cast<size_t>(3 * audio_plugin::kNumChannels)));
    const double naturalDelay =
        inputAdapter.filterGroupDelaySeconds() * static_cast<double>(hostRate) +
        outputAdapter.filterGroupDelayOutputSamples();
    const double extraDelay = std::ceil(naturalDelay - 1.0e-9) - naturalDelay;
    ASSERT_TRUE(outputAdapter.prepare(
        static_cast<double>(audio_plugin::kModelSampleRate),
        static_cast<double>(hostRate),
        static_cast<size_t>(3 * audio_plugin::kNumChannels), extraDelay));

    const size_t hostFramesPerHop = static_cast<size_t>(
        std::ceil(static_cast<double>(audio_plugin::kOutputChunkSize) *
                  static_cast<double>(hostRate) /
                  static_cast<double>(audio_plugin::kModelSampleRate)));
    std::array<std::vector<float>, audio_plugin::kNumChannels> hostInput;
    std::array<std::vector<float>, audio_plugin::kNumChannels> modelScratch;
    for (auto& channel : hostInput) {
      channel.assign(hostFramesPerHop, 0.1f);
    }
    const size_t modelScratchCapacity =
        inputAdapter.maxOutputForInput(hostFramesPerHop) + 1U;
    for (auto& channel : modelScratch) {
      channel.assign(modelScratchCapacity, 0.0f);
    }
    const std::array<const float*, audio_plugin::kNumChannels>
        hostInputPointers = {hostInput[0].data(), hostInput[1].data()};
    const std::array<float*, audio_plugin::kNumChannels> modelScratchPointers =
        {modelScratch[0].data(), modelScratch[1].data()};

    SeparatedChunk separated = makeSeparatedChunk();
    AudioChunk alignedInput = makeZeroAudioChunk();
    const AudioChunk modelInput = makeAudioChunk(0);
    bool outputValid = false;
    runtime.resetStreamingState();
    ASSERT_TRUE(
        runtime.runInference(modelInput, separated, alignedInput, outputValid));
    ASSERT_FALSE(outputValid);
    ASSERT_FLOAT_EQ(maxAbsoluteValue(alignedInput), 0.0f);

    const size_t hostOutputCapacity = hostFramesPerHop + 2U;
    std::array<std::vector<float>, 3 * audio_plugin::kNumChannels> hostOutput;
    for (auto& channel : hostOutput) {
      channel.assign(hostOutputCapacity, 0.0f);
    }
    std::array<const float*, 3 * audio_plugin::kNumChannels>
        modelOutputPointers{};
    std::array<float*, 3 * audio_plugin::kNumChannels> hostOutputPointers{};
    for (size_t retained = 0U; retained < kRetainedStems.size(); ++retained) {
      for (size_t channel = 0U;
           channel < static_cast<size_t>(audio_plugin::kNumChannels);
           ++channel) {
        const size_t flattened =
            retained * static_cast<size_t>(audio_plugin::kNumChannels) +
            channel;
        modelOutputPointers[flattened] =
            separated[static_cast<size_t>(kRetainedStems[retained])][channel]
                .data();
        hostOutputPointers[flattened] = hostOutput[flattened].data();
      }
    }

    std::vector<double> timings;
    timings.reserve(static_cast<size_t>(kMeasureIterations));
    for (int iteration = 0; iteration < kWarmupIterations + kMeasureIterations;
         ++iteration) {
      const auto begin = std::chrono::steady_clock::now();
      const auto inputConversion = inputAdapter.process(
          hostInputPointers.data(), hostFramesPerHop,
          modelScratchPointers.data(), modelScratchCapacity);
      const bool inferenceOk = runtime.runInference(modelInput, separated,
                                                    alignedInput, outputValid);
      const size_t requiredHostOutput = outputAdapter.maxOutputForInput(
          static_cast<size_t>(audio_plugin::kOutputChunkSize));
      const auto outputConversion = outputAdapter.process(
          modelOutputPointers.data(),
          static_cast<size_t>(audio_plugin::kOutputChunkSize),
          hostOutputPointers.data(), hostOutputCapacity);
      const auto end = std::chrono::steady_clock::now();

      ASSERT_TRUE(inputConversion.ok);
      ASSERT_TRUE(inferenceOk);
      ASSERT_TRUE(outputValid);
      ASSERT_FLOAT_EQ(maxChunkDifference(alignedInput, modelInput), 0.0f);
      ASSERT_TRUE(outputConversion.ok);
      ASSERT_EQ(outputConversion.outputProduced, requiredHostOutput);
      if (iteration >= kWarmupIterations) {
        timings.push_back(
            std::chrono::duration<double, std::milli>(end - begin).count());
      }
    }

    std::sort(timings.begin(), timings.end());
    const double mean = std::accumulate(timings.begin(), timings.end(), 0.0) /
                        static_cast<double>(timings.size());
    const size_t deadlineMisses = static_cast<size_t>(std::count_if(
        timings.begin(), timings.end(), [hopBudgetMilliseconds](double time) {
          return time > hopBudgetMilliseconds;
        }));
    std::cerr << "\nStateful CPU + SRC timing at " << hostRate
              << " Hz: mean=" << mean
              << " ms, p95=" << percentileFromSorted(timings, 0.95)
              << " ms, p99=" << percentileFromSorted(timings, 0.99)
              << " ms, max=" << timings.back()
              << " ms, deadline misses=" << deadlineMisses << "/"
              << timings.size() << "\n";
    EXPECT_LT(percentileFromSorted(timings, 0.95), hopBudgetMilliseconds);
  }
#endif
}

TEST(OrtStreamingRuntimeTest, DISABLED_BenchmarkStatefulCpuIntraOpThreadSweep) {
#if !(defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME)
  GTEST_SKIP() << "ONNX Runtime support not compiled";
#else
  constexpr int kWarmupIterations = 25;
  constexpr int kMeasureIterations = 500;
  constexpr int kPassCount = 3;
  constexpr std::array<std::array<int, 4>, kPassCount> kCandidateOrders{{
      {{1, 2, 3, 4}},
      {{4, 3, 2, 1}},
      {{2, 4, 1, 3}},
  }};
  constexpr std::array<int, 3> kRetainedStemIndices{
      audio_plugin::kStemDrums,
      audio_plugin::kStemBass,
      audio_plugin::kStemVocals,
  };

  const double hopBudgetMilliseconds =
      1000.0 * static_cast<double>(audio_plugin::kOutputChunkSize) /
      static_cast<double>(audio_plugin::kModelSampleRate);

  // Every fresh session sees the same stateful input sequence. Candidate
  // ordering changes each pass to reduce thermal and scheduler-order bias.
  std::vector<AudioChunk> workload;
  workload.reserve(
      static_cast<size_t>(1 + kWarmupIterations + kMeasureIterations));
  for (int hop = 0; hop < 1 + kWarmupIterations + kMeasureIterations; ++hop) {
    workload.push_back(makeAudioChunk(static_cast<std::int64_t>(hop) *
                                      audio_plugin::kOutputChunkSize));
  }

  struct CandidateMeasurements {
    std::vector<double> wallMilliseconds;
    std::vector<double> processCpuMilliseconds;
    std::array<double, kPassCount> passP99Milliseconds{};
    float maxRetainedStemDeltaFromOneThread{0.0f};
  };

  std::array<CandidateMeasurements, 4> measurements;
  for (auto& candidate : measurements) {
    candidate.wallMilliseconds.reserve(
        static_cast<size_t>(kPassCount * kMeasureIterations));
    candidate.processCpuMilliseconds.reserve(
        static_cast<size_t>(kPassCount * kMeasureIterations));
  }

  constexpr size_t kRetainedValuesPerResult =
      kRetainedStemIndices.size() *
      static_cast<size_t>(audio_plugin::kNumChannels) *
      static_cast<size_t>(audio_plugin::kOutputChunkSize);
  std::vector<float> oneThreadReference;
  oneThreadReference.reserve(static_cast<size_t>(kMeasureIterations) *
                             kRetainedValuesPerResult);

  std::string runtimeVersion;
  for (int pass = 0; pass < kPassCount; ++pass) {
    for (const int intraOpThreadCount :
         kCandidateOrders[static_cast<size_t>(pass)]) {
      SCOPED_TRACE("pass=" + std::to_string(pass + 1) +
                   ", intra-op threads=" + std::to_string(intraOpThreadCount));

      // ORT thread-pool size is immutable after session construction. Keep
      // sessions sequential so their pools do not oversubscribe one another.
      audio_plugin::OnnxRuntime runtime;
      std::string failureMessage;
      ASSERT_TRUE(prepareRuntime(runtime, failureMessage, intraOpThreadCount))
          << failureMessage;
      ASSERT_EQ(runtime.getExecutionProvider(), "CPU");
      if (runtimeVersion.empty()) {
        runtimeVersion = runtime.getRuntimeVersion();
      } else {
        ASSERT_EQ(runtime.getRuntimeVersion(), runtimeVersion);
      }

      SeparatedChunk separated = makeSeparatedChunk();
      AudioChunk alignedInput = makeZeroAudioChunk();
      bool outputValid = true;
      ASSERT_TRUE(runtime.runInference(workload.front(), separated,
                                       alignedInput, outputValid));
      ASSERT_FALSE(outputValid);

      const bool captureOneThreadReference =
          pass == 0 && intraOpThreadCount == 1;
      size_t retainedReferenceOffset = 0;
      std::vector<double> passWallMilliseconds;
      passWallMilliseconds.reserve(static_cast<size_t>(kMeasureIterations));
      CandidateMeasurements& candidate =
          measurements[static_cast<size_t>(intraOpThreadCount - 1)];

      for (int iteration = 0;
           iteration < kWarmupIterations + kMeasureIterations; ++iteration) {
        const std::clock_t cpuBegin = std::clock();
        const auto wallBegin = std::chrono::steady_clock::now();
        ASSERT_TRUE(
            runtime.runInference(workload[static_cast<size_t>(iteration + 1)],
                                 separated, alignedInput, outputValid));
        const auto wallEnd = std::chrono::steady_clock::now();
        const std::clock_t cpuEnd = std::clock();

        ASSERT_TRUE(outputValid);
        ASSERT_TRUE(std::isfinite(maxAbsoluteValue(separated)));
        ASSERT_LE(maxMixtureReconstructionError(separated, alignedInput),
                  1.0e-6f);

        if (iteration < kWarmupIterations) {
          continue;
        }

        const double wallMilliseconds =
            std::chrono::duration<double, std::milli>(wallEnd - wallBegin)
                .count();
        passWallMilliseconds.push_back(wallMilliseconds);
        candidate.wallMilliseconds.push_back(wallMilliseconds);

        constexpr std::clock_t kClockUnavailable =
            static_cast<std::clock_t>(-1);
        if (cpuBegin != kClockUnavailable && cpuEnd != kClockUnavailable &&
            cpuEnd >= cpuBegin) {
          candidate.processCpuMilliseconds.push_back(
              1000.0 * static_cast<double>(cpuEnd - cpuBegin) /
              static_cast<double>(CLOCKS_PER_SEC));
        }

        for (const int stemIndex : kRetainedStemIndices) {
          for (const auto& channel :
               separated[static_cast<size_t>(stemIndex)]) {
            for (const float sample : channel) {
              if (captureOneThreadReference) {
                oneThreadReference.push_back(sample);
              } else {
                ASSERT_LT(retainedReferenceOffset, oneThreadReference.size());
                const float difference = std::abs(
                    sample - oneThreadReference[retainedReferenceOffset]);
                ASSERT_TRUE(std::isfinite(difference));
                candidate.maxRetainedStemDeltaFromOneThread = std::max(
                    candidate.maxRetainedStemDeltaFromOneThread, difference);
                ++retainedReferenceOffset;
              }
            }
          }
        }
      }

      ASSERT_EQ(passWallMilliseconds.size(),
                static_cast<size_t>(kMeasureIterations));
      if (captureOneThreadReference) {
        ASSERT_EQ(
            oneThreadReference.size(),
            static_cast<size_t>(kMeasureIterations) * kRetainedValuesPerResult);
      } else {
        ASSERT_EQ(retainedReferenceOffset, oneThreadReference.size());
      }
      std::sort(passWallMilliseconds.begin(), passWallMilliseconds.end());
      candidate.passP99Milliseconds[static_cast<size_t>(pass)] =
          percentileFromSorted(passWallMilliseconds, 0.99);
    }
  }

  std::array<CpuHopTimingSummary, 4> summaries;
  std::array<double, 4> medianPassP99Milliseconds{};
  std::array<double, 4> meanProcessCpuMilliseconds{};
  for (size_t candidateIndex = 0; candidateIndex < measurements.size();
       ++candidateIndex) {
    const CandidateMeasurements& candidate = measurements[candidateIndex];
    ASSERT_EQ(candidate.wallMilliseconds.size(),
              static_cast<size_t>(kPassCount * kMeasureIterations));
    summaries[candidateIndex] = summarizeCpuHopTimings(
        candidate.wallMilliseconds, hopBudgetMilliseconds);

    auto sortedPassP99 = candidate.passP99Milliseconds;
    std::sort(sortedPassP99.begin(), sortedPassP99.end());
    medianPassP99Milliseconds[candidateIndex] = sortedPassP99[1];

    if (!candidate.processCpuMilliseconds.empty()) {
      meanProcessCpuMilliseconds[candidateIndex] =
          std::accumulate(candidate.processCpuMilliseconds.begin(),
                          candidate.processCpuMilliseconds.end(), 0.0) /
          static_cast<double>(candidate.processCpuMilliseconds.size());
    } else {
      meanProcessCpuMilliseconds[candidateIndex] =
          std::numeric_limits<double>::quiet_NaN();
    }

    EXPECT_GT(summaries[candidateIndex].meanMilliseconds, 0.0);
    EXPECT_TRUE(std::isfinite(candidate.maxRetainedStemDeltaFromOneThread));
    EXPECT_LE(candidate.maxRetainedStemDeltaFromOneThread, 2.0e-5f);
  }

  std::cerr << "\nStateful CPU intra-op thread sweep (ORT " << runtimeVersion
            << ", logical CPUs=" << std::thread::hardware_concurrency() << ", "
            << kPassCount << " order-balanced passes x " << kMeasureIterations
            << " measured hops, hop budget=" << hopBudgetMilliseconds
            << " ms)\n"
            << "threads mean_ms p50_ms p95_ms p99_ms median_pass_p99_ms "
               "max_ms misses cpu_ms_per_hop speedup_vs_1 "
               "retained_delta_vs_1\n";

  const double oneThreadMean = summaries[0].meanMilliseconds;
  int selectedThreadCount = 0;
  double selectedMedianPassP99 = std::numeric_limits<double>::infinity();
  for (size_t candidateIndex = 0; candidateIndex < measurements.size();
       ++candidateIndex) {
    const CpuHopTimingSummary& summary = summaries[candidateIndex];
    const CandidateMeasurements& candidate = measurements[candidateIndex];
    std::cerr << (candidateIndex + 1) << " " << summary.meanMilliseconds << " "
              << summary.p50Milliseconds << " " << summary.p95Milliseconds
              << " " << summary.p99Milliseconds << " "
              << medianPassP99Milliseconds[candidateIndex] << " "
              << summary.maximumMilliseconds << " " << summary.deadlineMisses
              << "/" << candidate.wallMilliseconds.size() << " "
              << meanProcessCpuMilliseconds[candidateIndex] << " "
              << oneThreadMean / summary.meanMilliseconds << " "
              << candidate.maxRetainedStemDeltaFromOneThread << "\n";

    if (summary.deadlineMisses == 0 &&
        medianPassP99Milliseconds[candidateIndex] < selectedMedianPassP99) {
      selectedThreadCount = static_cast<int>(candidateIndex + 1);
      selectedMedianPassP99 = medianPassP99Milliseconds[candidateIndex];
    }
  }

  ASSERT_NE(selectedThreadCount, 0)
      << "No candidate met the recurring stateful hop deadline";
  std::cerr << "Observed lowest zero-miss median pass p99: "
            << selectedThreadCount << " intra-op thread(s), "
            << selectedMedianPassP99 << " ms. This platform's production "
            << "automatic cap is "
            << audio_plugin::kOrtAutomaticIntraOpThreadCap << ".\n";
#endif
}
