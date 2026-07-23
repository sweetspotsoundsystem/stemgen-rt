#include <StemgenRT/Constants.h>
#include <StemgenRT/OutputWriter.h>
#include <StemgenRT/OverlapAddProcessor.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace audio_plugin_test {
namespace {

using StereoBuffer = std::array<std::vector<float>, audio_plugin::kNumChannels>;
using StemBuffers = std::array<StereoBuffer, audio_plugin::kNumStems>;

struct WriterOutput {
  StereoBuffer main;
  // Native bus order: Drums, Bass, Other, Vocals.
  std::array<StereoBuffer, audio_plugin::kNumStems> stems;
};

StereoBuffer makeConstantStereo(size_t sampleCount, float left, float right) {
  StereoBuffer result;
  result[0].assign(sampleCount, left);
  result[1].assign(sampleCount, right);
  return result;
}

StemBuffers makeConstantModelOutput(size_t sampleCount,
                                    float drumsLeft,
                                    float drumsRight,
                                    float bassLeft,
                                    float bassRight,
                                    float vocalsLeft,
                                    float vocalsRight) {
  StemBuffers result;
  result[static_cast<size_t>(audio_plugin::kStemDrums)] =
      makeConstantStereo(sampleCount, drumsLeft, drumsRight);
  result[static_cast<size_t>(audio_plugin::kStemBass)] =
      makeConstantStereo(sampleCount, bassLeft, bassRight);
  result[static_cast<size_t>(audio_plugin::kStemVocals)] =
      makeConstantStereo(sampleCount, vocalsLeft, vocalsRight);

  // OutputWriter must always derive Other from the final Main and the three
  // retained model stems. A conspicuous value catches accidental use of the
  // model's Other output.
  result[static_cast<size_t>(audio_plugin::kStemOther)] =
      makeConstantStereo(sampleCount, 123.0f, -456.0f);
  return result;
}

class OutputWriterHarness {
public:
  explicit OutputWriterHarness(
      size_t maximumHostBlockSize =
          static_cast<size_t>(audio_plugin::kOutputChunkSize),
      size_t latencySamples =
          static_cast<size_t>(audio_plugin::kPluginLatencySamples)) {
    overlapAdd_.allocate(maximumHostBlockSize, latencySamples);
    writer_.prepare(static_cast<double>(audio_plugin::kModelSampleRate));
    writer_.reset();
  }

  WriterOutput write(const StereoBuffer& alignedMain,
                     const StemBuffers& modelOutput,
                     size_t expectedUnderrunSamples = 0) {
    const size_t sampleCount = alignedMain[0].size();
    EXPECT_EQ(alignedMain[1].size(), sampleCount);
    EXPECT_LE(sampleCount, overlapAdd_.getOutputRingSize());

    scheduleModelOutput(0, alignedMain, modelOutput);

    return runWriter(sampleCount, expectedUnderrunSamples);
  }

  void scheduleModelOutput(size_t timelineOffset,
                           const StereoBuffer& alignedMain,
                           const StemBuffers& modelOutput) {
    const size_t sampleCount = alignedMain[0].size();
    EXPECT_EQ(alignedMain[1].size(), sampleCount);
    EXPECT_LE(sampleCount, overlapAdd_.getOutputRingSize());

    auto& ringBuffers = overlapAdd_.getOutputRingBuffers();
    auto& delayedInput = overlapAdd_.getDelayedInputBuffer();
    const uint64_t firstTimelineSample = overlapAdd_.getOutputTimelineSample() +
                                         static_cast<uint64_t>(timelineOffset);
    EXPECT_TRUE(
        overlapAdd_.canScheduleModelOutput(firstTimelineSample, sampleCount));

    for (size_t i = 0; i < sampleCount; ++i) {
      const size_t ringPos = overlapAdd_.getOutputRingPosition(
          firstTimelineSample + static_cast<uint64_t>(i));
      for (size_t ch = 0; ch < static_cast<size_t>(audio_plugin::kNumChannels);
           ++ch) {
        delayedInput[ch][ringPos] = alignedMain[ch][i];
        for (size_t stem = 0;
             stem < static_cast<size_t>(audio_plugin::kNumStems); ++stem) {
          EXPECT_EQ(modelOutput[stem][ch].size(), sampleCount);
          ringBuffers[stem][ch][ringPos] = modelOutput[stem][ch][i];
        }
      }
    }
    overlapAdd_.markModelOutputScheduled(firstTimelineSample, sampleCount);
  }

  void pushDryInput(const StereoBuffer& input) {
    const size_t sampleCount = input[0].size();
    EXPECT_EQ(input[1].size(), sampleCount);
    for (size_t i = 0; i < sampleCount; ++i) {
      for (int ch = 0; ch < audio_plugin::kNumChannels; ++ch) {
        overlapAdd_.pushInputSample(ch, input[static_cast<size_t>(ch)][i]);
      }
    }
  }

  WriterOutput writeDryFallback(const StereoBuffer& input) {
    const size_t sampleCount = input[0].size();
    EXPECT_EQ(input[1].size(), sampleCount);
    for (size_t i = 0; i < sampleCount; ++i) {
      for (int ch = 0; ch < audio_plugin::kNumChannels; ++ch) {
        overlapAdd_.pushInputSample(ch, input[static_cast<size_t>(ch)][i]);
      }
    }

    return runWriter(sampleCount,
                     expectedPostLatencyUnderrunSamples(sampleCount));
  }

  void resetWriter() { writer_.reset(); }

  WriterOutput render(size_t sampleCount,
                      size_t expectedUnderrunSamples,
                      bool modelOutputEnabled = true,
                      bool useScheduledModelReference = true) {
    return runWriter(sampleCount, expectedUnderrunSamples, modelOutputEnabled,
                     useScheduledModelReference);
  }

  const audio_plugin::OutputWriter::WriteResult& getLastWriteResult() const {
    return lastWriteResult_;
  }

private:
  size_t expectedPostLatencyUnderrunSamples(size_t sampleCount) const {
    const uint64_t firstTimelineSample = overlapAdd_.getOutputTimelineSample();
    const uint64_t endTimelineSample =
        firstTimelineSample + static_cast<uint64_t>(sampleCount);
    const uint64_t firstEligibleSample =
        std::max(firstTimelineSample,
                 static_cast<uint64_t>(overlapAdd_.getLatencySamples()));
    return endTimelineSample > firstEligibleSample
               ? static_cast<size_t>(endTimelineSample - firstEligibleSample)
               : 0U;
  }

  WriterOutput runWriter(size_t sampleCount,
                         size_t expectedUnderrunSamples,
                         bool modelOutputEnabled = true,
                         bool useScheduledModelReference = true) {
    auto& ringBuffers = overlapAdd_.getOutputRingBuffers();
    auto& delayedInput = overlapAdd_.getDelayedInputBuffer();
    const size_t ringSize = overlapAdd_.getOutputRingSize();

    WriterOutput output;
    for (auto& channel : output.main) {
      channel.assign(sampleCount, 0.0f);
    }
    for (auto& stem : output.stems) {
      for (auto& channel : stem) {
        channel.assign(sampleCount, 0.0f);
      }
    }

    float* mainWrite[audio_plugin::kNumChannels] = {output.main[0].data(),
                                                    output.main[1].data()};
    float* stemWrite[4][audio_plugin::kNumChannels] = {};
    int stemNumChannels[4] = {};
    for (size_t stem = 0; stem < static_cast<size_t>(audio_plugin::kNumStems);
         ++stem) {
      stemNumChannels[stem] = audio_plugin::kNumChannels;
      for (size_t ch = 0; ch < static_cast<size_t>(audio_plugin::kNumChannels);
           ++ch) {
        stemWrite[stem][ch] = output.stems[stem][ch].data();
      }
    }
    writer_.setOutputPointers(mainWrite, audio_plugin::kNumChannels, stemWrite,
                              stemNumChannels);
    lastWriteResult_ =
        writer_.writeBlock(overlapAdd_, ringBuffers, delayedInput, ringSize,
                           static_cast<int>(sampleCount), true,
                           modelOutputEnabled, useScheduledModelReference);
    EXPECT_EQ(lastWriteResult_.hadUnderrun, expectedUnderrunSamples > 0);
    EXPECT_EQ(lastWriteResult_.underrunSamples, expectedUnderrunSamples);
    return output;
  }

  audio_plugin::OverlapAddProcessor overlapAdd_;
  audio_plugin::OutputWriter writer_;
  audio_plugin::OutputWriter::WriteResult lastWriteResult_;
};

StereoBuffer sliceStereo(const StereoBuffer& source,
                         size_t offset,
                         size_t sampleCount) {
  StereoBuffer result = makeConstantStereo(sampleCount, 0.0f, 0.0f);
  for (size_t ch = 0; ch < static_cast<size_t>(audio_plugin::kNumChannels);
       ++ch) {
    EXPECT_LE(offset + sampleCount, source[ch].size());
    for (size_t i = 0; i < sampleCount; ++i) {
      result[ch][i] = source[ch][offset + i];
    }
  }
  return result;
}

StemBuffers sliceStems(const StemBuffers& source,
                       size_t offset,
                       size_t sampleCount) {
  StemBuffers result;
  for (size_t stem = 0; stem < static_cast<size_t>(audio_plugin::kNumStems);
       ++stem) {
    result[stem] = sliceStereo(source[stem], offset, sampleCount);
  }
  return result;
}

void expectOutputMatchesRange(const WriterOutput& expected,
                              size_t offset,
                              const WriterOutput& actual) {
  for (size_t ch = 0; ch < static_cast<size_t>(audio_plugin::kNumChannels);
       ++ch) {
    ASSERT_LE(offset + actual.main[ch].size(), expected.main[ch].size());
    for (size_t i = 0; i < actual.main[ch].size(); ++i) {
      EXPECT_FLOAT_EQ(actual.main[ch][i], expected.main[ch][offset + i])
          << "main channel=" << ch << " sample=" << offset + i;
      for (size_t bus = 0; bus < static_cast<size_t>(audio_plugin::kNumStems);
           ++bus) {
        EXPECT_FLOAT_EQ(actual.stems[bus][ch][i],
                        expected.stems[bus][ch][offset + i])
            << "bus=" << bus << " channel=" << ch << " sample=" << offset + i;
      }
    }
  }
}

void expectMixtureLossless(const WriterOutput& output, size_t beginSample) {
  for (size_t ch = 0; ch < static_cast<size_t>(audio_plugin::kNumChannels);
       ++ch) {
    for (size_t i = beginSample; i < output.main[ch].size(); ++i) {
      const float stemSum = output.stems[0][ch][i] + output.stems[1][ch][i] +
                            output.stems[2][ch][i] + output.stems[3][ch][i];
      EXPECT_NEAR(stemSum, output.main[ch][i], 1.0e-7f)
          << "channel=" << ch << " sample=" << i;
    }
  }
}

constexpr size_t kFirstFullySeparatedSample =
    static_cast<size_t>(audio_plugin::kUnderrunCrossfadeSamples - 1);

}  // namespace

TEST(OutputWriterTest, SuppressesModelFloorWhenAlignedMainIsSilent) {
  constexpr size_t kSampleCount = 256;
  const StereoBuffer alignedMain = makeConstantStereo(kSampleCount, 0.0f, 0.0f);
  const StemBuffers modelOutput = makeConstantModelOutput(
      kSampleCount, 1.1e-5f, -1.2e-5f, -9.0e-6f, 8.0e-6f, 1.3e-5f, -1.0e-5f);

  OutputWriterHarness harness;
  const WriterOutput output = harness.write(alignedMain, modelOutput);

  for (size_t bus = 0; bus < static_cast<size_t>(audio_plugin::kNumStems);
       ++bus) {
    for (size_t ch = 0; ch < static_cast<size_t>(audio_plugin::kNumChannels);
         ++ch) {
      for (size_t i = kFirstFullySeparatedSample; i < kSampleCount; ++i) {
        EXPECT_FLOAT_EQ(output.stems[bus][ch][i], 0.0f)
            << "bus=" << bus << " channel=" << ch << " sample=" << i;
      }
    }
  }
  expectMixtureLossless(output, 0);
}

TEST(OutputWriterTest, RoutesTinyAlignedMainEntirelyToOther) {
  constexpr size_t kSampleCount = 256;
  // Both channels are well below the -96 dBFS closed threshold.
  constexpr float kTinyLeft = 2.0e-6f;
  constexpr float kTinyRight = -7.0e-6f;
  const StereoBuffer alignedMain =
      makeConstantStereo(kSampleCount, kTinyLeft, kTinyRight);
  const StemBuffers modelOutput = makeConstantModelOutput(
      kSampleCount, 1.2e-5f, -1.1e-5f, -8.0e-6f, 9.0e-6f, 1.4e-5f, -1.3e-5f);

  OutputWriterHarness harness;
  const WriterOutput output = harness.write(alignedMain, modelOutput);

  for (size_t ch = 0; ch < static_cast<size_t>(audio_plugin::kNumChannels);
       ++ch) {
    const float expectedMain = ch == 0 ? kTinyLeft : kTinyRight;
    for (size_t i = kFirstFullySeparatedSample; i < kSampleCount; ++i) {
      EXPECT_FLOAT_EQ(output.main[ch][i], expectedMain);
      EXPECT_FLOAT_EQ(output.stems[0][ch][i], 0.0f);
      EXPECT_FLOAT_EQ(output.stems[1][ch][i], 0.0f);
      EXPECT_FLOAT_EQ(output.stems[3][ch][i], 0.0f);
      EXPECT_FLOAT_EQ(output.stems[2][ch][i], expectedMain);
    }
  }
  expectMixtureLossless(output, 0);
}

TEST(OutputWriterTest, RoutesLowLevelDryFallbackEntirelyToOther) {
  constexpr size_t kBlockSize = 512;
  constexpr float kTinyLeft = 8.0e-6f;
  constexpr float kTinyRight = -4.0e-6f;

  OutputWriterHarness harness;
  // The fallback delay is one 512-sample current-chunk hop. Feed one more
  // block so the first reaches the writer without model output available.
  harness.writeDryFallback(
      makeConstantStereo(kBlockSize, kTinyLeft, kTinyRight));
  const WriterOutput output =
      harness.writeDryFallback(makeConstantStereo(kBlockSize, 0.0f, 0.0f));

  for (size_t ch = 0; ch < static_cast<size_t>(audio_plugin::kNumChannels);
       ++ch) {
    const float expectedMain = ch == 0 ? kTinyLeft : kTinyRight;
    for (size_t i = 0; i < kBlockSize; ++i) {
      EXPECT_FLOAT_EQ(output.main[ch][i], expectedMain);
      EXPECT_FLOAT_EQ(output.stems[0][ch][i], 0.0f);
      EXPECT_FLOAT_EQ(output.stems[1][ch][i], 0.0f);
      EXPECT_FLOAT_EQ(output.stems[3][ch][i], 0.0f);
      EXPECT_FLOAT_EQ(output.stems[2][ch][i], expectedMain);
    }
  }
  expectMixtureLossless(output, 0);
}

TEST(OutputWriterTest, ReportsMissingModelOnlyAtOrAfterLatency) {
  constexpr size_t kBlockSize = 512;
  OutputWriterHarness harness;

  harness.writeDryFallback(makeConstantStereo(kBlockSize, 0.2f, -0.1f));
  EXPECT_FALSE(harness.getLastWriteResult().hadUnderrun);
  EXPECT_FALSE(harness.getLastWriteResult().underrunTransition);

  harness.writeDryFallback(makeConstantStereo(kBlockSize, 0.2f, -0.1f));
  EXPECT_TRUE(harness.getLastWriteResult().hadUnderrun);
  EXPECT_TRUE(harness.getLastWriteResult().isUnderrunNow);
  EXPECT_TRUE(harness.getLastWriteResult().underrunTransition);
  EXPECT_EQ(harness.getLastWriteResult().underrunSamples, kBlockSize);

  harness.writeDryFallback(makeConstantStereo(1, 0.2f, -0.1f));
  EXPECT_TRUE(harness.getLastWriteResult().hadUnderrun);
  EXPECT_TRUE(harness.getLastWriteResult().isUnderrunNow);
  EXPECT_FALSE(harness.getLastWriteResult().underrunTransition);
  EXPECT_EQ(harness.getLastWriteResult().underrunSamples, 1U);
}

TEST(OutputWriterTest,
     FutureFirstResultDoesNotCountPreLatencyPrefixAsUnderrun) {
  constexpr size_t kHostBlockSize = 1024;
  constexpr size_t kDynamicLatency = 1536;
  constexpr size_t kPreLatencyPrefix = kDynamicLatency - kHostBlockSize;
  constexpr size_t kModelSamples = kHostBlockSize - kPreLatencyPrefix;
  OutputWriterHarness harness(kHostBlockSize, kDynamicLatency);

  harness.writeDryFallback(makeConstantStereo(kHostBlockSize, 0.2f, -0.1f));
  ASSERT_FALSE(harness.getLastWriteResult().hadUnderrun);

  const StereoBuffer alignedMain =
      makeConstantStereo(kModelSamples, 0.4f, -0.3f);
  const StemBuffers modelOutput = makeConstantModelOutput(
      kModelSamples, 0.1f, 0.08f, -0.03f, 0.04f, 0.05f, -0.02f);
  harness.scheduleModelOutput(kPreLatencyPrefix, alignedMain, modelOutput);
  harness.render(kHostBlockSize, 0);

  EXPECT_FALSE(harness.getLastWriteResult().hadUnderrun);
  EXPECT_FALSE(harness.getLastWriteResult().isUnderrunNow);
  EXPECT_FALSE(harness.getLastWriteResult().underrunTransition);
}

TEST(OutputWriterTest, FirstUnderrunSampleUsesCompleteDryFallback) {
  constexpr size_t kFadeSamples =
      static_cast<size_t>(audio_plugin::kUnderrunCrossfadeSamples);
  constexpr size_t kLatency =
      static_cast<size_t>(audio_plugin::kPluginLatencySamples);
  constexpr float kDryLeft = 0.4f;
  constexpr float kDryRight = -0.3f;

  OutputWriterHarness harness;
  const StemBuffers modelOutput = makeConstantModelOutput(
      kFadeSamples, 0.1f, 0.08f, -0.03f, 0.04f, 0.05f, -0.02f);
  harness.write(makeConstantStereo(kFadeSamples, 0.0f, 0.0f), modelOutput);

  harness.pushDryInput(
      makeConstantStereo(kFadeSamples * 2, kDryLeft, kDryRight));
  const size_t samplesUntilDryArrival = kLatency - kFadeSamples;
  const StemBuffers fillerModel = makeConstantModelOutput(
      samplesUntilDryArrival, 0.1f, 0.08f, -0.03f, 0.04f, 0.05f, -0.02f);
  harness.write(makeConstantStereo(samplesUntilDryArrival, 0.0f, 0.0f),
                fillerModel);

  const WriterOutput output =
      harness.writeDryFallback(makeConstantStereo(1, 0.0f, 0.0f));
  EXPECT_FLOAT_EQ(output.main[0][0], kDryLeft);
  EXPECT_FLOAT_EQ(output.main[1][0], kDryRight);
  for (const size_t bus : {0U, 1U, 3U}) {
    EXPECT_FLOAT_EQ(output.stems[bus][0][0], 0.0f);
    EXPECT_FLOAT_EQ(output.stems[bus][1][0], 0.0f);
  }
  EXPECT_FLOAT_EQ(output.stems[2][0][0], kDryLeft);
  EXPECT_FLOAT_EQ(output.stems[2][1][0], kDryRight);
  expectMixtureLossless(output, 0);
}

TEST(OutputWriterTest, TimingGateRoutesScheduledModelEntirelyToOther) {
  constexpr size_t kLatency =
      static_cast<size_t>(audio_plugin::kPluginLatencySamples);
  constexpr size_t kSampleCount = 128;
  constexpr float kDryLeft = 0.4f;
  constexpr float kDryRight = -0.3f;

  OutputWriterHarness harness;
  harness.pushDryInput(
      makeConstantStereo(kLatency + kSampleCount, kDryLeft, kDryRight));
  harness.render(kLatency, 0);

  const StereoBuffer alignedMain =
      makeConstantStereo(kSampleCount, kDryLeft, kDryRight);
  const StemBuffers modelOutput = makeConstantModelOutput(
      kSampleCount, 0.10f, 0.08f, -0.03f, 0.04f, 0.05f, -0.02f);
  harness.scheduleModelOutput(0, alignedMain, modelOutput);
  const WriterOutput output = harness.render(kSampleCount, kSampleCount, false);

  for (size_t ch = 0; ch < static_cast<size_t>(audio_plugin::kNumChannels);
       ++ch) {
    const float expectedMain = ch == 0 ? kDryLeft : kDryRight;
    for (size_t sample = 0; sample < kSampleCount; ++sample) {
      EXPECT_FLOAT_EQ(output.main[ch][sample], expectedMain);
      EXPECT_FLOAT_EQ(output.stems[0][ch][sample], 0.0f);
      EXPECT_FLOAT_EQ(output.stems[1][ch][sample], 0.0f);
      EXPECT_FLOAT_EQ(output.stems[3][ch][sample], 0.0f);
      EXPECT_FLOAT_EQ(output.stems[2][ch][sample], expectedMain);
    }
  }
  expectMixtureLossless(output, 0);
}

TEST(OutputWriterTest,
     ConvertedModelPathUsesNativeDryMainInsteadOfBandlimitedReference) {
  constexpr size_t kSampleCount = 128;
  constexpr float kDryLeft = 0.4f;
  constexpr float kDryRight = -0.3f;
  OutputWriterHarness harness(kSampleCount, 0U);
  harness.pushDryInput(makeConstantStereo(kSampleCount, kDryLeft, kDryRight));
  harness.scheduleModelOutput(
      0U, makeConstantStereo(kSampleCount, 0.1f, 0.2f),
      makeConstantModelOutput(kSampleCount, 0.05f, -0.04f, 0.03f, 0.02f, -0.01f,
                              0.06f));

  const WriterOutput output = harness.render(kSampleCount, 0U, true, false);
  for (size_t channel = 0U;
       channel < static_cast<size_t>(audio_plugin::kNumChannels); ++channel) {
    const float expectedMain = channel == 0U ? kDryLeft : kDryRight;
    for (size_t sample = 0U; sample < kSampleCount; ++sample) {
      EXPECT_FLOAT_EQ(output.main[channel][sample], expectedMain);
    }
  }
  expectMixtureLossless(output, 0U);
}

TEST(OutputWriterTest, PassesModelStemsUnchangedAtOrdinaryLevel) {
  constexpr size_t kSampleCount = 256;
  constexpr float kMainLeft = 0.5f;
  constexpr float kMainRight = -0.4f;
  constexpr float kDrumsLeft = 0.12f;
  constexpr float kDrumsRight = 0.08f;
  constexpr float kBassLeft = -0.03f;
  constexpr float kBassRight = 0.04f;
  constexpr float kVocalsLeft = 0.05f;
  constexpr float kVocalsRight = -0.02f;
  const StereoBuffer alignedMain =
      makeConstantStereo(kSampleCount, kMainLeft, kMainRight);
  const StemBuffers modelOutput =
      makeConstantModelOutput(kSampleCount, kDrumsLeft, kDrumsRight, kBassLeft,
                              kBassRight, kVocalsLeft, kVocalsRight);

  OutputWriterHarness harness;
  const WriterOutput output = harness.write(alignedMain, modelOutput);

  for (size_t i = kFirstFullySeparatedSample; i < kSampleCount; ++i) {
    EXPECT_FLOAT_EQ(output.stems[0][0][i], kDrumsLeft);
    EXPECT_FLOAT_EQ(output.stems[0][1][i], kDrumsRight);
    EXPECT_FLOAT_EQ(output.stems[1][0][i], kBassLeft);
    EXPECT_FLOAT_EQ(output.stems[1][1][i], kBassRight);
    EXPECT_FLOAT_EQ(output.stems[3][0][i], kVocalsLeft);
    EXPECT_FLOAT_EQ(output.stems[3][1][i], kVocalsRight);
    EXPECT_FLOAT_EQ(output.stems[2][0][i],
                    kMainLeft - kDrumsLeft - kBassLeft - kVocalsLeft);
    EXPECT_FLOAT_EQ(output.stems[2][1][i],
                    kMainRight - kDrumsRight - kBassRight - kVocalsRight);
  }
  expectMixtureLossless(output, 0);
}

TEST(OutputWriterTest, ConfidenceUsesLinearAmplitudeSmoothstepThresholds) {
  constexpr size_t kSampleCount = 256;
  constexpr float kModelDrums = 0.2f;
  const float midpoint = 0.5f * (audio_plugin::kLowLevelSeparationClosed +
                                 audio_plugin::kLowLevelSeparationOpen);
  struct LevelCase {
    const char* label;
    float level;
    float expectedConfidence;
  };
  const std::array<LevelCase, 3> levelCases = {{
      {"closed endpoint", audio_plugin::kLowLevelSeparationClosed, 0.0f},
      {"linear-amplitude midpoint", midpoint, 0.5f},
      {"open endpoint", audio_plugin::kLowLevelSeparationOpen, 1.0f},
  }};

  for (const auto& levelCase : levelCases) {
    SCOPED_TRACE(levelCase.label);
    const StereoBuffer alignedMain =
        makeConstantStereo(kSampleCount, levelCase.level, -levelCase.level);
    const StemBuffers modelOutput = makeConstantModelOutput(
        kSampleCount, kModelDrums, -kModelDrums, 0.0f, 0.0f, 0.0f, 0.0f);

    OutputWriterHarness harness;
    const WriterOutput output = harness.write(alignedMain, modelOutput);
    const float expectedDrums = levelCase.expectedConfidence * kModelDrums;

    EXPECT_NEAR(output.stems[0][0].back(), expectedDrums, 1.0e-6f);
    EXPECT_NEAR(output.stems[0][1].back(), -expectedDrums, 1.0e-6f);
    expectMixtureLossless(output, 0);
  }
}

TEST(OutputWriterTest, PeakHoldKeepsThresholdToneFullyOpen) {
  constexpr size_t kSampleCount = 4410;
  constexpr float kFrequency = 20.0f;
  constexpr float kPi = 3.14159265358979323846f;
  constexpr float kModelDrums = 0.2f;
  const float amplitude = 1.01f * audio_plugin::kLowLevelSeparationOpen;
  StereoBuffer alignedMain = makeConstantStereo(kSampleCount, 0.0f, 0.0f);
  for (size_t i = 0; i < kSampleCount; ++i) {
    const float sample =
        amplitude *
        std::sin(2.0f * kPi * kFrequency * static_cast<float>(i) /
                 static_cast<float>(audio_plugin::kModelSampleRate));
    alignedMain[0][i] = sample;
    alignedMain[1][i] = -sample;
  }
  const StemBuffers modelOutput = makeConstantModelOutput(
      kSampleCount, kModelDrums, -kModelDrums, 0.0f, 0.0f, 0.0f, 0.0f);

  OutputWriterHarness harness;
  const WriterOutput output = harness.write(alignedMain, modelOutput);

  // Absolute peaks of a 20 Hz tone recur every 25 ms, within the 50 ms hold.
  // After one full period the confidence must remain continuously open.
  constexpr size_t kOnePeriod =
      static_cast<size_t>(audio_plugin::kModelSampleRate / 20);
  for (size_t i = kOnePeriod; i < kSampleCount; ++i) {
    EXPECT_FLOAT_EQ(output.stems[0][0][i], kModelDrums);
    EXPECT_FLOAT_EQ(output.stems[0][1][i], -kModelDrums);
  }
}

TEST(OutputWriterTest, ConfidenceReleaseMatchesConfiguredTiming) {
  constexpr size_t kOpenSamples =
      static_cast<size_t>(audio_plugin::kUnderrunCrossfadeSamples);
  const int holdSamples = static_cast<int>(
      std::lround(static_cast<double>(audio_plugin::kLowLevelHoldSeconds) *
                  static_cast<double>(audio_plugin::kModelSampleRate)));
  const double releaseCoefficient =
      std::exp(std::log(0.001) /
               (static_cast<double>(audio_plugin::kLowLevelReleaseSeconds) *
                static_cast<double>(audio_plugin::kModelSampleRate)));
  const double midpoint =
      0.5 * (static_cast<double>(audio_plugin::kLowLevelSeparationClosed) +
             static_cast<double>(audio_plugin::kLowLevelSeparationOpen));
  const int decaySamples = static_cast<int>(std::lround(
      std::log(midpoint /
               static_cast<double>(audio_plugin::kLowLevelSeparationOpen)) /
      std::log(releaseCoefficient)));
  const size_t checkpoint =
      kOpenSamples - 1 + static_cast<size_t>(holdSamples + decaySamples);
  const size_t sampleCount = checkpoint + 2;

  StereoBuffer alignedMain = makeConstantStereo(sampleCount, 0.0f, 0.0f);
  for (size_t i = 0; i < kOpenSamples; ++i) {
    alignedMain[0][i] = audio_plugin::kLowLevelSeparationOpen;
    alignedMain[1][i] = -audio_plugin::kLowLevelSeparationOpen;
  }
  constexpr float kModelDrums = 0.2f;
  const StemBuffers modelOutput = makeConstantModelOutput(
      sampleCount, kModelDrums, -kModelDrums, 0.0f, 0.0f, 0.0f, 0.0f);

  OutputWriterHarness harness;
  const WriterOutput output = harness.write(alignedMain, modelOutput);

  const double expectedEnvelope =
      static_cast<double>(audio_plugin::kLowLevelSeparationOpen) *
      std::pow(releaseCoefficient, decaySamples);
  const double linearConfidence =
      (expectedEnvelope -
       static_cast<double>(audio_plugin::kLowLevelSeparationClosed)) /
      (static_cast<double>(audio_plugin::kLowLevelSeparationOpen) -
       static_cast<double>(audio_plugin::kLowLevelSeparationClosed));
  const double expectedConfidence =
      linearConfidence * linearConfidence * (3.0 - 2.0 * linearConfidence);

  EXPECT_FLOAT_EQ(
      output.stems[0][0][kOpenSamples - 1 + static_cast<size_t>(holdSamples)],
      kModelDrums);
  EXPECT_NEAR(output.stems[0][0][checkpoint],
              static_cast<float>(expectedConfidence) * kModelDrums, 1.0e-5f);
}

TEST(OutputWriterTest, UsesStereoLinkedSeparationConfidence) {
  constexpr size_t kSampleCount = 256;
  const StereoBuffer alignedMain = makeConstantStereo(kSampleCount, 0.5f, 0.0f);
  const StemBuffers modelOutput = makeConstantModelOutput(
      kSampleCount, 0.10f, 0.12f, 0.03f, -0.04f, -0.02f, 0.025f);

  OutputWriterHarness harness;
  const WriterOutput output = harness.write(alignedMain, modelOutput);

  // The silent right channel retains its model contributions because the loud
  // left channel opens one linked confidence gain for the stereo pair.
  for (size_t i = kFirstFullySeparatedSample; i < kSampleCount; ++i) {
    EXPECT_FLOAT_EQ(output.stems[0][1][i], 0.12f);
    EXPECT_FLOAT_EQ(output.stems[1][1][i], -0.04f);
    EXPECT_FLOAT_EQ(output.stems[3][1][i], 0.025f);
    EXPECT_FLOAT_EQ(output.stems[2][1][i], -0.105f);
  }
  expectMixtureLossless(output, 0);
}

TEST(OutputWriterTest, OpensConfidenceOnTheFirstLoudSample) {
  constexpr size_t kQuietSamples =
      static_cast<size_t>(audio_plugin::kUnderrunCrossfadeSamples);
  constexpr size_t kSampleCount = kQuietSamples + 16;
  StereoBuffer alignedMain =
      makeConstantStereo(kSampleCount, 1.0e-6f, -1.0e-6f);
  for (size_t i = kQuietSamples; i < kSampleCount; ++i) {
    alignedMain[0][i] = 0.5f;
    alignedMain[1][i] = -0.4f;
  }
  const StemBuffers modelOutput = makeConstantModelOutput(
      kSampleCount, 0.10f, 0.08f, -0.03f, 0.04f, 0.05f, -0.02f);

  OutputWriterHarness harness;
  const WriterOutput output = harness.write(alignedMain, modelOutput);

  EXPECT_FLOAT_EQ(output.stems[0][0][kQuietSamples], 0.10f);
  EXPECT_FLOAT_EQ(output.stems[0][1][kQuietSamples], 0.08f);
  EXPECT_FLOAT_EQ(output.stems[1][0][kQuietSamples], -0.03f);
  EXPECT_FLOAT_EQ(output.stems[3][1][kQuietSamples], -0.02f);
  expectMixtureLossless(output, 0);
}

TEST(OutputWriterTest, RoutesNonFiniteModelFrameEntirelyToOther) {
  constexpr size_t kSampleCount = 128;
  constexpr float kMainLeft = 0.5f;
  constexpr float kMainRight = -0.4f;
  const StereoBuffer alignedMain =
      makeConstantStereo(kSampleCount, kMainLeft, kMainRight);
  StemBuffers modelOutput = makeConstantModelOutput(
      kSampleCount, 0.10f, 0.08f, -0.03f, 0.04f, 0.05f, -0.02f);
  modelOutput[static_cast<size_t>(audio_plugin::kStemBass)][0][17] =
      std::numeric_limits<float>::quiet_NaN();

  OutputWriterHarness harness;
  const WriterOutput output = harness.write(alignedMain, modelOutput);

  for (size_t ch = 0; ch < static_cast<size_t>(audio_plugin::kNumChannels);
       ++ch) {
    const float expectedMain = ch == 0 ? kMainLeft : kMainRight;
    EXPECT_FLOAT_EQ(output.main[ch][17], expectedMain);
    EXPECT_FLOAT_EQ(output.stems[0][ch][17], 0.0f);
    EXPECT_FLOAT_EQ(output.stems[1][ch][17], 0.0f);
    EXPECT_FLOAT_EQ(output.stems[3][ch][17], 0.0f);
    EXPECT_FLOAT_EQ(output.stems[2][ch][17], expectedMain);
  }
  expectMixtureLossless(output, 0);
}

TEST(OutputWriterTest, RecoversConfidenceAfterNonFiniteLevelReference) {
  constexpr size_t kLoudSamples =
      static_cast<size_t>(audio_plugin::kUnderrunCrossfadeSamples);
  constexpr size_t kSampleCount = kLoudSamples + 16;
  struct NonFiniteCase {
    const char* label;
    float value;
    size_t channel;
  };
  const std::array<NonFiniteCase, 3> nonFiniteCases = {{
      {"left NaN", std::numeric_limits<float>::quiet_NaN(), 0},
      {"right positive infinity", std::numeric_limits<float>::infinity(), 1},
      {"left negative infinity", -std::numeric_limits<float>::infinity(), 0},
  }};

  for (const auto& nonFiniteCase : nonFiniteCases) {
    SCOPED_TRACE(nonFiniteCase.label);
    StereoBuffer alignedMain =
        makeConstantStereo(kSampleCount, 1.0e-6f, -1.0e-6f);
    for (size_t i = 0; i < kLoudSamples; ++i) {
      alignedMain[0][i] = 0.5f;
      alignedMain[1][i] = -0.4f;
    }
    alignedMain[nonFiniteCase.channel][kLoudSamples] = nonFiniteCase.value;
    const StemBuffers modelOutput = makeConstantModelOutput(
        kSampleCount, 0.10f, 0.08f, -0.03f, 0.04f, 0.05f, -0.02f);

    OutputWriterHarness harness;
    const WriterOutput output = harness.write(alignedMain, modelOutput);

    // The invalid stereo frame is treated as an exact-timeline miss and uses
    // the complete (zero here) dry fallback. The following finite quiet frame
    // preserves Main while keeping unreliable model contributions closed.
    for (size_t ch = 0; ch < static_cast<size_t>(audio_plugin::kNumChannels);
         ++ch) {
      EXPECT_TRUE(std::isfinite(output.main[ch][kLoudSamples]));
      EXPECT_FLOAT_EQ(output.main[ch][kLoudSamples], 0.0f);
      EXPECT_FLOAT_EQ(output.stems[0][ch][kLoudSamples], 0.0f);
      EXPECT_FLOAT_EQ(output.stems[1][ch][kLoudSamples], 0.0f);
      EXPECT_FLOAT_EQ(output.stems[2][ch][kLoudSamples], 0.0f);
      EXPECT_FLOAT_EQ(output.stems[3][ch][kLoudSamples], 0.0f);

      const float expectedMain = ch == 0 ? 1.0e-6f : -1.0e-6f;
      EXPECT_FLOAT_EQ(output.main[ch][kLoudSamples + 1], expectedMain);
      EXPECT_FLOAT_EQ(output.stems[0][ch][kLoudSamples + 1], 0.0f);
      EXPECT_FLOAT_EQ(output.stems[1][ch][kLoudSamples + 1], 0.0f);
      EXPECT_FLOAT_EQ(output.stems[2][ch][kLoudSamples + 1], expectedMain);
      EXPECT_FLOAT_EQ(output.stems[3][ch][kLoudSamples + 1], 0.0f);
    }
  }
}

TEST(OutputWriterTest, ResetClearsConfidenceDeterministically) {
  constexpr size_t kPrimeSamples = 256;
  constexpr size_t kQuietSamples = 256;
  const StemBuffers primeModelOutput = makeConstantModelOutput(
      kPrimeSamples, 0.10f, 0.08f, -0.03f, 0.04f, 0.05f, -0.02f);
  const StemBuffers quietModelOutput = makeConstantModelOutput(
      kQuietSamples, 0.10f, 0.08f, -0.03f, 0.04f, 0.05f, -0.02f);
  const StereoBuffer quietMain =
      makeConstantStereo(kQuietSamples, 1.0e-6f, -1.0e-6f);

  OutputWriterHarness previouslyLoud;
  previouslyLoud.write(makeConstantStereo(kPrimeSamples, 0.5f, -0.4f),
                       primeModelOutput);
  const WriterOutput retained = previouslyLoud.write(
      sliceStereo(quietMain, 0, 1), sliceStems(quietModelOutput, 0, 1));
  EXPECT_GT(std::abs(retained.stems[0][0][0]), 0.09f);

  previouslyLoud.resetWriter();
  const WriterOutput afterReset =
      previouslyLoud.write(quietMain, quietModelOutput);

  OutputWriterHarness fresh;
  const WriterOutput freshOutput = fresh.write(quietMain, quietModelOutput);
  expectOutputMatchesRange(freshOutput, 0, afterReset);
  for (size_t i = kFirstFullySeparatedSample; i < kQuietSamples; ++i) {
    EXPECT_FLOAT_EQ(afterReset.stems[0][0][i], 0.0f);
    EXPECT_FLOAT_EQ(afterReset.stems[1][0][i], 0.0f);
    EXPECT_FLOAT_EQ(afterReset.stems[3][0][i], 0.0f);
  }
}

TEST(OutputWriterTest, RemainsMixtureLosslessDuringConfidenceRelease) {
  constexpr size_t kLoudSamples = 128;
  constexpr size_t kSampleCount = 9216;
  StereoBuffer alignedMain = makeConstantStereo(kSampleCount, 0.0f, 0.0f);
  for (size_t i = 0; i < kLoudSamples; ++i) {
    alignedMain[0][i] = 0.5f;
    alignedMain[1][i] = -0.4f;
  }
  const StemBuffers modelOutput = makeConstantModelOutput(
      kSampleCount, 0.10f, 0.08f, -0.03f, 0.04f, 0.05f, -0.02f);

  OutputWriterHarness harness;
  const WriterOutput output = harness.write(alignedMain, modelOutput);

  expectMixtureLossless(output, 0);

  bool observedPartialConfidence = false;
  for (size_t i = kLoudSamples; i < kSampleCount; ++i) {
    const float drums = output.stems[0][0][i];
    if (drums > 0.0f && drums < 0.10f) {
      observedPartialConfidence = true;
      break;
    }
  }
  EXPECT_TRUE(observedPartialConfidence);

  // The 50 ms hold plus 100 ms / 60 dB release has ample time to fall below
  // the -96 dBFS closed threshold by the end of this block.
  EXPECT_FLOAT_EQ(output.stems[0][0].back(), 0.0f);
  EXPECT_FLOAT_EQ(output.stems[1][0].back(), 0.0f);
  EXPECT_FLOAT_EQ(output.stems[3][0].back(), 0.0f);
  EXPECT_FLOAT_EQ(output.stems[2][0].back(), 0.0f);
}

TEST(OutputWriterTest, ConfidenceReleaseIsInvariantToHostBlockSplits) {
  constexpr size_t kLoudSamples = 128;
  constexpr size_t kSampleCount = 8192;
  StereoBuffer alignedMain = makeConstantStereo(kSampleCount, 0.0f, 0.0f);
  for (size_t i = 0; i < kLoudSamples; ++i) {
    alignedMain[0][i] = 0.5f;
    alignedMain[1][i] = -0.4f;
  }
  const StemBuffers modelOutput = makeConstantModelOutput(
      kSampleCount, 0.10f, 0.08f, -0.03f, 0.04f, 0.05f, -0.02f);

  OutputWriterHarness singleBlockHarness;
  const WriterOutput singleBlock =
      singleBlockHarness.write(alignedMain, modelOutput);

  OutputWriterHarness splitBlockHarness;
  const std::array<size_t, 10> blockSizes = {17,  47,  64,   1,    127,
                                             256, 511, 1024, 2049, 4096};
  size_t offset = 0;
  for (const size_t blockSize : blockSizes) {
    const WriterOutput splitBlock =
        splitBlockHarness.write(sliceStereo(alignedMain, offset, blockSize),
                                sliceStems(modelOutput, offset, blockSize));
    expectOutputMatchesRange(singleBlock, offset, splitBlock);
    offset += blockSize;
  }
  EXPECT_EQ(offset, kSampleCount);
}

}  // namespace audio_plugin_test
