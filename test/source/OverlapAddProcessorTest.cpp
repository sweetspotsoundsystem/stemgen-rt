#include <StemgenRT/Constants.h>
#include <StemgenRT/OverlapAddProcessor.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <limits>
#include <vector>

namespace audio_plugin {

class OverlapAddProcessorTestPeer {
public:
  static uint64_t outputGeneration(const OverlapAddProcessor& processor) {
    return processor.outputGeneration_;
  }

  static uint64_t timelineTag(const OverlapAddProcessor& processor,
                              size_t position) {
    return processor.outputTimelineTags_[position];
  }

  static uint64_t generationTag(const OverlapAddProcessor& processor,
                                size_t position) {
    return processor.outputGenerationTags_[position];
  }

  static void seedDryStorage(OverlapAddProcessor& processor,
                             size_t channel,
                             size_t position,
                             float value) {
    processor.dryDelayLine_[channel][position] = value;
  }

  static float dryStorage(const OverlapAddProcessor& processor,
                          size_t channel,
                          size_t position) {
    return processor.dryDelayLine_[channel][position];
  }
};

}  // namespace audio_plugin

namespace audio_plugin_test {
namespace {

using StereoBuffer = std::array<std::vector<float>, audio_plugin::kNumChannels>;

void appendDryBlock(audio_plugin::OverlapAddProcessor& processor,
                    const StereoBuffer& input,
                    size_t offset,
                    size_t count,
                    StereoBuffer& output) {
  ASSERT_LE(offset + count, input[0].size());
  ASSERT_EQ(input[0].size(), input[1].size());

  // Match PluginProcessor: the complete host block is accumulated before the
  // latency-aligned fallback is read.
  for (size_t i = 0; i < count; ++i) {
    for (int ch = 0; ch < audio_plugin::kNumChannels; ++ch) {
      processor.pushInputSample(ch, input[static_cast<size_t>(ch)][offset + i]);
    }
  }

  for (size_t i = 0; i < count; ++i) {
    for (int ch = 0; ch < audio_plugin::kNumChannels; ++ch) {
      output[static_cast<size_t>(ch)].push_back(
          processor.readDryDelaySample(ch));
    }
    processor.advanceDryDelayPos();
  }
}

StereoBuffer makeDistinctStereoInput(size_t sampleCount) {
  StereoBuffer input;
  for (auto& channel : input) {
    channel.resize(sampleCount);
  }
  for (size_t i = 0; i < sampleCount; ++i) {
    input[0][i] = static_cast<float>(i + 1);
    input[1][i] = -static_cast<float>(i + 1) - 0.25f;
  }
  return input;
}

void expectFixedLatency(const StereoBuffer& input, const StereoBuffer& output) {
  ASSERT_EQ(output[0].size(), input[0].size());
  ASSERT_EQ(output[1].size(), input[1].size());

  constexpr size_t kLatency =
      static_cast<size_t>(audio_plugin::kPluginLatencySamples);
  for (size_t ch = 0; ch < static_cast<size_t>(audio_plugin::kNumChannels);
       ++ch) {
    for (size_t i = 0; i < output[ch].size(); ++i) {
      const float expected = i < kLatency ? 0.0f : input[ch][i - kLatency];
      EXPECT_FLOAT_EQ(output[ch][i], expected)
          << "channel=" << ch << " sample=" << i;
    }
  }
}

}  // namespace

TEST(OverlapAddProcessorTest,
     DryFallbackHasFixedPluginLatencyAcrossDifferentHostBlocks) {
  constexpr size_t kMaximumHostBlock = 768;
  constexpr size_t kTotalSamples = 3072;
  const StereoBuffer input = makeDistinctStereoInput(kTotalSamples);
  StereoBuffer output;

  audio_plugin::OverlapAddProcessor processor;
  processor.allocate(kMaximumHostBlock);

  const std::array<size_t, 8> blockSizes = {64,  256, 512, 128,
                                            768, 320, 512, 512};
  size_t offset = 0;
  for (const size_t blockSize : blockSizes) {
    appendDryBlock(processor, input, offset, blockSize, output);
    offset += blockSize;
  }
  ASSERT_EQ(offset, kTotalSamples);

  expectFixedLatency(input, output);
}

TEST(OverlapAddProcessorTest,
     HostDryAndModelAccumulatorAdvanceOnIndependentClocks) {
  audio_plugin::OverlapAddProcessor processor;
  processor.allocate(1024U, 2048U);

  for (int sample = 0; sample < 1000; ++sample) {
    processor.pushDryInputSample(0, 0.25f);
    processor.pushDryInputSample(1, -0.25f);
  }
  EXPECT_EQ(processor.getInputAccumCount(), 0U);

  for (int sample = 0; sample < audio_plugin::kOutputChunkSize; ++sample) {
    processor.pushModelInputSample(0, static_cast<float>(sample));
    processor.pushModelInputSample(1, -static_cast<float>(sample));
  }
  EXPECT_TRUE(processor.readyForInference());
  EXPECT_EQ(processor.getInputAccumCount(),
            static_cast<size_t>(audio_plugin::kOutputChunkSize));
  constexpr size_t last = audio_plugin::kOutputChunkSize - 1U;
  EXPECT_FLOAT_EQ(processor.getInputAccumBuffer()[0].at(last),
                  static_cast<float>(last));
  EXPECT_FLOAT_EQ(processor.getInputAccumBuffer()[1].at(last),
                  -static_cast<float>(last));
}

TEST(OverlapAddProcessorTest,
     MaximumHostBlockDoesNotOverwriteLatencyAlignedDrySamples) {
  constexpr size_t kHostBlockSize = 2048;
  static_assert(kHostBlockSize >
                static_cast<size_t>(audio_plugin::kPluginLatencySamples));
  const StereoBuffer input = makeDistinctStereoInput(kHostBlockSize);
  StereoBuffer output;

  audio_plugin::OverlapAddProcessor processor;
  processor.allocate(kHostBlockSize);
  appendDryBlock(processor, input, 0, kHostBlockSize, output);

  expectFixedLatency(input, output);
}

TEST(OverlapAddProcessorTest,
     HostBlockLargerThanPreparedEstimateDoesNotOverwriteDrySamples) {
  constexpr size_t kPreparedHostBlockSize = 64;
  constexpr size_t kActualHostBlockSize = 8192;
  const StereoBuffer input = makeDistinctStereoInput(kActualHostBlockSize);
  StereoBuffer output;

  audio_plugin::OverlapAddProcessor processor;
  processor.allocate(kPreparedHostBlockSize);
  ASSERT_TRUE(processor.canProcessHostBlock(kActualHostBlockSize));
  appendDryBlock(processor, input, 0, kActualHostBlockSize, output);

  expectFixedLatency(input, output);
}

TEST(OverlapAddProcessorTest,
     OutputRingHoldsACompleteMaximumCallbackAndSchedulingHorizon) {
  constexpr size_t kMaximumHostBlock = 8192;
  constexpr size_t kLatency = 1472;
  constexpr size_t kRequiredCapacity =
      static_cast<size_t>(audio_plugin::kMinimumHostBlockCapacity) + kLatency +
      static_cast<size_t>(audio_plugin::kOutputChunkSize);

  audio_plugin::OverlapAddProcessor processor;
  processor.allocate(kMaximumHostBlock, kLatency);

  EXPECT_GE(processor.getOutputRingSize(), kRequiredCapacity);
  EXPECT_TRUE(processor.canScheduleModelOutput(
      kLatency, static_cast<size_t>(audio_plugin::kMinimumHostBlockCapacity) +
                    static_cast<size_t>(audio_plugin::kOutputChunkSize)));
}

TEST(OverlapAddProcessorTest, FutureModelOutputRemainsOnItsExactTimeline) {
  constexpr uint64_t kScheduledTimeline = 700;
  constexpr size_t kScheduledSamples = 128;

  audio_plugin::OverlapAddProcessor processor;
  processor.allocate();
  ASSERT_TRUE(
      processor.canScheduleModelOutput(kScheduledTimeline, kScheduledSamples));
  processor.markModelOutputScheduled(kScheduledTimeline, kScheduledSamples);

  for (uint64_t timeline = 0; timeline < kScheduledTimeline; ++timeline) {
    EXPECT_FALSE(processor.hasModelOutputForCurrentSample())
        << "timeline=" << timeline;
    processor.advanceOutputTimeline();
  }

  for (size_t i = 0; i < kScheduledSamples; ++i) {
    EXPECT_TRUE(processor.hasModelOutputForCurrentSample()) << "offset=" << i;
    processor.advanceOutputTimeline();
  }
  EXPECT_EQ(processor.getOutputSamplesAvailable(), 0U);
}

TEST(OverlapAddProcessorTest, RejectsModelOutputForElapsedTimeline) {
  audio_plugin::OverlapAddProcessor processor;
  processor.allocate();
  for (size_t i = 0; i < 100; ++i) {
    processor.advanceOutputTimeline();
  }

  EXPECT_FALSE(processor.canScheduleModelOutput(99, 1));
  EXPECT_TRUE(processor.canScheduleModelOutput(100, 1));
}

TEST(OverlapAddProcessorTest, ResetInvalidatesAllScheduledModelOutput) {
  audio_plugin::OverlapAddProcessor processor;
  processor.allocate();
  ASSERT_TRUE(processor.canScheduleModelOutput(0, 512));
  processor.markModelOutputScheduled(0, 512);
  ASSERT_EQ(processor.getOutputSamplesAvailable(), 512U);

  processor.resetIndices();

  EXPECT_EQ(processor.getOutputTimelineSample(), 0U);
  EXPECT_EQ(processor.getOutputSamplesAvailable(), 0U);
  EXPECT_FALSE(processor.hasModelOutputForCurrentSample());
}

TEST(OverlapAddProcessorTest,
     TransportResetInvalidatesStaleStorageWithoutClearingIt) {
  constexpr float kStaleLeft = 0.75f;
  constexpr float kStaleRight = -0.5f;

  audio_plugin::OverlapAddProcessor processor;
  processor.allocate();
  ASSERT_TRUE(processor.canScheduleModelOutput(0, 1));
  processor.markModelOutputScheduled(0, 1);

  const uint64_t staleGeneration =
      audio_plugin::OverlapAddProcessorTestPeer::outputGeneration(processor);
  ASSERT_EQ(
      audio_plugin::OverlapAddProcessorTestPeer::timelineTag(processor, 0), 0U);
  ASSERT_EQ(
      audio_plugin::OverlapAddProcessorTestPeer::generationTag(processor, 0),
      staleGeneration);
  audio_plugin::OverlapAddProcessorTestPeer::seedDryStorage(processor, 0, 0,
                                                            kStaleLeft);
  audio_plugin::OverlapAddProcessorTestPeer::seedDryStorage(processor, 1, 0,
                                                            kStaleRight);

  processor.resetIndices();

  // The backing tags and dry samples deliberately remain untouched. Only the
  // fixed-size generation/index state changes on the audio-thread reset.
  EXPECT_EQ(
      audio_plugin::OverlapAddProcessorTestPeer::timelineTag(processor, 0), 0U);
  EXPECT_EQ(
      audio_plugin::OverlapAddProcessorTestPeer::generationTag(processor, 0),
      staleGeneration);
  EXPECT_EQ(
      audio_plugin::OverlapAddProcessorTestPeer::outputGeneration(processor),
      staleGeneration + 1U);
  EXPECT_FLOAT_EQ(
      audio_plugin::OverlapAddProcessorTestPeer::dryStorage(processor, 0, 0),
      kStaleLeft);
  EXPECT_FLOAT_EQ(
      audio_plugin::OverlapAddProcessorTestPeer::dryStorage(processor, 1, 0),
      kStaleRight);

  EXPECT_FALSE(processor.hasModelOutputForCurrentSample());
  EXPECT_TRUE(processor.canScheduleModelOutput(0, 1));
  EXPECT_FLOAT_EQ(processor.readDryDelaySample(0), 0.0f);
  EXPECT_FLOAT_EQ(processor.readDryDelaySample(1), 0.0f);
}

TEST(OverlapAddProcessorTest,
     DryValidityCounterNeverExposesUnwrittenPostResetStorage) {
  constexpr float kNewLeft = 0.125f;
  constexpr float kNewRight = -0.25f;

  audio_plugin::OverlapAddProcessor processor;
  processor.allocate();
  audio_plugin::OverlapAddProcessorTestPeer::seedDryStorage(processor, 0, 0,
                                                            0.75f);
  audio_plugin::OverlapAddProcessorTestPeer::seedDryStorage(processor, 1, 0,
                                                            -0.5f);
  processor.resetIndices();

  // Even if output advances without matching input, stale storage stays hidden.
  for (size_t i = 0; i <= processor.getLatencySamples(); ++i) {
    EXPECT_FLOAT_EQ(processor.readDryDelaySample(0), 0.0f) << "sample=" << i;
    EXPECT_FLOAT_EQ(processor.readDryDelaySample(1), 0.0f) << "sample=" << i;
    if (i < processor.getLatencySamples()) {
      processor.advanceDryDelayPos();
    }
  }

  processor.pushInputSample(0, kNewLeft);
  processor.pushInputSample(1, kNewRight);
  EXPECT_FLOAT_EQ(processor.readDryDelaySample(0), kNewLeft);
  EXPECT_FLOAT_EQ(processor.readDryDelaySample(1), kNewRight);
}

TEST(OverlapAddProcessorTest, FullResetPhysicallyClearsBackingStorage) {
  audio_plugin::OverlapAddProcessor processor;
  processor.allocate();
  ASSERT_TRUE(processor.canScheduleModelOutput(0, 1));
  processor.markModelOutputScheduled(0, 1);
  audio_plugin::OverlapAddProcessorTestPeer::seedDryStorage(processor, 0, 0,
                                                            0.75f);
  audio_plugin::OverlapAddProcessorTestPeer::seedDryStorage(processor, 1, 0,
                                                            -0.5f);

  processor.reset();

  EXPECT_EQ(
      audio_plugin::OverlapAddProcessorTestPeer::timelineTag(processor, 0),
      std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(
      audio_plugin::OverlapAddProcessorTestPeer::generationTag(processor, 0),
      0U);
  EXPECT_FLOAT_EQ(
      audio_plugin::OverlapAddProcessorTestPeer::dryStorage(processor, 0, 0),
      0.0f);
  EXPECT_FLOAT_EQ(
      audio_plugin::OverlapAddProcessorTestPeer::dryStorage(processor, 1, 0),
      0.0f);
}

TEST(OverlapAddProcessorTest, ResetAndClearRestoreAZeroedDelayLine) {
  constexpr size_t kLatency =
      static_cast<size_t>(audio_plugin::kPluginLatencySamples);
  const StereoBuffer staleInput = makeDistinctStereoInput(kLatency * 2);
  StereoBuffer discarded;

  audio_plugin::OverlapAddProcessor processor;
  processor.allocate(kLatency);
  appendDryBlock(processor, staleInput, 0, kLatency, discarded);
  appendDryBlock(processor, staleInput, kLatency, kLatency, discarded);

  processor.resetIndices();
  processor.clearDryDelayBuffer();

  const StereoBuffer newInput = makeDistinctStereoInput(kLatency + 64);
  StereoBuffer output;
  appendDryBlock(processor, newInput, 0, kLatency, output);
  appendDryBlock(processor, newInput, kLatency, 64, output);

  expectFixedLatency(newInput, output);
}

}  // namespace audio_plugin_test
