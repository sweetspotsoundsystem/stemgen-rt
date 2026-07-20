#include <StemgenRT/Constants.h>
#include <StemgenRT/OverlapAddProcessor.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <vector>

namespace audio_plugin_test {
namespace {

using StereoBuffer =
    std::array<std::vector<float>, audio_plugin::kNumChannels>;

void appendDryBlock(audio_plugin::OverlapAddProcessor& processor,
                    const StereoBuffer& input, size_t offset, size_t count,
                    StereoBuffer& output) {
  ASSERT_LE(offset + count, input[0].size());
  ASSERT_EQ(input[0].size(), input[1].size());

  // Match PluginProcessor: the complete host block is accumulated before the
  // latency-aligned fallback is read.
  for (size_t i = 0; i < count; ++i) {
    for (int ch = 0; ch < audio_plugin::kNumChannels; ++ch) {
      processor.pushInputSample(
          ch, input[static_cast<size_t>(ch)][offset + i]);
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

void expectFixedLatency(const StereoBuffer& input,
                        const StereoBuffer& output) {
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

  const std::array<size_t, 8> blockSizes = {64, 256, 512, 128,
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
