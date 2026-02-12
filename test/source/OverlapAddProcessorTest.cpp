#include <StemgenRT/Constants.h>
#include <StemgenRT/OverlapAddProcessor.h>
#include <gtest/gtest.h>
#include <vector>

namespace audio_plugin_test {

namespace {

void pushDryBlock(audio_plugin::OverlapAddProcessor& processor,
                  const std::vector<float>& left,
                  const std::vector<float>& right) {
  ASSERT_EQ(left.size(), right.size());
  for (size_t i = 0; i < left.size(); ++i) {
    processor.pushInputSample(0, 0.0f, 0.0f, left[i]);
    processor.pushInputSample(1, 0.0f, 0.0f, right[i]);
  }
}

std::vector<float> readDrySamples(audio_plugin::OverlapAddProcessor& processor,
                                  int channel,
                                  size_t count) {
  std::vector<float> out(count);
  for (size_t i = 0; i < count; ++i) {
    out[i] = processor.readDryDelaySample(channel);
    processor.advanceDryDelayPos();
  }
  return out;
}

}  // namespace

TEST(OverlapAddProcessorTest, PrimeDryDelayDoesNotTileShortHostBlock) {
  audio_plugin::OverlapAddProcessor processor;
  processor.allocate();

  // Simulate RT reset behavior: indices are reset but dry delay storage is retained.
  std::vector<float> stale(audio_plugin::kOutputChunkSize * 2, -1.0f);
  pushDryBlock(processor, stale, stale);
  processor.resetIndices();

  constexpr int kHostBlockSize = 64;
  std::vector<float> input(static_cast<size_t>(kHostBlockSize));
  for (int i = 0; i < kHostBlockSize; ++i) {
    input[static_cast<size_t>(i)] = static_cast<float>(i + 1);
  }

  pushDryBlock(processor, input, input);

  const float* inputPointers[audio_plugin::kNumChannels] = {
      input.data(), input.data()};
  processor.primeDryDelayFromInput(inputPointers, kHostBlockSize);

  auto primed = readDrySamples(processor, 0, audio_plugin::kOutputChunkSize);

  for (int i = 0; i < kHostBlockSize; ++i) {
    EXPECT_FLOAT_EQ(primed[static_cast<size_t>(i)], input[static_cast<size_t>(i)]);
  }
  for (int i = kHostBlockSize; i < audio_plugin::kOutputChunkSize; ++i) {
    EXPECT_FLOAT_EQ(primed[static_cast<size_t>(i)], 0.0f);
  }
}

TEST(OverlapAddProcessorTest, PrimeDryDelayKeepsNewestWrappedSamplesForLargeHostBlock) {
  audio_plugin::OverlapAddProcessor withPriming;
  withPriming.allocate();
  withPriming.resetIndices();

  audio_plugin::OverlapAddProcessor withoutPriming;
  withoutPriming.allocate();
  withoutPriming.resetIndices();

  constexpr int kHostBlockSize = 1024;
  static_assert(kHostBlockSize > audio_plugin::kOutputChunkSize);
  std::vector<float> input(static_cast<size_t>(kHostBlockSize));
  for (int i = 0; i < kHostBlockSize; ++i) {
    input[static_cast<size_t>(i)] = static_cast<float>(i + 1);
  }

  pushDryBlock(withPriming, input, input);
  pushDryBlock(withoutPriming, input, input);

  const float* inputPointers[audio_plugin::kNumChannels] = {
      input.data(), input.data()};
  withPriming.primeDryDelayFromInput(inputPointers, kHostBlockSize);

  auto primed = readDrySamples(withPriming, 0, audio_plugin::kOutputChunkSize);
  auto baseline = readDrySamples(withoutPriming, 0, audio_plugin::kOutputChunkSize);

  for (int i = 0; i < audio_plugin::kOutputChunkSize; ++i) {
    EXPECT_FLOAT_EQ(primed[static_cast<size_t>(i)], baseline[static_cast<size_t>(i)]);
    EXPECT_FLOAT_EQ(primed[static_cast<size_t>(i)],
                    input[static_cast<size_t>(i + audio_plugin::kOutputChunkSize)]);
  }
}

}  // namespace audio_plugin_test
