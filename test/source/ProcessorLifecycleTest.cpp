#include <StemgenRT/PluginProcessor.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>

#include "AudioPluginProcessorTestPeer.h"

#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
namespace audio_plugin_test {
namespace {
using audio_plugin::AudioPluginAudioProcessor;
using audio_plugin::AudioPluginProcessorTestPeer;

float inputSample(int sample, int channel) {
  return static_cast<float>((sample * 7 + channel * 13) % 257 - 128) / 512.0f;
}

void fillInput(juce::AudioBuffer<float>& buffer, int firstSample) {
  buffer.clear();
  for (int ch = 0; ch < 2; ++ch) {
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
      buffer.setSample(ch, i, inputSample(firstSample + i, ch));
    }
  }
}

bool waitForSubmission(AudioPluginAudioProcessor& processor) {
  // Control-thread correctness synchronization only. The following callback
  // still claims results through the production scheduler, without waiting.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!AudioPluginProcessorTestPeer::submissionCompleted(processor) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return AudioPluginProcessorTestPeer::submissionCompleted(processor);
}

void checkOutput(AudioPluginAudioProcessor& processor,
                 juce::AudioBuffer<float>& buffer,
                 int firstSample,
                 bool bypassed) {
  auto main = processor.getBusBuffer(buffer, false, 0);
  float maximumMainError = 0.0f;
  float maximumReconstructionError = 0.0f;
  float maximumBypassError = 0.0f;
  for (int ch = 0; ch < 2; ++ch) {
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
      const int sourceSample = firstSample + i - processor.getLatencySamples();
      const float expected =
          sourceSample < 0 ? 0.0f : inputSample(sourceSample, ch);
      maximumMainError = std::max(maximumMainError,
                                  std::abs(main.getSample(ch, i) - expected));
      float sum = 0.0f;
      for (int bus = 1; bus < 5; ++bus) {
        auto stem = processor.getBusBuffer(buffer, false, bus);
        if (stem.getNumChannels() == 0) {
          continue;
        }
        const float value = stem.getSample(ch, i);
        ASSERT_TRUE(std::isfinite(value));
        sum += value;
        if (bypassed) {
          maximumBypassError =
              std::max(maximumBypassError,
                       std::abs(value - (bus == 3 ? expected : 0.0f)));
        }
      }
      // Sparse routing cannot reconstruct sources on disabled buses, except
      // during complete bypass when the enabled Other bus holds the mixture.
      if (bypassed || processor.getTotalNumOutputChannels() == 10) {
        maximumReconstructionError =
            std::max(maximumReconstructionError, std::abs(sum - expected));
      }
    }
  }
  EXPECT_FLOAT_EQ(maximumMainError, 0.0f);
  EXPECT_LE(maximumReconstructionError, 1.0e-6f);
  EXPECT_FLOAT_EQ(maximumBypassError, 0.0f);
}

class PositionlessPlayHead final : public juce::AudioPlayHead {
public:
  juce::Optional<PositionInfo> getPosition() const override {
    PositionInfo position;
    position.setIsPlaying(true);
    return position;
  }
};
}  // namespace

class BypassLifecycleTest : public ::testing::TestWithParam<int> {};

TEST_P(BypassLifecycleTest, PreservesDelayAndRoutingAcrossBypassTransitions) {
  const int blockSize = GetParam();
  for (const bool sparse : {false, true}) {
    SCOPED_TRACE(sparse);
    AudioPluginAudioProcessor processor;
    if (sparse) {
      auto layout = processor.getBusesLayout();
      layout.outputBuses.set(1, juce::AudioChannelSet::disabled());
      layout.outputBuses.set(2, juce::AudioChannelSet::disabled());
      ASSERT_TRUE(processor.setBusesLayout(layout));
    }
    processor.setNonRealtime(false);
    processor.prepareToPlay(44100.0, blockSize);
    const int latency = processor.getLatencySamples();
    ASSERT_GT(latency, 0) << processor.getOrtStatusString();
    auto& host = static_cast<juce::AudioProcessor&>(processor);
    juce::AudioBuffer<float> buffer(processor.getTotalNumOutputChannels(),
                                    blockSize);
    juce::MidiBuffer midi;
    float resumedStemPeak = 0.0f;
    for (int block = 0; block < 18; ++block) {
      // Start bypassed, resume, bypass again, then resume. Distinct sample
      // values reveal stale input, an extra delay or a timeline jump.
      const bool bypassed = block < 4 || (block >= 8 && block < 12);
      fillInput(buffer, block * blockSize);
      if (bypassed) {
        host.processBlockBypassed(buffer, midi);
      } else {
        host.processBlock(buffer, midi);
      }
      EXPECT_EQ(host.getLatencySamples(), latency);
      checkOutput(processor, buffer, block * blockSize, bypassed);
      if (block >= 12) {
        auto vocals = processor.getBusBuffer(buffer, false, 4);
        resumedStemPeak =
            std::max(resumedStemPeak, vocals.getMagnitude(0, blockSize));
      }
      ASSERT_TRUE(waitForSubmission(processor));
    }
    EXPECT_GT(resumedStemPeak, 1.0e-6f);
    EXPECT_EQ(processor.getUnderrunSampleCount(), 0U);
    EXPECT_EQ(processor.getSameCallbackTimeoutCount(), 0U);
    EXPECT_EQ(processor.getQueueFullChunkDropCount(), 0U);
    processor.releaseResources();
  }
}

INSTANTIATE_TEST_SUITE_P(HostBuffers,
                         BypassLifecycleTest,
                         ::testing::Values(128, 256));

TEST(ProcessorLifecycleTest, HostResetMatchesFreshStreamWithQueuedResults) {
  // A playing host without sample positions cannot repair a missed reset
  // through the processor's separate seek detector.
  PositionlessPlayHead playHead;
  AudioPluginAudioProcessor processor;
  AudioPluginAudioProcessor fresh;
  for (auto* instance : {&processor, &fresh}) {
    instance->setPlayHead(&playHead);
    instance->setNonRealtime(false);
    instance->prepareToPlay(44100.0, 128);
    ASSERT_EQ(instance->getLatencySamples(), 256)
        << instance->getOrtStatusString();
  }
  juce::AudioBuffer<float> buffer(10, 128);
  juce::AudioBuffer<float> reference(10, 128);
  juce::MidiBuffer midi;
  for (int block = 0; block < 4; ++block) {
    fillInput(buffer, 10000 + block * 128);
    processor.processBlock(buffer, midi);
    ASSERT_TRUE(waitForSubmission(processor));
  }
  // The previous output is published but has not yet been claimed. Reset
  // must invalidate it, the dry history and all model state before use.
  static_cast<juce::AudioProcessor&>(processor).reset();
  for (int block = 0; block < 8; ++block) {
    fillInput(buffer, block * 128);
    fillInput(reference, block * 128);
    processor.processBlock(buffer, midi);
    fresh.processBlock(reference, midi);
    checkOutput(processor, buffer, block * 128, false);
    for (int ch = 0; ch < 10; ++ch) {
      for (int i = 0; i < 128; ++i) {
        ASSERT_NEAR(buffer.getSample(ch, i), reference.getSample(ch, i),
                    1.0e-7f);
      }
    }
    ASSERT_TRUE(waitForSubmission(processor));
    ASSERT_TRUE(waitForSubmission(fresh));
  }
  EXPECT_EQ(processor.getLatencySamples(), 256);
  EXPECT_EQ(processor.getUnderrunSampleCount(), 0U);
  processor.releaseResources();
  fresh.releaseResources();
}

TEST(ProcessorLifecycleTest, ResetIsSafeOutsideAnActiveStream) {
  AudioPluginAudioProcessor processor;
  auto& host = static_cast<juce::AudioProcessor&>(processor);
  host.reset();
  processor.prepareToPlay(48000.0, 128);
  ASSERT_EQ(processor.getLatencySamples(), 0);
  juce::AudioBuffer<float> buffer(10, 128);
  juce::MidiBuffer midi;
  fillInput(buffer, 0);
  host.processBlockBypassed(buffer, midi);
  checkOutput(processor, buffer, 0, true);
  processor.releaseResources();
  host.reset();
  fillInput(buffer, 128);
  host.processBlock(buffer, midi);
  checkOutput(processor, buffer, 128, true);
}

class LargeCallbackTest : public ::testing::TestWithParam<int> {};

TEST_P(LargeCallbackTest, EntirePreparedBurstSurvivesQueueWraps) {
  const int blockSize = GetParam();
  AudioPluginAudioProcessor processor;
  processor.setNonRealtime(false);
  processor.prepareToPlay(44100.0, blockSize);
  ASSERT_GT(processor.getLatencySamples(), 0) << processor.getOrtStatusString();
  juce::AudioBuffer<float> buffer(10, blockSize);
  juce::MidiBuffer midi;
  float retainedPeak = 0.0f;
  // Four full bursts wrap the dynamically sized queue multiple times.
  for (int block = 0; block < 4; ++block) {
    fillInput(buffer, block * blockSize);
    processor.processBlock(buffer, midi);
    checkOutput(processor, buffer, block * blockSize, false);
    retainedPeak = std::max(retainedPeak, buffer.getMagnitude(2, 0, blockSize));
    ASSERT_TRUE(waitForSubmission(processor));
  }
  EXPECT_GT(retainedPeak, 1.0e-6f);
  EXPECT_FALSE(processor.isRealtimeCallbackTimingUnsafe());
  EXPECT_EQ(processor.getQueueFullChunkDropCount(), 0U);
  EXPECT_EQ(processor.getUnderrunSampleCount(), 0U);
  processor.releaseResources();
  // Resizing back down must also reset every ring index.
  processor.prepareToPlay(44100.0, 128);
  buffer.setSize(10, 128);
  for (int block = 0; block < 4; ++block) {
    fillInput(buffer, block * 128);
    processor.processBlock(buffer, midi);
    checkOutput(processor, buffer, block * 128, false);
    ASSERT_TRUE(waitForSubmission(processor));
  }
  EXPECT_EQ(processor.getQueueFullChunkDropCount(), 0U);
  EXPECT_EQ(processor.getUnderrunSampleCount(), 0U);
  processor.releaseResources();
}

INSTANTIATE_TEST_SUITE_P(HostBuffers,
                         LargeCallbackTest,
                         ::testing::Values(2048, 4095, 4096, 65536));
}  // namespace audio_plugin_test
#endif
