#include <StemgenRT/PluginProcessor.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <thread>

namespace audio_plugin_test {
namespace {

class TransportPlayHead final : public juce::AudioPlayHead {
public:
  void setPosition(bool isPlaying, int64_t timeInSamples) {
    position_.setIsPlaying(isPlaying);
    position_.setTimeInSamples(timeInSamples);
  }

  juce::Optional<PositionInfo> getPosition() const override {
    return position_;
  }

private:
  PositionInfo position_;
};

constexpr double kSampleRate = 44100.0;
constexpr int kBlockSize = audio_plugin::kOutputChunkSize;
constexpr int kTotalChannels = 12;  // Stereo input + five stereo outputs.
constexpr float kPi = 3.14159265358979323846f;

float sineAtSample(int64_t sampleIndex, float frequency, float amplitude) {
  const float time =
      static_cast<float>(sampleIndex) / static_cast<float>(kSampleRate);
  return amplitude * std::sin(2.0f * kPi * frequency * time);
}

}  // namespace

TEST(TransportUnderrunE2ETest,
     StoppedTransportDoesNotReportUnderrunsAndPlaybackRecovers) {
  audio_plugin::AudioPluginAudioProcessor processor;
  processor.prepareToPlay(kSampleRate, kBlockSize);
  if (processor.getLatencySamples() <= 0) {
    GTEST_SKIP() << "Qualified model is unavailable";
  }

  TransportPlayHead playHead;
  playHead.setPosition(false, 0);
  processor.setPlayHead(&playHead);
  juce::MidiBuffer midiBuffer;

  // Output is consumed before requests from the same callback are published.
  // A callback extending beyond PDC therefore has no possible model result for
  // its post-latency samples. While stopped, that is idle pipeline fill rather
  // than a missed real-time playback deadline.
  constexpr int kStoppedCallbackSize = 4096;
  juce::AudioBuffer<float> stoppedBuffer(kTotalChannels, kStoppedCallbackSize);
  stoppedBuffer.clear();
  processor.processBlock(stoppedBuffer, midiBuffer);

  EXPECT_FALSE(processor.isUnderrunActive());
  EXPECT_EQ(processor.getUnderrunSamplesInLastBlock(), 0U);
  EXPECT_EQ(processor.getUnderrunSampleCount(), 0U);
  EXPECT_EQ(processor.getUnderrunBlockCount(), 0U);

  // Starting transport must create a clean stream generation. Give the worker
  // comfortably more than one hop between callbacks so this validates state
  // and timeline recovery rather than benchmarking the test machine.
  constexpr int kWarmupBlocks = 8;
  constexpr int kMeasureBlocks = 16;
  constexpr auto kWorkerAllowance = std::chrono::milliseconds(20);
  int64_t playbackSample = 0;
  float maximumRetainedStemMagnitude = 0.0f;
  float maximumReconstructionError = 0.0f;

  for (int block = 0; block < kWarmupBlocks + kMeasureBlocks; ++block) {
    playHead.setPosition(true, playbackSample);
    juce::AudioBuffer<float> buffer(kTotalChannels, kBlockSize);
    buffer.clear();
    auto inputBus = processor.getBusBuffer(buffer, true, 0);
    for (int sample = 0; sample < kBlockSize; ++sample) {
      const int64_t timelineSample = playbackSample + sample;
      inputBus.setSample(0, sample,
                         sineAtSample(timelineSample, 73.0f, 0.30f) +
                             sineAtSample(timelineSample, 509.0f, 0.20f));
      inputBus.setSample(1, sample,
                         sineAtSample(timelineSample, 97.0f, 0.30f) +
                             sineAtSample(timelineSample, 761.0f, 0.20f));
    }

    processor.processBlock(buffer, midiBuffer);

    if (block >= kWarmupBlocks) {
      auto mainBus = processor.getBusBuffer(buffer, false, 0);
      auto drumsBus = processor.getBusBuffer(buffer, false, 1);
      auto bassBus = processor.getBusBuffer(buffer, false, 2);
      auto otherBus = processor.getBusBuffer(buffer, false, 3);
      auto vocalsBus = processor.getBusBuffer(buffer, false, 4);
      for (int channel = 0; channel < audio_plugin::kNumChannels; ++channel) {
        for (int sample = 0; sample < kBlockSize; ++sample) {
          const float drums = drumsBus.getSample(channel, sample);
          const float bass = bassBus.getSample(channel, sample);
          const float other = otherBus.getSample(channel, sample);
          const float vocals = vocalsBus.getSample(channel, sample);
          maximumRetainedStemMagnitude =
              std::max(maximumRetainedStemMagnitude, std::abs(drums));
          maximumRetainedStemMagnitude =
              std::max(maximumRetainedStemMagnitude, std::abs(bass));
          maximumRetainedStemMagnitude =
              std::max(maximumRetainedStemMagnitude, std::abs(vocals));
          maximumReconstructionError =
              std::max(maximumReconstructionError,
                       std::abs(mainBus.getSample(channel, sample) -
                                (drums + bass + other + vocals)));
        }
      }
    }

    playbackSample += kBlockSize;
    std::this_thread::sleep_for(kWorkerAllowance);
  }

  EXPECT_GT(maximumRetainedStemMagnitude, 1.0e-3f)
      << "Playback remained on complete Other fallback after transport start";
  EXPECT_LE(maximumReconstructionError, 1.0e-6f);
  EXPECT_FALSE(processor.isUnderrunActive());
  EXPECT_EQ(processor.getUnderrunSamplesInLastBlock(), 0U);
  EXPECT_EQ(processor.getUnderrunSampleCount(), 0U);
  EXPECT_EQ(processor.getUnderrunBlockCount(), 0U);

  // A play-to-stop transition must discard the old recurrent/output and dry
  // timelines. A silent stopped callback must neither leak the preceding tone
  // nor turn its intentionally missing model output into underrun telemetry.
  playHead.setPosition(false, playbackSample);
  juce::AudioBuffer<float> afterStop(kTotalChannels, kStoppedCallbackSize);
  afterStop.clear();
  processor.processBlock(afterStop, midiBuffer);
  const auto stoppedMain = processor.getBusBuffer(afterStop, false, 0);
  for (int channel = 0; channel < stoppedMain.getNumChannels(); ++channel) {
    for (int sample = 0; sample < stoppedMain.getNumSamples(); ++sample) {
      EXPECT_FLOAT_EQ(stoppedMain.getSample(channel, sample), 0.0f);
    }
  }
  EXPECT_FALSE(processor.isUnderrunActive());
  EXPECT_EQ(processor.getUnderrunSamplesInLastBlock(), 0U);
  EXPECT_EQ(processor.getUnderrunSampleCount(), 0U);
  EXPECT_EQ(processor.getUnderrunBlockCount(), 0U);

  processor.setPlayHead(nullptr);
  processor.releaseResources();
}

TEST(TransportUnderrunE2ETest,
     SmallHostBlocksKeepModelOnExactTimelineWithoutDistortion) {
  constexpr int kSmallBlockSize = 64;
  constexpr int kExpectedLatency =
      3 * audio_plugin::kOutputChunkSize - kSmallBlockSize;
  constexpr int kTotalBlocks = 96;
  constexpr int kMeasurementStartBlock = 32;
  constexpr auto kCallbackInterval = std::chrono::microseconds(1400);

  audio_plugin::AudioPluginAudioProcessor processor;
  processor.prepareToPlay(kSampleRate, kSmallBlockSize);
  if (processor.getLatencySamples() <= 0) {
    GTEST_SKIP() << "Qualified model is unavailable";
  }
  ASSERT_EQ(processor.getLatencySamples(), kExpectedLatency);

  TransportPlayHead playHead;
  processor.setPlayHead(&playHead);
  juce::MidiBuffer midiBuffer;
  int64_t inputTimelineSample = 0;
  float maximumMainDelayError = 0.0f;
  float maximumRetainedStemMagnitude = 0.0f;
  float maximumReconstructionError = 0.0f;

  for (int block = 0; block < kTotalBlocks; ++block) {
    playHead.setPosition(true, inputTimelineSample);
    juce::AudioBuffer<float> buffer(kTotalChannels, kSmallBlockSize);
    buffer.clear();
    auto inputBus = processor.getBusBuffer(buffer, true, 0);
    for (int sample = 0; sample < kSmallBlockSize; ++sample) {
      const int64_t timelineSample = inputTimelineSample + sample;
      inputBus.setSample(0, sample,
                         sineAtSample(timelineSample, 73.0f, 0.30f) +
                             sineAtSample(timelineSample, 509.0f, 0.20f));
      inputBus.setSample(1, sample,
                         sineAtSample(timelineSample, 97.0f, 0.30f) +
                             sineAtSample(timelineSample, 761.0f, 0.20f));
    }

    processor.processBlock(buffer, midiBuffer);

    auto mainBus = processor.getBusBuffer(buffer, false, 0);
    auto drumsBus = processor.getBusBuffer(buffer, false, 1);
    auto bassBus = processor.getBusBuffer(buffer, false, 2);
    auto otherBus = processor.getBusBuffer(buffer, false, 3);
    auto vocalsBus = processor.getBusBuffer(buffer, false, 4);
    for (int channel = 0; channel < audio_plugin::kNumChannels; ++channel) {
      for (int sample = 0; sample < kSmallBlockSize; ++sample) {
        const int64_t outputTimelineSample = inputTimelineSample + sample;
        float expectedMain = 0.0f;
        if (outputTimelineSample >= kExpectedLatency) {
          const int64_t delayedSample = outputTimelineSample - kExpectedLatency;
          expectedMain = channel == 0
                             ? sineAtSample(delayedSample, 73.0f, 0.30f) +
                                   sineAtSample(delayedSample, 509.0f, 0.20f)
                             : sineAtSample(delayedSample, 97.0f, 0.30f) +
                                   sineAtSample(delayedSample, 761.0f, 0.20f);
        }
        maximumMainDelayError = std::max(
            maximumMainDelayError,
            std::abs(mainBus.getSample(channel, sample) - expectedMain));

        const float drums = drumsBus.getSample(channel, sample);
        const float bass = bassBus.getSample(channel, sample);
        const float other = otherBus.getSample(channel, sample);
        const float vocals = vocalsBus.getSample(channel, sample);
        maximumReconstructionError =
            std::max(maximumReconstructionError,
                     std::abs(mainBus.getSample(channel, sample) -
                              (drums + bass + other + vocals)));
        if (block >= kMeasurementStartBlock) {
          maximumRetainedStemMagnitude =
              std::max(maximumRetainedStemMagnitude, std::abs(drums));
          maximumRetainedStemMagnitude =
              std::max(maximumRetainedStemMagnitude, std::abs(bass));
          maximumRetainedStemMagnitude =
              std::max(maximumRetainedStemMagnitude, std::abs(vocals));
        }
      }
    }

    inputTimelineSample += kSmallBlockSize;
    std::this_thread::sleep_for(kCallbackInterval);
  }

  EXPECT_LE(maximumMainDelayError, 1.0e-6f)
      << "Main was not an exact latency-aligned copy of the input";
  EXPECT_GT(maximumRetainedStemMagnitude, 1.0e-3f)
      << "Small callbacks remained on complete Other fallback";
  EXPECT_LE(maximumReconstructionError, 1.0e-6f);
  EXPECT_FALSE(processor.isUnderrunActive());
  EXPECT_EQ(processor.getUnderrunSamplesInLastBlock(), 0U);
  EXPECT_EQ(processor.getUnderrunSampleCount(), 0U);
  EXPECT_EQ(processor.getUnderrunBlockCount(), 0U);
  EXPECT_EQ(processor.getQueueFullChunkDropCount(), 0U);
  EXPECT_EQ(processor.getRingOverflowEventCount(), 0U);
  EXPECT_FALSE(processor.isRealtimeCallbackTimingUnsafe());
  EXPECT_EQ(processor.getUnsafeRealtimeCallbackCount(), 0U);

  processor.setPlayHead(nullptr);
  processor.releaseResources();
}

TEST(TransportUnderrunE2ETest,
     PreparedBlockPdcMismatchFallsBackLosslesslyAndReportsUnsafeTiming) {
  constexpr int kPreparedBlockSize = 512;
  constexpr int kActualBlockSize = 64;
  constexpr int kTotalBlocks = 48;
  constexpr int kRequiredLatency = 1472;

  audio_plugin::AudioPluginAudioProcessor processor;
  processor.prepareToPlay(kSampleRate, kPreparedBlockSize);
  if (processor.getLatencySamples() <= 0) {
    GTEST_SKIP() << "Qualified model is unavailable";
  }
  ASSERT_EQ(processor.getLatencySamples(), audio_plugin::kPluginLatencySamples);
  ASSERT_EQ(processor.getPreparedHostBlockSize(), kPreparedBlockSize);
  ASSERT_EQ(processor.getUnsafeRealtimeCallbackCount(), 0U);

  TransportPlayHead playHead;
  processor.setPlayHead(&playHead);
  juce::MidiBuffer midiBuffer;
  int64_t timelineSample = 0;
  uint64_t fallbackFramesChecked = 0;
  float maximumMainDelayError = 0.0f;
  float maximumReconstructionError = 0.0f;

  for (int block = 0; block < kTotalBlocks; ++block) {
    playHead.setPosition(true, timelineSample);
    juce::AudioBuffer<float> buffer(kTotalChannels, kActualBlockSize);
    buffer.clear();
    auto inputBus = processor.getBusBuffer(buffer, true, 0);
    for (int sample = 0; sample < kActualBlockSize; ++sample) {
      const int64_t inputSample = timelineSample + sample;
      inputBus.setSample(0, sample,
                         sineAtSample(inputSample, 73.0f, 0.30f) +
                             sineAtSample(inputSample, 509.0f, 0.20f));
      inputBus.setSample(1, sample,
                         sineAtSample(inputSample, 97.0f, 0.30f) +
                             sineAtSample(inputSample, 761.0f, 0.20f));
    }

    // Deliberately do not pace these callbacks. The configured PDC is too
    // short for this real-time callback size, so exact-timeline misses must use
    // complete Other fallback without changing the host's reported latency.
    processor.processBlock(buffer, midiBuffer);

    EXPECT_EQ(processor.getLatencySamples(),
              audio_plugin::kPluginLatencySamples);
    EXPECT_EQ(processor.getLastHostBlockSize(), kActualBlockSize);
    EXPECT_EQ(processor.getRequiredLatencySamplesForLastHostBlock(),
              kRequiredLatency);
    EXPECT_TRUE(processor.isRealtimeCallbackTimingUnsafe());
    EXPECT_EQ(processor.getUnsafeRealtimeCallbackCount(),
              static_cast<uint64_t>(block + 1));

    const size_t unavailableSamples = processor.getUnderrunSamplesInLastBlock();
    if (unavailableSamples > 0) {
      // Model output is hop-aligned and 64 divides 512, so a callback cannot
      // contain a partial model-availability transition in this scenario.
      ASSERT_EQ(unavailableSamples, static_cast<size_t>(kActualBlockSize));
    }

    auto mainBus = processor.getBusBuffer(buffer, false, 0);
    auto drumsBus = processor.getBusBuffer(buffer, false, 1);
    auto bassBus = processor.getBusBuffer(buffer, false, 2);
    auto otherBus = processor.getBusBuffer(buffer, false, 3);
    auto vocalsBus = processor.getBusBuffer(buffer, false, 4);
    for (int channel = 0; channel < audio_plugin::kNumChannels; ++channel) {
      for (int sample = 0; sample < kActualBlockSize; ++sample) {
        const int64_t outputSample = timelineSample + sample;
        float expectedMain = 0.0f;
        if (outputSample >= audio_plugin::kPluginLatencySamples) {
          const int64_t delayedSample =
              outputSample - audio_plugin::kPluginLatencySamples;
          expectedMain = channel == 0
                             ? sineAtSample(delayedSample, 73.0f, 0.30f) +
                                   sineAtSample(delayedSample, 509.0f, 0.20f)
                             : sineAtSample(delayedSample, 97.0f, 0.30f) +
                                   sineAtSample(delayedSample, 761.0f, 0.20f);
        }

        const float main = mainBus.getSample(channel, sample);
        const float drums = drumsBus.getSample(channel, sample);
        const float bass = bassBus.getSample(channel, sample);
        const float other = otherBus.getSample(channel, sample);
        const float vocals = vocalsBus.getSample(channel, sample);
        EXPECT_TRUE(std::isfinite(main));
        EXPECT_TRUE(std::isfinite(drums));
        EXPECT_TRUE(std::isfinite(bass));
        EXPECT_TRUE(std::isfinite(other));
        EXPECT_TRUE(std::isfinite(vocals));
        maximumMainDelayError =
            std::max(maximumMainDelayError, std::abs(main - expectedMain));
        maximumReconstructionError =
            std::max(maximumReconstructionError,
                     std::abs(main - (drums + bass + other + vocals)));

        // Unsafe callback timing disables presentation of every scheduled
        // model sample for the complete callback. The worker may continue to
        // advance, but no late suffix may leak into a retained source.
        EXPECT_FLOAT_EQ(drums, 0.0f);
        EXPECT_FLOAT_EQ(bass, 0.0f);
        EXPECT_FLOAT_EQ(vocals, 0.0f);
        EXPECT_FLOAT_EQ(other, main);
        ++fallbackFramesChecked;
      }
    }

    timelineSample += kActualBlockSize;
  }

  EXPECT_EQ(fallbackFramesChecked,
            static_cast<uint64_t>(kTotalBlocks * kActualBlockSize *
                                  audio_plugin::kNumChannels));
  EXPECT_LE(maximumMainDelayError, 1.0e-6f);
  EXPECT_LE(maximumReconstructionError, 1.0e-6f);
  EXPECT_TRUE(processor.getOrtStatusString().containsIgnoreCase("PDC warning"));
  EXPECT_EQ(processor.getUnsafeRealtimeCallbackCount(),
            static_cast<uint64_t>(kTotalBlocks));
  EXPECT_EQ(processor.getLatencySamples(), audio_plugin::kPluginLatencySamples);

  processor.setPlayHead(nullptr);
  processor.releaseResources();
}

TEST(TransportUnderrunE2ETest,
     OfflineRenderWaitsForModelAcrossMixedCallbackSizes) {
  constexpr std::array<int, 8> kCallbackSizes = {512, 64,   960, 128,
                                                 37,  1024, 255, 512};
  constexpr int kPatternRepeats = 8;
  constexpr int64_t kMeasurementStartSample = 4096;

  audio_plugin::AudioPluginAudioProcessor processor;
  processor.setNonRealtime(true);
  processor.prepareToPlay(kSampleRate, kBlockSize);
  if (processor.getLatencySamples() <= 0) {
    GTEST_SKIP() << "Qualified model is unavailable";
  }
  ASSERT_EQ(processor.getLatencySamples(), audio_plugin::kPluginLatencySamples);

  const int latency = processor.getLatencySamples();
  juce::MidiBuffer midiBuffer;
  int64_t timelineSample = 0;
  float maximumMainDelayError = 0.0f;
  float maximumRetainedStemMagnitude = 0.0f;
  float maximumReconstructionError = 0.0f;

  for (int repeat = 0; repeat < kPatternRepeats; ++repeat) {
    for (const int callbackSize : kCallbackSizes) {
      juce::AudioBuffer<float> buffer(kTotalChannels, callbackSize);
      buffer.clear();
      auto inputBus = processor.getBusBuffer(buffer, true, 0);
      for (int sample = 0; sample < callbackSize; ++sample) {
        const int64_t inputSample = timelineSample + sample;
        inputBus.setSample(0, sample,
                           sineAtSample(inputSample, 73.0f, 0.30f) +
                               sineAtSample(inputSample, 509.0f, 0.20f));
        inputBus.setSample(1, sample,
                           sineAtSample(inputSample, 97.0f, 0.30f) +
                               sineAtSample(inputSample, 761.0f, 0.20f));
      }

      // Deliberately no sleep: an offline host may call processBlock as fast
      // as the CPU can render and may vary callback sizes within one bounce.
      processor.processBlock(buffer, midiBuffer);

      auto mainBus = processor.getBusBuffer(buffer, false, 0);
      auto drumsBus = processor.getBusBuffer(buffer, false, 1);
      auto bassBus = processor.getBusBuffer(buffer, false, 2);
      auto otherBus = processor.getBusBuffer(buffer, false, 3);
      auto vocalsBus = processor.getBusBuffer(buffer, false, 4);
      for (int channel = 0; channel < audio_plugin::kNumChannels; ++channel) {
        for (int sample = 0; sample < callbackSize; ++sample) {
          const int64_t outputSample = timelineSample + sample;
          float expectedMain = 0.0f;
          if (outputSample >= latency) {
            const int64_t delayedSample = outputSample - latency;
            expectedMain = channel == 0
                               ? sineAtSample(delayedSample, 73.0f, 0.30f) +
                                     sineAtSample(delayedSample, 509.0f, 0.20f)
                               : sineAtSample(delayedSample, 97.0f, 0.30f) +
                                     sineAtSample(delayedSample, 761.0f, 0.20f);
          }
          maximumMainDelayError = std::max(
              maximumMainDelayError,
              std::abs(mainBus.getSample(channel, sample) - expectedMain));

          const float drums = drumsBus.getSample(channel, sample);
          const float bass = bassBus.getSample(channel, sample);
          const float other = otherBus.getSample(channel, sample);
          const float vocals = vocalsBus.getSample(channel, sample);
          maximumReconstructionError =
              std::max(maximumReconstructionError,
                       std::abs(mainBus.getSample(channel, sample) -
                                (drums + bass + other + vocals)));
          if (outputSample >= kMeasurementStartSample) {
            maximumRetainedStemMagnitude =
                std::max(maximumRetainedStemMagnitude, std::abs(drums));
            maximumRetainedStemMagnitude =
                std::max(maximumRetainedStemMagnitude, std::abs(bass));
            maximumRetainedStemMagnitude =
                std::max(maximumRetainedStemMagnitude, std::abs(vocals));
          }
        }
      }

      timelineSample += callbackSize;
    }
  }

  EXPECT_LE(maximumMainDelayError, 1.0e-6f);
  EXPECT_GT(maximumRetainedStemMagnitude, 1.0e-3f)
      << "Offline bounce never consumed separated model output";
  EXPECT_LE(maximumReconstructionError, 1.0e-6f);
  EXPECT_EQ(processor.getQueueFullChunkDropCount(), 0U);
  EXPECT_EQ(processor.getRingOverflowEventCount(), 0U);
  EXPECT_EQ(processor.getUnderrunSampleCount(), 0U);
  EXPECT_EQ(processor.getUnderrunBlockCount(), 0U);
  EXPECT_FALSE(processor.isRealtimeCallbackTimingUnsafe());
  EXPECT_EQ(processor.getUnsafeRealtimeCallbackCount(), 0U);

  processor.releaseResources();
}

}  // namespace audio_plugin_test
