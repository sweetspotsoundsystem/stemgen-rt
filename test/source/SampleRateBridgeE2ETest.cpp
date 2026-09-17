#include <StemgenRT/PluginProcessor.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <thread>
#include <utility>
#include <vector>

#include "AudioPluginProcessorTestPeer.h"

namespace audio_plugin_test {
namespace {

constexpr int kBlockSize = 512;
constexpr double kPi = 3.1415926535897932384626433832795;

float makeInputSample(std::size_t timelineSample,
                      double sampleRate,
                      int channel) {
  const double time = static_cast<double>(timelineSample) / sampleRate;
  const double polarity = channel == 0 ? 1.0 : -0.73;
  return static_cast<float>(polarity *
                            (0.22 * std::sin(2.0 * kPi * 83.0 * time) +
                             0.16 * std::sin(2.0 * kPi * 997.0 * time) +
                             0.08 * std::sin(2.0 * kPi * 7311.0 * time)));
}

void expectRateRoundTrip(double sampleRate, int blockSize) {
  audio_plugin::AudioPluginAudioProcessor processor;
  processor.setNonRealtime(true);
  processor.prepareToPlay(sampleRate, blockSize);
  ASSERT_GT(processor.getLatencySamples(), 0) << processor.getOrtStatusString();

  const int latency = processor.getLatencySamples();
  ASSERT_GT(latency, audio_plugin::calculateModelSchedulingLatencySamples(
                         static_cast<int>(sampleRate), blockSize));
  EXPECT_NEAR(processor.getLatencyMs(),
              1000.0 * static_cast<double>(latency) / sampleRate, 1.0e-12);
  const std::size_t totalSamples =
      static_cast<std::size_t>(latency + 8 * blockSize);
  std::array<std::vector<float>, audio_plugin::kNumChannels> input;
  std::array<std::vector<float>, audio_plugin::kNumChannels> main;
  for (auto& channel : input) {
    channel.reserve(totalSamples);
  }
  for (auto& channel : main) {
    channel.reserve(totalSamples);
  }

  double retainedPeak = 0.0;
  juce::MidiBuffer midi;
  for (std::size_t blockStart = 0U; blockStart < totalSamples;
       blockStart += static_cast<std::size_t>(blockSize)) {
    juce::AudioBuffer<float> buffer(12, blockSize);
    buffer.clear();
    auto inputBus = processor.getBusBuffer(buffer, true, 0);
    for (int sample = 0; sample < blockSize; ++sample) {
      const std::size_t timeline =
          blockStart + static_cast<std::size_t>(sample);
      for (int channel = 0; channel < audio_plugin::kNumChannels; ++channel) {
        const float value = makeInputSample(timeline, sampleRate, channel);
        input[static_cast<std::size_t>(channel)].push_back(value);
        inputBus.setSample(channel, sample, value);
      }
    }

    processor.processBlock(buffer, midi);
    const auto mainBus = processor.getBusBuffer(buffer, false, 0);
    const auto drumsBus = processor.getBusBuffer(buffer, false, 1);
    const auto bassBus = processor.getBusBuffer(buffer, false, 2);
    const auto otherBus = processor.getBusBuffer(buffer, false, 3);
    const auto vocalsBus = processor.getBusBuffer(buffer, false, 4);
    for (int sample = 0; sample < blockSize; ++sample) {
      for (int channel = 0; channel < audio_plugin::kNumChannels; ++channel) {
        const float mainSample = mainBus.getSample(channel, sample);
        main[static_cast<std::size_t>(channel)].push_back(mainSample);
        const float drums = drumsBus.getSample(channel, sample);
        const float bass = bassBus.getSample(channel, sample);
        const float other = otherBus.getSample(channel, sample);
        const float vocals = vocalsBus.getSample(channel, sample);
        EXPECT_TRUE(std::isfinite(mainSample));
        EXPECT_TRUE(std::isfinite(drums));
        EXPECT_TRUE(std::isfinite(bass));
        EXPECT_TRUE(std::isfinite(other));
        EXPECT_TRUE(std::isfinite(vocals));
        EXPECT_NEAR(mainSample, drums + bass + other + vocals, 1.0e-6f);
        retainedPeak = std::max(
            retainedPeak,
            static_cast<double>(
                std::max({std::abs(drums), std::abs(bass), std::abs(vocals)})));
      }
    }
  }

  for (std::size_t channel = 0U;
       channel < static_cast<std::size_t>(audio_plugin::kNumChannels);
       ++channel) {
    ASSERT_EQ(main[channel].size(), input[channel].size());
    for (std::size_t timeline = 0U; timeline < main[channel].size();
         ++timeline) {
      const float expected =
          timeline < static_cast<std::size_t>(latency)
              ? 0.0f
              : input[channel][timeline - static_cast<std::size_t>(latency)];
      EXPECT_FLOAT_EQ(main[channel][timeline], expected)
          << "rate=" << sampleRate << " channel=" << channel
          << " timeline=" << timeline;
    }
  }
  EXPECT_GT(retainedPeak, 1.0e-5) << "rate=" << sampleRate;
  processor.releaseResources();
}

void expectFiniteRenderCompletion(double sampleRate) {
  audio_plugin::AudioPluginAudioProcessor processor;
  processor.setNonRealtime(true);
  processor.prepareToPlay(sampleRate, kBlockSize);
  ASSERT_GT(processor.getLatencySamples(), 0) << processor.getOrtStatusString();

  constexpr std::size_t kSignalSamples =
      2U * static_cast<std::size_t>(kBlockSize) + 137U;
  const int latency = processor.getLatencySamples();
  const std::size_t declaredTailSamples = static_cast<std::size_t>(
      std::ceil(processor.getTailLengthSeconds() * sampleRate - 1.0e-9));
  ASSERT_EQ(declaredTailSamples, static_cast<std::size_t>(latency));
  const std::size_t renderSamples = kSignalSamples + declaredTailSamples;

  std::array<std::vector<float>, audio_plugin::kNumChannels> renderedMain;
  for (auto& channel : renderedMain) {
    channel.reserve(renderSamples);
  }
  double retainedPeakInFinalAlignedHop = 0.0;
  double maximumReconstructionError = 0.0;
  juce::MidiBuffer midi;
  std::size_t timeline = 0U;
  while (timeline < renderSamples) {
    const int callbackSamples = static_cast<int>(std::min(
        static_cast<std::size_t>(kBlockSize), renderSamples - timeline));
    juce::AudioBuffer<float> buffer(12, callbackSamples);
    buffer.clear();
    auto inputBus = processor.getBusBuffer(buffer, true, 0);
    for (int sample = 0; sample < callbackSamples; ++sample) {
      const std::size_t inputTimeline =
          timeline + static_cast<std::size_t>(sample);
      if (inputTimeline >= kSignalSamples) {
        continue;
      }
      for (int channel = 0; channel < audio_plugin::kNumChannels; ++channel) {
        inputBus.setSample(channel, sample,
                           makeInputSample(inputTimeline, sampleRate, channel));
      }
    }

    processor.processBlock(buffer, midi);
    const auto mainBus = processor.getBusBuffer(buffer, false, 0);
    const auto drumsBus = processor.getBusBuffer(buffer, false, 1);
    const auto bassBus = processor.getBusBuffer(buffer, false, 2);
    const auto otherBus = processor.getBusBuffer(buffer, false, 3);
    const auto vocalsBus = processor.getBusBuffer(buffer, false, 4);
    for (int sample = 0; sample < callbackSamples; ++sample) {
      const std::size_t outputTimeline =
          timeline + static_cast<std::size_t>(sample);
      for (int channel = 0; channel < audio_plugin::kNumChannels; ++channel) {
        const float main = mainBus.getSample(channel, sample);
        const float drums = drumsBus.getSample(channel, sample);
        const float bass = bassBus.getSample(channel, sample);
        const float other = otherBus.getSample(channel, sample);
        const float vocals = vocalsBus.getSample(channel, sample);
        renderedMain[static_cast<std::size_t>(channel)].push_back(main);
        maximumReconstructionError =
            std::max(maximumReconstructionError,
                     static_cast<double>(
                         std::abs(main - (drums + bass + other + vocals))));
        if (outputTimeline + static_cast<std::size_t>(kBlockSize) >=
            renderSamples) {
          retainedPeakInFinalAlignedHop = std::max(
              retainedPeakInFinalAlignedHop,
              static_cast<double>(std::max(
                  {std::abs(drums), std::abs(bass), std::abs(vocals)})));
        }
      }
    }
    timeline += static_cast<std::size_t>(callbackSamples);
  }

  for (std::size_t channel = 0U;
       channel < static_cast<std::size_t>(audio_plugin::kNumChannels);
       ++channel) {
    ASSERT_EQ(renderedMain[channel].size(), renderSamples);
    for (std::size_t outputTimeline = 0U; outputTimeline < renderSamples;
         ++outputTimeline) {
      const bool hasAlignedSignal =
          outputTimeline >= static_cast<std::size_t>(latency) &&
          outputTimeline - static_cast<std::size_t>(latency) < kSignalSamples;
      const float expected =
          hasAlignedSignal
              ? makeInputSample(
                    outputTimeline - static_cast<std::size_t>(latency),
                    sampleRate, static_cast<int>(channel))
              : 0.0f;
      EXPECT_FLOAT_EQ(renderedMain[channel][outputTimeline], expected)
          << "rate=" << sampleRate << " channel=" << channel
          << " timeline=" << outputTimeline;
    }
    EXPECT_FLOAT_EQ(renderedMain[channel].back(),
                    makeInputSample(kSignalSamples - 1U, sampleRate,
                                    static_cast<int>(channel)));
  }
  EXPECT_GT(retainedPeakInFinalAlignedHop, 1.0e-5);
  EXPECT_LE(maximumReconstructionError, 1.0e-6);
  EXPECT_EQ(processor.getQueueFullChunkDropCount(), 0U);
  EXPECT_EQ(processor.getRingOverflowEventCount(), 0U);
  processor.releaseResources();
}

void expectRealtimeWorkerBridge(double sampleRate, int blockSize) {
  audio_plugin::AudioPluginAudioProcessor processor;
  processor.prepareToPlay(sampleRate, blockSize);
  ASSERT_GT(processor.getLatencySamples(), 0) << processor.getOrtStatusString();

  const int latency = processor.getLatencySamples();
  const int warmupBlocks = (latency + blockSize - 1) / blockSize + 8;
  constexpr int kMeasureBlocks = 16;
  juce::MidiBuffer midi;
  std::size_t timeline = 0U;
  double retainedPeak = 0.0;
  double maximumMainError = 0.0;
  double maximumReconstructionError = 0.0;

  for (int block = 0; block < warmupBlocks + kMeasureBlocks; ++block) {
    juce::AudioBuffer<float> buffer(12, blockSize);
    buffer.clear();
    auto inputBus = processor.getBusBuffer(buffer, true, 0);
    for (int sample = 0; sample < blockSize; ++sample) {
      const std::size_t inputTimeline =
          timeline + static_cast<std::size_t>(sample);
      for (int channel = 0; channel < audio_plugin::kNumChannels; ++channel) {
        inputBus.setSample(channel, sample,
                           makeInputSample(inputTimeline, sampleRate, channel));
      }
    }

    processor.processBlock(buffer, midi);
    const auto mainBus = processor.getBusBuffer(buffer, false, 0);
    const auto drumsBus = processor.getBusBuffer(buffer, false, 1);
    const auto bassBus = processor.getBusBuffer(buffer, false, 2);
    const auto otherBus = processor.getBusBuffer(buffer, false, 3);
    const auto vocalsBus = processor.getBusBuffer(buffer, false, 4);
    for (int sample = 0; sample < blockSize; ++sample) {
      const std::size_t outputTimeline =
          timeline + static_cast<std::size_t>(sample);
      for (int channel = 0; channel < audio_plugin::kNumChannels; ++channel) {
        const float main = mainBus.getSample(channel, sample);
        const float drums = drumsBus.getSample(channel, sample);
        const float bass = bassBus.getSample(channel, sample);
        const float other = otherBus.getSample(channel, sample);
        const float vocals = vocalsBus.getSample(channel, sample);
        const float expectedMain =
            outputTimeline >= static_cast<std::size_t>(latency)
                ? makeInputSample(
                      outputTimeline - static_cast<std::size_t>(latency),
                      sampleRate, channel)
                : 0.0f;
        maximumMainError =
            std::max(maximumMainError,
                     static_cast<double>(std::abs(main - expectedMain)));
        maximumReconstructionError =
            std::max(maximumReconstructionError,
                     static_cast<double>(
                         std::abs(main - (drums + bass + other + vocals))));
        if (block >= warmupBlocks) {
          retainedPeak = std::max(
              retainedPeak,
              static_cast<double>(std::max(
                  {std::abs(drums), std::abs(bass), std::abs(vocals)})));
        }
      }
    }
    timeline += static_cast<std::size_t>(blockSize);
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
    // Wait only on the test control thread. The next callback consumes the
    // worker publication normally, independently of CI machine speed.
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!audio_plugin::AudioPluginProcessorTestPeer::allSubmissionsCompleted(
               processor) &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(
        audio_plugin::AudioPluginProcessorTestPeer::allSubmissionsCompleted(
            processor));
#endif
  }

  EXPECT_LE(maximumMainError, 1.0e-6);
  EXPECT_LE(maximumReconstructionError, 1.0e-6);
  EXPECT_GT(retainedPeak, 1.0e-5);
  EXPECT_FALSE(processor.isUnderrunActive());
  EXPECT_EQ(processor.getUnderrunSampleCount(), 0U);
  EXPECT_EQ(processor.getUnderrunBlockCount(), 0U);
  EXPECT_EQ(processor.getQueueFullChunkDropCount(), 0U);
  EXPECT_EQ(processor.getRingOverflowEventCount(), 0U);
  EXPECT_FALSE(processor.isRealtimeCallbackTimingUnsafe());
  EXPECT_EQ(processor.getUnsafeRealtimeCallbackCount(), 0U);
  processor.releaseResources();
}

TEST(SampleRateBridgeE2ETest, HigherRatePathPreservesMainAndExactResidual) {
  for (const double sampleRate :
       {48000.0, 88200.0, 96000.0, 176400.0, 192000.0}) {
    SCOPED_TRACE(::testing::Message() << "sampleRate=" << sampleRate);
    for (const int blockSize : {128, 512}) {
      SCOPED_TRACE(::testing::Message() << "blockSize=" << blockSize);
      expectRateRoundTrip(sampleRate, blockSize);
    }
  }
}

TEST(SampleRateBridgeE2ETest, FiniteRenderCompletesPartialCurrentChunk) {
  for (const double sampleRate :
       {48000.0, 88200.0, 96000.0, 176400.0, 192000.0}) {
    SCOPED_TRACE(::testing::Message() << "sampleRate=" << sampleRate);
    expectFiniteRenderCompletion(sampleRate);
  }
}

TEST(SampleRateBridgeE2ETest, RealtimeWorkerPublishesExactConvertedRanges) {
  for (const double sampleRate : {48000.0, 96000.0, 192000.0}) {
    for (const int blockSize : {64, 128, 512}) {
      SCOPED_TRACE(::testing::Message()
                   << "rate=" << sampleRate << " block=" << blockSize);
      expectRealtimeWorkerBridge(sampleRate, blockSize);
    }
  }
}

TEST(SampleRateBridgeE2ETest, ReportsDocumented128SampleBufferLatency) {
  audio_plugin::AudioPluginAudioProcessor processor;
  const std::array<std::pair<int, int>, 6> expected = {{{44100, 256},
                                                        {48000, 702},
                                                        {88200, 947},
                                                        {96000, 1276},
                                                        {176400, 2021},
                                                        {192000, 2422}}};
  for (const auto& [rate, delay] : expected) {
    SCOPED_TRACE(::testing::Message() << "rate=" << rate);
    processor.prepareToPlay(static_cast<double>(rate), 128);
    EXPECT_EQ(processor.getLatencySamples(), delay)
        << processor.getOrtStatusString();
    EXPECT_NEAR(processor.getTailLengthSeconds(),
                static_cast<double>(delay) / static_cast<double>(rate),
                1.0e-12);
  }
  processor.prepareToPlay(48000.0, 512);
  EXPECT_EQ(processor.getLatencySamples(), 958);
  processor.releaseResources();
}

TEST(SampleRateBridgeE2ETest, RejectsUnqualifiedHostRateFailClosed) {
  audio_plugin::AudioPluginAudioProcessor processor;
  for (const double sampleRate : {22050.0, 48001.0, 48000.25, 384000.0}) {
    SCOPED_TRACE(::testing::Message() << "sampleRate=" << sampleRate);
    processor.prepareToPlay(sampleRate, kBlockSize);
    EXPECT_EQ(processor.getLatencySamples(), 0);
    EXPECT_TRUE(processor.getOrtStatusString().containsIgnoreCase(
        "Unsupported audio configuration"));
    processor.releaseResources();
  }
}

TEST(SampleRateBridgeE2ETest, RejectsUnqualifiedPreparedBlockSizeFailClosed) {
  audio_plugin::AudioPluginAudioProcessor processor;
  processor.prepareToPlay(44100.0, 0);
  EXPECT_EQ(processor.getLatencySamples(), 0);
  EXPECT_TRUE(processor.getOrtStatusString().containsIgnoreCase(
      "a buffer of 1 to 65536 samples"));
  processor.releaseResources();
}

class BridgePlayHead final : public juce::AudioPlayHead {
 public:
  void setPosition(bool playing, int64_t sample) {
    position_.setIsPlaying(playing);
    position_.setTimeInSamples(sample);
  }
  juce::Optional<PositionInfo> getPosition() const override {
    return position_;
  }

 private:
  PositionInfo position_;
};

std::vector<float> renderThroughTransportStop(double sampleRate, bool stop) {
  audio_plugin::AudioPluginAudioProcessor processor;
  processor.setNonRealtime(true);
  processor.prepareToPlay(sampleRate, kBlockSize);
  EXPECT_GT(processor.getLatencySamples(), 0) << processor.getOrtStatusString();
  if (processor.getLatencySamples() <= 0) {
    return {};
  }
  const std::size_t signalSamples = 2U * kBlockSize + 137U;
  const std::size_t totalSamples =
      signalSamples + static_cast<std::size_t>(processor.getLatencySamples());
  BridgePlayHead playHead;
  processor.setPlayHead(&playHead);
  juce::MidiBuffer midi;
  std::vector<float> rendered;
  std::size_t timeline = 0U;
  while (timeline < totalSamples) {
    const auto boundary =
        timeline < signalSamples ? signalSamples : totalSamples;
    const auto count =
        std::min(static_cast<std::size_t>(kBlockSize), boundary - timeline);
    const bool stopped = stop && timeline >= signalSamples;
    playHead.setPosition(
        !stopped, static_cast<int64_t>(stopped ? signalSamples : timeline));
    juce::AudioBuffer<float> buffer(10, static_cast<int>(count));
    buffer.clear();
    if (timeline < signalSamples) {
      for (int ch = 0; ch < 2; ++ch) {
        for (std::size_t sample = 0U; sample < count; ++sample) {
          buffer.setSample(ch, static_cast<int>(sample),
                           makeInputSample(timeline + sample, sampleRate, ch));
        }
      }
    }
    processor.processBlock(buffer, midi);
    for (std::size_t sample = 0U; sample < count; ++sample) {
      for (int ch = 0; ch < 10; ++ch) {
        const float value = buffer.getSample(ch, static_cast<int>(sample));
        EXPECT_TRUE(std::isfinite(value));
        rendered.push_back(value);
      }
    }
    timeline += count;
  }
  processor.releaseResources();
  return rendered;
}

TEST(SampleRateBridgeE2ETest, TransportStopPreservesCompleteConvertedTail) {
  for (const double sampleRate :
       {44100.0, 48000.0, 88200.0, 96000.0, 176400.0, 192000.0}) {
    SCOPED_TRACE(::testing::Message() << "sampleRate=" << sampleRate);
    const auto continuous = renderThroughTransportStop(sampleRate, false);
    const auto stopped = renderThroughTransportStop(sampleRate, true);
    ASSERT_FALSE(continuous.empty());
    ASSERT_EQ(stopped.size(), continuous.size());
    for (std::size_t sample = 0U; sample < stopped.size(); ++sample) {
      ASSERT_FLOAT_EQ(stopped[sample], continuous[sample])
          << "sample=" << sample;
    }
  }
}

TEST(SampleRateBridgeE2ETest,
     RateChangesResetStateAndPreserveVariableOfflineCallbacks) {
  audio_plugin::AudioPluginAudioProcessor processor;
  processor.setNonRealtime(true);
  std::vector<float> firstRender;
  // Return to 48 kHz after both a converted clock and an exact model clock.
  for (const double sampleRate : {48000.0, 96000.0, 44100.0, 48000.0}) {
    SCOPED_TRACE(::testing::Message() << "sampleRate=" << sampleRate);
    processor.prepareToPlay(sampleRate, kBlockSize);
    const int latency = processor.getLatencySamples();
    ASSERT_GT(latency, 0) << processor.getOrtStatusString();
    std::size_t timeline = 0U;
    std::vector<float> rendered;
    juce::MidiBuffer midi;
    for (const int callbackSize : {256, 1024, 137, 512, 256, 1024, 137, 512}) {
      juce::AudioBuffer<float> buffer(10, callbackSize);
      buffer.clear();
      for (int sample = 0; sample < callbackSize; ++sample) {
        for (int channel = 0; channel < audio_plugin::kNumChannels; ++channel) {
          buffer.setSample(
              channel, sample,
              makeInputSample(timeline + static_cast<std::size_t>(sample),
                              sampleRate, channel));
        }
      }
      processor.processBlock(buffer, midi);
      EXPECT_FALSE(processor.isRealtimeCallbackTimingUnsafe());
      EXPECT_EQ(processor.getLatencySamples(), latency);
      for (int sample = 0; sample < callbackSize; ++sample) {
        const auto position = timeline + static_cast<std::size_t>(sample);
        for (int channel = 0; channel < audio_plugin::kNumChannels; ++channel) {
          const float expected =
              position < static_cast<std::size_t>(latency)
                  ? 0.0f
                  : makeInputSample(
                        position - static_cast<std::size_t>(latency),
                        sampleRate, channel);
          EXPECT_FLOAT_EQ(buffer.getSample(channel, sample), expected);
          float sum = 0.0f;
          for (int bus = 1; bus < 5; ++bus) {
            const float value = buffer.getSample(2 * bus + channel, sample);
            EXPECT_TRUE(std::isfinite(value));
            sum += value;
            rendered.push_back(value);
          }
          EXPECT_NEAR(sum, expected, 1.0e-6f);
        }
      }
      timeline += static_cast<std::size_t>(callbackSize);
    }
    EXPECT_EQ(processor.getQueueFullChunkDropCount(), 0U);
    if (static_cast<int>(sampleRate) == 48000) {
      if (firstRender.empty()) {
        firstRender = rendered;
      } else {
        EXPECT_EQ(rendered, firstRender);
      }
    }
  }
  processor.releaseResources();
}

}  // namespace
}  // namespace audio_plugin_test
