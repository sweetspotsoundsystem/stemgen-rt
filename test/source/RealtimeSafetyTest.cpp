#include <StemgenRT/PluginProcessor.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <new>
#include <thread>

#include "RealtimeAllocationGuard.h"

namespace audio_plugin {
class AudioPluginProcessorTestPeer {
public:
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  static void stopWorker(AudioPluginAudioProcessor& processor) {
    processor.inferenceQueue_.stopThread();
  }
  static bool submissionCompleted(AudioPluginAudioProcessor& processor) {
    const auto& queue = processor.inferenceQueue_;
    const size_t index = (queue.writeIdx_.load(std::memory_order_acquire) +
                          queue.queue_.size() - 1U) %
                         queue.queue_.size();
    return queue.queue_[index]->isProcessed();
  }
#endif
};
}  // namespace audio_plugin

namespace audio_plugin_test {

TEST(RealtimeSafetyTest,
     HeapProbeObservesAllocationAndDeletionOnCallingThread) {
  void* (*volatile allocate)(size_t, std::align_val_t) = &::operator new;
  void (*volatile deallocate)(void*, std::align_val_t) noexcept =
      &::operator delete;
  RealtimeAllocationGuard guard;
  void* memory = allocate(64U, std::align_val_t{64});
  deallocate(memory, std::align_val_t{64});
  const auto traffic = guard.finish();
  EXPECT_EQ(traffic.allocations, 1U);
  EXPECT_EQ(traffic.deallocations, 1U);
}

#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
namespace {
class RealtimeTestPlayHead final : public juce::AudioPlayHead {
public:
  void setSample(int64_t sample) {
    position_.setIsPlaying(true);
    position_.setTimeInSamples(sample);
  }
  juce::Optional<PositionInfo> getPosition() const override {
    return position_;
  }

private:
  PositionInfo position_;
};
constexpr int kBlockSize = audio_plugin::kOutputChunkSize;
constexpr int kPdc = audio_plugin::kPluginLatencySamples;
float sampleAt(int sample, int channel) {
  return static_cast<float>((sample * 7 + channel * 13) % 257 - 128) / 512.0f;
}
void fillInput(juce::AudioBuffer<float>& buffer, int start) {
  buffer.clear();
  for (int ch = 0; ch < 2; ++ch) {
    for (int i = 0; i < kBlockSize; ++i) {
      buffer.setSample(ch, i, sampleAt(start + i, ch));
    }
  }
}
void checkMainAndReconstruction(const juce::AudioBuffer<float>& buffer,
                                int start,
                                bool fallback) {
  for (int ch = 0; ch < 2; ++ch) {
    for (int i = 0; i < kBlockSize; ++i) {
      const int source = start + i - kPdc;
      const float expected = source < 0 ? 0.0f : sampleAt(source, ch);
      EXPECT_FLOAT_EQ(buffer.getSample(ch, i), expected);
      float sum = 0.0f;
      for (int bus = 1; bus < 5; ++bus) {
        const float value = buffer.getSample(bus * 2 + ch, i);
        EXPECT_TRUE(std::isfinite(value));
        sum += value;
        if (fallback) {
          EXPECT_FLOAT_EQ(value, bus == 3 ? expected : 0.0f);
        }
      }
      EXPECT_NEAR(sum, expected, 1.0e-6f);
    }
  }
}
void processGuarded(audio_plugin::AudioPluginAudioProcessor& processor,
                    juce::AudioBuffer<float>& buffer,
                    juce::MidiBuffer& midi) {
  RealtimeAllocationGuard guard;
  processor.processBlock(buffer, midi);
  const auto traffic = guard.finish();
  EXPECT_EQ(traffic.allocations, 0U);
  EXPECT_EQ(traffic.deallocations, 0U);
  EXPECT_EQ(processor.getLatencySamples(), kPdc);
  EXPECT_EQ(processor.getMaximumSameCallbackWaitMicroseconds(), 0);
}
}  // namespace
#endif

TEST(RealtimeSafetyTest,
     StoppedWorkerSaturatesQueueWithoutCallbackHeapTraffic) {
#if !(defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME)
  GTEST_SKIP() << "ONNX Runtime support not compiled";
#else
  audio_plugin::AudioPluginAudioProcessor processor;
  processor.setNonRealtime(false);
  processor.prepareToPlay(44100.0, kBlockSize);
  ASSERT_EQ(processor.getConfiguredOrtIntraOpThreads(), 1);
  ASSERT_EQ(processor.getLatencySamples(), kPdc)
      << processor.getOrtStatusString();
  audio_plugin::AudioPluginProcessorTestPeer::stopWorker(processor);
  juce::AudioBuffer<float> buffer(processor.getTotalNumOutputChannels(),
                                  kBlockSize);
  ASSERT_EQ(buffer.getNumChannels(), 10);
  juce::MidiBuffer midi;
  RealtimeTestPlayHead playHead;
  processor.setPlayHead(&playHead);
  for (int block = 0; block < 512; ++block) {
    // Seek once with a saturated queue; the callback must also reset its
    // timeline in bounded time with no heap traffic and preserve exact PDC.
    const int start = (block % 256) * kBlockSize;
    playHead.setSample(start + (block >= 256 ? 65536 : 0));
    fillInput(buffer, start);
    processGuarded(processor, buffer, midi);
    checkMainAndReconstruction(buffer, start, true);
  }
  EXPECT_GT(processor.getQueueFullChunkDropCount(), 0U);
  EXPECT_GT(processor.getSameCallbackTimeoutCount(), 0U);
  EXPECT_GT(processor.getUnderrunSampleCount(), 0U);
  processor.releaseResources();
#endif
}

TEST(RealtimeSafetyTest, ReadyResultsRecoverWithoutCallbackHeapTraffic) {
#if !(defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME)
  GTEST_SKIP() << "ONNX Runtime support not compiled";
#else
  audio_plugin::AudioPluginAudioProcessor processor;
  processor.setNonRealtime(false);
  processor.prepareToPlay(44100.0, kBlockSize);
  ASSERT_EQ(processor.getLatencySamples(), kPdc)
      << processor.getOrtStatusString();
  ASSERT_EQ(processor.getConfiguredOrtIntraOpThreads(), 1);
  juce::AudioBuffer<float> buffer(processor.getTotalNumOutputChannels(),
                                  kBlockSize);
  ASSERT_EQ(buffer.getNumChannels(), 10);
  juce::MidiBuffer midi;
  float retainedPeak = 0.0f;
  for (int block = 0; block < 16; ++block) {
    const int start = block * kBlockSize;
    fillInput(buffer, start);
    processGuarded(processor, buffer, midi);
    checkMainAndReconstruction(buffer, start, false);
    retainedPeak =
        std::max(retainedPeak, buffer.getMagnitude(2, 0, kBlockSize));
    // Test control thread only. Make results observable before the next call
    // so correctness of the ready-output branch is independent of CI speed.
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!audio_plugin::AudioPluginProcessorTestPeer::submissionCompleted(
               processor) &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(audio_plugin::AudioPluginProcessorTestPeer::submissionCompleted(
        processor));
  }
  EXPECT_GT(retainedPeak, 1.0e-6f);
  EXPECT_EQ(processor.getSameCallbackTimeoutCount(), 0U);
  EXPECT_EQ(processor.getQueueFullChunkDropCount(), 0U);
  processor.releaseResources();
#endif
}

}  // namespace audio_plugin_test
