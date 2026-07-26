#include <StemgenRT/PluginProcessor.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

namespace audio_plugin_test {
namespace {

constexpr double kSampleRate = 44100.0;
constexpr int kBlockSize = audio_plugin::kOutputChunkSize;
constexpr int kTotalChannels = 12;  // 2 input + 10 output (5 buses * 2ch)

constexpr float kPi = 3.14159265358979323846f;

float sineAtSample(int64_t sampleIndex, float freqHz, float amplitude) {
  const float t =
      static_cast<float>(sampleIndex) / static_cast<float>(kSampleRate);
  return amplitude * std::sin(2.0f * kPi * freqHz * t);
}

}  // namespace

// Real-time paced sanity check: with enough wall-clock time for the inference
// thread to keep up, at least one retained model source must be present.
// Complete fallback routes the mixture only to Other.
TEST(RealtimeStemSanityTest,
     DISABLED_StemsAreNotAllIdenticalWhenRealtimePaced) {
  audio_plugin::AudioPluginAudioProcessor processor;
  processor.prepareToPlay(kSampleRate, kBlockSize);

  if (processor.getLatencySamples() <= 0) {
    GTEST_SKIP()
        << "Model not loaded; skipping real-time paced stem sanity check";
  }

  juce::MidiBuffer midiBuffer;

  constexpr int kWarmupBlocks = 8;
  constexpr int kMeasureBlocks = 64;

  int64_t sampleIndex = 0;
  float maxAbsRetainedStem = 0.0f;
  float maxAbsReconstructionError = 0.0f;
  std::vector<double> completeCallbackMicroseconds;
  completeCallbackMicroseconds.reserve(kMeasureBlocks);
  uint64_t completeCallbackDeadlineMisses = 0U;
  auto nextDeadline = std::chrono::steady_clock::now();
  const auto blockDuration = std::chrono::duration<double>(
      static_cast<double>(kBlockSize) / kSampleRate);

  for (int b = 0; b < (kWarmupBlocks + kMeasureBlocks); ++b) {
    juce::AudioBuffer<float> buffer(kTotalChannels, kBlockSize);
    buffer.clear();

    // Fill input bus with a continuous multitone to encourage non-trivial stem
    // output.
    auto inputBus = processor.getBusBuffer(buffer, true /* isInput */, 0);
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
      const int64_t si = sampleIndex + i;
      const float l = sineAtSample(si, 55.0f, 0.30f) +
                      sineAtSample(si, 220.0f, 0.25f) +
                      sineAtSample(si, 880.0f, 0.20f);
      const float r = sineAtSample(si, 65.0f, 0.30f) +
                      sineAtSample(si, 330.0f, 0.25f) +
                      sineAtSample(si, 1320.0f, 0.20f);
      inputBus.setSample(0, i, l);
      inputBus.setSample(1, i, r);
    }

    const auto processStarted = std::chrono::steady_clock::now();
    processor.processBlock(buffer, midiBuffer);
    const auto processFinished = std::chrono::steady_clock::now();

    if (b >= kWarmupBlocks) {
      const double processMicroseconds =
          std::chrono::duration<double, std::micro>(
              processFinished - processStarted)
              .count();
      completeCallbackMicroseconds.push_back(processMicroseconds);
      if (processFinished - processStarted > blockDuration) {
        ++completeCallbackDeadlineMisses;
      }

      const int numOutputBuses = processor.getBusCount(false /* isInput */);
      ASSERT_GE(numOutputBuses, 5)
          << "Expected 5 output buses (Main + 4 stems)";

      auto drumsBus = processor.getBusBuffer(buffer, false /* isInput */, 1);
      auto bassBus = processor.getBusBuffer(buffer, false /* isInput */, 2);
      auto otherBus = processor.getBusBuffer(buffer, false /* isInput */, 3);
      auto vocalsBus = processor.getBusBuffer(buffer, false /* isInput */, 4);
      auto mainBus = processor.getBusBuffer(buffer, false /* isInput */, 0);

      for (int i = 0; i < buffer.getNumSamples(); ++i) {
        for (int ch = 0; ch < 2; ++ch) {
          const float d = drumsBus.getSample(ch, i);
          const float b0 = bassBus.getSample(ch, i);
          const float o = otherBus.getSample(ch, i);
          const float v = vocalsBus.getSample(ch, i);

          maxAbsRetainedStem = std::max(
              {maxAbsRetainedStem, std::abs(d), std::abs(b0), std::abs(v)});
          maxAbsReconstructionError =
              std::max(maxAbsReconstructionError,
                       std::abs(mainBus.getSample(ch, i) - (d + b0 + o + v)));
        }
      }
    }

    sampleIndex += buffer.getNumSamples();

    // Pace at the real 512-sample callback interval. sleep_until returns
    // immediately after a complete processBlock overrun; the explicit timing
    // above covers drain/write time beyond the inference-only 10 ms gate.
    nextDeadline +=
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            blockDuration);
    std::this_thread::sleep_until(nextDeadline);
  }

  EXPECT_GT(maxAbsRetainedStem, 1.0e-3f)
      << "Drums, Bass, and Vocals remained silent (complete fallback only); "
         "maxAbsRetainedStem="
      << maxAbsRetainedStem;
  EXPECT_LE(maxAbsReconstructionError, 1.0e-6f)
      << "Stem buses did not reconstruct latency-aligned Main";
  EXPECT_EQ(processor.getQueueFullChunkDropCount(), 0u);
  EXPECT_EQ(processor.getRingOverflowEventCount(), 0u);
  EXPECT_EQ(processor.getUnderrunBlockCount(), 0u);
  ASSERT_FALSE(completeCallbackMicroseconds.empty());
  std::sort(completeCallbackMicroseconds.begin(),
            completeCallbackMicroseconds.end());
  const double maximumCompleteCallbackMicroseconds =
      completeCallbackMicroseconds.back();
  EXPECT_EQ(completeCallbackDeadlineMisses, 0U)
      << "Complete processBlock deadline misses; maximum callback was "
      << maximumCompleteCallbackMicroseconds << " us";
  EXPECT_LT(maximumCompleteCallbackMicroseconds,
            1.0e6 * static_cast<double>(kBlockSize) / kSampleRate);

  processor.releaseResources();
}

}  // namespace audio_plugin_test
