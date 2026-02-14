#include <StemgenRT/PluginProcessor.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <thread>

namespace audio_plugin_test {
namespace {

constexpr double kSampleRate = 44100.0;
constexpr int kBlockSize = 128;
constexpr int kTotalChannels = 12;  // 2 input + 10 output (5 buses * 2ch)

constexpr float kPi = 3.14159265358979323846f;

float sineAtSample(int64_t sampleIndex, float freqHz, float amplitude) {
  const float t = static_cast<float>(sampleIndex) / static_cast<float>(kSampleRate);
  return amplitude * std::sin(2.0f * kPi * freqHz * t);
}

}  // namespace

// Real-time paced sanity check: with enough wall-clock time for the inference thread
// to keep up, the 4 stem buses should not all be identical (dry fallback signature).
TEST(RealtimeStemSanityTest, DISABLED_StemsAreNotAllIdenticalWhenRealtimePaced) {
  audio_plugin::AudioPluginAudioProcessor processor;
  processor.prepareToPlay(kSampleRate, kBlockSize);

  if (processor.getLatencySamples() <= 0) {
    GTEST_SKIP() << "Model not loaded; skipping real-time paced stem sanity check";
  }

  juce::MidiBuffer midiBuffer;

  constexpr int kWarmupBlocks = 40;
  constexpr int kMeasureBlocks = 80;

  int64_t sampleIndex = 0;
  float maxAbsStemDiff = 0.0f;

  for (int b = 0; b < (kWarmupBlocks + kMeasureBlocks); ++b) {
    juce::AudioBuffer<float> buffer(kTotalChannels, kBlockSize);
    buffer.clear();

    // Fill input bus with a continuous multitone to encourage non-trivial stem output.
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

    processor.processBlock(buffer, midiBuffer);

    if (b >= kWarmupBlocks) {
      const int numOutputBuses = processor.getBusCount(false /* isInput */);
      ASSERT_GE(numOutputBuses, 5) << "Expected 5 output buses (Main + 4 stems)";

      auto drumsBus = processor.getBusBuffer(buffer, false /* isInput */, 1);
      auto bassBus = processor.getBusBuffer(buffer, false /* isInput */, 2);
      auto otherBus = processor.getBusBuffer(buffer, false /* isInput */, 3);
      auto vocalsBus = processor.getBusBuffer(buffer, false /* isInput */, 4);

      for (int i = 0; i < buffer.getNumSamples(); ++i) {
        for (int ch = 0; ch < 2; ++ch) {
          const float d = drumsBus.getSample(ch, i);
          const float b0 = bassBus.getSample(ch, i);
          const float o = otherBus.getSample(ch, i);
          const float v = vocalsBus.getSample(ch, i);

          maxAbsStemDiff = std::max(maxAbsStemDiff, std::abs(d - b0));
          maxAbsStemDiff = std::max(maxAbsStemDiff, std::abs(d - o));
          maxAbsStemDiff = std::max(maxAbsStemDiff, std::abs(d - v));
        }
      }
    }

    sampleIndex += buffer.getNumSamples();

    // Pace processing so the background inference thread can keep up.
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  // If stems were always falling back to delayed dry/4, they would be bit-identical
  // across buses, making maxAbsStemDiff ~= 0.
  EXPECT_GT(maxAbsStemDiff, 1.0e-3f)
      << "Stem buses appear identical (likely underrun fallback only); maxAbsStemDiff="
      << maxAbsStemDiff;

  processor.releaseResources();
}

}  // namespace audio_plugin_test

