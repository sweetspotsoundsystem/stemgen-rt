#include <StemgenRT/OnnxRuntime.h>
#include <StemgenRT/PluginProcessor.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {
using namespace audio_plugin;
using Audio = std::array<std::vector<float>, kNumChannels>;
using Stems = std::array<Audio, kNumStems>;

struct Reference {
  size_t frames{};
  Audio input;
  Stems output;
};

std::vector<Reference> readReferences() {
  const juce::File fixture(juce::String(STEMGENRT_TEST_FIXTURE_DIR) +
                           "/cropped1024-pytorch.bin");
  auto stream = fixture.createInputStream();
  EXPECT_NE(stream, nullptr);
  if (!stream)
    return {};
  std::array<char, 8> magic{};
  EXPECT_EQ(stream->read(magic.data(), static_cast<int>(magic.size())), 8);
  EXPECT_EQ(std::string(magic.data(), magic.size()), "SGRTG001");
  const int count = stream->readInt();
  EXPECT_EQ(count, 8);
  if (count != 8)
    return {};
  std::vector<Reference> references;
  for (int index = 0; index < count; ++index) {
    Reference reference;
    const int frames = stream->readInt();
    EXPECT_GT(frames, 0);
    EXPECT_LE(frames, 20000);
    if (frames <= 0 || frames > 20000)
      return {};
    reference.frames = static_cast<size_t>(frames);
    const auto readAudio = [&](Audio& audio) {
      for (auto& channel : audio) {
        channel.resize(reference.frames);
        for (float& sample : channel)
          sample = stream->readFloat();
      }
    };
    readAudio(reference.input);
    for (auto& stem : reference.output)
      readAudio(stem);
    references.push_back(std::move(reference));
  }
  EXPECT_TRUE(stream->isExhausted());
  return references;
}

juce::File modelPath() {
  return juce::File::getSpecialLocation(juce::File::currentExecutableFile)
      .getParentDirectory()
      .getParentDirectory()
      .getChildFile("Resources/model.onnx");
}

class PlayHead final : public juce::AudioPlayHead {
public:
  void position(bool playing, size_t sample) {
    position_.setIsPlaying(playing);
    position_.setTimeInSamples(static_cast<int64_t>(sample));
  }
  juce::Optional<PositionInfo> getPosition() const override {
    return position_;
  }

private:
  PositionInfo position_;
};

TEST(CroppedModelParityTest,
     FourNativeStemsMatchIndependentPytorchWithOneFlush) {
  const auto references = readReferences();
  ASSERT_EQ(references.size(), 8U);
  OnnxRuntime runtime;
  ASSERT_TRUE(runtime.isInitialized());
  juce::String error;
  ASSERT_TRUE(runtime.loadModel(modelPath().getFullPathName(), error, 1))
      << error.toStdString();
  ASSERT_TRUE(runtime.prepareForInference(error)) << error.toStdString();
  Audio input, aligned;
  Stems output;
  for (auto& channel : input)
    channel.resize(kOutputChunkSize);
  for (auto& channel : aligned)
    channel.resize(kOutputChunkSize);
  for (auto& stem : output)
    for (auto& channel : stem)
      channel.resize(kOutputChunkSize);

  for (const auto& reference : references) {
    SCOPED_TRACE(reference.frames);
    runtime.resetStreamingState();
    const size_t hops =
        (reference.frames + kOutputChunkSize - 1U) / kOutputChunkSize;
    float maxError = 0.0f;
    float maxAlignmentError = 0.0f;
    // The final iteration is exactly one all-zero graph flush.
    for (size_t hop = 0; hop <= hops; ++hop) {
      for (size_t ch = 0; ch < input.size(); ++ch) {
        for (size_t i = 0; i < input[ch].size(); ++i) {
          const size_t sample = hop * kOutputChunkSize + i;
          input[ch][i] =
              sample < reference.frames ? reference.input[ch][sample] : 0.0f;
        }
      }
      bool valid = false;
      ASSERT_TRUE(runtime.runInference(input, output, aligned, valid));
      ASSERT_EQ(valid, hop != 0U);
      if (!valid)
        continue;
      for (size_t ch = 0; ch < input.size(); ++ch) {
        for (size_t i = 0; i < input[ch].size(); ++i) {
          const size_t sample = (hop - 1U) * kOutputChunkSize + i;
          if (sample >= reference.frames)
            continue;
          maxAlignmentError =
              std::max(maxAlignmentError,
                       std::abs(aligned[ch][i] - reference.input[ch][sample]));
          for (size_t stem = 0; stem < output.size(); ++stem) {
            ASSERT_TRUE(std::isfinite(output[stem][ch][i]));
            maxError = std::max(maxError,
                                std::abs(output[stem][ch][i] -
                                         reference.output[stem][ch][sample]));
          }
        }
      }
    }
    EXPECT_FLOAT_EQ(maxAlignmentError, 0.0f);
    EXPECT_LE(maxError, 1.0e-5f);
  }
}

TEST(CroppedModelParityTest,
     OfflineBusesPreservePytorchStemsThroughPartialEof) {
  const auto references = readReferences();
  ASSERT_EQ(references.size(), 8U);
  AudioPluginAudioProcessor processor;
  processor.setNonRealtime(true);
  processor.prepareToPlay(44100.0, 128);
  ASSERT_EQ(processor.getLatencySamples(), 256)
      << processor.getOrtStatusString().toStdString();
  EXPECT_NEAR(processor.getTailLengthSeconds(), 256.0 / 44100.0, 1.0e-12);
  PlayHead playHead;
  processor.setPlayHead(&playHead);
  juce::MidiBuffer midi;
  constexpr std::array<int, 4> buses = {1, 2, 4, 3};
  constexpr std::array<size_t, 5> variableBlocks = {73, 511, 128, 37, 1024};

  for (const auto& reference : references) {
    for (const bool variable : {false, true}) {
      SCOPED_TRACE(reference.frames);
      SCOPED_TRACE(variable);
      processor.resetStreamingBuffers();
      std::array<Audio, 5> rendered;
      for (auto& bus : rendered)
        for (auto& channel : bus)
          channel.reserve(reference.frames + 256U);
      const auto render = [&](size_t offset, size_t count, bool playing) {
        playHead.position(playing, playing ? offset : reference.frames);
        juce::AudioBuffer<float> buffer(12, static_cast<int>(count));
        buffer.clear();
        if (playing) {
          auto input = processor.getBusBuffer(buffer, true, 0);
          for (int ch = 0; ch < 2; ++ch)
            for (size_t i = 0; i < count; ++i)
              input.setSample(
                  ch, static_cast<int>(i),
                  reference.input[static_cast<size_t>(ch)][offset + i]);
        }
        processor.processBlock(buffer, midi);
        for (int bus = 0; bus < 5; ++bus) {
          auto output = processor.getBusBuffer(buffer, false, bus);
          for (int ch = 0; ch < 2; ++ch) {
            auto& destination =
                rendered[static_cast<size_t>(bus)][static_cast<size_t>(ch)];
            destination.insert(destination.end(), output.getReadPointer(ch),
                               output.getReadPointer(ch) + count);
          }
        }
      };
      size_t offset = 0, block = 0;
      while (offset < reference.frames) {
        const size_t count = std::min(
            reference.frames - offset,
            variable ? variableBlocks[block++ % variableBlocks.size()] : 128U);
        render(offset, count, true);
        offset += count;
      }
      // Short stopped callbacks exercise partial padding and queue-only drain.
      for (const size_t count : {31U, 97U, 128U})
        render(reference.frames, count, false);
      ASSERT_EQ(rendered[0][0].size(), reference.frames + 256U);
      float maxMainError = 0.0f, maxStemError = 0.0f, maxClosureError = 0.0f;
      for (size_t ch = 0; ch < 2U; ++ch) {
        for (size_t i = 0; i < reference.frames + 256U; ++i) {
          const bool preroll = i < 256U;
          const size_t sample = preroll ? 0U : i - 256U;
          const float expectedMain =
              preroll ? 0.0f : reference.input[ch][sample];
          const float gain =
              preroll ? 0.0f
                      : std::min(1.0f, static_cast<float>(sample + 1U) / 64.0f);
          float expectedOther = expectedMain;
          float sum = 0.0f;
          for (size_t stem = 0; stem < 4U; ++stem) {
            const float expected =
                stem == 3U
                    ? expectedOther
                    : (preroll ? 0.0f
                               : gain * reference.output[stem][ch][sample]);
            if (stem < 3U)
              expectedOther -= expected;
            const float actual =
                rendered[static_cast<size_t>(buses[stem])][ch][i];
            ASSERT_TRUE(std::isfinite(actual));
            maxStemError = std::max(maxStemError, std::abs(actual - expected));
            sum += actual;
          }
          maxMainError = std::max(maxMainError,
                                  std::abs(rendered[0][ch][i] - expectedMain));
          maxClosureError =
              std::max(maxClosureError, std::abs(sum - expectedMain));
        }
      }
      EXPECT_LE(maxMainError, 1.0e-7f);
      EXPECT_LE(maxStemError, 1.0e-5f);
      EXPECT_LE(maxClosureError, 1.0e-6f);
      EXPECT_EQ(processor.getQueueFullChunkDropCount(), 0U);
      EXPECT_EQ(processor.getUnsafeRealtimeCallbackCount(), 0U);
      // The callback after the tail must start a clean silent generation.
      juce::AudioBuffer<float> afterTail(12, 31);
      afterTail.clear();
      processor.processBlock(afterTail, midi);
      EXPECT_FLOAT_EQ(afterTail.getMagnitude(0, 31), 0.0f);
    }
  }
  processor.setPlayHead(nullptr);
  processor.releaseResources();
}
}  // namespace
