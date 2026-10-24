#include <StemgenRT/OnnxRuntime.h>
#include <StemgenRT/PluginProcessor.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>

namespace audio_plugin {
class OnnxRuntimeTestPeer {
public:
  static const auto& states(const OnnxRuntime& runtime) {
    return runtime.states_;
  }
};
}  // namespace audio_plugin
namespace {
using namespace audio_plugin;
using Audio = std::array<std::vector<float>, kNumChannels>;
using Stems = std::array<Audio, kNumStems>;
struct Buffers {
  Audio input, aligned;
  Stems output;
  Buffers() {
    for (auto* a : {&input, &aligned})
      for (auto& c : *a)
        c.resize(kOutputChunkSize);
    for (auto& s : output)
      for (auto& c : s)
        c.resize(kOutputChunkSize);
  }
};
juce::String modelPath() {
  return juce::File::getSpecialLocation(juce::File::currentExecutableFile)
      .getParentDirectory()
      .getParentDirectory()
      .getChildFile("Resources/model.onnx")
      .getFullPathName();
}
void equalFloats(const std::vector<float>& a, const std::vector<float>& b) {
  ASSERT_EQ(a.size(), b.size());
  EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)), 0);
}
void compareTrajectory(float phase) {
  OnnxRuntime baseline, parallel;
  ASSERT_TRUE(parallel.setIndependentWorkerCount(3));
  juce::String error;
  ASSERT_TRUE(baseline.loadModel(modelPath(), error, 1)) << error.toStdString();
  ASSERT_TRUE(parallel.loadModel(modelPath(), error, 1)) << error.toStdString();
  ASSERT_TRUE(baseline.prepareForInference(error));
  ASSERT_TRUE(parallel.prepareForInference(error));
  ASSERT_TRUE(parallel.startAuxiliaryWorkers(false, {nullptr, nullptr}));
  Buffers a, b;
  for (int hop = 0; hop <= 96; ++hop) {
    SCOPED_TRACE(hop);
    if (hop == 33 || hop == 61) {
      baseline.resetStreamingState();
      parallel.resetStreamingState();
    }
    for (size_t ch = 0; ch < a.input.size(); ++ch)
      for (size_t i = 0; i < a.input[ch].size(); ++i) {
        const float x =
            static_cast<float>(hop * kOutputChunkSize + static_cast<int>(i));
        a.input[ch][i] =
            hop == 96 ? 0.0f
                      : .25f * std::sin(x * (ch == 0 ? .017f : .033f) + phase);
      }
    b.input = a.input;
    bool av = false, bv = false;
    ASSERT_TRUE(baseline.runInference(a.input, a.output, a.aligned, av));
    ASSERT_TRUE(parallel.runInference(b.input, b.output, b.aligned, bv));
    EXPECT_EQ(av, bv);
    EXPECT_EQ(bv, hop != 0 && hop != 33 && hop != 61);
    for (size_t ch = 0; ch < a.aligned.size(); ++ch)
      equalFloats(a.aligned[ch], b.aligned[ch]);
    for (size_t s = 0; s < a.output.size(); ++s)
      for (size_t ch = 0; ch < a.output[s].size(); ++ch)
        equalFloats(a.output[s][ch], b.output[s][ch]);
    const auto& as = OnnxRuntimeTestPeer::states(baseline);
    const auto& bs = OnnxRuntimeTestPeer::states(parallel);
    ASSERT_EQ(as.size(), 13U);
    for (size_t state = 0; state < as.size(); ++state)
      equalFloats(as[state], bs[state]);
  }
  a.input[0][0] = b.input[0][0] = std::numeric_limits<float>::quiet_NaN();
  bool av = true, bv = true;
  EXPECT_FALSE(baseline.runInference(a.input, a.output, a.aligned, av));
  EXPECT_FALSE(parallel.runInference(b.input, b.output, b.aligned, bv));
  a.input[0][0] = b.input[0][0] = 0.0f;
  ASSERT_TRUE(baseline.runInference(a.input, a.output, a.aligned, av));
  ASSERT_TRUE(parallel.runInference(b.input, b.output, b.aligned, bv));
  EXPECT_FALSE(av);
  EXPECT_FALSE(bv);
  for (size_t state = 0; state < 13; ++state)
    equalFloats(OnnxRuntimeTestPeer::states(baseline)[state],
                OnnxRuntimeTestPeer::states(parallel)[state]);
  parallel.stopAuxiliaryWorkers();
}
TEST(ParallelInferenceTest, AllFourOutputsAndThirteenStatesMatchEveryHop) {
  compareTrajectory(0.0f);
}
TEST(ParallelInferenceTest, ConcurrentInstancesKeepIndependentStates) {
  std::thread a([] { compareTrajectory(.2f); });
  std::thread b([] { compareTrajectory(1.7f); });
  a.join();
  b.join();
}
TEST(ParallelInferenceTest, HelpersRestartAcrossRealtimeAndOfflinePreparation) {
  OnnxRuntime runtime;
  juce::String error;
  ASSERT_TRUE(runtime.setIndependentWorkerCount(3));
  ASSERT_TRUE(runtime.loadModel(modelPath(), error, 1)) << error.toStdString();
  Buffers b;
  for (int pass = 0; pass < 4; ++pass) {
    ASSERT_TRUE(runtime.prepareForInference(error));
    ASSERT_TRUE(
        runtime.startAuxiliaryWorkers(pass % 2 == 0, {nullptr, nullptr}));
    bool valid = true;
    ASSERT_TRUE(runtime.runInference(b.input, b.output, b.aligned, valid));
    EXPECT_FALSE(valid);
    runtime.stopAuxiliaryWorkers();
    for (size_t helper = 0; helper < 2; ++helper) {
#if defined(__APPLE__)
      EXPECT_EQ(runtime.auxiliaryInitialPolicy(helper), 1);
      EXPECT_EQ(runtime.auxiliaryFinalPolicy(helper), 1);
#else
      EXPECT_EQ(runtime.auxiliaryInitialPolicy(helper), 2);
      EXPECT_EQ(runtime.auxiliaryFinalPolicy(helper), 2);
#endif
    }
  }
}
TEST(ParallelInferenceTest, ModeAndNestedThreadCountsFailClosed) {
  OnnxRuntime runtime;
  juce::String error;
  EXPECT_FALSE(runtime.setIndependentWorkerCount(2));
  EXPECT_FALSE(runtime.setIndependentWorkerCount(0));
  ASSERT_TRUE(runtime.setIndependentWorkerCount(3));
  EXPECT_FALSE(runtime.loadModel(modelPath(), error, 2));
  EXPECT_FALSE(runtime.isModelLoaded());
  ASSERT_TRUE(runtime.loadModel(modelPath(), error, 1)) << error.toStdString();
  EXPECT_FALSE(runtime.setIndependentWorkerCount(1));
  ASSERT_TRUE(runtime.prepareForInference(error));
  Buffers b;
  bool valid = true;
  EXPECT_FALSE(runtime.runInference(b.input, b.output, b.aligned,
                                    valid));  // Helpers were not started.
  EXPECT_FALSE(valid);
}
TEST(ParallelInferenceTest, ProcessorModeMustBeSelectedBeforePreparation) {
  AudioPluginAudioProcessor processor;
  ASSERT_TRUE(processor.setDiagnosticIndependentWorkers(3));
  processor.setNonRealtime(true);
  processor.prepareToPlay(44100.0, 128);
  EXPECT_EQ(processor.getIndependentWorkerCount(), 3);
  EXPECT_EQ(processor.getLatencySamples(), 256);
  EXPECT_FALSE(processor.setDiagnosticIndependentWorkers(1));
  processor.releaseResources();
}
}  // namespace
