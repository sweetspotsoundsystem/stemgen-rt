#include <StemgenRT/Constants.h>
#include <StemgenRT/OnnxRuntime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <juce_core/juce_core.h>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace {

#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME

using AudioChunk =
    std::array<std::vector<float>, audio_plugin::kNumChannels>;
using SeparatedChunk = std::array<
    std::array<std::vector<float>, audio_plugin::kNumChannels>,
    audio_plugin::kNumStems>;

juce::File resolveModelPathForTestBinary() {
  // Keep in sync with PluginProcessor::prepareToPlay() model lookup contract.
  const juce::File executable =
      juce::File::getSpecialLocation(juce::File::currentExecutableFile);
  return executable.getParentDirectory()
      .getParentDirectory()
      .getChildFile("Resources/model.onnx");
}

bool prepareRuntime(audio_plugin::OnnxRuntime& runtime,
                    std::string& failureMessage) {
  if (!runtime.isInitialized()) {
    failureMessage = "ONNX Runtime did not initialize";
    return false;
  }

  const juce::File modelFile = resolveModelPathForTestBinary();
  if (!modelFile.existsAsFile()) {
    failureMessage =
        "Model file not found: " + modelFile.getFullPathName().toStdString();
    return false;
  }

  juce::String loadError;
  if (!runtime.loadModel(modelFile.getFullPathName(), loadError)) {
    failureMessage = "Model load failed: " + loadError.toStdString();
    return false;
  }

  runtime.prepareForInference();
  return true;
}

AudioChunk makeAudioChunk(std::int64_t firstSample) {
  AudioChunk chunk;
  for (auto& channel : chunk) {
    channel.resize(static_cast<size_t>(audio_plugin::kOutputChunkSize));
  }

  constexpr double kTwoPi = 6.28318530717958647692;
  constexpr double kSampleRate =
      static_cast<double>(audio_plugin::kModelSampleRate);
  for (int i = 0; i < audio_plugin::kOutputChunkSize; ++i) {
    const double sample = static_cast<double>(firstSample + i);
    const double time = sample / kSampleRate;
    chunk[0][static_cast<size_t>(i)] = static_cast<float>(
        0.17 * std::sin(kTwoPi * 173.0 * time) +
        0.09 * std::cos(kTwoPi * 997.0 * time));
    chunk[1][static_cast<size_t>(i)] = static_cast<float>(
        -0.13 * std::cos(kTwoPi * 251.0 * time) +
        0.07 * std::sin(kTwoPi * 1301.0 * time));
  }
  return chunk;
}

AudioChunk makeZeroAudioChunk() {
  AudioChunk chunk;
  for (auto& channel : chunk) {
    channel.assign(static_cast<size_t>(audio_plugin::kOutputChunkSize),
                   0.0f);
  }
  return chunk;
}

SeparatedChunk makeSeparatedChunk() {
  SeparatedChunk chunk;
  for (auto& stem : chunk) {
    for (auto& channel : stem) {
      channel.resize(static_cast<size_t>(audio_plugin::kOutputChunkSize));
    }
  }
  return chunk;
}

float maxAbsoluteValue(const AudioChunk& chunk) {
  float maximum = 0.0f;
  for (const auto& channel : chunk) {
    for (const float value : channel) {
      maximum = std::max(maximum, std::abs(value));
    }
  }
  return maximum;
}

float maxAbsoluteValue(const SeparatedChunk& chunk) {
  float maximum = 0.0f;
  for (const auto& stem : chunk) {
    for (const auto& channel : stem) {
      for (const float value : channel) {
        maximum = std::max(maximum, std::abs(value));
      }
    }
  }
  return maximum;
}

float maxChunkDifference(const AudioChunk& lhs, const AudioChunk& rhs) {
  float maximum = 0.0f;
  for (size_t ch = 0; ch < lhs.size(); ++ch) {
    EXPECT_EQ(lhs[ch].size(), rhs[ch].size());
    const size_t count = std::min(lhs[ch].size(), rhs[ch].size());
    for (size_t i = 0; i < count; ++i) {
      maximum = std::max(maximum, std::abs(lhs[ch][i] - rhs[ch][i]));
    }
  }
  return maximum;
}

float maxChunkDifference(const SeparatedChunk& lhs,
                         const SeparatedChunk& rhs) {
  float maximum = 0.0f;
  for (size_t stem = 0; stem < lhs.size(); ++stem) {
    for (size_t ch = 0; ch < lhs[stem].size(); ++ch) {
      EXPECT_EQ(lhs[stem][ch].size(), rhs[stem][ch].size());
      const size_t count =
          std::min(lhs[stem][ch].size(), rhs[stem][ch].size());
      for (size_t i = 0; i < count; ++i) {
        maximum = std::max(
            maximum, std::abs(lhs[stem][ch][i] - rhs[stem][ch][i]));
      }
    }
  }
  return maximum;
}

float maxMixtureReconstructionError(const SeparatedChunk& separated,
                                    const AudioChunk& alignedInput) {
  float maximum = 0.0f;
  for (size_t ch = 0; ch < alignedInput.size(); ++ch) {
    for (size_t i = 0; i < alignedInput[ch].size(); ++i) {
      float reconstructed = 0.0f;
      for (size_t stem = 0; stem < separated.size(); ++stem) {
        reconstructed += separated[stem][ch][i];
      }
      maximum = std::max(
          maximum, std::abs(reconstructed - alignedInput[ch][i]));
    }
  }
  return maximum;
}

double percentileFromSorted(const std::vector<double>& sorted,
                            double percentile) {
  if (sorted.empty()) return 0.0;
  const double position =
      percentile * static_cast<double>(sorted.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

#endif

}  // namespace

TEST(OrtStreamingRuntimeTest,
     StatefulSequenceHonorsPrerollFlushResetAndMixtureSum) {
#if !(defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME)
  GTEST_SKIP() << "ONNX Runtime support not compiled";
#else
  audio_plugin::OnnxRuntime runtime;
  std::string failureMessage;
  ASSERT_TRUE(prepareRuntime(runtime, failureMessage)) << failureMessage;

  const AudioChunk first = makeAudioChunk(0);
  const AudioChunk second =
      makeAudioChunk(audio_plugin::kOutputChunkSize);
  const AudioChunk zero = makeZeroAudioChunk();
  SeparatedChunk separated = makeSeparatedChunk();
  AudioChunk alignedInput = makeZeroAudioChunk();
  bool outputValid = true;

  ASSERT_TRUE(
      runtime.runInference(first, separated, alignedInput, outputValid));
  EXPECT_FALSE(outputValid);
  EXPECT_FLOAT_EQ(maxAbsoluteValue(alignedInput), 0.0f);
  EXPECT_FLOAT_EQ(maxAbsoluteValue(separated), 0.0f);

  ASSERT_TRUE(
      runtime.runInference(second, separated, alignedInput, outputValid));
  ASSERT_TRUE(outputValid);
  EXPECT_FLOAT_EQ(maxChunkDifference(alignedInput, first), 0.0f);
  EXPECT_LE(maxMixtureReconstructionError(separated, alignedInput), 1.0e-6f);
  const SeparatedChunk firstResult = separated;

  // A single zero hop flushes the second real input hop.
  ASSERT_TRUE(runtime.runInference(zero, separated, alignedInput, outputValid));
  ASSERT_TRUE(outputValid);
  EXPECT_FLOAT_EQ(maxChunkDifference(alignedInput, second), 0.0f);
  EXPECT_LE(maxMixtureReconstructionError(separated, alignedInput), 1.0e-6f);

  runtime.resetStreamingState();

  ASSERT_TRUE(
      runtime.runInference(first, separated, alignedInput, outputValid));
  EXPECT_FALSE(outputValid);
  EXPECT_FLOAT_EQ(maxAbsoluteValue(alignedInput), 0.0f);
  EXPECT_FLOAT_EQ(maxAbsoluteValue(separated), 0.0f);

  ASSERT_TRUE(
      runtime.runInference(second, separated, alignedInput, outputValid));
  ASSERT_TRUE(outputValid);
  EXPECT_FLOAT_EQ(maxChunkDifference(alignedInput, first), 0.0f);
  EXPECT_LE(maxMixtureReconstructionError(separated, alignedInput), 1.0e-6f);
  EXPECT_LE(maxChunkDifference(separated, firstResult), 2.0e-5f);
#endif
}

TEST(OrtStreamingRuntimeTest,
     UsesBundledRuntimeWhenASystemOrtIsAlreadyLoaded) {
#if !(defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME)
  GTEST_SKIP() << "ONNX Runtime support not compiled";
#elif !defined(_WIN32)
  GTEST_SKIP() << "Windows DLL-isolation test";
#else
  // Some Windows installations expose an older onnxruntime.dll in System32.
  // Preload it deliberately, then require StemgenRT to resolve the exact DLL
  // bundled beside this test binary instead of reusing the process-global one.
  wchar_t systemDirectory[MAX_PATH] = {};
  const UINT directoryLength =
      GetSystemDirectoryW(systemDirectory, MAX_PATH);
  HMODULE competingOrt = nullptr;
  if (directoryLength > 0 && directoryLength < MAX_PATH) {
    std::wstring systemOrtPath(systemDirectory, directoryLength);
    systemOrtPath += L"\\onnxruntime.dll";
    competingOrt = LoadLibraryW(systemOrtPath.c_str());
  }

  {
    audio_plugin::OnnxRuntime runtime;
    std::string failureMessage;
    ASSERT_TRUE(prepareRuntime(runtime, failureMessage)) << failureMessage;
    EXPECT_EQ(runtime.getRuntimeVersion(), "1.26.0");
    EXPECT_EQ(runtime.getExecutionProvider(), "CPU");
  }

  if (competingOrt != nullptr) {
    FreeLibrary(competingOrt);
  }
#endif
}

TEST(OrtStreamingRuntimeTest, DISABLED_BenchmarkStatefulCpuPerHop) {
#if !(defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME)
  GTEST_SKIP() << "ONNX Runtime support not compiled";
#else
  audio_plugin::OnnxRuntime runtime;
  std::string failureMessage;
  ASSERT_TRUE(prepareRuntime(runtime, failureMessage)) << failureMessage;
  if (runtime.getExecutionProvider() != "CPU") {
    GTEST_SKIP() << "CPU timing requested; active provider is "
                 << runtime.getExecutionProvider();
  }

  SeparatedChunk separated = makeSeparatedChunk();
  AudioChunk alignedInput = makeZeroAudioChunk();
  bool outputValid = false;

  constexpr int kWarmupIterations = 5;
  constexpr int kMeasureIterations = 500;
  const AudioChunk preroll = makeAudioChunk(0);
  ASSERT_TRUE(
      runtime.runInference(preroll, separated, alignedInput, outputValid));
  ASSERT_FALSE(outputValid);

  std::vector<double> runMilliseconds;
  runMilliseconds.reserve(static_cast<size_t>(kMeasureIterations));
  for (int iteration = 0;
       iteration < kWarmupIterations + kMeasureIterations; ++iteration) {
    const AudioChunk input = makeAudioChunk(
        static_cast<std::int64_t>(iteration + 1) *
        audio_plugin::kOutputChunkSize);
    const auto begin = std::chrono::steady_clock::now();
    ASSERT_TRUE(
        runtime.runInference(input, separated, alignedInput, outputValid));
    const auto end = std::chrono::steady_clock::now();
    ASSERT_TRUE(outputValid);
    ASSERT_LE(maxMixtureReconstructionError(separated, alignedInput),
              1.0e-6f);

    if (iteration >= kWarmupIterations) {
      runMilliseconds.push_back(
          std::chrono::duration<double, std::milli>(end - begin).count());
    }
  }

  ASSERT_EQ(runMilliseconds.size(),
            static_cast<size_t>(kMeasureIterations));
  std::sort(runMilliseconds.begin(), runMilliseconds.end());
  const double mean =
      std::accumulate(runMilliseconds.begin(), runMilliseconds.end(), 0.0) /
      static_cast<double>(runMilliseconds.size());
  const double hopBudgetMilliseconds =
      1000.0 * static_cast<double>(audio_plugin::kOutputChunkSize) /
      static_cast<double>(audio_plugin::kModelSampleRate);
  const auto deadlineMisses = static_cast<size_t>(std::count_if(
      runMilliseconds.begin(), runMilliseconds.end(),
      [hopBudgetMilliseconds](double run) {
        return run > hopBudgetMilliseconds;
      }));

  std::cerr << "\nStateful CPU hop timing (ORT "
            << runtime.getRuntimeVersion() << "): mean=" << mean
            << " ms, p50=" << percentileFromSorted(runMilliseconds, 0.50)
            << " ms, p95=" << percentileFromSorted(runMilliseconds, 0.95)
            << " ms, p99=" << percentileFromSorted(runMilliseconds, 0.99)
            << " ms, max=" << runMilliseconds.back()
            << " ms, hop budget=" << hopBudgetMilliseconds
            << " ms, deadline misses=" << deadlineMisses << "/"
            << runMilliseconds.size() << "\n";

  EXPECT_GT(mean, 0.0);
  EXPECT_LT(percentileFromSorted(runMilliseconds, 0.95),
            hopBudgetMilliseconds);
#endif
}
