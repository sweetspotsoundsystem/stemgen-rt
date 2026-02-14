#include <StemgenRT/Constants.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <juce_core/juce_core.h>

#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
#if __has_include(<onnxruntime_c_api.h>)
#include <onnxruntime_c_api.h>
#elif __has_include(<onnxruntime/core/session/onnxruntime_c_api.h>)
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#else
#error "ONNX Runtime headers not found. Ensure include paths are set."
#endif
#endif

namespace {

#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME

const OrtApiBase* ortApiBase() noexcept {
  return OrtGetApiBase();
}

const OrtApi* ortApi() noexcept {
  const OrtApiBase* base = ortApiBase();
  return (base != nullptr) ? base->GetApi(ORT_API_VERSION) : nullptr;
}

struct OrtEnvDeleter {
  void operator()(OrtEnv* p) const noexcept {
    if (p == nullptr) return;
    const OrtApi* api = ortApi();
    if (api == nullptr) return;
    api->ReleaseEnv(p);
  }
};

struct OrtSessionOptionsDeleter {
  void operator()(OrtSessionOptions* p) const noexcept {
    if (p == nullptr) return;
    const OrtApi* api = ortApi();
    if (api == nullptr) return;
    api->ReleaseSessionOptions(p);
  }
};

struct OrtSessionDeleter {
  void operator()(OrtSession* p) const noexcept {
    if (p == nullptr) return;
    const OrtApi* api = ortApi();
    if (api == nullptr) return;
    api->ReleaseSession(p);
  }
};

struct OrtMemoryInfoDeleter {
  void operator()(OrtMemoryInfo* p) const noexcept {
    if (p == nullptr) return;
    const OrtApi* api = ortApi();
    if (api == nullptr) return;
    api->ReleaseMemoryInfo(p);
  }
};

struct OrtValueDeleter {
  void operator()(OrtValue* p) const noexcept {
    if (p == nullptr) return;
    const OrtApi* api = ortApi();
    if (api == nullptr) return;
    api->ReleaseValue(p);
  }
};

using OrtEnvPtr = std::unique_ptr<OrtEnv, OrtEnvDeleter>;
using OrtSessionOptionsPtr =
    std::unique_ptr<OrtSessionOptions, OrtSessionOptionsDeleter>;
using OrtSessionPtr = std::unique_ptr<OrtSession, OrtSessionDeleter>;
using OrtMemoryInfoPtr = std::unique_ptr<OrtMemoryInfo, OrtMemoryInfoDeleter>;
using OrtValuePtr = std::unique_ptr<OrtValue, OrtValueDeleter>;

bool ortCheckOk(const OrtApi* api, OrtStatus* status, const char* what) {
  if (status == nullptr) return true;
  const char* msg = (api != nullptr) ? api->GetErrorMessage(status) : nullptr;
  const std::string msgStr = (msg != nullptr) ? msg : "unknown";
  if (api != nullptr) api->ReleaseStatus(status);
  ADD_FAILURE() << "[ORT] " << what << " failed: " << msgStr;
  return false;
}

double percentileFromSorted(const std::vector<double>& sorted, double pct) {
  if (sorted.empty()) return 0.0;
  if (pct <= 0.0) return sorted.front();
  if (pct >= 1.0) return sorted.back();

  const double idx = pct * static_cast<double>(sorted.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(idx));
  const size_t hi = static_cast<size_t>(std::ceil(idx));
  const double frac = idx - static_cast<double>(lo);
  if (hi <= lo) return sorted[lo];
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

juce::File resolveModelPathForTestBinary() {
  // Keep in sync with PluginProcessor::prepareToPlay() model lookup contract.
  const juce::File exe =
      juce::File::getSpecialLocation(juce::File::currentExecutableFile);
  return exe.getParentDirectory()
      .getParentDirectory()
      .getChildFile("Resources/model.onnx");
}

#endif  // STEMGENRT_USE_ONNXRUNTIME

}  // namespace

TEST(OrtRunBenchmarkTest, DISABLED_BenchmarkOrtRunPerChunk) {
#if !(defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME)
  GTEST_SKIP() << "ONNX Runtime support not compiled";
#else
  const OrtApiBase* base = ortApiBase();
  ASSERT_NE(base, nullptr) << "OrtGetApiBase returned null";
  const OrtApi* api = ortApi();
  ASSERT_NE(api, nullptr) << "Failed to get OrtApi";

  const juce::File modelFile = resolveModelPathForTestBinary();
  ASSERT_TRUE(modelFile.existsAsFile())
      << "Model file not found: " << modelFile.getFullPathName();

  OrtEnv* rawEnv = nullptr;
  if (!ortCheckOk(api,
                  api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "StemgenRTBench",
                                 &rawEnv),
                  "CreateEnv")) {
    return;
  }
  OrtEnvPtr env(rawEnv);

  OrtSessionOptions* rawSessionOptions = nullptr;
  if (!ortCheckOk(api, api->CreateSessionOptions(&rawSessionOptions),
                  "CreateSessionOptions")) {
    return;
  }
  OrtSessionOptionsPtr sessionOptions(rawSessionOptions);

  // Match plugin defaults as closely as possible.
  (void)ortCheckOk(api,
                   api->SetSessionGraphOptimizationLevel(rawSessionOptions,
                                                        ORT_ENABLE_ALL),
                   "SetSessionGraphOptimizationLevel");

  const int numHardwareThreads =
      static_cast<int>(std::thread::hardware_concurrency());
  const int numIntraOpThreads =
      std::min(std::max(numHardwareThreads / 2, 2), 4);
  (void)ortCheckOk(api,
                   api->SetIntraOpNumThreads(rawSessionOptions,
                                             numIntraOpThreads),
                   "SetIntraOpNumThreads");
  (void)ortCheckOk(api, api->SetInterOpNumThreads(rawSessionOptions, 1),
                   "SetInterOpNumThreads");

  // Try to match plugin execution provider selection (TensorRT -> CUDA -> CPU).
  bool gpuEnabled = false;
  std::string executionProvider = "CPU";

  // Priority 1: TensorRT
  {
    OrtTensorRTProviderOptionsV2* trtOptions = nullptr;
    OrtStatus* status = api->CreateTensorRTProviderOptions(&trtOptions);
    if (status == nullptr && trtOptions != nullptr) {
      const char* trtKeys[] = {"device_id", "trt_fp16_enable",
                               "trt_builder_optimization_level",
                               "trt_engine_cache_enable"};
      const char* trtValues[] = {"0", "1", "3", "1"};

      OrtStatus* updateStatus = api->UpdateTensorRTProviderOptions(
          trtOptions, trtKeys, trtValues, 4);
      if (updateStatus != nullptr) {
        api->ReleaseStatus(updateStatus);
        updateStatus = nullptr;
      }

      OrtStatus* appendStatus =
          api->SessionOptionsAppendExecutionProvider_TensorRT_V2(
              rawSessionOptions, trtOptions);
      if (appendStatus == nullptr) {
        gpuEnabled = true;
        executionProvider = "TensorRT";
      } else {
        api->ReleaseStatus(appendStatus);
        appendStatus = nullptr;
      }

      api->ReleaseTensorRTProviderOptions(trtOptions);
    } else {
      if (status != nullptr) {
        api->ReleaseStatus(status);
        status = nullptr;
      }
    }
  }

  // Priority 2: CUDA
  if (!gpuEnabled) {
    OrtCUDAProviderOptionsV2* cudaOptions = nullptr;
    OrtStatus* status = api->CreateCUDAProviderOptions(&cudaOptions);
    if (status == nullptr && cudaOptions != nullptr) {
      const char* keys[] = {"device_id", "arena_extend_strategy",
                            "cudnn_conv_algo_search",
                            "cudnn_conv_use_max_workspace",
                            "do_copy_in_default_stream"};
      const char* values[] = {"0", "kSameAsRequested", "EXHAUSTIVE", "1", "1"};

      OrtStatus* updateStatus =
          api->UpdateCUDAProviderOptions(cudaOptions, keys, values, 5);
      if (updateStatus != nullptr) {
        api->ReleaseStatus(updateStatus);
        updateStatus = nullptr;
      }

      OrtStatus* appendStatus =
          api->SessionOptionsAppendExecutionProvider_CUDA_V2(
              rawSessionOptions, cudaOptions);
      if (appendStatus == nullptr) {
        gpuEnabled = true;
        executionProvider = "CUDA";
      } else {
        api->ReleaseStatus(appendStatus);
        appendStatus = nullptr;
      }

      api->ReleaseCUDAProviderOptions(cudaOptions);
    } else {
      if (status != nullptr) {
        api->ReleaseStatus(status);
        status = nullptr;
      }
    }
  }

  OrtSession* rawSession = nullptr;
  {
    const juce::String modelPath = modelFile.getFullPathName();
#ifdef _WIN32
    std::wstring wideModelPath(modelPath.toWideCharPointer());
    if (!ortCheckOk(api,
                    api->CreateSession(env.get(), wideModelPath.c_str(),
                                       rawSessionOptions, &rawSession),
                    "CreateSession")) {
      return;
    }
#else
    if (!ortCheckOk(api,
                    api->CreateSession(env.get(), modelPath.toRawUTF8(),
                                       rawSessionOptions, &rawSession),
                    "CreateSession")) {
      return;
    }
#endif
  }
  OrtSessionPtr session(rawSession);

  OrtMemoryInfo* rawMemoryInfo = nullptr;
  if (!ortCheckOk(api,
                  api->CreateCpuMemoryInfo(OrtArenaAllocator,
                                           OrtMemTypeDefault, &rawMemoryInfo),
                  "CreateCpuMemoryInfo")) {
    return;
  }
  OrtMemoryInfoPtr memoryInfo(rawMemoryInfo);

  // Build a fixed input tensor (1, 2, kInternalChunkSize).
  std::vector<float> audioInput(
      static_cast<size_t>(audio_plugin::kNumChannels *
                          audio_plugin::kInternalChunkSize));
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& v : audioInput) v = dist(rng);

  const std::int64_t dims[3] = {
      1, static_cast<std::int64_t>(audio_plugin::kNumChannels),
      static_cast<std::int64_t>(audio_plugin::kInternalChunkSize)};

  OrtValue* rawInputTensor = nullptr;
  if (!ortCheckOk(
          api,
          api->CreateTensorWithDataAsOrtValue(
              memoryInfo.get(), audioInput.data(),
              audioInput.size() * sizeof(float), dims, 3,
              ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &rawInputTensor),
          "CreateTensorWithDataAsOrtValue")) {
    return;
  }
  OrtValuePtr inputTensor(rawInputTensor);

  const char* inputNames[1] = {"audio"};
  const char* outputNames[1] = {"separated"};
  const OrtValue* inputValues[1] = {inputTensor.get()};

  constexpr int kWarmupIters = 10;
  constexpr int kIters = 100;
  std::vector<double> runMs;
  runMs.reserve(static_cast<size_t>(kIters));

  for (int i = 0; i < kWarmupIters + kIters; ++i) {
    OrtValue* outputTensor = nullptr;

    const auto t0 = std::chrono::steady_clock::now();
    OrtStatus* status = api->Run(session.get(), nullptr, inputNames, inputValues,
                                 1, outputNames, 1, &outputTensor);
    const auto t1 = std::chrono::steady_clock::now();

    if (!ortCheckOk(api, status, "Run")) {
      if (outputTensor != nullptr) api->ReleaseValue(outputTensor);
      return;
    }

    if (i == 0 && outputTensor != nullptr) {
      OrtTensorTypeAndShapeInfo* shapeInfo = nullptr;
      OrtStatus* shapeStatus = api->GetTensorTypeAndShape(outputTensor, &shapeInfo);
      if (shapeStatus == nullptr && shapeInfo != nullptr) {
        size_t numDims = 0;
        if (api->GetDimensionsCount(shapeInfo, &numDims) == nullptr &&
            numDims > 0 && numDims < 16) {
          std::vector<std::int64_t> outDims(numDims);
          if (api->GetDimensions(shapeInfo, outDims.data(), numDims) == nullptr) {
            std::cerr << "  Output shape: (";
            for (size_t d = 0; d < numDims; ++d) {
              std::cerr << outDims[d];
              if (d + 1 < numDims) std::cerr << ", ";
            }
            std::cerr << ")\n";
          }
        }
        api->ReleaseTensorTypeAndShapeInfo(shapeInfo);
      } else {
        if (shapeStatus != nullptr) {
          api->ReleaseStatus(shapeStatus);
          shapeStatus = nullptr;
        }
        if (shapeInfo != nullptr) {
          api->ReleaseTensorTypeAndShapeInfo(shapeInfo);
          shapeInfo = nullptr;
        }
      }
    }

    if (outputTensor != nullptr) api->ReleaseValue(outputTensor);

    if (i >= kWarmupIters) {
      const std::chrono::duration<double, std::milli> dt = t1 - t0;
      runMs.push_back(dt.count());
    }
  }

  ASSERT_EQ(runMs.size(), static_cast<size_t>(kIters));

  const auto minmax = std::minmax_element(runMs.begin(), runMs.end());
  double sum = 0.0;
  for (double v : runMs) sum += v;
  const double mean = sum / static_cast<double>(runMs.size());

  std::vector<double> sorted = runMs;
  std::sort(sorted.begin(), sorted.end());
  const double p50 = percentileFromSorted(sorted, 0.50);
  const double p90 = percentileFromSorted(sorted, 0.90);
  const double p95 = percentileFromSorted(sorted, 0.95);
  const double p99 = percentileFromSorted(sorted, 0.99);

  const double chunkMs441 =
      (static_cast<double>(audio_plugin::kOutputChunkSize) / 44100.0) * 1000.0;
  const double chunkMs480 =
      (static_cast<double>(audio_plugin::kOutputChunkSize) / 48000.0) * 1000.0;

  std::cerr << "\n";
  std::cerr << "====================================================================\n";
  std::cerr << "  ORT RUN BENCHMARK (per chunk)\n";
  std::cerr << "====================================================================\n";
  std::cerr << "  Model: " << modelFile.getFullPathName() << "\n";
  std::cerr << "  ORT version: " << base->GetVersionString() << "\n";
  std::cerr << "  Graph optimization: ORT_ENABLE_ALL\n";
  std::cerr << "  Threads: intra=" << numIntraOpThreads
            << " inter=1 (hw=" << numHardwareThreads << ")\n";
  std::cerr << "  Execution provider: " << executionProvider << "\n";
  std::cerr << "  Input shape: (1, " << audio_plugin::kNumChannels << ", "
            << audio_plugin::kInternalChunkSize << ")\n";
  std::cerr << "  Chunk: " << audio_plugin::kOutputChunkSize << " samples"
            << " (11.6ms @ 44.1k, 10.7ms @ 48k)\n";
  std::cerr << "  Chunk duration: " << chunkMs441 << "ms @ 44.1kHz, " << chunkMs480
            << "ms @ 48kHz\n";
  std::cerr << "  Warmup: " << kWarmupIters << " iters\n";
  std::cerr << "  Measure: " << kIters << " iters\n";
  std::cerr << "--------------------------------------------------------------------\n";
  std::cerr << "  Run time (OrtApi::Run):\n";
  std::cerr << "    mean=" << mean << " ms\n";
  std::cerr << "    min =" << *minmax.first << " ms\n";
  std::cerr << "    p50 =" << p50 << " ms\n";
  std::cerr << "    p90 =" << p90 << " ms\n";
  std::cerr << "    p95 =" << p95 << " ms\n";
  std::cerr << "    p99 =" << p99 << " ms\n";
  std::cerr << "    max =" << *minmax.second << " ms\n";
  std::cerr << "--------------------------------------------------------------------\n";
  std::cerr << "  Real-time ratio (mean): " << (mean / chunkMs441) << "x of chunk @ 44.1kHz\n";
  std::cerr << "====================================================================\n\n";

#endif
}
