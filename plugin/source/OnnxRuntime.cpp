#include "StemgenRT/OnnxRuntime.h"
#include <StemgenRT/QualifiedModelContract.h>
#include <juce_cryptography/juce_cryptography.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cwchar>
#include <cstring>
#include <exception>
#include <limits>
#include <thread>
#include <utility>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <libloaderapi.h>
#endif

#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
#if __has_include(<onnxruntime_c_api.h>)
#include <onnxruntime_c_api.h>
#elif __has_include(<onnxruntime/core/session/onnxruntime_c_api.h>)
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#else
#error "ONNX Runtime headers not found. Ensure include paths are set."
#endif
#endif

namespace audio_plugin {

std::string OnnxRuntime::getExecutionProvider() const {
  std::lock_guard<std::mutex> lock(statusMutex_);
  return executionProvider_;
}

std::string OnnxRuntime::getRuntimeVersion() const {
  std::lock_guard<std::mutex> lock(statusMutex_);
  return runtimeVersion_;
}

#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME

namespace {

template <size_t Size>
std::vector<std::int64_t> toShapeVector(
    const std::array<std::int64_t, Size>& shape) {
  return {shape.begin(), shape.end()};
}

juce::String toJuceString(std::string_view text) {
  return juce::String::fromUTF8(text.data(), static_cast<int>(text.size()));
}

bool checkOrtStatus(const OrtApi* api,
                    OrtStatus* status,
                    juce::String& errorMessage,
                    const char* operation) {
  if (status == nullptr) {
    return true;
  }

  const char* detail = api != nullptr ? api->GetErrorMessage(status) : nullptr;
  errorMessage = juce::String(operation) + " failed: " +
                 (detail != nullptr ? juce::String(detail)
                                    : juce::String("unknown error"));
  if (api != nullptr) {
    api->ReleaseStatus(status);
  }
  return false;
}

}  // namespace

#ifdef _WIN32
static HMODULE g_ortDllHandle = nullptr;
static std::once_flag g_ortDllLoadOnce;
static std::atomic<bool> g_ortDllLoadFailed{false};

void* OnnxRuntime::ensureOrtDllLoaded() noexcept {
  if (g_ortDllLoadFailed.load(std::memory_order_acquire)) {
    return nullptr;
  }

  try {
    std::call_once(g_ortDllLoadOnce, []() {
      // Resolve the ORT DLL adjacent to this plugin module. Every failure
      // returns normally from the one-time callable, permanently latching
      // the loader into a consistent fail-closed state.
      HMODULE thisModule = nullptr;
      if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                  GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              reinterpret_cast<LPCWSTR>(&g_ortDllHandle),
                              &thisModule)) {
        DBG("[ORT] GetModuleHandleExW failed with error "
            << static_cast<int>(GetLastError()));
        g_ortDllLoadFailed.store(true, std::memory_order_release);
        return;
      }

      wchar_t modulePath[MAX_PATH]{};
      constexpr DWORD modulePathCapacity =
          static_cast<DWORD>(sizeof(modulePath) / sizeof(modulePath[0]));
      const DWORD pathLength =
          GetModuleFileNameW(thisModule, modulePath, modulePathCapacity);
      if (pathLength == 0 || pathLength >= modulePathCapacity) {
        DBG("[ORT] GetModuleFileNameW failed or truncated with error "
            << static_cast<int>(GetLastError()));
        g_ortDllLoadFailed.store(true, std::memory_order_release);
        return;
      }

      DBG("[ORT] Plugin module path: " << juce::String(modulePath));

      wchar_t* const lastSlash = std::wcsrchr(modulePath, L'\\');
      if (lastSlash == nullptr) {
        DBG("[ORT] Could not find backslash in module path");
        g_ortDllLoadFailed.store(true, std::memory_order_release);
        return;
      }

      constexpr wchar_t ortDllName[] = L"onnxruntime.dll";
      const size_t directoryLength =
          static_cast<size_t>(lastSlash - modulePath) + 1;
      constexpr size_t ortDllNameLength =
          sizeof(ortDllName) / sizeof(ortDllName[0]);
      if (directoryLength + ortDllNameLength > modulePathCapacity) {
        DBG("[ORT] Adjacent onnxruntime.dll path is too long");
        g_ortDllLoadFailed.store(true, std::memory_order_release);
        return;
      }
      std::wmemcpy(modulePath + directoryLength, ortDllName, ortDllNameLength);
      DBG("[ORT] Attempting to load: " << juce::String(modulePath));

      const DWORD fileAttributes = GetFileAttributesW(modulePath);
      if (fileAttributes == INVALID_FILE_ATTRIBUTES) {
        DBG("[ORT] Bundled onnxruntime.dll not found at path (error "
            << static_cast<int>(GetLastError()) << ")");
        g_ortDllLoadFailed.store(true, std::memory_order_release);
        return;
      }

      g_ortDllHandle =
          LoadLibraryExW(modulePath, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
      if (g_ortDllHandle == nullptr) {
        DBG("[ORT] LoadLibraryExW failed with error "
            << static_cast<int>(GetLastError()));
        g_ortDllLoadFailed.store(true, std::memory_order_release);
        return;
      }
      DBG("[ORT] Successfully loaded onnxruntime.dll");
    });
  } catch (...) {
    // ensureOrtDllLoaded is noexcept and ORT initialization must fail closed.
    g_ortDllLoadFailed.store(true, std::memory_order_release);
  }

  if (g_ortDllLoadFailed.load(std::memory_order_acquire)) {
    return nullptr;
  }
    return g_ortDllHandle;
}
#endif  // _WIN32

const OrtApiBase* OnnxRuntime::getSafeOrtApiBase() noexcept {
#ifdef _WIN32
    const auto module = static_cast<HMODULE>(ensureOrtDllLoaded());
    if (module == nullptr) {
        return nullptr;
    }
    using GetApiBaseFunction = const OrtApiBase* (ORT_API_CALL*)();
    const auto getApiBase = reinterpret_cast<GetApiBaseFunction>(
        GetProcAddress(module, "OrtGetApiBase"));
    if (getApiBase == nullptr) {
        DBG("[ORT] Bundled DLL does not export OrtGetApiBase");
        return nullptr;
    }
    return getApiBase();
#else
    return OrtGetApiBase();
#endif
}

const OrtApi* OnnxRuntime::getSafeOrtApi() noexcept {
    const OrtApiBase* apiBase = getSafeOrtApiBase();
    if (apiBase == nullptr) {
        DBG("[ORT] OrtGetApiBase() returned nullptr");
        return nullptr;
    }
    return apiBase->GetApi(ORT_API_VERSION);
}

bool OnnxRuntime::validateModelContract(juce::String& errorMessage) const {
  const OrtApi* api = getSafeOrtApi();
  if (api == nullptr || !ortSession_) {
    errorMessage = "Cannot validate model without an ORT session";
    return false;
  }

  OrtAllocator* allocator = nullptr;
  if (!checkOrtStatus(api, api->GetAllocatorWithDefaultOptions(&allocator),
                      errorMessage, "GetAllocatorWithDefaultOptions") ||
      allocator == nullptr) {
    if (errorMessage.isEmpty()) {
      errorMessage = "ORT returned a null default allocator";
    }
    return false;
  }

  const std::array<std::vector<std::int64_t>, 5> expectedInputShapes = {{
      toShapeVector(qualified_model::kInputAudioShape),
      toShapeVector(qualified_model::kInputHistoryShape),
      toShapeVector(qualified_model::kInputHiddenShape),
      toShapeVector(qualified_model::kInputSpectralTailShape),
      toShapeVector(qualified_model::kInputWaveformTailShape),
  }};
  const std::array<std::vector<std::int64_t>, 5> expectedOutputShapes = {{
      toShapeVector(qualified_model::kOutputSeparatedShape),
      toShapeVector(qualified_model::kOutputHistoryShape),
      toShapeVector(qualified_model::kOutputHiddenShape),
      toShapeVector(qualified_model::kOutputSpectralTailShape),
      toShapeVector(qualified_model::kOutputWaveformTailShape),
  }};

  size_t inputCount = 0;
  size_t outputCount = 0;
  if (!checkOrtStatus(api,
                      api->SessionGetInputCount(ortSession_.get(), &inputCount),
                      errorMessage, "SessionGetInputCount") ||
      !checkOrtStatus(
          api, api->SessionGetOutputCount(ortSession_.get(), &outputCount),
          errorMessage, "SessionGetOutputCount")) {
    return false;
  }
  if (inputCount != qualified_model::kInputNames.size() ||
      outputCount != qualified_model::kOutputNames.size()) {
    errorMessage = juce::String("Unexpected model I/O count: inputs=") +
                   juce::String(static_cast<int>(inputCount)) +
                   " outputs=" + juce::String(static_cast<int>(outputCount));
    return false;
  }

  const auto validateTensor =
      [&](bool isInput, size_t index, std::string_view expectedName,
          const std::vector<std::int64_t>& expectedShape) {
        char* rawName = nullptr;
        OrtStatus* nameStatus =
            isInput ? api->SessionGetInputName(ortSession_.get(), index,
                                               allocator, &rawName)
                    : api->SessionGetOutputName(ortSession_.get(), index,
                                                allocator, &rawName);
        if (!checkOrtStatus(
                api, nameStatus, errorMessage,
                isInput ? "SessionGetInputName" : "SessionGetOutputName")) {
          return false;
        }

        const std::string actualName = rawName != nullptr ? rawName : "";
        if (rawName != nullptr) {
          OrtStatus* freeStatus = api->AllocatorFree(allocator, rawName);
          if (!checkOrtStatus(api, freeStatus, errorMessage, "AllocatorFree")) {
            return false;
          }
        }
        if (actualName != expectedName) {
          errorMessage = juce::String(isInput ? "Unexpected input name: "
                                              : "Unexpected output name: ") +
                         juce::String(actualName) + " (expected " +
                         toJuceString(expectedName) + ")";
          return false;
        }

        OrtTypeInfo* typeInfo = nullptr;
        OrtStatus* typeStatus = isInput
                                    ? api->SessionGetInputTypeInfo(
                                          ortSession_.get(), index, &typeInfo)
                                    : api->SessionGetOutputTypeInfo(
                                          ortSession_.get(), index, &typeInfo);
        if (!checkOrtStatus(api, typeStatus, errorMessage,
                            isInput ? "SessionGetInputTypeInfo"
                                    : "SessionGetOutputTypeInfo") ||
            typeInfo == nullptr) {
          if (typeInfo != nullptr) {
            api->ReleaseTypeInfo(typeInfo);
          }
          return false;
        }

        const OrtTensorTypeAndShapeInfo* tensorInfo = nullptr;
        bool ok = checkOrtStatus(
            api, api->CastTypeInfoToTensorInfo(typeInfo, &tensorInfo),
            errorMessage, "CastTypeInfoToTensorInfo");
        if (!ok || tensorInfo == nullptr) {
          api->ReleaseTypeInfo(typeInfo);
          if (ok) {
            errorMessage = toJuceString(expectedName) + " is not a tensor";
          }
          return false;
        }

        ONNXTensorElementDataType elementType =
            ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        size_t rank = 0;
        ok = checkOrtStatus(api,
                            api->GetTensorElementType(tensorInfo, &elementType),
                            errorMessage, "GetTensorElementType") &&
             checkOrtStatus(api, api->GetDimensionsCount(tensorInfo, &rank),
                            errorMessage, "GetDimensionsCount");
        std::vector<std::int64_t> actualShape(rank);
        if (ok && rank > 0) {
          ok = checkOrtStatus(
              api, api->GetDimensions(tensorInfo, actualShape.data(), rank),
              errorMessage, "GetDimensions");
        }
        api->ReleaseTypeInfo(typeInfo);

        if (!ok) {
          return false;
        }
        if (elementType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            actualShape != expectedShape) {
          errorMessage = juce::String("Unexpected type or shape for ") +
                         toJuceString(expectedName);
          return false;
        }
        return true;
      };

  for (size_t index = 0; index < qualified_model::kInputNames.size(); ++index) {
    if (!validateTensor(true, index, qualified_model::kInputNames[index],
                        expectedInputShapes[index])) {
      return false;
    }
  }
  for (size_t index = 0; index < qualified_model::kOutputNames.size();
       ++index) {
    if (!validateTensor(false, index, qualified_model::kOutputNames[index],
                        expectedOutputShapes[index])) {
      return false;
    }
  }

  OrtModelMetadata* metadata = nullptr;
  if (!checkOrtStatus(
          api, api->SessionGetModelMetadata(ortSession_.get(), &metadata),
          errorMessage, "SessionGetModelMetadata") ||
      metadata == nullptr) {
    if (metadata != nullptr) {
      api->ReleaseModelMetadata(metadata);
    }
    return false;
  }

  for (const auto& [key, expectedValue] : qualified_model::kMetadata) {
    char* rawValue = nullptr;
    const bool lookupOk =
        checkOrtStatus(api,
                       api->ModelMetadataLookupCustomMetadataMap(
                           metadata, allocator, key.data(), &rawValue),
                       errorMessage, "ModelMetadataLookupCustomMetadataMap");
    if (!lookupOk) {
      api->ReleaseModelMetadata(metadata);
      return false;
    }
    // ONNX Runtime returns nullptr when the key is absent. The cropped1024
    // contract validates only metadata fields embedded by the accepted
    // exporter.
    if (rawValue == nullptr) {
      errorMessage = juce::String("Missing model metadata ") +
                     toJuceString(key);
      api->ReleaseModelMetadata(metadata);
      return false;
    }
    const std::string actualValue = rawValue;
    OrtStatus* freeStatus = api->AllocatorFree(allocator, rawValue);
    if (!checkOrtStatus(api, freeStatus, errorMessage, "AllocatorFree")) {
      api->ReleaseModelMetadata(metadata);
      return false;
    }
    if (actualValue != expectedValue) {
      errorMessage = juce::String("Unexpected model metadata ") +
                     toJuceString(key) + "=" + juce::String(actualValue) +
                     " (expected " + toJuceString(expectedValue) + ")";
      api->ReleaseModelMetadata(metadata);
      return false;
    }
  }

  api->ReleaseModelMetadata(metadata);
  return true;
}

void OnnxRuntime::OrtEnvDeleter::operator()(OrtEnv* p) const noexcept {
    if (p == nullptr) return;
    const OrtApi* api = getSafeOrtApi();
    if (api == nullptr) return;
    api->ReleaseEnv(p);
}

void OnnxRuntime::OrtSessionDeleter::operator()(OrtSession* p) const noexcept {
    if (p == nullptr) return;
    const OrtApi* api = getSafeOrtApi();
    if (api == nullptr) return;
    api->ReleaseSession(p);
}

OnnxRuntime::OnnxRuntime() {
    const OrtApi* api = getSafeOrtApi();
    if (api != nullptr) {
        const OrtApiBase* apiBase = getSafeOrtApiBase();
        const std::string runtimeVersion =
            (apiBase != nullptr) ? apiBase->GetVersionString() : "unknown";
        {
          std::lock_guard<std::mutex> lock(statusMutex_);
          runtimeVersion_ = runtimeVersion;
        }

        OrtEnv* rawEnv = nullptr;
        OrtStatus* status = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "StemgenRT", &rawEnv);
        if (status == nullptr) {
            ortEnv_.reset(rawEnv);
            {
              std::lock_guard<std::mutex> lock(statusMutex_);
              ortInitialized_.store(true, std::memory_order_release);
            }
            DBG("[ORT] Initialized. Version: " << juce::String(runtimeVersion));
        } else {
            DBG("[ORT] Failed to create OrtEnv: " << api->GetErrorMessage(status));
            api->ReleaseStatus(status);
        }
    } else {
        DBG("[ORT] getSafeOrtApi() returned null");
    }
}

OnnxRuntime::~OnnxRuntime() {
  releasePreallocatedTensorValues();
  if (ortMemoryInfo_ != nullptr) {
    const OrtApi* api = getSafeOrtApi();
    if (api != nullptr) {
      api->ReleaseMemoryInfo(ortMemoryInfo_);
    }
    ortMemoryInfo_ = nullptr;
  }
}

bool OnnxRuntime::loadModel(const juce::String& modelPath,
                            juce::String& errorMessage,
                            std::optional<int> intraOpThreadCount) {
  // A new load attempt invalidates any tensor bindings associated with the
  // preceding session, even if this attempt later fails.
  inferenceReady_.store(false, std::memory_order_release);
  modelLoaded_.store(false, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(statusMutex_);
    inferencePreparationError_.clear();
  }
  if (!ortInitialized_.load(std::memory_order_acquire) || !ortEnv_) {
    errorMessage = "ORT not initialized";
    return false;
  }

  const OrtApi* api = getSafeOrtApi();
  if (api == nullptr) {
    errorMessage = "ORT API not available";
    return false;
  }

  if (intraOpThreadCount.has_value() && *intraOpThreadCount < 1) {
    errorMessage = "Explicit intra-op thread count must be positive";
    std::lock_guard<std::mutex> lock(statusMutex_);
    modelLoadError_ = errorMessage;
    return false;
  }

  // Verify the deployed bytes before ORT parses the graph. CMake performs the
  // same check at packaging time; this closes the gap if a bundle is damaged
  // or its Resources payload is replaced after the build.
  const juce::File modelFile(modelPath);
  if (!modelFile.existsAsFile()) {
    errorMessage = juce::String("Model file not found: ") + modelPath;
    std::lock_guard<std::mutex> lock(statusMutex_);
    modelLoadError_ = errorMessage;
    return false;
  }
  if (modelFile.getSize() != qualified_model::kModelByteSize) {
    errorMessage = juce::String("Unexpected model size: expected ") +
                   juce::String(qualified_model::kModelByteSize) +
                   " bytes, got " + juce::String(modelFile.getSize());
    std::lock_guard<std::mutex> lock(statusMutex_);
    modelLoadError_ = errorMessage;
    return false;
  }
  const juce::String actualModelSha = juce::SHA256(modelFile).toHexString();
  if (!actualModelSha.equalsIgnoreCase(
          toJuceString(qualified_model::kModelSha256))) {
    errorMessage = juce::String("Unexpected model SHA-256: ") + actualModelSha;
    std::lock_guard<std::mutex> lock(statusMutex_);
    modelLoadError_ = errorMessage;
    return false;
  }

  // Create session options
  OrtSessionOptions* sessionOptions = nullptr;
  OrtStatus* status = api->CreateSessionOptions(&sessionOptions);
  if (status != nullptr) {
    errorMessage =
        juce::String("Session options error: ") + api->GetErrorMessage(status);
    api->ReleaseStatus(status);
    return false;
  }

  const auto applySessionOption = [&](OrtStatus* optionStatus,
                                      const char* operation) {
    if (checkOrtStatus(api, optionStatus, errorMessage, operation)) {
      return true;
    }
    api->ReleaseSessionOptions(sessionOptions);
    return false;
  };

  if (!applySessionOption(
          api->SetSessionGraphOptimizationLevel(sessionOptions, ORT_ENABLE_ALL),
          "SetSessionGraphOptimizationLevel")) {
    return false;
  }

  // Apply the platform-qualified production policy when no override is
  // supplied. A positive override creates a distinct immutable ORT thread pool
  // for that session, allowing an apples-to-apples benchmark.
  const int numHardwareThreads =
      std::max(static_cast<int>(std::thread::hardware_concurrency()), 1);
  const int automaticIntraOpThreads =
      calculateAutomaticOrtIntraOpThreadCount(numHardwareThreads);
  const int numIntraOpThreads =
      intraOpThreadCount.value_or(automaticIntraOpThreads);

  if (!applySessionOption(
          api->SetIntraOpNumThreads(sessionOptions, numIntraOpThreads),
          "SetIntraOpNumThreads")) {
    return false;
  }
  if (!applySessionOption(api->SetInterOpNumThreads(sessionOptions, 1),
                          "SetInterOpNumThreads")) {
    return false;
  }

  // Avoid busy-spinning ORT worker threads between audio hops. Inference runs
  // on the plugin's dedicated worker and must leave the host audio thread and
  // other plugins schedulable.
  if (!applySessionOption(
          api->AddSessionConfigEntry(sessionOptions,
                                     "session.intra_op.allow_spinning", "0"),
          "DisableIntraOpSpinning")) {
    return false;
  }
  if (!applySessionOption(
          api->AddSessionConfigEntry(sessionOptions,
                                     "session.inter_op.allow_spinning", "0"),
          "DisableInterOpSpinning")) {
    return false;
  }

  DBG("[ORT] Using " << numIntraOpThreads << " intra-op threads (of "
                     << numHardwareThreads << " available)");

  // The shipping plugin is deliberately CPU-only. This is the universally
  // available path, matches the deployment target, and avoids silently
  // changing numerical/runtime behavior with host-specific accelerators.
  DBG("[ORT] Using CPU execution provider");

  // Create the inference session
  OrtSession* rawSession = nullptr;
#ifdef _WIN32
  std::wstring wideModelPath(modelPath.toWideCharPointer());
  status = api->CreateSession(ortEnv_.get(), wideModelPath.c_str(),
                              sessionOptions, &rawSession);
#else
  status = api->CreateSession(ortEnv_.get(), modelPath.toRawUTF8(),
                              sessionOptions, &rawSession);
#endif

  api->ReleaseSessionOptions(sessionOptions);

  if (status != nullptr) {
    errorMessage = juce::String("Session creation failed: ") +
                   api->GetErrorMessage(status);
    api->ReleaseStatus(status);
    return false;
  }

  ortSession_.reset(rawSession);
  if (!validateModelContract(errorMessage)) {
    {
      std::lock_guard<std::mutex> lock(statusMutex_);
      modelLoadError_ = errorMessage;
      modelLoaded_.store(false, std::memory_order_release);
    }
    ortSession_.reset();
    DBG("[ORT] Model contract validation failed: " << errorMessage);
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(statusMutex_);
    modelLoadError_.clear();
    executionProvider_ = "CPU";
    modelLoaded_.store(true, std::memory_order_release);
  }

  DBG("[ORT] Model loaded successfully from: " << modelPath);
  DBG("[ORT] Execution provider: CPU");

  return true;
}

void OnnxRuntime::releasePreallocatedTensorValues() noexcept {
  const OrtApi* api = getSafeOrtApi();
  if (api != nullptr) {
    for (OrtValue*& value : inputTensorValues_) {
      if (value != nullptr) {
        api->ReleaseValue(value);
      }
      value = nullptr;
    }
    for (OrtValue*& value : outputTensorValues_) {
      if (value != nullptr) {
        api->ReleaseValue(value);
      }
      value = nullptr;
    }
    return;
  }

  // Values cannot have been created without a valid API. Still clear the
  // bookkeeping so a failed initialization remains consistently unusable.
  inputTensorValues_.fill(nullptr);
  outputTensorValues_.fill(nullptr);
}

bool OnnxRuntime::createPreallocatedTensorValues(juce::String& errorMessage) {
  const OrtApi* api = getSafeOrtApi();
  if (api == nullptr || ortMemoryInfo_ == nullptr) {
    errorMessage = "ORT memory services are unavailable";
    return false;
  }

  const auto createTensor = [&](OrtValue*& value, std::vector<float>& data,
                                const auto& shape, const char* name) {
    OrtStatus* status = api->CreateTensorWithDataAsOrtValue(
        ortMemoryInfo_, data.data(), data.size() * sizeof(float), shape.data(),
        shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &value);
    if (status == nullptr) {
      return true;
    }
    const char* detail = api->GetErrorMessage(status);
    errorMessage = juce::String("Failed to create preallocated tensor ") +
                   juce::String(name) + ": " +
                   juce::String(detail != nullptr ? detail : "unknown error");
    api->ReleaseStatus(status);
    return false;
  };

  if (!createTensor(inputTensorValues_[0], audioChunkBuffer_,
                    qualified_model::kInputAudioShape,
                    qualified_model::kInputNames[0].data()) ||
      !createTensor(inputTensorValues_[1], audioHistory_,
                    qualified_model::kInputHistoryShape,
                    qualified_model::kInputNames[1].data()) ||
      !createTensor(inputTensorValues_[2], fusionHidden_,
                    qualified_model::kInputHiddenShape,
                    qualified_model::kInputNames[2].data()) ||
      !createTensor(inputTensorValues_[3], spectralNumeratorTail_,
                    qualified_model::kInputSpectralTailShape,
                    qualified_model::kInputNames[3].data()) ||
      !createTensor(inputTensorValues_[4], waveformTail_,
                    qualified_model::kInputWaveformTailShape,
                    qualified_model::kInputNames[4].data()) ||
      !createTensor(outputTensorValues_[0], separatedOutputBuffer_,
                    qualified_model::kOutputSeparatedShape,
                    qualified_model::kOutputNames[0].data()) ||
      !createTensor(outputTensorValues_[1], nextAudioHistoryBuffer_,
                    qualified_model::kOutputHistoryShape,
                    qualified_model::kOutputNames[1].data()) ||
      !createTensor(outputTensorValues_[2], nextFusionHiddenBuffer_,
                    qualified_model::kOutputHiddenShape,
                    qualified_model::kOutputNames[2].data()) ||
      !createTensor(outputTensorValues_[3], nextSpectralNumeratorTail_,
                    qualified_model::kOutputSpectralTailShape,
                    qualified_model::kOutputNames[3].data()) ||
      !createTensor(outputTensorValues_[4], nextWaveformTail_,
                    qualified_model::kOutputWaveformTailShape,
                    qualified_model::kOutputNames[4].data())) {
    releasePreallocatedTensorValues();
    return false;
  }

  return true;
}

bool OnnxRuntime::prepareForInference(juce::String& errorMessage) {
  inferenceReady_.store(false, std::memory_order_release);
  errorMessage.clear();

  const auto failPreparation = [this,
                                &errorMessage](const juce::String& message) {
    errorMessage = message;
    {
      std::lock_guard<std::mutex> statusLock(statusMutex_);
      inferencePreparationError_ = message;
    }
    DBG("[ORT] Inference preparation failed: " << message);
    return false;
  };

  if (!modelLoaded_.load(std::memory_order_acquire) || !ortSession_) {
    return failPreparation("Cannot prepare inference: model is not loaded");
  }

  const OrtApi* api = getSafeOrtApi();
  if (api == nullptr) {
    return failPreparation("Cannot prepare inference: ORT API is unavailable");
  }

  std::lock_guard<std::mutex> lock(streamingStateMutex_);
  releasePreallocatedTensorValues();

  // Pre-allocate memory info.
  if (ortMemoryInfo_ == nullptr) {
    OrtStatus* status = api->CreateCpuMemoryInfo(
        OrtArenaAllocator, OrtMemTypeDefault, &ortMemoryInfo_);
    if (status != nullptr) {
      const char* detail = api->GetErrorMessage(status);
      const juce::String message =
          juce::String("Failed to create CPU memory info: ") +
          juce::String(detail != nullptr ? detail : "unknown error");
      api->ReleaseStatus(status);
      return failPreparation(message);
    }
  }

  const size_t audioElements =
      static_cast<size_t>(kNumChannels * kOutputChunkSize);
  const size_t separatedElements =
      static_cast<size_t>(kNumStems * kNumChannels * kOutputChunkSize);
  const size_t tailElements =
      static_cast<size_t>(kNumStems * kNumChannels * kOutputChunkSize);
  const size_t historyElements =
      static_cast<size_t>(kNumChannels * kAnalysisHistorySize);
  const size_t hiddenElements =
      static_cast<size_t>(kFusionHiddenLayers * kFusionHiddenSize);

  try {
    audioChunkBuffer_.resize(audioElements);
    audioHistory_.resize(historyElements);
    spectralNumeratorTail_.resize(tailElements);
    waveformTail_.resize(tailElements);
    fusionHidden_.resize(hiddenElements);
    previousAlignedInput_.resize(audioElements);
    separatedOutputBuffer_.resize(separatedElements);
    nextAudioHistoryBuffer_.resize(historyElements);
    nextSpectralNumeratorTail_.resize(tailElements);
    nextWaveformTail_.resize(tailElements);
    nextFusionHiddenBuffer_.resize(hiddenElements);
  } catch (const std::exception& exception) {
    releasePreallocatedTensorValues();
    return failPreparation(
        juce::String("Failed to allocate streaming buffers: ") +
        juce::String(exception.what()));
  } catch (...) {
    releasePreallocatedTensorValues();
    return failPreparation("Failed to allocate streaming buffers");
  }

  resetStreamingStateUnlocked();
  if (!createPreallocatedTensorValues(errorMessage)) {
    const juce::String message =
        errorMessage.isNotEmpty()
            ? errorMessage
            : juce::String("Failed to prepare fixed streaming tensors");
    return failPreparation(message);
  }

  {
    std::lock_guard<std::mutex> statusLock(statusMutex_);
    inferencePreparationError_.clear();
  }
  inferenceReady_.store(true, std::memory_order_release);
  return true;
}

void OnnxRuntime::resetStreamingStateUnlocked() {
  std::fill(audioHistory_.begin(), audioHistory_.end(), 0.0f);
  std::fill(spectralNumeratorTail_.begin(), spectralNumeratorTail_.end(), 0.0f);
  std::fill(waveformTail_.begin(), waveformTail_.end(), 0.0f);
  std::fill(fusionHidden_.begin(), fusionHidden_.end(), 0.0f);
  std::fill(previousAlignedInput_.begin(), previousAlignedInput_.end(), 0.0f);
  hasPreviousAlignedInput_ = false;
}

void OnnxRuntime::resetStreamingState() {
    std::lock_guard<std::mutex> lock(streamingStateMutex_);
    resetStreamingStateUnlocked();
}

bool OnnxRuntime::runInference(
    const std::array<std::vector<float>, kNumChannels>& inputChunk,
    std::array<std::array<std::vector<float>, kNumChannels>, kNumStems>& outputChunks,
    std::array<std::vector<float>, kNumChannels>& alignedInput,
    bool& outputValid) {
  outputValid = false;
  if (!inferenceReady_.load(std::memory_order_acquire) ||
      !modelLoaded_.load(std::memory_order_acquire) || !ortSession_ ||
      !ortMemoryInfo_) {
    return false;
  }

    const OrtApi* api = getSafeOrtApi();
    if (api == nullptr) return false;

    std::lock_guard<std::mutex> lock(streamingStateMutex_);
    if (!inferenceReady_.load(std::memory_order_acquire)) {
      return false;
    }

    const size_t audioElements =
        static_cast<size_t>(kNumChannels * kOutputChunkSize);
    const size_t separatedElements =
        static_cast<size_t>(kNumStems * kNumChannels * kOutputChunkSize);
    const size_t tailElements =
        static_cast<size_t>(kNumStems * kNumChannels * kOutputChunkSize);
    const size_t historyElements =
        static_cast<size_t>(kNumChannels * kAnalysisHistorySize);
    const size_t hiddenElements =
        static_cast<size_t>(kFusionHiddenLayers * kFusionHiddenSize);

    if (audioChunkBuffer_.size() != audioElements ||
        audioHistory_.size() != historyElements ||
        spectralNumeratorTail_.size() != tailElements ||
        waveformTail_.size() != tailElements ||
        fusionHidden_.size() != hiddenElements ||
        previousAlignedInput_.size() != audioElements ||
        separatedOutputBuffer_.size() != separatedElements ||
        nextAudioHistoryBuffer_.size() != historyElements ||
        nextSpectralNumeratorTail_.size() != tailElements ||
        nextWaveformTail_.size() != tailElements ||
        nextFusionHiddenBuffer_.size() != hiddenElements ||
        std::any_of(inputTensorValues_.begin(), inputTensorValues_.end(),
                    [](const OrtValue* value) { return value == nullptr; }) ||
        std::any_of(outputTensorValues_.begin(), outputTensorValues_.end(),
                    [](const OrtValue* value) { return value == nullptr; })) {
      DBG("[ORT] Streaming buffers were not prepared");
      return false;
    }

    // Feed the graph at the exact native input level used by its frozen
    // quality evaluation. The graph emits the preceding input hop, so retain
    // a separate raw-domain copy for Main and the final residual instead of
    // deriving presentation alignment from provider-owned recurrent state.
    for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
      if (inputChunk[ch].size() != static_cast<size_t>(kOutputChunkSize) ||
          alignedInput[ch].size() != static_cast<size_t>(kOutputChunkSize)) {
        DBG("[ORT] Invalid streaming input or aligned-output shape");
        resetStreamingStateUnlocked();
        return false;
      }
      const size_t channelOffset =
          ch * static_cast<size_t>(kOutputChunkSize);
      for (size_t i = 0; i < static_cast<size_t>(kOutputChunkSize); ++i) {
        const float sample = inputChunk[ch][i];
        if (!std::isfinite(sample)) {
          DBG("[ORT] Non-finite streaming input");
          resetStreamingStateUnlocked();
          return false;
        }
        audioChunkBuffer_[channelOffset + i] = sample;
        alignedInput[ch][i] =
            hasPreviousAlignedInput_
                ? previousAlignedInput_[channelOffset + i]
                : 0.0f;
      }
    }

    const std::array<const char*, 5> inputNames = {
        qualified_model::kInputNames[0].data(),
        qualified_model::kInputNames[1].data(),
        qualified_model::kInputNames[2].data(),
        qualified_model::kInputNames[3].data(),
        qualified_model::kInputNames[4].data()};
    const std::array<const char*, 5> outputNames = {
        qualified_model::kOutputNames[0].data(),
        qualified_model::kOutputNames[1].data(),
        qualified_model::kOutputNames[2].data(),
        qualified_model::kOutputNames[3].data(),
        qualified_model::kOutputNames[4].data()};
    const OrtValue* constInputValues[5] = {
        inputTensorValues_[0], inputTensorValues_[1], inputTensorValues_[2],
        inputTensorValues_[3], inputTensorValues_[4]};
    std::array<OrtValue*, 5> runOutputValues = outputTensorValues_;

    OrtStatus* runStatus =
        api->Run(ortSession_.get(), nullptr, inputNames.data(),
                 constInputValues, inputNames.size(), outputNames.data(),
                 outputNames.size(), runOutputValues.data());

    // With every output pre-bound, ORT must retain the supplied OrtValue
    // pointers. Release any unexpected replacement without disturbing the
    // persistent bindings, then fail closed.
    bool outputBindingChanged = false;
    for (size_t index = 0; index < runOutputValues.size(); ++index) {
      if (runOutputValues[index] == outputTensorValues_[index]) {
        continue;
      }
      outputBindingChanged = true;
      const bool aliasesPersistentBinding =
          runOutputValues[index] != nullptr &&
          std::find(outputTensorValues_.begin(), outputTensorValues_.end(),
                    runOutputValues[index]) != outputTensorValues_.end();
      if (runOutputValues[index] != nullptr && !aliasesPersistentBinding) {
        api->ReleaseValue(runOutputValues[index]);
      }
    }

    if (runStatus != nullptr) {
      DBG("[ORT] Inference failed: " << api->GetErrorMessage(runStatus));
      api->ReleaseStatus(runStatus);
      resetStreamingStateUnlocked();
      return false;
    }
    if (outputBindingChanged) {
      DBG("[ORT] Run replaced a preallocated streaming output binding");
      resetStreamingStateUnlocked();
      return false;
    }

    const auto allFinite = [](const float* data, size_t count) {
      return std::all_of(data, data + count,
                         [](float value) { return std::isfinite(value); });
    };
    if (!allFinite(separatedOutputBuffer_.data(), separatedElements) ||
        !allFinite(nextAudioHistoryBuffer_.data(), historyElements) ||
        !allFinite(nextSpectralNumeratorTail_.data(), tailElements) ||
        !allFinite(nextWaveformTail_.data(), tailElements) ||
        !allFinite(nextFusionHiddenBuffer_.data(), hiddenElements)) {
      DBG("[ORT] Non-finite streaming output; resetting recurrent state");
      resetStreamingStateUnlocked();
      return false;
    }

    outputValid = hasPreviousAlignedInput_;
    for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
      for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
        auto& destination = outputChunks[stem][ch];
        if (destination.size() != static_cast<size_t>(kOutputChunkSize)) {
          resetStreamingStateUnlocked();
          outputValid = false;
          return false;
        }
        const size_t offset = (stem * static_cast<size_t>(kNumChannels) + ch) *
                              static_cast<size_t>(kOutputChunkSize);
        if (outputValid) {
          for (size_t i = 0; i < static_cast<size_t>(kOutputChunkSize); ++i) {
            destination[i] = separatedOutputBuffer_[offset + i];
          }
        } else {
          std::fill(destination.begin(), destination.end(), 0.0f);
        }
      }
    }

    // The accepted export already returns all four deployed stems. Preserve
    // Other exactly here; an extra residual rewrite would hide graph errors.
    // OutputWriter applies its final residual only after presentation fades.
    if (outputValid) {
      for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
        for (size_t i = 0; i < static_cast<size_t>(kOutputChunkSize); ++i) {
          double mixture = 0.0;
          double magnitude = 1.0;
          for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
            const double sample =
                static_cast<double>(outputChunks[stem][ch][i]);
            mixture += sample;
            magnitude += std::abs(sample);
          }
          // Scale the rounding allowance for valid floating-point audio above
          // unity and cancellation between stems; do not alter input gain.
          const double tolerance =
              8.0 * static_cast<double>(std::numeric_limits<float>::epsilon()) *
              magnitude;
          if (!std::isfinite(mixture) ||
              std::abs(mixture - static_cast<double>(alignedInput[ch][i])) >
                  tolerance) {
            resetStreamingStateUnlocked();
            outputValid = false;
            return false;
          }
        }
      }
    }

    // Validate the actual native-domain values returned to the queue, without
    // changing the graph's complete deployed output.
    bool finalOutputIsFinite = true;
    for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
      finalOutputIsFinite =
          finalOutputIsFinite &&
          allFinite(alignedInput[ch].data(), alignedInput[ch].size());
      for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
        finalOutputIsFinite =
            finalOutputIsFinite && allFinite(outputChunks[stem][ch].data(),
                                             outputChunks[stem][ch].size());
      }
    }
    if (!finalOutputIsFinite) {
      DBG("[ORT] Non-finite native streaming output; resetting recurrent "
          "state");
      outputValid = false;
      for (auto& channel : alignedInput) {
        std::fill(channel.begin(), channel.end(), 0.0f);
      }
      for (auto& stem : outputChunks) {
        for (auto& channel : stem) {
          std::fill(channel.begin(), channel.end(), 0.0f);
        }
      }
      resetStreamingStateUnlocked();
      return false;
    }

    std::memcpy(audioHistory_.data(), nextAudioHistoryBuffer_.data(),
                historyElements * sizeof(float));
    std::memcpy(spectralNumeratorTail_.data(),
                nextSpectralNumeratorTail_.data(),
                tailElements * sizeof(float));
    std::memcpy(waveformTail_.data(), nextWaveformTail_.data(),
                tailElements * sizeof(float));
    std::memcpy(fusionHidden_.data(), nextFusionHiddenBuffer_.data(),
                hiddenElements * sizeof(float));
    std::memcpy(previousAlignedInput_.data(), audioChunkBuffer_.data(),
                audioElements * sizeof(float));
    hasPreviousAlignedInput_ = true;

    return true;
}

juce::String OnnxRuntime::getStatusString() const {
  std::lock_guard<std::mutex> lock(statusMutex_);
  if (modelLoaded_.load(std::memory_order_acquire)) {
    if (!inferenceReady_.load(std::memory_order_acquire)) {
      if (inferencePreparationError_.isNotEmpty()) {
        return juce::String("Inference preparation error: ") +
               inferencePreparationError_;
      }
      return juce::String("HS-TasNet loaded (") +
             juce::String(executionProvider_) + ", ORT v" +
             juce::String(runtimeVersion_) + "; inference not prepared)";
    }
    return juce::String("HS-TasNet loaded (") +
           juce::String(executionProvider_) + ", ORT v" +
           juce::String(runtimeVersion_) + ")";
  }
  if (ortInitialized_.load(std::memory_order_acquire)) {
    if (modelLoadError_.isNotEmpty()) {
      return juce::String("Model error: ") + modelLoadError_;
    }
    if (inferencePreparationError_.isNotEmpty()) {
      return juce::String("Inference preparation error: ") +
             inferencePreparationError_;
    }
    return juce::String("ORT v") + juce::String(runtimeVersion_) +
           " ready (model not loaded)";
  }
  return "ONNX Runtime not available";
}

#else  // !STEMGENRT_USE_ONNXRUNTIME

// Stub implementations when ONNX Runtime is disabled
OnnxRuntime::OnnxRuntime() {}
OnnxRuntime::~OnnxRuntime() {}
bool OnnxRuntime::loadModel(const juce::String&,
                            juce::String& errorMessage,
                            std::optional<int>) {
  errorMessage = "ONNX Runtime support not compiled";
  return false;
}
bool OnnxRuntime::prepareForInference(juce::String& errorMessage) {
  errorMessage = "ONNX Runtime support not compiled";
  return false;
}
void OnnxRuntime::resetStreamingState() {}
bool OnnxRuntime::runInference(
    const std::array<std::vector<float>, kNumChannels>&,
    std::array<std::array<std::vector<float>, kNumChannels>, kNumStems>&,
    std::array<std::vector<float>, kNumChannels>&,
    bool& outputValid) {
  outputValid = false;
  return false;
}
juce::String OnnxRuntime::getStatusString() const {
  return "ONNX Runtime not available";
}

void OnnxRuntime::OrtEnvDeleter::operator()(OrtEnv*) const noexcept {}
void OnnxRuntime::OrtSessionDeleter::operator()(OrtSession*) const noexcept {}

#endif  // STEMGENRT_USE_ONNXRUNTIME

}  // namespace audio_plugin
