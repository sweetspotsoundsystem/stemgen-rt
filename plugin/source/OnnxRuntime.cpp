#include "StemgenRT/OnnxRuntime.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>

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

#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME

#ifdef _WIN32
static HMODULE g_ortDllHandle = nullptr;
static bool g_ortDllLoadAttempted = false;

void OnnxRuntime::ensureOrtDllLoaded() noexcept {
    if (g_ortDllLoadAttempted) return;
    g_ortDllLoadAttempted = true;

    // Check if onnxruntime.dll is already loaded
    g_ortDllHandle = GetModuleHandleW(L"onnxruntime.dll");
    if (g_ortDllHandle != nullptr) {
        DBG("[ORT] onnxruntime.dll already loaded in process");
        return;
    }

    // Get the path to this DLL (the plugin itself)
    HMODULE thisModule = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&ensureOrtDllLoaded), &thisModule)) {
        DBG("[ORT] GetModuleHandleExW failed with error " << static_cast<int>(GetLastError()));
        return;
    }

    wchar_t modulePath[MAX_PATH];
    DWORD pathLen = GetModuleFileNameW(thisModule, modulePath, MAX_PATH);
    juce::ignoreUnused(pathLen);
    if (pathLen == 0) {
        DBG("[ORT] GetModuleFileNameW failed with error " << static_cast<int>(GetLastError()));
        return;
    }

    DBG("[ORT] Plugin module path: " << juce::String(modulePath));

    std::wstring dllPath(modulePath);
    size_t lastSlash = dllPath.rfind(L'\\');
    if (lastSlash == std::wstring::npos) {
        DBG("[ORT] Could not find backslash in module path");
        return;
    }

    dllPath = dllPath.substr(0, lastSlash + 1) + L"onnxruntime.dll";
    DBG("[ORT] Attempting to load: " << juce::String(dllPath.c_str()));

    DWORD fileAttrib = GetFileAttributesW(dllPath.c_str());
    if (fileAttrib == INVALID_FILE_ATTRIBUTES) {
        DBG("[ORT] onnxruntime.dll not found at path (error " << static_cast<int>(GetLastError()) << ")");
        return;
    }

    g_ortDllHandle = LoadLibraryExW(dllPath.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (g_ortDllHandle != nullptr) {
        DBG("[ORT] Successfully loaded onnxruntime.dll");
    } else {
        DBG("[ORT] LoadLibraryExW failed with error " << static_cast<int>(GetLastError()));
    }
}
#endif  // _WIN32

const OrtApi* OnnxRuntime::getSafeOrtApi() noexcept {
#ifdef _WIN32
    ensureOrtDllLoaded();
#endif
    const OrtApiBase* apiBase = OrtGetApiBase();
    if (apiBase == nullptr) {
        DBG("[ORT] OrtGetApiBase() returned nullptr");
        return nullptr;
    }
    return apiBase->GetApi(ORT_API_VERSION);
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
        const OrtApiBase* apiBase = OrtGetApiBase();
        runtimeVersion_ = (apiBase != nullptr) ? apiBase->GetVersionString() : "unknown";

        OrtEnv* rawEnv = nullptr;
        OrtStatus* status = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "StemgenRT", &rawEnv);
        if (status == nullptr) {
            ortEnv_.reset(rawEnv);
            ortInitialized_ = true;
            DBG("[ORT] Initialized. Version: " << juce::String(runtimeVersion_));
        } else {
            DBG("[ORT] Failed to create OrtEnv: " << api->GetErrorMessage(status));
            api->ReleaseStatus(status);
        }
    } else {
        DBG("[ORT] getSafeOrtApi() returned null");
    }
}

OnnxRuntime::~OnnxRuntime() {
    if (ortMemoryInfo_ != nullptr) {
        const OrtApi* api = getSafeOrtApi();
        if (api != nullptr) {
            api->ReleaseMemoryInfo(ortMemoryInfo_);
        }
        ortMemoryInfo_ = nullptr;
    }
}

bool OnnxRuntime::loadModel(const juce::String& modelPath, juce::String& errorMessage) {
    if (!ortInitialized_ || !ortEnv_) {
        errorMessage = "ORT not initialized";
        return false;
    }

    const OrtApi* api = getSafeOrtApi();
    if (api == nullptr) {
        errorMessage = "ORT API not available";
        return false;
    }

    // Create session options
    OrtSessionOptions* sessionOptions = nullptr;
    OrtStatus* status = api->CreateSessionOptions(&sessionOptions);
    if (status != nullptr) {
        errorMessage = juce::String("Session options error: ") + api->GetErrorMessage(status);
        api->ReleaseStatus(status);
        return false;
    }

    // Set graph optimization level
    status = api->SetSessionGraphOptimizationLevel(sessionOptions, ORT_ENABLE_ALL);
    if (status != nullptr) {
        api->ReleaseStatus(status);
    }

    // Configure threading for real-time audio
    int numHardwareThreads = static_cast<int>(std::thread::hardware_concurrency());
    int numIntraOpThreads = std::min(std::max(numHardwareThreads / 2, 2), 4);

    status = api->SetIntraOpNumThreads(sessionOptions, numIntraOpThreads);
    if (status != nullptr) api->ReleaseStatus(status);

    status = api->SetInterOpNumThreads(sessionOptions, 1);
    if (status != nullptr) api->ReleaseStatus(status);

    DBG("[ORT] Using " << numIntraOpThreads << " intra-op threads (of " << numHardwareThreads << " available)");

    // Try GPU execution providers
    bool gpuEnabled = false;
    std::string gpuProviderName;

    // Priority 1: TensorRT
    {
        OrtTensorRTProviderOptionsV2* trtOptions = nullptr;
        status = api->CreateTensorRTProviderOptions(&trtOptions);
        if (status == nullptr && trtOptions != nullptr) {
            const char* trtKeys[] = {"device_id", "trt_fp16_enable", "trt_builder_optimization_level",
                                     "trt_engine_cache_enable"};
            const char* trtValues[] = {"0", "1", "3", "1"};

            status = api->UpdateTensorRTProviderOptions(trtOptions, trtKeys, trtValues, 4);
            if (status != nullptr) {
                api->ReleaseStatus(status);
                status = nullptr;
            }

            status = api->SessionOptionsAppendExecutionProvider_TensorRT_V2(sessionOptions, trtOptions);
            if (status == nullptr) {
                gpuEnabled = true;
                gpuProviderName = "TensorRT";
                DBG("[ORT] TensorRT execution provider enabled (FP16)");
            } else {
                DBG("[ORT] TensorRT not available: " << api->GetErrorMessage(status) << " - trying CUDA");
                api->ReleaseStatus(status);
                status = nullptr;
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
        status = api->CreateCUDAProviderOptions(&cudaOptions);
        if (status == nullptr && cudaOptions != nullptr) {
            const char* keys[] = {"device_id", "arena_extend_strategy", "cudnn_conv_algo_search",
                                  "cudnn_conv_use_max_workspace", "do_copy_in_default_stream"};
            const char* values[] = {"0", "kSameAsRequested", "EXHAUSTIVE", "1", "1"};

            status = api->UpdateCUDAProviderOptions(cudaOptions, keys, values, 5);
            if (status != nullptr) {
                api->ReleaseStatus(status);
                status = nullptr;
            }

            status = api->SessionOptionsAppendExecutionProvider_CUDA_V2(sessionOptions, cudaOptions);
            if (status == nullptr) {
                gpuEnabled = true;
                gpuProviderName = "CUDA";
                DBG("[ORT] CUDA execution provider enabled");
            } else {
                DBG("[ORT] CUDA not available: " << api->GetErrorMessage(status) << " - falling back to CPU");
                api->ReleaseStatus(status);
                status = nullptr;
            }
            api->ReleaseCUDAProviderOptions(cudaOptions);
        } else {
            if (status != nullptr) {
                api->ReleaseStatus(status);
                status = nullptr;
            }
        }
    }

    if (!gpuEnabled) {
        gpuProviderName = "CPU";
        DBG("[ORT] Using CPU execution provider");
    }

    // Create the inference session
    OrtSession* rawSession = nullptr;
#ifdef _WIN32
    std::wstring wideModelPath(modelPath.toWideCharPointer());
    status = api->CreateSession(ortEnv_.get(), wideModelPath.c_str(), sessionOptions, &rawSession);
#else
    status = api->CreateSession(ortEnv_.get(), modelPath.toRawUTF8(), sessionOptions, &rawSession);
#endif

    api->ReleaseSessionOptions(sessionOptions);

    if (status != nullptr) {
        errorMessage = juce::String("Session creation failed: ") + api->GetErrorMessage(status);
        api->ReleaseStatus(status);
        return false;
    }

    ortSession_.reset(rawSession);
    modelLoaded_ = true;
    modelLoadError_.clear();
    usingGPU_ = gpuEnabled;
    executionProvider_ = gpuProviderName;

    DBG("[ORT] Model loaded successfully from: " << modelPath);
    DBG("[ORT] Execution provider: " << juce::String(executionProvider_));

    return true;
}

void OnnxRuntime::prepareForInference() {
    if (!modelLoaded_) return;

    const OrtApi* api = getSafeOrtApi();
    if (api == nullptr) return;

    // Pre-allocate memory info
    if (ortMemoryInfo_ == nullptr) {
        OrtStatus* status = api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ortMemoryInfo_);
        if (status != nullptr) {
            DBG("[ORT] Failed to create memory info: " << api->GetErrorMessage(status));
            api->ReleaseStatus(status);
        }
    }

    // Pre-allocate scratch buffer
    scratchBuffer_.resize(static_cast<size_t>(kNumChannels * kInternalChunkSize));
}

bool OnnxRuntime::runInference(
    const std::array<std::vector<float>, kNumChannels>& contextSnapshot,
    const std::array<std::vector<float>, kNumChannels>& inputChunk,
    const std::array<std::vector<float>, kNumChannels>& lowFreqChunk,
    float normalizationGain,
    std::array<std::array<std::vector<float>, kNumChannels>, kNumStems>& outputChunks) {

    if (!modelLoaded_ || !ortSession_ || !ortMemoryInfo_) return false;

    const OrtApi* api = getSafeOrtApi();
    if (api == nullptr) return false;

    // Build the internal chunk: [left_context | new_samples | right_padding]
    float* audioInput = scratchBuffer_.data();

    for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
        size_t offset = ch * static_cast<size_t>(kInternalChunkSize);

        // Left context
        for (size_t i = 0; i < static_cast<size_t>(kContextSize); ++i) {
            audioInput[offset + i] = contextSnapshot[ch][i];
        }

        // New samples
        for (size_t i = 0; i < static_cast<size_t>(kOutputChunkSize); ++i) {
            audioInput[offset + static_cast<size_t>(kContextSize) + i] = inputChunk[ch][i];
        }

        // Right context: Reflection padding with smooth blend
        float lastSample = inputChunk[ch][kOutputChunkSize - 1];
        for (size_t i = 0; i < static_cast<size_t>(kContextSize); ++i) {
            size_t distFromEnd = (i + 1) % static_cast<size_t>(kOutputChunkSize);
            if (distFromEnd == 0) distFromEnd = kOutputChunkSize;
            size_t mirrorIdx = static_cast<size_t>(kOutputChunkSize) - distFromEnd;
            float reflected = inputChunk[ch][mirrorIdx];

            float t = static_cast<float>(i) / static_cast<float>(kContextSize);
            float blendFactor = 1.0f - std::exp(-4.0f * t);

            audioInput[offset + static_cast<size_t>(kContextSize + kOutputChunkSize) + i] =
                (1.0f - blendFactor) * reflected + blendFactor * lastSample;
        }
    }

    // Create input tensor
    OrtValue* inputTensor = nullptr;
    std::int64_t audioDims[3] = {1, kNumChannels, kInternalChunkSize};

    OrtStatus* status = api->CreateTensorWithDataAsOrtValue(
        ortMemoryInfo_, audioInput, scratchBuffer_.size() * sizeof(float),
        audioDims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputTensor);

    if (status != nullptr) {
        DBG("[ORT] Failed to create input tensor: " << api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        return false;
    }

    const char* inputNames[1] = {"audio"};
    const char* outputNames[1] = {"separated"};
    OrtValue* outputTensor = nullptr;

    status = api->Run(ortSession_.get(), nullptr, inputNames, &inputTensor, 1, outputNames, 1, &outputTensor);

    if (inputTensor) api->ReleaseValue(inputTensor);

    if (status != nullptr) {
        DBG("[ORT] Inference failed: " << api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        if (outputTensor) api->ReleaseValue(outputTensor);
        return false;
    }

    // Extract output
    float* separatedData = nullptr;
    status = api->GetTensorMutableData(outputTensor, reinterpret_cast<void**>(&separatedData));

    if (status == nullptr && separatedData != nullptr) {
        float invNormGain = 1.0f / normalizationGain;

        for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
            for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
                size_t dataOffset = stem * static_cast<size_t>(kNumChannels * kInternalChunkSize)
                                  + ch * static_cast<size_t>(kInternalChunkSize)
                                  + static_cast<size_t>(kContextSize);

                for (size_t i = 0; i < static_cast<size_t>(kOutputChunkSize); ++i) {
                    float sample = separatedData[dataOffset + i];
                    if (std::isnan(sample) || std::isinf(sample)) {
                        sample = 0.0f;
                    }
                    outputChunks[stem][ch][i] = sample * invNormGain;
                }
            }
        }

        // Add LP to bass stem
        constexpr size_t kBassStemIndex = 1;
        for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
            for (size_t i = 0; i < static_cast<size_t>(kOutputChunkSize); ++i) {
                outputChunks[kBassStemIndex][ch][i] += lowFreqChunk[ch][i];
            }
        }
    }

    if (outputTensor) api->ReleaseValue(outputTensor);
    return true;
}

juce::String OnnxRuntime::getStatusString() const {
    if (modelLoaded_) {
        return juce::String("HS-TasNet loaded (") + juce::String(executionProvider_)
               + ", ORT v" + juce::String(runtimeVersion_) + ")";
    }
    if (ortInitialized_) {
        if (modelLoadError_.isNotEmpty()) {
            return juce::String("Model error: ") + modelLoadError_;
        }
        return juce::String("ORT v") + juce::String(runtimeVersion_) + " ready (model not loaded)";
    }
    return "ONNX Runtime not available";
}

#else  // !STEMGENRT_USE_ONNXRUNTIME

// Stub implementations when ONNX Runtime is disabled
OnnxRuntime::OnnxRuntime() {}
OnnxRuntime::~OnnxRuntime() {}
bool OnnxRuntime::loadModel(const juce::String&, juce::String& errorMessage) {
    errorMessage = "ONNX Runtime support not compiled";
    return false;
}
void OnnxRuntime::prepareForInference() {}
bool OnnxRuntime::runInference(
    const std::array<std::vector<float>, kNumChannels>&,
    const std::array<std::vector<float>, kNumChannels>&,
    const std::array<std::vector<float>, kNumChannels>&,
    float,
    std::array<std::array<std::vector<float>, kNumChannels>, kNumStems>&) {
    return false;
}
juce::String OnnxRuntime::getStatusString() const {
    return "ONNX Runtime not available";
}

void OnnxRuntime::OrtEnvDeleter::operator()(OrtEnv*) const noexcept {}
void OnnxRuntime::OrtSessionDeleter::operator()(OrtSession*) const noexcept {}

#endif  // STEMGENRT_USE_ONNXRUNTIME

}  // namespace audio_plugin
