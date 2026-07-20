#include "StemgenRT/OnnxRuntime.h"
#include <juce_cryptography/juce_cryptography.h>
#include <algorithm>
#include <cmath>
#include <cstring>
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

#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME

namespace {

constexpr char kExpectedCheckpointSha256[] =
    "e966c7e98c9fa05ed6eaebfd3ee56bf82a5b7f7ef502b1a8389a50d9da40901d";
constexpr char kExpectedModelSha256[] =
    "52fdc46d015819821dae19ef272b6bc4ccf441a0274d7d9c8bb44af50eafef8c";
constexpr std::int64_t kExpectedModelSize = 129088022;

bool checkOrtStatus(const OrtApi* api, OrtStatus* status,
                    juce::String& errorMessage, const char* operation) {
    if (status == nullptr) {
        return true;
    }

    const char* detail = api != nullptr ? api->GetErrorMessage(status) : nullptr;
    errorMessage = juce::String(operation) + " failed: "
                   + (detail != nullptr ? juce::String(detail) : juce::String("unknown error"));
    if (api != nullptr) {
        api->ReleaseStatus(status);
    }
    return false;
}

}  // namespace

#ifdef _WIN32
static HMODULE g_ortDllHandle = nullptr;
static bool g_ortDllLoadAttempted = false;

void* OnnxRuntime::ensureOrtDllLoaded() noexcept {
    if (g_ortDllLoadAttempted) return g_ortDllHandle;
    g_ortDllLoadAttempted = true;

    // Get the path to this DLL (the plugin itself)
    HMODULE thisModule = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&ensureOrtDllLoaded), &thisModule)) {
        DBG("[ORT] GetModuleHandleExW failed with error " << static_cast<int>(GetLastError()));
        return nullptr;
    }

    wchar_t modulePath[MAX_PATH];
    DWORD pathLen = GetModuleFileNameW(thisModule, modulePath, MAX_PATH);
    juce::ignoreUnused(pathLen);
    if (pathLen == 0) {
        DBG("[ORT] GetModuleFileNameW failed with error " << static_cast<int>(GetLastError()));
        return nullptr;
    }

    DBG("[ORT] Plugin module path: " << juce::String(modulePath));

    std::wstring dllPath(modulePath);
    size_t lastSlash = dllPath.rfind(L'\\');
    if (lastSlash == std::wstring::npos) {
        DBG("[ORT] Could not find backslash in module path");
        return nullptr;
    }

    dllPath = dllPath.substr(0, lastSlash + 1) + L"onnxruntime.dll";
    DBG("[ORT] Attempting to load: " << juce::String(dllPath.c_str()));

    DWORD fileAttrib = GetFileAttributesW(dllPath.c_str());
    if (fileAttrib == INVALID_FILE_ATTRIBUTES) {
        DBG("[ORT] Bundled onnxruntime.dll not found at path (error "
            << static_cast<int>(GetLastError()) << ")");
        return nullptr;
    }

    g_ortDllHandle = LoadLibraryExW(dllPath.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (g_ortDllHandle != nullptr) {
        DBG("[ORT] Successfully loaded onnxruntime.dll");
    } else {
        DBG("[ORT] LoadLibraryExW failed with error " << static_cast<int>(GetLastError()));
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

    const std::array<const char*, 4> expectedInputNames = {
        "audio_chunk", "past_audio", "overlap_add_buffer", "fusion_hidden"};
    const std::array<std::vector<std::int64_t>, 4> expectedInputShapes = {{
        {1, kNumChannels, kOutputChunkSize},
        {1, kNumChannels, kOutputChunkSize},
        {1, kNumStems, kNumChannels, kAnalysisWindowSize},
        {kFusionHiddenLayers, 1, kFusionHiddenSize},
    }};
    const std::array<const char*, 4> expectedOutputNames = {
        "separated_chunk", "next_past_audio", "next_overlap_add_buffer",
        "next_fusion_hidden"};
    const std::array<std::vector<std::int64_t>, 4> expectedOutputShapes = {{
        {1, kNumStems, kNumChannels, kOutputChunkSize},
        {1, kNumChannels, kOutputChunkSize},
        {1, kNumStems, kNumChannels, kAnalysisWindowSize},
        {kFusionHiddenLayers, 1, kFusionHiddenSize},
    }};

    size_t inputCount = 0;
    size_t outputCount = 0;
    if (!checkOrtStatus(api, api->SessionGetInputCount(ortSession_.get(), &inputCount),
                        errorMessage, "SessionGetInputCount") ||
        !checkOrtStatus(api, api->SessionGetOutputCount(ortSession_.get(), &outputCount),
                        errorMessage, "SessionGetOutputCount")) {
        return false;
    }
    if (inputCount != expectedInputNames.size() ||
        outputCount != expectedOutputNames.size()) {
        errorMessage = juce::String("Unexpected model I/O count: inputs=")
                       + juce::String(static_cast<int>(inputCount)) + " outputs="
                       + juce::String(static_cast<int>(outputCount));
        return false;
    }

    const auto validateTensor = [&](bool isInput, size_t index,
                                    const char* expectedName,
                                    const std::vector<std::int64_t>& expectedShape) {
        char* rawName = nullptr;
        OrtStatus* nameStatus = isInput
            ? api->SessionGetInputName(ortSession_.get(), index, allocator, &rawName)
            : api->SessionGetOutputName(ortSession_.get(), index, allocator, &rawName);
        if (!checkOrtStatus(api, nameStatus, errorMessage,
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
                                                : "Unexpected output name: ")
                           + juce::String(actualName) + " (expected "
                           + juce::String(expectedName) + ")";
            return false;
        }

        OrtTypeInfo* typeInfo = nullptr;
        OrtStatus* typeStatus = isInput
            ? api->SessionGetInputTypeInfo(ortSession_.get(), index, &typeInfo)
            : api->SessionGetOutputTypeInfo(ortSession_.get(), index, &typeInfo);
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
        bool ok = checkOrtStatus(api,
                                 api->CastTypeInfoToTensorInfo(typeInfo, &tensorInfo),
                                 errorMessage, "CastTypeInfoToTensorInfo");
        if (!ok || tensorInfo == nullptr) {
            api->ReleaseTypeInfo(typeInfo);
            if (ok) {
                errorMessage = juce::String(expectedName) + " is not a tensor";
            }
            return false;
        }

        ONNXTensorElementDataType elementType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        size_t rank = 0;
        ok = checkOrtStatus(api, api->GetTensorElementType(tensorInfo, &elementType),
                            errorMessage, "GetTensorElementType") &&
             checkOrtStatus(api, api->GetDimensionsCount(tensorInfo, &rank),
                            errorMessage, "GetDimensionsCount");
        std::vector<std::int64_t> actualShape(rank);
        if (ok && rank > 0) {
            ok = checkOrtStatus(api,
                                api->GetDimensions(tensorInfo, actualShape.data(), rank),
                                errorMessage, "GetDimensions");
        }
        api->ReleaseTypeInfo(typeInfo);

        if (!ok) {
            return false;
        }
        if (elementType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            actualShape != expectedShape) {
            errorMessage = juce::String("Unexpected type or shape for ") + expectedName;
            return false;
        }
        return true;
    };

    for (size_t index = 0; index < expectedInputNames.size(); ++index) {
        if (!validateTensor(true, index, expectedInputNames[index],
                            expectedInputShapes[index])) {
            return false;
        }
    }
    for (size_t index = 0; index < expectedOutputNames.size(); ++index) {
        if (!validateTensor(false, index, expectedOutputNames[index],
                            expectedOutputShapes[index])) {
            return false;
        }
    }

    OrtModelMetadata* metadata = nullptr;
    if (!checkOrtStatus(api,
                        api->SessionGetModelMetadata(ortSession_.get(), &metadata),
                        errorMessage, "SessionGetModelMetadata") ||
        metadata == nullptr) {
        if (metadata != nullptr) {
            api->ReleaseModelMetadata(metadata);
        }
        return false;
    }

    const std::array<std::pair<const char*, const char*>, 7> expectedMetadata = {{
        {"hs_tasnet.checkpoint_sha256", kExpectedCheckpointSha256},
        {"hs_tasnet.export.mode", "streaming"},
        {"hs_tasnet.export.mixture_consistency", "route_residual"},
        {"hs_tasnet.export.residual_source_index", "3"},
        {"hs_tasnet.external_data", "false"},
        {"hs_tasnet.streaming.chunk_samples", "512"},
        {"hs_tasnet.streaming.output_alignment", "previous_input_chunk"},
    }};

    for (const auto& [key, expectedValue] : expectedMetadata) {
        char* rawValue = nullptr;
        const bool lookupOk = checkOrtStatus(
            api,
            api->ModelMetadataLookupCustomMetadataMap(metadata, allocator, key,
                                                       &rawValue),
            errorMessage, "ModelMetadataLookupCustomMetadataMap");
        if (!lookupOk) {
            api->ReleaseModelMetadata(metadata);
            return false;
        }
        const std::string actualValue = rawValue != nullptr ? rawValue : "";
        if (rawValue != nullptr) {
            OrtStatus* freeStatus = api->AllocatorFree(allocator, rawValue);
            if (!checkOrtStatus(api, freeStatus, errorMessage, "AllocatorFree")) {
                api->ReleaseModelMetadata(metadata);
                return false;
            }
        }
        if (actualValue != expectedValue) {
            errorMessage = juce::String("Unexpected model metadata ")
                           + juce::String(key) + "="
                           + juce::String(actualValue) + " (expected "
                           + juce::String(expectedValue) + ")";
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

    // Verify the deployed bytes before ORT parses the graph. CMake performs the
    // same check at packaging time; this closes the gap if a bundle is damaged
    // or its Resources payload is replaced after the build.
    const juce::File modelFile(modelPath);
    if (!modelFile.existsAsFile()) {
        errorMessage = juce::String("Model file not found: ") + modelPath;
        modelLoadError_ = errorMessage;
        return false;
    }
    if (modelFile.getSize() != kExpectedModelSize) {
        errorMessage = juce::String("Unexpected model size: expected ")
                       + juce::String(kExpectedModelSize) + " bytes, got "
                       + juce::String(modelFile.getSize());
        modelLoadError_ = errorMessage;
        return false;
    }
    const juce::String actualModelSha = juce::SHA256(modelFile).toHexString();
    if (!actualModelSha.equalsIgnoreCase(kExpectedModelSha256)) {
        errorMessage = juce::String("Unexpected model SHA-256: ")
                       + actualModelSha;
        modelLoadError_ = errorMessage;
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

    // Avoid busy-spinning ORT worker threads between audio hops. Inference runs
    // on the plugin's dedicated worker and must leave the host audio thread and
    // other plugins schedulable.
    status = api->AddSessionConfigEntry(
        sessionOptions, "session.intra_op.allow_spinning", "0");
    if (status != nullptr) api->ReleaseStatus(status);
    status = api->AddSessionConfigEntry(
        sessionOptions, "session.inter_op.allow_spinning", "0");
    if (status != nullptr) api->ReleaseStatus(status);

    DBG("[ORT] Using " << numIntraOpThreads << " intra-op threads (of " << numHardwareThreads << " available)");

    // The shipping plugin is deliberately CPU-only. This is the universally
    // available path, matches the deployment target, and avoids silently
    // changing numerical/runtime behavior with host-specific accelerators.
    DBG("[ORT] Using CPU execution provider");

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
    if (!validateModelContract(errorMessage)) {
        modelLoadError_ = errorMessage;
        modelLoaded_ = false;
        ortSession_.reset();
        DBG("[ORT] Model contract validation failed: " << errorMessage);
        return false;
    }

    modelLoaded_ = true;
    modelLoadError_.clear();
    executionProvider_ = "CPU";

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

    audioChunkBuffer_.resize(
        static_cast<size_t>(kNumChannels * kOutputChunkSize));
    pastAudio_.resize(static_cast<size_t>(kNumChannels * kOutputChunkSize));
    overlapAddBuffer_.resize(static_cast<size_t>(
        kNumStems * kNumChannels * kAnalysisWindowSize));
    fusionHidden_.resize(
        static_cast<size_t>(kFusionHiddenLayers * kFusionHiddenSize));
    resetStreamingState();
}

void OnnxRuntime::resetStreamingStateUnlocked() {
    std::fill(pastAudio_.begin(), pastAudio_.end(), 0.0f);
    std::fill(overlapAddBuffer_.begin(), overlapAddBuffer_.end(), 0.0f);
    std::fill(fusionHidden_.begin(), fusionHidden_.end(), 0.0f);
    hasPastAudio_ = false;
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

    if (!modelLoaded_ || !ortSession_ || !ortMemoryInfo_) return false;

    const OrtApi* api = getSafeOrtApi();
    if (api == nullptr) return false;

    std::lock_guard<std::mutex> lock(streamingStateMutex_);
    outputValid = false;

    const size_t audioElements =
        static_cast<size_t>(kNumChannels * kOutputChunkSize);
    const size_t separatedElements =
        static_cast<size_t>(kNumStems * kNumChannels * kOutputChunkSize);
    const size_t overlapElements = static_cast<size_t>(
        kNumStems * kNumChannels * kAnalysisWindowSize);
    const size_t hiddenElements =
        static_cast<size_t>(kFusionHiddenLayers * kFusionHiddenSize);

    if (audioChunkBuffer_.size() != audioElements ||
        pastAudio_.size() != audioElements ||
        overlapAddBuffer_.size() != overlapElements ||
        fusionHidden_.size() != hiddenElements) {
        DBG("[ORT] Streaming buffers were not prepared");
        return false;
    }

    for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
        if (inputChunk[ch].size() != static_cast<size_t>(kOutputChunkSize) ||
            alignedInput[ch].size() != static_cast<size_t>(kOutputChunkSize)) {
            DBG("[ORT] Unexpected streaming chunk size");
            return false;
        }
        std::memcpy(audioChunkBuffer_.data() + ch * kOutputChunkSize,
                    inputChunk[ch].data(),
                    static_cast<size_t>(kOutputChunkSize) * sizeof(float));
        if (hasPastAudio_) {
            std::memcpy(alignedInput[ch].data(),
                        pastAudio_.data() + ch * kOutputChunkSize,
                        static_cast<size_t>(kOutputChunkSize) * sizeof(float));
        } else {
            std::fill(alignedInput[ch].begin(), alignedInput[ch].end(), 0.0f);
        }
    }

    const std::int64_t audioDims[3] = {1, kNumChannels, kOutputChunkSize};
    const std::int64_t overlapDims[4] = {
        1, kNumStems, kNumChannels, kAnalysisWindowSize};
    const std::int64_t hiddenDims[3] = {
        kFusionHiddenLayers, 1, kFusionHiddenSize};

    std::array<OrtValue*, 4> inputValues = {nullptr, nullptr, nullptr, nullptr};
    const auto releaseValues = [&](auto& values) {
        for (OrtValue*& value : values) {
            if (value != nullptr) {
                api->ReleaseValue(value);
                value = nullptr;
            }
        }
    };
    const auto createInput = [&](size_t index, std::vector<float>& data,
                                 const std::int64_t* dims, size_t rank) {
        OrtStatus* status = api->CreateTensorWithDataAsOrtValue(
            ortMemoryInfo_, data.data(), data.size() * sizeof(float), dims, rank,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputValues[index]);
        if (status == nullptr) {
            return true;
        }
        DBG("[ORT] Failed to create streaming input " << static_cast<int>(index)
            << ": " << api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        return false;
    };

    if (!createInput(0, audioChunkBuffer_, audioDims, 3) ||
        !createInput(1, pastAudio_, audioDims, 3) ||
        !createInput(2, overlapAddBuffer_, overlapDims, 4) ||
        !createInput(3, fusionHidden_, hiddenDims, 3)) {
        releaseValues(inputValues);
        resetStreamingStateUnlocked();
        return false;
    }

    const char* inputNames[4] = {
        "audio_chunk", "past_audio", "overlap_add_buffer", "fusion_hidden"};
    const char* outputNames[4] = {
        "separated_chunk", "next_past_audio", "next_overlap_add_buffer",
        "next_fusion_hidden"};
    const OrtValue* constInputValues[4] = {
        inputValues[0], inputValues[1], inputValues[2], inputValues[3]};
    std::array<OrtValue*, 4> outputValues = {nullptr, nullptr, nullptr, nullptr};

    OrtStatus* runStatus = api->Run(
        ortSession_.get(), nullptr, inputNames, constInputValues, 4,
        outputNames, 4, outputValues.data());
    releaseValues(inputValues);

    if (runStatus != nullptr) {
        DBG("[ORT] Inference failed: " << api->GetErrorMessage(runStatus));
        api->ReleaseStatus(runStatus);
        releaseValues(outputValues);
        resetStreamingStateUnlocked();
        return false;
    }

    std::array<float*, 4> outputData = {nullptr, nullptr, nullptr, nullptr};
    for (size_t index = 0; index < outputValues.size(); ++index) {
        OrtStatus* dataStatus = api->GetTensorMutableData(
            outputValues[index], reinterpret_cast<void**>(&outputData[index]));
        if (dataStatus != nullptr || outputData[index] == nullptr) {
            if (dataStatus != nullptr) {
                DBG("[ORT] Failed to access streaming output "
                    << static_cast<int>(index) << ": "
                    << api->GetErrorMessage(dataStatus));
                api->ReleaseStatus(dataStatus);
            }
            releaseValues(outputValues);
            resetStreamingStateUnlocked();
            return false;
        }
    }

    const auto allFinite = [](const float* data, size_t count) {
        return std::all_of(data, data + count,
                           [](float value) { return std::isfinite(value); });
    };
    if (!allFinite(outputData[0], separatedElements) ||
        !allFinite(outputData[1], audioElements) ||
        !allFinite(outputData[2], overlapElements) ||
        !allFinite(outputData[3], hiddenElements)) {
        DBG("[ORT] Non-finite streaming output; resetting recurrent state");
        releaseValues(outputValues);
        resetStreamingStateUnlocked();
        return false;
    }

    outputValid = hasPastAudio_;
    for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
        for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
            auto& destination = outputChunks[stem][ch];
            if (destination.size() != static_cast<size_t>(kOutputChunkSize)) {
                releaseValues(outputValues);
                resetStreamingStateUnlocked();
                outputValid = false;
                return false;
            }
            const size_t offset =
                (stem * static_cast<size_t>(kNumChannels) + ch)
                * static_cast<size_t>(kOutputChunkSize);
            if (outputValid) {
                std::memcpy(destination.data(), outputData[0] + offset,
                            static_cast<size_t>(kOutputChunkSize) * sizeof(float));
            } else {
                std::fill(destination.begin(), destination.end(), 0.0f);
            }
        }
    }

    // Enforce the deployment invariant again after provider-specific numerical
    // differences: keep drums/bass/vocals and route the final residual to Other.
    if (outputValid) {
        for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
            for (size_t i = 0; i < static_cast<size_t>(kOutputChunkSize); ++i) {
                outputChunks[kStemOther][ch][i] =
                    alignedInput[ch][i]
                    - outputChunks[kStemDrums][ch][i]
                    - outputChunks[kStemBass][ch][i]
                    - outputChunks[kStemVocals][ch][i];
            }
        }
    }

    std::memcpy(pastAudio_.data(), outputData[1], audioElements * sizeof(float));
    std::memcpy(overlapAddBuffer_.data(), outputData[2],
                overlapElements * sizeof(float));
    std::memcpy(fusionHidden_.data(), outputData[3], hiddenElements * sizeof(float));
    hasPastAudio_ = true;

    releaseValues(outputValues);
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
void OnnxRuntime::resetStreamingState() {}
bool OnnxRuntime::runInference(
    const std::array<std::vector<float>, kNumChannels>&,
    std::array<std::array<std::vector<float>, kNumChannels>, kNumStems>&,
    std::array<std::vector<float>, kNumChannels>&,
    bool&) {
    return false;
}
juce::String OnnxRuntime::getStatusString() const {
    return "ONNX Runtime not available";
}

void OnnxRuntime::OrtEnvDeleter::operator()(OrtEnv*) const noexcept {}
void OnnxRuntime::OrtSessionDeleter::operator()(OrtSession*) const noexcept {}

#endif  // STEMGENRT_USE_ONNXRUNTIME

}  // namespace audio_plugin
