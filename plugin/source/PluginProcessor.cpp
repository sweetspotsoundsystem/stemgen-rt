#include "StemgenRT/PluginProcessor.h"
#include "StemgenRT/PluginEditor.h"
#include <algorithm>
#include <vector>
#include <cstdint>
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
// On Windows, VST3 plugins need to explicitly load onnxruntime.dll from the plugin's
// directory because Windows' DLL search path doesn't include the plugin bundle location.
// The DAW's working directory is typically its own install folder, not the plugin's.
static HMODULE g_ortDllHandle = nullptr;
static bool g_ortDllLoadAttempted = false;

// Attempts to load onnxruntime.dll from the plugin's bundle directory.
// Must be called before any OrtGetApiBase() calls.
static void ensureOrtDllLoaded() noexcept {
  if (g_ortDllLoadAttempted) return;  // Only try once
  g_ortDllLoadAttempted = true;
  
  // First check if onnxruntime.dll is already loaded (e.g., by another plugin or the host)
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
  
  // modulePath is now something like:
  //   C:\...\StemgenRT.vst3\Contents\x86_64-win\StemgenRT.vst3
  // We need to find onnxruntime.dll in the same directory
  
  std::wstring dllPath(modulePath);
  size_t lastSlash = dllPath.rfind(L'\\');
  if (lastSlash == std::wstring::npos) {
    DBG("[ORT] Could not find backslash in module path");
    return;
  }
  
  dllPath = dllPath.substr(0, lastSlash + 1) + L"onnxruntime.dll";
  DBG("[ORT] Attempting to load: " << juce::String(dllPath.c_str()));
  
  // Check if the file exists
  DWORD fileAttrib = GetFileAttributesW(dllPath.c_str());
  if (fileAttrib == INVALID_FILE_ATTRIBUTES) {
    DBG("[ORT] onnxruntime.dll not found at path (error " << static_cast<int>(GetLastError()) << ")");
    return;
  }
  
  // Use LOAD_WITH_ALTERED_SEARCH_PATH so the DLL can find its own dependencies
  g_ortDllHandle = LoadLibraryExW(dllPath.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  
  if (g_ortDllHandle != nullptr) {
    DBG("[ORT] Successfully loaded onnxruntime.dll");
  } else {
    DBG("[ORT] LoadLibraryExW failed with error " << static_cast<int>(GetLastError()));
  }
}
#endif  // _WIN32

// Helper to safely get ORT API, returns nullptr if ORT is not available
static const OrtApi* getSafeOrtApi() noexcept {
#ifdef _WIN32
  // Ensure the DLL is loaded from the plugin's directory before calling ORT functions
  ensureOrtDllLoaded();
#endif
  const OrtApiBase* apiBase = OrtGetApiBase();
  if (apiBase == nullptr) {
    DBG("[ORT] OrtGetApiBase() returned nullptr");
    return nullptr;
  }
  return apiBase->GetApi(ORT_API_VERSION);
}

// Define deleters for OrtEnv and OrtSession unique_ptr
void AudioPluginAudioProcessor::OrtEnvDeleter::operator()(void* p) const noexcept {
  if (p == nullptr) return;
  const OrtApi* api = getSafeOrtApi();
  if (api == nullptr) return;  // ORT not available, leak rather than crash
  api->ReleaseEnv(reinterpret_cast<OrtEnv*>(p));
}

void AudioPluginAudioProcessor::OrtSessionDeleter::operator()(void* p) const noexcept {
  if (p == nullptr) return;
  const OrtApi* api = getSafeOrtApi();
  if (api == nullptr) return;  // ORT not available, leak rather than crash
  api->ReleaseSession(reinterpret_cast<OrtSession*>(p));
}
#endif

AudioPluginAudioProcessor::AudioPluginAudioProcessor()
    : AudioProcessor(
          BusesProperties()
#if !JucePlugin_IsMidiEffect
#if !JucePlugin_IsSynth
              .withInput("Input", juce::AudioChannelSet::stereo(), true)
#endif
              .withOutput("Main", juce::AudioChannelSet::stereo(), true)
              .withOutput("Drums", juce::AudioChannelSet::stereo(), true)
              .withOutput("Bass", juce::AudioChannelSet::stereo(), true)
              .withOutput("Other", juce::AudioChannelSet::stereo(), true)
              .withOutput("Vocals", juce::AudioChannelSet::stereo(), true)
#endif
      ) {
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  // Early initialization of ONNX Runtime environment (for accurate status display).
  // The full model loading and session creation happens in prepareToPlay().
  const OrtApi* api = getSafeOrtApi();
  if (api != nullptr) {
    const OrtApiBase* apiBase = OrtGetApiBase();
    ortRuntimeVersion = (apiBase != nullptr) ? apiBase->GetVersionString() : "unknown";
    
    if (!ortEnv) {
      OrtEnv* rawEnv = nullptr;
      OrtStatus* status = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "StemgenRT", &rawEnv);
      if (status == nullptr) {
        ortEnv.reset(static_cast<void*>(rawEnv));
        ortInitialized = true;
        DBG("[ORT] Early init OK. Version: " << juce::String(ortRuntimeVersion));
      } else {
        DBG("[ORT] Early init failed: " << api->GetErrorMessage(status));
        api->ReleaseStatus(status);
      }
    }
  } else {
    DBG("[ORT] Early init: getSafeOrtApi() returned null");
  }
#endif
}

AudioPluginAudioProcessor::~AudioPluginAudioProcessor() {
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  stopInferenceThread();
  
  // Release pre-allocated ORT memory info
  if (ortMemoryInfo != nullptr) {
    const OrtApi* api = getSafeOrtApi();
    if (api != nullptr) {
      api->ReleaseMemoryInfo(reinterpret_cast<OrtMemoryInfo*>(ortMemoryInfo));
    }
    ortMemoryInfo = nullptr;
  }
#endif
}

const juce::String AudioPluginAudioProcessor::getName() const {
  return JucePlugin_Name;
}

bool AudioPluginAudioProcessor::acceptsMidi() const {
#if JucePlugin_WantsMidiInput
  return true;
#else
  return false;
#endif
}

bool AudioPluginAudioProcessor::producesMidi() const {
#if JucePlugin_ProducesMidiOutput
  return true;
#else
  return false;
#endif
}

bool AudioPluginAudioProcessor::isMidiEffect() const {
#if JucePlugin_IsMidiEffect
  return true;
#else
  return false;
#endif
}

double AudioPluginAudioProcessor::getTailLengthSeconds() const {
  return 0.0;
}

int AudioPluginAudioProcessor::getNumPrograms() {
  return 1;  // NB: some hosts don't cope very well if you tell them there are 0
             // programs, so this should be at least 1, even if you're not
             // really implementing programs.
}

int AudioPluginAudioProcessor::getCurrentProgram() {
  return 0;
}

void AudioPluginAudioProcessor::setCurrentProgram(int index) {
  juce::ignoreUnused(index);
}

const juce::String AudioPluginAudioProcessor::getProgramName(int index) {
  juce::ignoreUnused(index);
  return {};
}

void AudioPluginAudioProcessor::changeProgramName(int index,
                                                  const juce::String& newName) {
  juce::ignoreUnused(index, newName);
}

juce::String AudioPluginAudioProcessor::getOrtStatusString() const {
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  if (modelLoaded) {
    // Show execution provider: TensorRT, CUDA, or CPU
    return juce::String("HS-TasNet loaded (") + juce::String(ortExecutionProvider) + ", ORT v" + juce::String(ortRuntimeVersion) + ")";
  }
  if (ortEnv && ortInitialized) {
    if (modelLoadError.isNotEmpty()) {
      return juce::String("Model error: ") + modelLoadError;
    }
    return juce::String("ONNX Runtime: OK (v") + juce::String(ortRuntimeVersion) + ") - model not loaded";
  }
  return "ONNX Runtime: not initialized";
#else
  return "ONNX Runtime: not linked";
#endif
}

int AudioPluginAudioProcessor::getLatencySamples() const {
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  if (!modelLoaded) {
    return 0;
  }
  
  // The fixed algorithmic latency is kOutputChunkSize samples.
  // This is the constant delay between input and output in steady state:
  // - We accumulate kOutputChunkSize input samples before processing
  // - Inference runs in background (pipelined, doesn't add to latency in steady state)
  // - Output is read from ring buffer as it becomes available
  //
  // This fixed value is what hosts use for Plugin Delay Compensation (PDC).
  return kOutputChunkSize;
#else
  return 0;
#endif
}

double AudioPluginAudioProcessor::getLatencyMs() const {
  double sampleRate = getSampleRate();
  if (sampleRate <= 0.0) {
    sampleRate = 44100.0;  // Fallback if not yet initialized
  }
  return (static_cast<double>(getLatencySamples()) / sampleRate) * 1000.0;
}

#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
void AudioPluginAudioProcessor::allocateStreamingBuffers() {
  // Allocate context buffer (holds kContextSize samples of history)
  for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
    contextBuffer[ch].resize(static_cast<size_t>(kContextSize), 0.0f);
    inputAccumBuffer[ch].resize(static_cast<size_t>(kOutputChunkSize), 0.0f);
  }
  inputAccumCount = 0;

  // Allocate output ring buffers (larger to handle latency)
  const size_t ringBufferSize = static_cast<size_t>(kOutputChunkSize * 16);
  for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
    for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
      outputRingBuffers[stem][ch].resize(ringBufferSize, 0.0f);
    }
  }
  outputReadPos = 0;
  outputSamplesAvailable = 0;
  hasPendingChunk = false;
  pendingChunkCopyOffset = 0;

  // Allocate delayed input buffer (same size as output ring buffer)
  for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
    delayedInputBuffer[ch].resize(ringBufferSize, 0.0f);
  }
  delayedInputWritePos = 0;

  // Allocate dry delay line for underrun fallback
  // Size must be at least kOutputChunkSize to provide latency-aligned dry signal
  const size_t dryDelaySize = static_cast<size_t>(kOutputChunkSize * 2);  // Extra room for safety
  for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
    dryDelayLine[ch].resize(dryDelaySize, 0.0f);
  }
  // Initialize positions so readPos lags behind writePos by kOutputChunkSize
  // This provides latency-aligned dry signal from the start
  // writePos starts ahead by kOutputChunkSize, readPos at 0
  dryDelayWritePos = static_cast<size_t>(kOutputChunkSize);
  dryDelayReadPos = 0;
  crossfadeGain = 1.0f;  // Start assuming separated audio is available
  wasInUnderrun = false;

  // Allocate inference queue buffers
  for (size_t i = 0; i < kNumInferenceBuffers; ++i) {
    inferenceQueue[i] = std::make_unique<InferenceRequest>();
    for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
      inferenceQueue[i]->inputChunk[ch].resize(static_cast<size_t>(kOutputChunkSize), 0.0f);
      inferenceQueue[i]->contextSnapshot[ch].resize(static_cast<size_t>(kContextSize), 0.0f);
      inferenceQueue[i]->originalInput[ch].resize(static_cast<size_t>(kOutputChunkSize), 0.0f);
      inferenceQueue[i]->lowFreqChunk[ch].resize(static_cast<size_t>(kOutputChunkSize), 0.0f);
    }
    for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
      for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
        inferenceQueue[i]->outputChunk[stem][ch].resize(static_cast<size_t>(kOutputChunkSize), 0.0f);
      }
    }
    inferenceQueue[i]->ready.store(false, std::memory_order_release);
    inferenceQueue[i]->processed.store(false, std::memory_order_release);
  }
  inferenceWriteIdx.store(0, std::memory_order_release);
  inferenceReadIdx.store(0, std::memory_order_release);
  outputConsumeIdx.store(0, std::memory_order_release);

  // Pre-allocate inference scratch buffer (avoids per-inference heap allocation)
  inferenceScratchBuffer.resize(static_cast<size_t>(kNumChannels * kInternalChunkSize), 0.0f);

  // Allocate low-frequency accumulation buffer for LR4 crossover
  for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
    lowFreqAccumBuffer[ch].resize(static_cast<size_t>(kOutputChunkSize), 0.0f);
  }
  
  DBG("[HS-TasNet] LR4 crossover buffers allocated");

  // Pre-allocate ORT memory info (avoids per-inference allocation)
  if (ortMemoryInfo == nullptr) {
    const OrtApi* api = getSafeOrtApi();
    if (api != nullptr) {
      OrtMemoryInfo* memInfo = nullptr;
      OrtStatus* status = api->CreateCpuMemoryInfo(OrtAllocatorType::OrtArenaAllocator,
                                                   OrtMemTypeDefault, &memInfo);
      if (status == nullptr) {
        ortMemoryInfo = memInfo;
      } else {
        api->ReleaseStatus(status);
      }
    }
  }

  DBG("[HS-TasNet] Streaming buffers allocated:");
  DBG("  Context size: " << kContextSize << " samples");
  DBG("  Output chunk size: " << kOutputChunkSize << " samples");
  DBG("  Internal chunk size: " << kInternalChunkSize << " samples");
  DBG("  Inference queue size: " << kNumInferenceBuffers << " slots");
}

void AudioPluginAudioProcessor::startInferenceThread() {
  if (inferenceThread && inferenceThread->joinable()) {
    return;  // Already running
  }
  
  shouldStopInference.store(false, std::memory_order_release);
  inferenceThread = std::make_unique<std::thread>([this]() {
    // Set high thread priority
#if JUCE_MAC || JUCE_IOS
    pthread_t thisThread = pthread_self();
    struct sched_param param;
    param.sched_priority = 45;  // macOS priority range is 15-47 for SCHED_RR
    pthread_setschedparam(thisThread, SCHED_RR, &param);
#elif JUCE_WINDOWS
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#elif JUCE_LINUX
    pthread_t thisThread = pthread_self();
    struct sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_RR) / 2;
    pthread_setschedparam(thisThread, SCHED_RR, &param);
#endif
    this->inferenceThreadFunc();
  });
  DBG("[HS-TasNet] Inference thread started");
}

void AudioPluginAudioProcessor::stopInferenceThread() {
  if (!inferenceThread || !inferenceThread->joinable()) {
    return;
  }
  
  shouldStopInference.store(true);
  inferenceCV.notify_all();
  inferenceThread->join();
  inferenceThread.reset();
  DBG("[HS-TasNet] Inference thread stopped");
}

void AudioPluginAudioProcessor::inferenceThreadFunc() {
  DBG("[HS-TasNet] Inference thread running");
  
  // Track the epoch we last synchronized to (for detecting resets)
  uint32_t lastSeenEpoch = resetEpoch.load(std::memory_order_acquire);
  
  while (!shouldStopInference.load(std::memory_order_acquire)) {
    // Check for epoch change (reset occurred)
    uint32_t currentEpoch = resetEpoch.load(std::memory_order_acquire);
    if (currentEpoch != lastSeenEpoch) {
      // Reset occurred - reinitialize our read index to 0
      // This is safe because inference thread owns inferenceReadIdx
      inferenceReadIdx.store(0, std::memory_order_release);
      lastSeenEpoch = currentEpoch;
    }
    
    // Wait for work (polling with timeout - no notify_one needed from audio thread)
    {
      std::unique_lock<std::mutex> lock(inferenceMutex);
      inferenceCV.wait_for(lock, std::chrono::milliseconds(5), [this] {
        size_t readIdx = inferenceReadIdx.load(std::memory_order_acquire);
        return shouldStopInference.load(std::memory_order_acquire) || 
               (inferenceQueue[readIdx] && inferenceQueue[readIdx]->ready.load(std::memory_order_acquire));
      });
    }
    
    if (shouldStopInference.load(std::memory_order_acquire)) break;
    
    // Process all ready requests
    while (true) {
      size_t readIdx = inferenceReadIdx.load(std::memory_order_acquire);
      auto& request = inferenceQueue[readIdx];
      
      if (!request || !request->ready.load(std::memory_order_acquire)) {
        break;  // No more work
      }
      
      // Capture epoch before inference
      uint32_t requestEpoch = request->epoch;
      
      // Check if this request is stale (from before a reset) - skip it without running inference
      currentEpoch = resetEpoch.load(std::memory_order_acquire);
      if (requestEpoch != currentEpoch) {
        // Stale request - discard without running inference
        request->ready.store(false, std::memory_order_release);
        inferenceReadIdx.store((readIdx + 1) % kNumInferenceBuffers, std::memory_order_release);
        lastSeenEpoch = currentEpoch;
        continue;  // Check next slot
      }
      
      // Run inference on this request
      runOverlapAddInference(*request);
      
      // Check if a reset occurred during inference - if so, discard this result
      currentEpoch = resetEpoch.load(std::memory_order_acquire);
      if (requestEpoch != currentEpoch) {
        // Epoch changed during inference - discard result, don't publish
        request->ready.store(false, std::memory_order_release);
        inferenceReadIdx.store((readIdx + 1) % kNumInferenceBuffers, std::memory_order_release);
        lastSeenEpoch = currentEpoch;
        continue;
      }
      
      // Mark as processed (release ensures output data is visible before flag)
      request->ready.store(false, std::memory_order_release);
      request->processed.store(true, std::memory_order_release);
      
      // Advance read index (inference thread owns this index)
      inferenceReadIdx.store((readIdx + 1) % kNumInferenceBuffers, std::memory_order_release);
    }
  }
  
  DBG("[HS-TasNet] Inference thread exiting");
}
#endif

void AudioPluginAudioProcessor::resetStreamingBuffers() {
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  // Clear context and input accumulation buffers
  for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
    std::fill(contextBuffer[ch].begin(), contextBuffer[ch].end(), 0.0f);
    std::fill(inputAccumBuffer[ch].begin(), inputAccumBuffer[ch].end(), 0.0f);
  }
  inputAccumCount = 0;

  // Clear output ring buffers
  for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
    for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
      std::fill(outputRingBuffers[stem][ch].begin(), outputRingBuffers[stem][ch].end(), 0.0f);
    }
  }
  outputReadPos = 0;
  outputSamplesAvailable = 0;
  hasPendingChunk = false;
  pendingChunkCopyOffset = 0;

  // Clear delayed input buffer
  for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
    std::fill(delayedInputBuffer[ch].begin(), delayedInputBuffer[ch].end(), 0.0f);
  }
  delayedInputWritePos = 0;

  // Clear dry delay line for underrun fallback
  for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
    std::fill(dryDelayLine[ch].begin(), dryDelayLine[ch].end(), 0.0f);
  }
  // Initialize positions so readPos lags behind writePos by kOutputChunkSize
  dryDelayWritePos = static_cast<size_t>(kOutputChunkSize);
  dryDelayReadPos = 0;
  crossfadeGain = 1.0f;
  wasInUnderrun = false;

  // Reset inference queue state
  for (size_t i = 0; i < kNumInferenceBuffers; ++i) {
    if (inferenceQueue[i]) {
      inferenceQueue[i]->ready.store(false, std::memory_order_release);
      inferenceQueue[i]->processed.store(false, std::memory_order_release);
    }
  }
  inferenceWriteIdx.store(0, std::memory_order_release);
  inferenceReadIdx.store(0, std::memory_order_release);
  outputConsumeIdx.store(0, std::memory_order_release);

  DBG("[HS-TasNet] Streaming buffers reset");
#endif
}

void AudioPluginAudioProcessor::resetStreamingBuffersRT() {
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  // RT-safe reset for transport start/stop.
  // 
  // Design principles:
  //   1. Audio thread only writes to indices it owns (inferenceWriteIdx, outputConsumeIdx)
  //   2. Inference thread owns inferenceReadIdx and will reset it when it sees epoch change
  //   3. We clear contextBuffer here (small: 2ch × 2048 floats = 16KB) to avoid stale context
  //   4. inputAccumBuffer is NOT cleared - audio thread overwrites from index 0 anyway
  //   5. Epoch increment invalidates all in-flight inference results
  //
  // This avoids data races: inference thread never touches audio thread's buffers.
  
  // Reset audio thread's indices and pending state
  inputAccumCount = 0;
  outputReadPos = 0;
  outputSamplesAvailable = 0;
  delayedInputWritePos = 0;
  hasPendingChunk = false;
  pendingChunkCopyOffset = 0;
  
  // Reset dry delay line state for underrun fallback
  // Initialize positions so readPos lags behind writePos by kOutputChunkSize
  dryDelayWritePos = static_cast<size_t>(kOutputChunkSize);
  dryDelayReadPos = 0;
  crossfadeGain = 1.0f;
  wasInUnderrun = false;
  
  // Clear context buffer to avoid audio artifacts from stale history.
  // This is ~16KB which takes <1µs on modern CPUs. It only happens on transport start,
  // which is a discontinuity anyway, so a brief stall is acceptable.
  for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
    std::memset(contextBuffer[ch].data(), 0, contextBuffer[ch].size() * sizeof(float));
  }
  
  // Reset LR4 crossover filter states to avoid clicks from stale filter memory
  for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
    lpFilter1[ch].reset();
    lpFilter2[ch].reset();
    inputHpFilter1[ch].reset();
    inputHpFilter2[ch].reset();
  }

  // Increment epoch BEFORE invalidating queue slots.
  // This ensures inference thread sees the new epoch and discards stale work.
  uint32_t newEpoch = resetEpoch.fetch_add(1, std::memory_order_acq_rel) + 1;
  juce::ignoreUnused(newEpoch);

  // Invalidate all queue slots (O(kNumInferenceBuffers) = O(4) = O(1))
  // This prevents consumption of stale results that were already marked processed.
  for (size_t i = 0; i < kNumInferenceBuffers; ++i) {
    if (inferenceQueue[i]) {
      inferenceQueue[i]->ready.store(false, std::memory_order_release);
      inferenceQueue[i]->processed.store(false, std::memory_order_release);
    }
  }
  
  // Reset only the indices owned by audio thread
  inferenceWriteIdx.store(0, std::memory_order_release);
  outputConsumeIdx.store(0, std::memory_order_release);
  // NOTE: inferenceReadIdx is owned by inference thread - it will reset itself on epoch change
#endif
}

void AudioPluginAudioProcessor::prepareToPlay(double sampleRate,
                                              int samplesPerBlock) {
  // Use this method as the place to do any pre-playback
  // initialisation that you need..
  
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  // Initialize LR4 crossover filters (Linkwitz-Riley 4th order = 2x cascaded Butterworth)
  // Q = 0.707107 (1/sqrt(2)) for Butterworth. Two cascaded = LR4 with -6dB at crossover.
  // Strategy: Split input into LP + HP before inference. Model sees only HP.
  // LP is added to bass stem after inference to reconstruct full-spectrum bass.
  constexpr float butterworthQ = 0.7071067811865476f;  // 1/sqrt(2)
  
  auto lpCoeffs = juce::dsp::IIR::Coefficients<float>::makeLowPass(sampleRate, 
      static_cast<double>(kCrossoverFreqHz), butterworthQ);
  auto hpCoeffs = juce::dsp::IIR::Coefficients<float>::makeHighPass(sampleRate, 
      static_cast<double>(kCrossoverFreqHz), butterworthQ);
  
  for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
    // LP filters on input (audio thread): capture low frequencies for bass stem
    lpFilter1[ch].coefficients = lpCoeffs;
    lpFilter2[ch].coefficients = lpCoeffs;
    lpFilter1[ch].reset();
    lpFilter2[ch].reset();
    
    // HP filters on input (audio thread): remove low frequencies before model
    inputHpFilter1[ch].coefficients = hpCoeffs;
    inputHpFilter2[ch].coefficients = hpCoeffs;
    inputHpFilter1[ch].reset();
    inputHpFilter2[ch].reset();
  }
  
  // Calculate vocals gate attack/release coefficients based on actual sample rate
  // coeff = exp(-1 / (sampleRate * timeInSeconds))
  vocalsGateAttackCoeff = std::exp(-1.0f / (static_cast<float>(sampleRate) * kVocalsGateAttackTimeSec));
  vocalsGateReleaseCoeff = std::exp(-1.0f / (static_cast<float>(sampleRate) * kVocalsGateReleaseTimeSec));
  
  DBG("[HS-TasNet] LR4 crossover initialized at " << kCrossoverFreqHz << " Hz (HP to model, LP to bass)");
  // Warn about small buffer sizes that may cause real-time issues.
  // The plugin has bursty CPU usage when copying inference results (~32KB per chunk).
  // At very small buffer sizes (<64 samples), these bursts may exceed the callback deadline.
  constexpr int kMinRecommendedBufferSize = 128;
  if (samplesPerBlock < kMinRecommendedBufferSize) {
    DBG("[HS-TasNet] WARNING: Buffer size " << samplesPerBlock << " samples is below the recommended minimum of "
        << kMinRecommendedBufferSize << " samples. This may cause audio dropouts due to bursty CPU usage "
        << "when copying inference results. Consider increasing buffer size to " << kMinRecommendedBufferSize << "+ samples.");
  }

  // Get ORT API safely - may be null if ORT DLL not loaded
  const OrtApi* api = getSafeOrtApi();
  if (api == nullptr) {
    DBG("[ORT] ONNX Runtime not available - OrtGetApiBase() returned null");
    return;
  }
  
  const OrtApiBase* apiBase = OrtGetApiBase();
  ortRuntimeVersion = (apiBase != nullptr) ? apiBase->GetVersionString() : "unknown";

  // Initialize ONNX Runtime environment (only once)
  if (!ortEnv) {
    OrtEnv* rawEnv = nullptr;
    OrtStatus* status = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "StemgenRT", &rawEnv);
    if (status != nullptr) {
      DBG("[ORT] Failed to create OrtEnv: " << api->GetErrorMessage(status));
      api->ReleaseStatus(status);
      return;
    }
    ortEnv.reset(static_cast<void*>(rawEnv));
    ortInitialized = true;
    DBG("[ORT] Initialized. Version: " << juce::String(ortRuntimeVersion));
  }

  // Load the HS-TasNet model (only once)
  if (!modelLoaded && ortEnv) {
    // Construct model path relative to the plugin binary
    juce::File pluginFile = juce::File::getSpecialLocation(
        juce::File::currentExecutableFile);
    
    // Try several possible model locations (bundled Resources folder first)
    std::vector<juce::File> modelPaths = {
      // Primary: Bundled in plugin's Resources folder (macOS bundle structure)
      pluginFile.getParentDirectory().getParentDirectory().getChildFile("Resources/model.onnx"),
    };

    juce::String modelPath;
    juce::String searchedPaths;
    for (const auto& path : modelPaths) {
      if (searchedPaths.isNotEmpty()) searchedPaths += "; ";
      searchedPaths += path.getFullPathName();
      DBG("[HS-TasNet] Checking path: " << path.getFullPathName() << " exists=" << (path.existsAsFile() ? "yes" : "no"));
      if (path.existsAsFile()) {
        modelPath = path.getFullPathName();
        DBG("[HS-TasNet] Found model at: " << modelPath);
        break;
      }
    }

    if (modelPath.isEmpty()) {
      modelLoadError = juce::String("Model file not found. Searched: ") + searchedPaths;
      DBG("[HS-TasNet] Model file not found in any expected location");
      return;
    }

    // Create session options
    OrtSessionOptions* sessionOptions = nullptr;
    OrtStatus* status = api->CreateSessionOptions(&sessionOptions);
    if (status != nullptr) {
      modelLoadError = juce::String("Session options error: ") + api->GetErrorMessage(status);
      DBG("[ORT] Failed to create session options: " << api->GetErrorMessage(status));
      api->ReleaseStatus(status);
      return;
    }

    // Set graph optimization level
    status = api->SetSessionGraphOptimizationLevel(sessionOptions, ORT_ENABLE_ALL);
    if (status != nullptr) {
      api->ReleaseStatus(status);
    }
    
    // Configure ORT threading for real-time audio workloads.
    // ORT has two threading settings:
    //   - Intra-op: parallelism WITHIN a single operation (e.g., matrix multiplication)
    //   - Inter-op: parallelism BETWEEN operations (running graph nodes concurrently)
    //
    // For real-time audio, predictability matters more than throughput, so we:
    //   1. Cap intra-op threads to reduce CPU contention with the audio thread
    //   2. Disable inter-op parallelism for sequential, predictable graph execution
    int numHardwareThreads = static_cast<int>(std::thread::hardware_concurrency());
    int numIntraOpThreads = std::min(std::max(numHardwareThreads / 2, 2), 4);  // Half of cores, capped 2-4
    
    status = api->SetIntraOpNumThreads(sessionOptions, numIntraOpThreads);
    if (status != nullptr) {
      api->ReleaseStatus(status);
    }
    
    status = api->SetInterOpNumThreads(sessionOptions, 1);  // Sequential node execution
    if (status != nullptr) {
      api->ReleaseStatus(status);
    }
    
    DBG("[ORT] Using " << numIntraOpThreads << " intra-op threads (of " << numHardwareThreads << " available)");
    
    // Try to enable GPU execution providers for acceleration.
    // Priority: TensorRT > CUDA > CPU
    // TensorRT provides the best performance through graph optimizations and FP16.
    // CUDA/cuDNN is the fallback GPU option.
    // CPU is the final fallback if no GPU is available.
    bool gpuEnabled = false;
    std::string gpuProviderName;
    
    // Priority 1: Try TensorRT (best performance, requires TensorRT SDK)
    // TensorRT provides graph-level optimizations, layer fusion, and FP16/INT8 support.
    // It can be 10-40% faster than CUDA/cuDNN for inference.
    {
      OrtTensorRTProviderOptionsV2* trtOptions = nullptr;
      status = api->CreateTensorRTProviderOptions(&trtOptions);
      if (status == nullptr && trtOptions != nullptr) {
        // Configure TensorRT options for real-time audio:
        // - device_id=0: Use first NVIDIA GPU
        // - trt_fp16_enable=1: Enable FP16 for faster inference on RTX GPUs
        // - trt_builder_optimization_level=3: Maximum optimization
        // - trt_engine_cache_enable=1: Cache compiled engines for faster startup
        const char* trtKeys[] = {"device_id", "trt_fp16_enable", "trt_builder_optimization_level", 
                                 "trt_engine_cache_enable"};
        const char* trtValues[] = {"0", "1", "3", "1"};
        
        status = api->UpdateTensorRTProviderOptions(trtOptions, trtKeys, trtValues, 4);
        if (status != nullptr) {
          DBG("[ORT] Failed to configure TensorRT options: " << api->GetErrorMessage(status));
          api->ReleaseStatus(status);
          status = nullptr;  // Continue with default TensorRT options
        }
        
        // Append TensorRT provider to session options
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
          DBG("[ORT] CreateTensorRTProviderOptions failed: " << api->GetErrorMessage(status));
          api->ReleaseStatus(status);
          status = nullptr;
        }
        DBG("[ORT] TensorRT provider options not available - trying CUDA");
      }
    }
    
    // Priority 2: Try CUDA/cuDNN (good performance, more compatible)
    if (!gpuEnabled) {
      OrtCUDAProviderOptionsV2* cudaOptions = nullptr;
      status = api->CreateCUDAProviderOptions(&cudaOptions);
      if (status == nullptr && cudaOptions != nullptr) {
        // Configure CUDA options for real-time audio:
        // - device_id=0: Use first NVIDIA GPU
        // - arena_extend_strategy=kSameAsRequested: Predictable memory allocation
        // - cudnn_conv_algo_search=EXHAUSTIVE: Find fastest algorithm (slower startup, faster inference)
        // - cudnn_conv_use_max_workspace=1: Allow cuDNN to use more memory for faster algorithms
        // - do_copy_in_default_stream=1: Reduce synchronization overhead
        const char* keys[] = {"device_id", "arena_extend_strategy", "cudnn_conv_algo_search", 
                              "cudnn_conv_use_max_workspace", "do_copy_in_default_stream"};
        const char* values[] = {"0", "kSameAsRequested", "EXHAUSTIVE", "1", "1"};
        
        status = api->UpdateCUDAProviderOptions(cudaOptions, keys, values, 5);
        if (status != nullptr) {
          DBG("[ORT] Failed to configure CUDA options: " << api->GetErrorMessage(status));
          api->ReleaseStatus(status);
          status = nullptr;  // Continue with default CUDA options
        }
        
        // Append CUDA provider to session options
        status = api->SessionOptionsAppendExecutionProvider_CUDA_V2(sessionOptions, cudaOptions);
        if (status == nullptr) {
          gpuEnabled = true;
          gpuProviderName = "CUDA";
          DBG("[ORT] CUDA execution provider enabled");
        } else {
          DBG("[ORT] CUDA not available: " << api->GetErrorMessage(status) << " - falling back to CPU");
          api->ReleaseStatus(status);
          status = nullptr;  // Not an error, just fall back to CPU
        }
        
        api->ReleaseCUDAProviderOptions(cudaOptions);
      } else {
        if (status != nullptr) {
          DBG("[ORT] CreateCUDAProviderOptions failed: " << api->GetErrorMessage(status));
          api->ReleaseStatus(status);
          status = nullptr;
        }
        DBG("[ORT] CUDA provider options not available - using CPU");
      }
    }
    
    // If no GPU provider succeeded, we'll use CPU (always available)
    if (!gpuEnabled) {
      gpuProviderName = "CPU";
      DBG("[ORT] Using CPU execution provider");
    }

    // Create the inference session
    OrtSession* rawSession = nullptr;
#ifdef _WIN32
    // Windows uses wide strings
    std::wstring wideModelPath(modelPath.toWideCharPointer());
    status = api->CreateSession(
        reinterpret_cast<OrtEnv*>(ortEnv.get()),
        wideModelPath.c_str(),
        sessionOptions,
        &rawSession);
#else
    status = api->CreateSession(
        reinterpret_cast<OrtEnv*>(ortEnv.get()),
        modelPath.toRawUTF8(),
        sessionOptions,
        &rawSession);
#endif

    api->ReleaseSessionOptions(sessionOptions);

    if (status != nullptr) {
      modelLoadError = juce::String("Session creation failed: ") + api->GetErrorMessage(status);
      DBG("[ORT] Failed to create session: " << api->GetErrorMessage(status));
      api->ReleaseStatus(status);
      return;
    }

    ortSession.reset(static_cast<void*>(rawSession));
    modelLoaded = true;
    modelLoadError.clear();
    usingGPU = gpuEnabled;
    ortExecutionProvider = gpuProviderName;
    DBG("[HS-TasNet] Model loaded successfully from: " << modelPath);
    DBG("[HS-TasNet] Execution provider: " << juce::String(ortExecutionProvider));

    // Allocate streaming buffers for overlap-add inference
    allocateStreamingBuffers();
  }
  
  // Always start the inference thread if model is loaded.
  // This must be outside the model-loading block because hosts (especially VST3)
  // may call releaseResources() + prepareToPlay() cycles during transport changes,
  // sample rate changes, or buffer size changes. releaseResources() stops the
  // inference thread, so we must restart it here.
  if (modelLoaded) {
    // Reset streaming buffers to clean state for new playback session.
    // This must happen before starting the inference thread to avoid a race
    // condition where the thread accesses buffers while they're being cleared.
    resetStreamingBuffers();
    
    // Start the background inference thread (no-op if already running)
    startInferenceThread();
    
    // Report latency to host for Plugin Delay Compensation (PDC)
    setLatencySamples(kOutputChunkSize);

    // Warm up ORT: queue a dummy inference to trigger lazy initialization
    // (kernel compilation, memory pool setup) before real audio arrives.
    // This happens on the message thread, which is fine - we're in prepareToPlay().
    if (inferenceQueue[0]) {
      auto& warmup = inferenceQueue[0];
      // Fill with zeros (buffers already allocated)
      for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
        std::memset(warmup->inputChunk[ch].data(), 0, 
                    static_cast<size_t>(kOutputChunkSize) * sizeof(float));
        std::memset(warmup->contextSnapshot[ch].data(), 0, 
                    static_cast<size_t>(kContextSize) * sizeof(float));
      }
      warmup->epoch = resetEpoch.load(std::memory_order_acquire);
      warmup->ready.store(true, std::memory_order_release);
      
      // Wait for it to complete (blocking is OK here, we're in prepareToPlay)
      while (!warmup->processed.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      warmup->processed.store(false, std::memory_order_release);
      
      // Increment epoch to reset inference thread's read index back to 0.
      // Warmup advanced inferenceReadIdx to 1 (after processing slot 0), but 
      // inferenceWriteIdx stayed at 0. Without this reset, real audio would queue
      // at slot 0 while inference thread looks at slot 1, causing out-of-order
      // processing and audio discontinuity on hosts without playhead info.
      resetEpoch.fetch_add(1, std::memory_order_acq_rel);
      
      DBG("[HS-TasNet] ORT warmup inference complete");
    }
  }
#endif
}

void AudioPluginAudioProcessor::releaseResources() {
  // When playback stops, you can use this as an opportunity to free up any
  // spare memory, etc.
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  // Stop the inference thread
  stopInferenceThread();
#endif
  // Reset streaming buffers when playback stops
  resetStreamingBuffers();
}

bool AudioPluginAudioProcessor::isBusesLayoutSupported(
    const BusesLayout& layouts) const {
#if JucePlugin_IsMidiEffect
  juce::ignoreUnused(layouts);
  return true;
#else
  // Require 1 stereo input bus and up to 5 stereo output buses.
  // Allow additional output buses (1..4) to be disabled if the host chooses.

  // Input: bus 0 must be stereo and enabled
#if !JucePlugin_IsSynth
  if (layouts.getChannelSet(true /* isInput */, 0) !=
      juce::AudioChannelSet::stereo())
    return false;
#endif

  // Outputs: bus 0 must be stereo; buses 1..4 may be stereo or disabled
  const int numOutputBuses = getBusCount(false /* isInput */);
  if (numOutputBuses < 1)
    return false;

  // Enforce exactly 5 output buses configured on this processor
  if (numOutputBuses != 5)
    return false;

  for (int busIndex = 0; busIndex < numOutputBuses; ++busIndex) {
    const auto set = layouts.getChannelSet(false /* isInput */, busIndex);
    if (busIndex == 0) {
      if (set != juce::AudioChannelSet::stereo())
        return false;
    } else {
      if (!(set == juce::AudioChannelSet::stereo() ||
            set == juce::AudioChannelSet::disabled()))
        return false;
    }
  }

  return true;
#endif
}

#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
void AudioPluginAudioProcessor::runOverlapAddInference(InferenceRequest& request) {
  // Overlap-add streaming inference for non-streaming model
  // This runs on the background thread
  // Input data comes from request.inputChunk and request.contextSnapshot
  // Output goes to request.outputChunk
  
  if (!modelLoaded || !ortSession || !ortMemoryInfo) return;

  const OrtApi* api = getSafeOrtApi();
  if (api == nullptr) return;  // ORT not available
  
  // Use pre-allocated memory info to avoid per-inference allocation overhead
  OrtMemoryInfo* memInfo = reinterpret_cast<OrtMemoryInfo*>(ortMemoryInfo);
  OrtStatus* status = nullptr;

  // Build the internal chunk: [left_context | new_samples | right_padding]
  // Use pre-allocated scratch buffer to avoid per-inference heap allocation
  float* audioInput = inferenceScratchBuffer.data();
  
  for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
    size_t offset = ch * static_cast<size_t>(kInternalChunkSize);
    
    // Copy left context (kContextSize samples from context snapshot)
    for (size_t i = 0; i < static_cast<size_t>(kContextSize); ++i) {
      audioInput[offset + i] = request.contextSnapshot[ch][i];
    }
    
    // Copy new samples (kOutputChunkSize samples)
    for (size_t i = 0; i < static_cast<size_t>(kOutputChunkSize); ++i) {
      audioInput[offset + static_cast<size_t>(kContextSize) + i] = request.inputChunk[ch][i];
    }
    
    // Right context: Reflection padding with smooth blend
    // This avoids the discontinuity of zero-padding while not creating artificial trends.
    // We reflect the signal around the last sample and apply a smooth crossfade
    // to prevent the cusp that pure reflection would create at the boundary.
    float lastSample = request.inputChunk[ch][kOutputChunkSize - 1];
    
    for (size_t i = 0; i < static_cast<size_t>(kContextSize); ++i) {
      // Reflection index: mirror around the end of the signal
      // i=0 -> kOutputChunkSize-2, i=1 -> kOutputChunkSize-3, etc.
      // We wrap around if kContextSize > kOutputChunkSize
      size_t distFromEnd = (i + 1) % static_cast<size_t>(kOutputChunkSize);
      if (distFromEnd == 0) distFromEnd = kOutputChunkSize;
      size_t mirrorIdx = static_cast<size_t>(kOutputChunkSize) - distFromEnd;
      float reflected = request.inputChunk[ch][mirrorIdx];
      
      // Smooth blend: exponential crossfade from reflected signal to constant (last sample)
      // This prevents the cusp at the reflection point while maintaining natural decay
      float t = static_cast<float>(i) / static_cast<float>(kContextSize);
      float blendFactor = 1.0f - std::exp(-4.0f * t);  // ~98% blended by end
      
      // Blend between reflected signal (maintains frequency content) and 
      // constant hold (prevents runaway reflections)
      audioInput[offset + static_cast<size_t>(kContextSize + kOutputChunkSize) + i] = 
          (1.0f - blendFactor) * reflected + blendFactor * lastSample;
    }
  }

  // Create input tensor [1, 2, kInternalChunkSize]
  // Input is HP-filtered before reaching this function (LR4 crossover in processBlock).
  // The model only sees high frequencies, avoiding sub-bass artifacts from chunked processing.
  // LP content is stored in request.lowFreqChunk and added to bass stem below.
  OrtValue* inputTensor = nullptr;
  std::int64_t audioDims[3] = {1, kNumChannels, kInternalChunkSize};
  
  status = api->CreateTensorWithDataAsOrtValue(
      memInfo, audioInput, inferenceScratchBuffer.size() * sizeof(float),
      audioDims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputTensor);
  
  if (status != nullptr) {
    DBG("[HS-TasNet] Failed to create input tensor: " << api->GetErrorMessage(status));
    api->ReleaseStatus(status);
    // NOTE: Don't release memInfo here - it's pre-allocated and shared
    return;
  }

  // Input/output names for non-streaming model
  const char* inputNames[1] = {"audio"};
  const char* outputNames[1] = {"separated"};

  // Output tensor
  OrtValue* outputTensor = nullptr;

  // Run inference
  status = api->Run(
      reinterpret_cast<OrtSession*>(ortSession.get()),
      nullptr,  // run options
      inputNames, &inputTensor, 1,
      outputNames, 1, &outputTensor);

  // Release input tensor (memInfo is pre-allocated, don't release here)
  if (inputTensor) api->ReleaseValue(inputTensor);

  if (status != nullptr) {
    DBG("[HS-TasNet] Overlap-add inference failed: " << api->GetErrorMessage(status));
    api->ReleaseStatus(status);
    if (outputTensor) api->ReleaseValue(outputTensor);
    return;
  }

  // Extract separated audio output
  // Output shape: [1, 4, 2, kInternalChunkSize] = [batch, stems, channels, samples]
  float* separatedData = nullptr;
  status = api->GetTensorMutableData(outputTensor, reinterpret_cast<void**>(&separatedData));
  
  if (status == nullptr && separatedData != nullptr) {
    // Debug: log sample range on first few inferences
    static int overlapAddInferenceCount = 0;
    float minVal = std::numeric_limits<float>::max();
    float maxVal = std::numeric_limits<float>::lowest();
    int nanCount = 0;
    int infCount = 0;
    
    // Apply inverse normalization gain to restore original level
    // Input was normalized by normalizationGain, so output needs to be divided by it
    float invNormGain = 1.0f / request.normalizationGain;
    
    // Extract only the middle portion (avoiding edge artifacts)
    // The valid output is at offset kContextSize, with length kOutputChunkSize
    for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
      for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
        // Data layout: [batch, stem, channel, samples]
        size_t dataOffset = stem * static_cast<size_t>(kNumChannels * kInternalChunkSize) 
                          + ch * static_cast<size_t>(kInternalChunkSize)
                          + static_cast<size_t>(kContextSize);  // Skip left context
        
        for (size_t i = 0; i < static_cast<size_t>(kOutputChunkSize); ++i) {
          float sample = separatedData[dataOffset + i];
          
          // Check for NaN/Inf and clamp
          if (std::isnan(sample)) {
            nanCount++;
            sample = 0.0f;
          } else if (std::isinf(sample)) {
            infCount++;
            sample = 0.0f;
          } else {
            minVal = std::min(minVal, sample);
            maxVal = std::max(maxVal, sample);
          }
          // Apply inverse normalization to restore original level
          request.outputChunk[stem][ch][i] = sample * invNormGain;
        }
      }
    }
    
    // Add LP to bass stem to reconstruct full-spectrum bass.
    // Since the model received HP-filtered input, its outputs are already HP-filtered.
    // We simply add the LP component to bass. This keeps the crossover coherent:
    //   - Model outputs: drums_HP, bass_HP, other_HP, vocals_HP
    //   - After LP addition: drums_HP, bass_HP + LP, other_HP, vocals_HP
    //   - Sum: HP_sum + LP = fullband (with LR4 phase shift)
    constexpr size_t kBassStemIndex = 1;
    for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
      for (size_t i = 0; i < static_cast<size_t>(kOutputChunkSize); ++i) {
        request.outputChunk[kBassStemIndex][ch][i] += request.lowFreqChunk[ch][i];
      }
    }
    
    if (overlapAddInferenceCount < 5) {
      DBG("[HS-TasNet] Overlap-add inference #" << overlapAddInferenceCount 
          << " - Output range: [" << minVal << ", " << maxVal 
          << "], NaN: " << nanCount << ", Inf: " << infCount);
      overlapAddInferenceCount++;
    }
    juce::ignoreUnused(nanCount, infCount);
  }

  // Release output tensor
  if (outputTensor) api->ReleaseValue(outputTensor);
}
#endif

void AudioPluginAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                             juce::MidiBuffer& midiMessages) {
  juce::ignoreUnused(midiMessages);

  juce::ScopedNoDenormals noDenormals;
  const int numSamples = buffer.getNumSamples();

  // Check for playback state change to reset streaming buffers.
  // Note: getPlayHead()->getPosition() is generally safe but not strictly RT-guaranteed
  // in all hosts (some may take locks). We only call it when we need the information,
  // and we degrade gracefully if it fails.
  // The playhead check is relatively infrequent (once per block) and essential for
  // proper transport sync. If a host's implementation is problematic, the user can
  // increase buffer size. We prioritize correct behavior over the edge case of a
  // blocking playhead implementation.
  if (juce::AudioPlayHead* currentPlayHead = getPlayHead()) {
    if (auto posInfo = currentPlayHead->getPosition()) {
      bool isPlaying = posInfo->getIsPlaying();
      bool wasPlayingBefore = wasPlaying.exchange(isPlaying, std::memory_order_acq_rel);
      
      // Reset streaming buffers when playback starts (was stopped, now playing)
      // Use RT-safe reset: O(1) index reset, defer memory clearing to inference thread
      if (isPlaying && !wasPlayingBefore) {
        resetStreamingBuffersRT();
      }
    }
  }

#if !JucePlugin_IsSynth
  // Extract input channel pointers directly (RT-safe: getBusBuffer returns a view,
  // but we avoid storing the AudioBuffer object to sidestep copy ambiguity)
  const float* inputChannelPtrs[kNumChannels] = { nullptr, nullptr };
  {
    auto inputBus = getBusBuffer(buffer, true /* isInput */, 0);
    for (int ch = 0; ch < std::min(kNumChannels, inputBus.getNumChannels()); ++ch)
      inputChannelPtrs[ch] = inputBus.getReadPointer(ch);
  }
#endif

#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  if (modelLoaded && ortSession) {
    const size_t outRingSize = outputRingBuffers[0][0].size();

    // Amortized chunk consumption: instead of copying full 1024-sample chunks in bursts,
    // we copy at most numSamples worth of data per callback. This bounds worst-case CPU
    // work proportionally to the host buffer size, enabling stable operation at small buffers.
    //
    // The algorithm:
    //   1. If no pending chunk, check if a processed result is ready
    //   2. Copy up to numSamples from the pending chunk to the ring buffer
    //   3. When chunk is fully copied, move to next processed result
    
    size_t samplesToProcess = static_cast<size_t>(numSamples);
    
    while (samplesToProcess > 0) {
      // If no pending chunk, try to acquire one
      if (!hasPendingChunk) {
        size_t consumeIdx = outputConsumeIdx.load(std::memory_order_acquire);
        auto& consumeRequest = inferenceQueue[consumeIdx];
        
        if (!consumeRequest || !consumeRequest->processed.load(std::memory_order_acquire)) {
          break;  // No more results ready
        }
        
        // Check if this result is from an old epoch (stale after reset)
        uint32_t currentEpoch = resetEpoch.load(std::memory_order_acquire);
        if (consumeRequest->epoch != currentEpoch) {
          // Stale result from before a reset - discard it
          consumeRequest->processed.store(false, std::memory_order_release);
          outputConsumeIdx.store((consumeIdx + 1) % kNumInferenceBuffers, std::memory_order_release);
          continue;  // Check next slot
        }
        
        // Check ring buffer capacity before writing
        if (outputSamplesAvailable + static_cast<size_t>(kOutputChunkSize) > outRingSize) {
          // Ring buffer would overflow - drop oldest samples to make room
          size_t overflow = (outputSamplesAvailable + static_cast<size_t>(kOutputChunkSize)) - outRingSize;
          outputReadPos = (outputReadPos + overflow) % outRingSize;
          outputSamplesAvailable -= overflow;
        }
        
        // We have a valid pending chunk to copy
        hasPendingChunk = true;
        pendingChunkCopyOffset = 0;
      }
      
      // Copy portion of pending chunk - at most samplesToProcess samples
      size_t consumeIdx = outputConsumeIdx.load(std::memory_order_acquire);
      auto& consumeRequest = inferenceQueue[consumeIdx];
      
      size_t remainingInChunk = static_cast<size_t>(kOutputChunkSize) - pendingChunkCopyOffset;
      size_t samplesToCopy = std::min(samplesToProcess, remainingInChunk);
      
      // ===== Optimized ring buffer write (avoids modulo in hot loop) =====
      // Copy output to ring buffer and compute residuals for lossless reconstruction.
      // Residual = original_input - sum(all_stems)
      //
      // We distribute residual using energy-weighted mixture consistency among
      // drums, bass, and other stems. Vocals receives NO residual to keep it clean
      // when there are no vocals in the track (avoids pollution from bleed).
      //
      // VOCALS GATE: When vocals energy is tiny relative to the total mix (likely
      // spurious output on instrumental tracks), we transfer that energy to "other"
      // instead. This keeps the vocals stem silent on instrumental content.
      //
      // Additionally, we apply an INPUT-FOLLOWING SOFT GATE to eliminate model noise floor:
      // Neural networks output small non-zero values even for silent input. These errors
      // are often correlated across stems (sum≈0) but individually audible. The gate
      // attenuates all stems when input is very quiet, eliminating this noise.
      constexpr float kResidualEpsilon = 1e-8f;  // Avoid division by zero

      // Compute base write position once, then increment with branch
      size_t writePos = (outputReadPos + outputSamplesAvailable) % outRingSize;
      const size_t srcBase = pendingChunkCopyOffset;

      // Get raw pointers for source data (avoid repeated operator[] on unique_ptr)
      const float* origL = consumeRequest->originalInput[0].data() + srcBase;
      const float* origR = consumeRequest->originalInput[1].data() + srcBase;
      const float* stemData[kNumStems][kNumChannels];
      for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
        stemData[stem][0] = consumeRequest->outputChunk[stem][0].data() + srcBase;
        stemData[stem][1] = consumeRequest->outputChunk[stem][1].data() + srcBase;
      }

      for (size_t i = 0; i < samplesToCopy; ++i) {
        // ===== Input-following soft gate =====
        // Compute input magnitude (max of L/R for simplicity and to avoid sqrt)
        float origLSample = origL[i];
        float origRSample = origR[i];
        float inputMag = std::max(std::abs(origLSample), std::abs(origRSample));
        
        // Soft gate gain: 1.0 above threshold, ramps to 0 below floor
        float gateGain = std::clamp((inputMag - kSoftGateFloor) * kSoftGateInvRange, 0.0f, 1.0f);
        // ===== end soft gate =====

        // Get all stem samples for both channels
        float drums_L = stemData[0][0][i];
        float drums_R = stemData[0][1][i];
        float bass_L = stemData[1][0][i];
        float bass_R = stemData[1][1][i];
        float vocals_L = stemData[2][0][i];
        float vocals_R = stemData[2][1][i];
        float other_L = stemData[3][0][i];
        float other_R = stemData[3][1][i];

        // ===== Vocals gate =====
        // Two criteria to detect spurious vocals content:
        // 1. Ratio-based: vocals are tiny fraction of total mix (likely instrumental section)
        // 2. Level-based: vocals are very quiet in absolute terms (real vocals are rarely < -35dB)
        // Transfer gated vocals energy to "other" stem to preserve total energy
        
        // Criterion 1: Ratio of vocals energy to total mix
        float vocalsEnergy = vocals_L * vocals_L + vocals_R * vocals_R;
        float totalStemEnergy = drums_L * drums_L + drums_R * drums_R +
                                bass_L * bass_L + bass_R * bass_R +
                                vocalsEnergy +
                                other_L * other_L + other_R * other_R + kResidualEpsilon;
        float vocalsRatio = vocalsEnergy / totalStemEnergy;

        float vocalsGateRatio = std::clamp((vocalsRatio - kVocalsGateRatioFloor) * kVocalsGateRatioInvRange, 0.0f, 1.0f);

        // Criterion 2: Absolute vocals level (peak of L/R)
        float vocalsPeak = std::max(std::abs(vocals_L), std::abs(vocals_R));
        float vocalsGateLevel = std::clamp((vocalsPeak - kVocalsGateLevelFloor) * kVocalsGateLevelInvRange, 0.0f, 1.0f);

        // Combined gate: most restrictive wins (minimum of both criteria)
        float vocalsGateTarget = std::min(vocalsGateRatio, vocalsGateLevel);

        // Asymmetric attack/release smoothing to avoid pumping artifacts
        // Fast attack (gate opening) so vocals come in quickly
        // Slow release (gate closing) to avoid pumping on short gaps
        float smoothingCoeff = (vocalsGateTarget > vocalsGateGainSmoothed)
                                   ? vocalsGateAttackCoeff   // Opening: fast attack
                                   : vocalsGateReleaseCoeff; // Closing: slow release
        vocalsGateGainSmoothed = smoothingCoeff * vocalsGateGainSmoothed +
                                 (1.0f - smoothingCoeff) * vocalsGateTarget;

        // Apply vocals gate: transfer gated vocals to "other"
        float vocalsGated_L = vocals_L * vocalsGateGainSmoothed;
        float vocalsGated_R = vocals_R * vocalsGateGainSmoothed;
        float vocalsToOther_L = vocals_L - vocalsGated_L;  // What we're removing from vocals
        float vocalsToOther_R = vocals_R - vocalsGated_R;
        float other_L_adj = other_L + vocalsToOther_L;  // Add to other
        float other_R_adj = other_R + vocalsToOther_R;
        // ===== end vocals gate =====

        // Channel 0 (Left)
        {
          float originalSample = origLSample;
          delayedInputBuffer[0][writePos] = originalSample;

          // Compute residual (using original stems, before vocals gate adjustment)
          float residual = originalSample - (drums_L + bass_L + vocals_L + other_L);

          // Power-weighted distribution (excluding vocals to keep it clean)
          float p_drums = drums_L * drums_L;
          float p_bass = bass_L * bass_L;
          float p_other = other_L_adj * other_L_adj;
          float totalPower = p_drums + p_bass + p_other + kResidualEpsilon;

          // Distribute residual proportionally to non-vocal stem power
          // Apply soft gate to eliminate model noise floor on quiet passages
          outputRingBuffers[0][0][writePos] = (drums_L + (p_drums / totalPower) * residual) * gateGain;
          outputRingBuffers[1][0][writePos] = (bass_L + (p_bass / totalPower) * residual) * gateGain;
          outputRingBuffers[2][0][writePos] = vocalsGated_L * gateGain;  // Gated vocals, no residual
          outputRingBuffers[3][0][writePos] = (other_L_adj + (p_other / totalPower) * residual) * gateGain;
        }

        // Channel 1 (Right)
        {
          float originalSample = origRSample;
          delayedInputBuffer[1][writePos] = originalSample;

          // Compute residual
          float residual = originalSample - (drums_R + bass_R + vocals_R + other_R);

          // Power-weighted distribution (excluding vocals to keep it clean)
          float p_drums = drums_R * drums_R;
          float p_bass = bass_R * bass_R;
          float p_other = other_R_adj * other_R_adj;
          float totalPower = p_drums + p_bass + p_other + kResidualEpsilon;

          // Distribute residual proportionally to non-vocal stem power
          // Apply soft gate to eliminate model noise floor on quiet passages
          outputRingBuffers[0][1][writePos] = (drums_R + (p_drums / totalPower) * residual) * gateGain;
          outputRingBuffers[1][1][writePos] = (bass_R + (p_bass / totalPower) * residual) * gateGain;
          outputRingBuffers[2][1][writePos] = vocalsGated_R * gateGain;  // Gated vocals, no residual
          outputRingBuffers[3][1][writePos] = (other_R_adj + (p_other / totalPower) * residual) * gateGain;
        }
        
        // Advance write position with branch instead of modulo
        ++writePos;
        if (writePos == outRingSize) writePos = 0;
      }
      // ===== end optimized ring buffer write =====
      
      outputSamplesAvailable += samplesToCopy;
      pendingChunkCopyOffset += samplesToCopy;
      samplesToProcess -= samplesToCopy;
      
      // Check if chunk is fully copied
      if (pendingChunkCopyOffset >= static_cast<size_t>(kOutputChunkSize)) {
        // Mark as consumed and move to next slot
        consumeRequest->processed.store(false, std::memory_order_release);
        outputConsumeIdx.store((consumeIdx + 1) % kNumInferenceBuffers, std::memory_order_release);
        hasPendingChunk = false;
        pendingChunkCopyOffset = 0;
      }
    }

    // Accumulate input samples until we have kOutputChunkSize
    // Strategy: Split input into LP + HP using LR4 crossover.
    //   - HP goes to model (model struggles with low frequencies in chunked inference)
    //   - LP is stored and added to bass stem after inference
    // This keeps the crossover coherent: LP + HP = original (LR4 property)
    const size_t dryDelaySize = dryDelayLine[0].size();
    for (int i = 0; i < numSamples; ++i) {
      for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
#if !JucePlugin_IsSynth
        float sample = (inputChannelPtrs[ch] != nullptr) ? inputChannelPtrs[ch][i] : 0.0f;
#else
        float sample = 0.0f;
#endif
        // Write raw input to dry delay line for underrun fallback
        // This provides a latency-aligned dry signal when separated output is unavailable
        dryDelayLine[ch][dryDelayWritePos] = sample;
        
        // HP-filter input before model (remove sub-bass that chunked inference handles poorly)
        float hpSample = inputHpFilter2[ch].processSample(inputHpFilter1[ch].processSample(sample));
        inputAccumBuffer[ch][inputAccumCount] = hpSample;
        
        // LP-filter input for bass stem reconstruction
        // (added to bass after inference to restore full-spectrum bass)
        float lpSample = lpFilter2[ch].processSample(lpFilter1[ch].processSample(sample));
        lowFreqAccumBuffer[ch][inputAccumCount] = lpSample;
      }
      
      // Advance dry delay write position
      ++dryDelayWritePos;
      if (dryDelayWritePos >= dryDelaySize) dryDelayWritePos = 0;
      
      inputAccumCount++;

      // When we have enough samples, queue for inference
      if (inputAccumCount >= static_cast<size_t>(kOutputChunkSize)) {
        // Get the next write slot
        size_t writeIdx = inferenceWriteIdx.load(std::memory_order_acquire);
        auto& request = inferenceQueue[writeIdx];
        
        // Check if slot is available (not being processed or waiting to be consumed)
        if (request && !request->ready.load(std::memory_order_acquire) 
                    && !request->processed.load(std::memory_order_acquire)) {
          // Calculate RMS of HP input for normalization
          float sumSquares = 0.0f;
          for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
            for (size_t j = 0; j < static_cast<size_t>(kOutputChunkSize); ++j) {
              float sample = inputAccumBuffer[ch][j];
              sumSquares += sample * sample;
            }
          }
          float rms = std::sqrt(sumSquares / static_cast<float>(kNumChannels * kOutputChunkSize));
          
          // Compute normalization gain (clamped to avoid extreme amplification of silence)
          float normGain = 1.0f;
          if (rms >= kNormMinInputRms) {
            normGain = std::min(kNormTargetRms / rms, kNormMaxGain);
          }
          request->normalizationGain = normGain;
          
          // Copy and normalize HP input for model, HP context, LP for bass, and fullband for residual.
          for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
            // HP-filtered input for model (normalized)
            for (size_t j = 0; j < static_cast<size_t>(kOutputChunkSize); ++j) {
              request->inputChunk[ch][j] = inputAccumBuffer[ch][j] * normGain;
            }
            // HP-filtered context for model (normalized)
            for (size_t j = 0; j < static_cast<size_t>(kContextSize); ++j) {
              request->contextSnapshot[ch][j] = contextBuffer[ch][j] * normGain;
            }
            // LP-filtered samples for bass stem (NOT normalized - added after inference)
            std::memcpy(request->lowFreqChunk[ch].data(), lowFreqAccumBuffer[ch].data(),
                        static_cast<size_t>(kOutputChunkSize) * sizeof(float));
            // Fullband input for residual calculation: HP + LP = original (LR4 property)
            // NOT normalized - residual calc uses original levels
            for (size_t j = 0; j < static_cast<size_t>(kOutputChunkSize); ++j) {
              request->originalInput[ch][j] = inputAccumBuffer[ch][j] + lowFreqAccumBuffer[ch][j];
            }
          }
          
          // Update context buffer for next chunk (stores HP-filtered samples, NOT normalized).
          if constexpr (kOutputChunkSize >= kContextSize) {
            const size_t srcOffset = static_cast<size_t>(kOutputChunkSize - kContextSize);
            for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
              std::memcpy(contextBuffer[ch].data(),
                          inputAccumBuffer[ch].data() + srcOffset,
                          static_cast<size_t>(kContextSize) * sizeof(float));
            }
          } else {
            const size_t samplesToKeep = static_cast<size_t>(kContextSize - kOutputChunkSize);
            for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
              std::memmove(contextBuffer[ch].data(),
                           contextBuffer[ch].data() + static_cast<size_t>(kOutputChunkSize),
                           samplesToKeep * sizeof(float));
              std::memcpy(contextBuffer[ch].data() + samplesToKeep,
                          inputAccumBuffer[ch].data(),
                          static_cast<size_t>(kOutputChunkSize) * sizeof(float));
            }
          }
          
          // Stamp request with current epoch (for stale detection after reset)
          request->epoch = resetEpoch.load(std::memory_order_acquire);
          
          // Mark as ready and advance write index
          request->ready.store(true, std::memory_order_release);
          inferenceWriteIdx.store((writeIdx + 1) % kNumInferenceBuffers, std::memory_order_release);
        }
        // else: queue is full, drop this chunk (will cause audio dropout)
        
        inputAccumCount = 0;
      }
    }

    // ===== Write separated stems to output buses (optimized RT-friendly hot path) =====
    const int numOutputBuses = getBusCount(false /* isInput */);

    // Bus 0 (Main): sum of all stems
    // Extract write pointers directly (RT-safe: avoids storing AudioBuffer object)
    float* mainWrite[kNumChannels] = { nullptr, nullptr };
    int mainNumCh = 0;
    {
        auto mainBus = getBusBuffer(buffer, false /* isInput */, 0);
        mainNumCh = mainBus.getNumChannels();
        for (int ch = 0; ch < std::min(kNumChannels, mainNumCh); ++ch)
            mainWrite[ch] = mainBus.getWritePointer(ch);
    }

    // Buses 1-4: individual stems
    // Model output order: 0=drums, 1=bass, 2=vocals, 3=other
    // Bus order: 1=Drums, 2=Bass, 3=Other, 4=Vocals
    static constexpr size_t busToStemMap[4] = { 0, 1, 3, 2 };

    // Grab write pointers directly from bus buffers (RT-safe: avoids AudioBuffer copies)
    // If the host disabled them, they typically have 0 channels
    float* stemWrite[4][kNumChannels] = { {nullptr, nullptr}, {nullptr, nullptr}, {nullptr, nullptr}, {nullptr, nullptr} };
    int stemNumCh[4] = { 0, 0, 0, 0 };

    for (int b = 0; b < 4; ++b) {
        if (b + 1 < numOutputBuses) {
            auto stemBus = getBusBuffer(buffer, false, b + 1);
            stemNumCh[b] = stemBus.getNumChannels();
            for (int ch = 0; ch < std::min(kNumChannels, stemNumCh[b]); ++ch)
                stemWrite[b][ch] = stemBus.getWritePointer(ch);
        }
    }

    // Use locals for ring state (audio thread owns these, so no atomics needed)
    size_t readPos = outputReadPos;
    size_t avail   = outputSamplesAvailable;
    const size_t ringSize = outRingSize;
    
    // Dry delay read position lags behind write position by kOutputChunkSize samples
    // This provides latency-aligned dry signal for underrun fallback
    size_t dryReadPos = dryDelayReadPos;
    const size_t drySize = dryDelayLine[0].size();
    
    // Local crossfade gain for smooth transitions
    float xfadeGain = crossfadeGain;
    constexpr float xfadeDelta = 1.0f / static_cast<float>(kCrossfadeSamples);

    for (int i = 0; i < numSamples; ++i) {
        const bool have = (avail > 0);
        
        // Update crossfade gain: ramp toward 1.0 (separated) or 0.0 (dry)
        // When separated audio is available (have=true), ramp up toward 1.0
        // When underrun (have=false), ramp down toward 0.0
        if (have) {
            xfadeGain = std::min(1.0f, xfadeGain + xfadeDelta);
        } else {
            xfadeGain = std::max(0.0f, xfadeGain - xfadeDelta);
        }
        
        // Get dry signal (latency-aligned input) for fallback/crossfade
        float dry[kNumChannels];
        for (int ch = 0; ch < kNumChannels; ++ch) {
            dry[ch] = dryDelayLine[static_cast<size_t>(ch)][dryReadPos];
        }
        
        // Get separated signal (sum of stems) when available
        float separated[kNumChannels] = {0.0f, 0.0f};
        if (have) {
            for (int ch = 0; ch < kNumChannels; ++ch) {
                for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
                    separated[ch] += outputRingBuffers[stem][static_cast<size_t>(ch)][readPos];
                }
            }
        }
        
        // Main bus = crossfade between separated and dry
        // When xfadeGain=1.0: output=separated, When xfadeGain=0.0: output=dry
        for (int ch = 0; ch < std::min(kNumChannels, mainNumCh); ++ch) {
            float output = xfadeGain * separated[ch] + (1.0f - xfadeGain) * dry[ch];
            mainWrite[ch][i] = output;
        }

        // Stem buses (if enabled)
        // During underrun, output dry/4 to each stem (approximate equal split)
        // During normal operation, output separated stems with crossfade
        for (int busIdx = 0; busIdx < 4; ++busIdx) {
            if (stemNumCh[busIdx] <= 0)
                continue;

            const size_t stemIndex = busToStemMap[busIdx];
            for (int ch = 0; ch < std::min(kNumChannels, stemNumCh[busIdx]); ++ch) {
                float stemSample = 0.0f;
                if (have) {
                    stemSample = outputRingBuffers[stemIndex][static_cast<size_t>(ch)][readPos];
                }
                // Crossfade: separated stem when available, dry/4 as fallback
                // The /4 distributes dry signal equally across 4 stems
                float dryStem = dry[ch] * 0.25f;
                float output = xfadeGain * stemSample + (1.0f - xfadeGain) * dryStem;
                stemWrite[busIdx][ch][i] = output;
            }
        }

        // Advance ring read position only if we consumed a sample
        if (have) {
            ++readPos;
            if (readPos == ringSize) readPos = 0;
            --avail;
        }
        
        // Always advance dry delay read position (dry input flows continuously)
        ++dryReadPos;
        if (dryReadPos >= drySize) dryReadPos = 0;
    }

    // Commit ring state back to members
    outputReadPos = readPos;
    outputSamplesAvailable = avail;
    dryDelayReadPos = dryReadPos;
    crossfadeGain = xfadeGain;
    // ===== end optimized section =====

    return;
  }
#endif

  // Fallback: copy input to all outputs (no model loaded)
  // Extract pointers directly to avoid AudioBuffer copy ambiguity (RT-safe)
  const int numOutputBuses = getBusCount(false /* isInput */);
  for (int busIndex = 0; busIndex < numOutputBuses; ++busIndex) {
#if !JucePlugin_IsSynth
    auto outputBus = getBusBuffer(buffer, false /* isInput */, busIndex);
    const int outNumCh = outputBus.getNumChannels();
    const int channelsToCopy = std::min(kNumChannels, outNumCh);
    
    for (int ch = 0; ch < channelsToCopy; ++ch) {
      float* outPtr = outputBus.getWritePointer(ch);
      if (inputChannelPtrs[ch] != nullptr) {
        std::memcpy(outPtr, inputChannelPtrs[ch], static_cast<size_t>(numSamples) * sizeof(float));
      } else {
        std::memset(outPtr, 0, static_cast<size_t>(numSamples) * sizeof(float));
      }
    }

    // If the output bus has more channels than the input, clear the extras
    for (int ch = channelsToCopy; ch < outNumCh; ++ch) {
      float* outPtr = outputBus.getWritePointer(ch);
      std::memset(outPtr, 0, static_cast<size_t>(numSamples) * sizeof(float));
    }
#endif
  }
}

bool AudioPluginAudioProcessor::hasEditor() const {
  return true;  // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* AudioPluginAudioProcessor::createEditor() {
  return new AudioPluginAudioProcessorEditor(*this);
}

void AudioPluginAudioProcessor::getStateInformation(
    juce::MemoryBlock& destData) {
  // You should use this method to store your parameters in the memory block.
  // You could do that either as raw data, or use the XML or ValueTree classes
  // as intermediaries to make it easy to save and load complex data.
  juce::ignoreUnused(destData);
}

void AudioPluginAudioProcessor::setStateInformation(const void* data,
                                                    int sizeInBytes) {
  // You should use this method to restore your parameters from this memory
  // block, whose contents will have been created by the getStateInformation()
  // call.
  juce::ignoreUnused(data, sizeInBytes);
}
}  // namespace audio_plugin

// This creates new instances of the plugin.
// This function definition must be in the global namespace.
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
  return new audio_plugin::AudioPluginAudioProcessor();
}
