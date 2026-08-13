#include "StemgenRT/PluginProcessor.h"
#include "StemgenRT/PluginEditor.h"
#include "StemgenRT/ModelOutputScheduler.h"
#include <algorithm>
#include <vector>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <thread>

namespace audio_plugin {

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
  // Initialize the environment early so availability is visible before the
  // host starts playback. Model loading still happens in prepareToPlay().
  onnxRuntime_ = std::make_unique<OnnxRuntime>();
#endif
}

AudioPluginAudioProcessor::~AudioPluginAudioProcessor() {
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  inferenceQueue_.stopThread();
  // OnnxRuntime handles its own cleanup via RAII
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
  // c236 has no graph tail, but hosts still need the declared 256-sample PDC
  // horizon to drain already scheduled current-chunk output.
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  const int activeLatency =
      activeLatencySamples_.load(std::memory_order_acquire);
  // The host bridge's prepared rate is authoritative. Some direct plugin
  // lifecycles call prepareToPlay() without populating JUCE's base-class rate.
  if (activeLatency > 0 && hostSampleRate_ > 0) {
    return static_cast<double>(activeLatency) /
           static_cast<double>(hostSampleRate_);
  }
#endif
  return static_cast<double>(kPluginLatencySamples) /
         static_cast<double>(kModelSampleRate);
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
  const auto appendTimingWarning = [this](juce::String status) {
    if (isRealtimeCallbackTimingUnsafe()) {
      status += juce::String::formatted(
          " | PDC warning: %d-sample callback needs %d samples",
          getLastHostBlockSize(), getRequiredLatencySamplesForLastHostBlock());
    }
    return status;
  };
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  if (onnxRuntime_) {
    if (!sampleRateSupported_.load(std::memory_order_acquire)) {
      juce::String sampleRateError;
      {
        const std::lock_guard<std::mutex> lock(statusMutex_);
        sampleRateError = modelLoadError_;
      }
      return appendTimingWarning(
          sampleRateError.isNotEmpty()
              ? sampleRateError
              : juce::String::formatted(
                    "This c236 candidate requires %d Hz / %d samples",
                    kCurrentChunkQualifiedHostSampleRate,
                    kCurrentChunkQualifiedHostBlockSize));
    }
    juce::String modelLoadError;
    {
      const std::lock_guard<std::mutex> lock(statusMutex_);
      modelLoadError = modelLoadError_;
    }
    if (modelLoadError.isNotEmpty() &&
        activeLatencySamples_.load(std::memory_order_acquire) == 0) {
      return appendTimingWarning(juce::String("Model error: ") +
                                 modelLoadError);
    }
    return appendTimingWarning(onnxRuntime_->getStatusString());
  }
  return appendTimingWarning("ONNX Runtime: not initialized");
#else
  return appendTimingWarning("ONNX Runtime: not linked");
#endif
}

int AudioPluginAudioProcessor::getLatencySamples() const {
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  if (!sampleRateSupported_.load(std::memory_order_acquire) || !onnxRuntime_ ||
      !onnxRuntime_->isReadyForInference()) {
    return 0;
  }

  return activeLatencySamples_.load(std::memory_order_acquire);
#else
  return 0;
#endif
}

double AudioPluginAudioProcessor::getLatencyMs() const {
  double sampleRate = getSampleRate();
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  if (activeLatencySamples_.load(std::memory_order_acquire) > 0 &&
      hostSampleRate_ > 0) {
    sampleRate = static_cast<double>(hostSampleRate_);
  }
#endif
  if (sampleRate <= 0.0) {
    sampleRate = 44100.0;  // Fallback if not yet initialized
  }
  return (static_cast<double>(getLatencySamples()) / sampleRate) * 1000.0;
}

#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
size_t AudioPluginAudioProcessor::getUnderrunSamplesInLastBlock() const {
  return lastUnderrunSamplesInLastBlock_.load(std::memory_order_acquire);
}

uint64_t AudioPluginAudioProcessor::getUnderrunSampleCount() const {
  return totalUnderrunSamples_.load(std::memory_order_acquire);
}

uint64_t AudioPluginAudioProcessor::getUnderrunBlockCount() const {
  return totalUnderrunBlocks_.load(std::memory_order_acquire);
}

bool AudioPluginAudioProcessor::isUnderrunActive() const {
  return underrunActive_.load(std::memory_order_acquire);
}

size_t AudioPluginAudioProcessor::getRingFillLevel() const {
  return ringFillLevel_.load(std::memory_order_acquire);
}

uint64_t AudioPluginAudioProcessor::getRingOverflowEventCount() const {
  return totalRingOverflowEvents_.load(std::memory_order_acquire);
}

uint64_t AudioPluginAudioProcessor::getRingOverflowSampleDropCount() const {
  return totalRingOverflowSamplesDropped_.load(std::memory_order_acquire);
}

uint64_t AudioPluginAudioProcessor::getQueueFullChunkDropCount() const {
  return totalQueueFullChunkDrops_.load(std::memory_order_acquire);
}
#else
size_t AudioPluginAudioProcessor::getUnderrunSamplesInLastBlock() const {
  return 0;
}

uint64_t AudioPluginAudioProcessor::getUnderrunSampleCount() const {
  return 0;
}

uint64_t AudioPluginAudioProcessor::getUnderrunBlockCount() const {
  return 0;
}

bool AudioPluginAudioProcessor::isUnderrunActive() const {
  return false;
}

size_t AudioPluginAudioProcessor::getRingFillLevel() const {
  return 0;
}

uint64_t AudioPluginAudioProcessor::getRingOverflowEventCount() const {
  return 0;
}

uint64_t AudioPluginAudioProcessor::getRingOverflowSampleDropCount() const {
  return 0;
}

uint64_t AudioPluginAudioProcessor::getQueueFullChunkDropCount() const {
  return 0;
}
#endif

int AudioPluginAudioProcessor::getPreparedHostBlockSize() const {
  return preparedHostBlockSize_.load(std::memory_order_acquire);
}

int AudioPluginAudioProcessor::getLastHostBlockSize() const {
  return lastHostBlockSize_.load(std::memory_order_acquire);
}

int AudioPluginAudioProcessor::getRequiredLatencySamplesForLastHostBlock()
    const {
  return requiredLatencySamplesForLastHostBlock_.load(
      std::memory_order_acquire);
}

bool AudioPluginAudioProcessor::isRealtimeCallbackTimingUnsafe() const {
  return realtimeCallbackTimingUnsafe_.load(std::memory_order_acquire);
}

uint64_t AudioPluginAudioProcessor::getUnsafeRealtimeCallbackCount() const {
  return unsafeRealtimeCallbackCount_.load(std::memory_order_acquire);
}

InferenceQueue::WorkerPriorityStatus
AudioPluginAudioProcessor::getInferenceWorkerPriorityStatus() const {
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  return inferenceQueue_.getWorkerPriorityStatus();
#else
  return InferenceQueue::WorkerPriorityStatus::Unsupported;
#endif
}

#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
void AudioPluginAudioProcessor::allocateStreamingBuffers(
    int maximumHostBlockSize,
    double hostSampleRate) {
  const int safeHostBlockSize = std::max(maximumHostBlockSize, 1);
  hostSampleRate_ = static_cast<int>(std::lround(hostSampleRate));
  sampleRateConversionActive_ = hostSampleRate_ != kModelSampleRate;

  if (!inputSampleRateAdapter_.prepare(hostSampleRate,
                                       static_cast<double>(kModelSampleRate),
                                       static_cast<size_t>(kNumChannels))) {
    throw std::runtime_error("input sample-rate converter preparation failed");
  }

  sampleRateConversionDelaySamples_ = 0;
  inferenceQueue_.disableOutputSampleRateConversion();
  if (sampleRateConversionActive_) {
    if (!inferenceQueue_.prepareOutputSampleRate(hostSampleRate)) {
      throw std::runtime_error(
          "output sample-rate converter preparation failed");
    }

    const double naturalDelaySamples =
        inputSampleRateAdapter_.filterGroupDelaySeconds() * hostSampleRate +
        inferenceQueue_.getOutputSampleRateConversionDelaySamples();
    const double integralDelay = std::ceil(naturalDelaySamples - 1.0e-9);
    const double fractionalCompensation = integralDelay - naturalDelaySamples;
    if (fractionalCompensation < -1.0e-9 || fractionalCompensation >= 1.0) {
      throw std::runtime_error("invalid sample-rate converter phase delay");
    }
    if (!inferenceQueue_.prepareOutputSampleRate(
            hostSampleRate, std::max(0.0, fractionalCompensation))) {
      throw std::runtime_error(
          "output sample-rate converter phase alignment failed");
    }

    const double correctedDelaySamples =
        inputSampleRateAdapter_.filterGroupDelaySeconds() * hostSampleRate +
        inferenceQueue_.getOutputSampleRateConversionDelaySamples();
    const double roundedDelaySamples = std::round(correctedDelaySamples);
    if (!std::isfinite(correctedDelaySamples) ||
        std::abs(correctedDelaySamples - roundedDelaySamples) > 1.0e-6 ||
        roundedDelaySamples < 0.0 ||
        roundedDelaySamples >
            static_cast<double>(std::numeric_limits<int>::max())) {
      throw std::runtime_error(
          "sample-rate converter delay is not host-sample aligned");
    }
    sampleRateConversionDelaySamples_ = static_cast<int>(roundedDelaySamples);
  }

  modelSchedulingLatencySamples_ = calculateModelSchedulingLatencySamples(
      hostSampleRate_, safeHostBlockSize);
  const int latencySamples = calculatePluginLatencySamples(
      hostSampleRate_, safeHostBlockSize, sampleRateConversionDelaySamples_);
  activeLatencySamples_.store(latencySamples, std::memory_order_release);
  overlapAdd_.allocate(static_cast<size_t>(safeHostBlockSize),
                       static_cast<size_t>(latencySamples),
                       inferenceQueue_.getMaximumHostOutputSamplesPerHop());

  const size_t hostScratchCapacity = static_cast<size_t>(
      std::max(safeHostBlockSize, kMinimumHostBlockCapacity));
  const size_t modelScratchCapacity =
      inputSampleRateAdapter_.maxOutputForInput(hostScratchCapacity);
  if (modelScratchCapacity == std::numeric_limits<size_t>::max() ||
      modelScratchCapacity == std::numeric_limits<size_t>::max() - 1U) {
    throw std::runtime_error("sample-rate converter scratch size overflow");
  }
  for (size_t channel = 0U; channel < static_cast<size_t>(kNumChannels);
       ++channel) {
    sanitizedHostInputScratch_[channel].assign(hostScratchCapacity, 0.0f);
    modelInputScratch_[channel].assign(modelScratchCapacity + 1U, 0.0f);
  }

  // Reset output writer (initializes crossfade state)
  outputWriter_.reset();

  // Allocate inference queue buffers
  inferenceQueue_.allocate();

  DBG("[HS-TasNet] Streaming buffers allocated:");
  DBG("  Model chunk size: " << kOutputChunkSize << " samples");
  DBG("  Model analysis window: " << kAnalysisWindowSize << " samples");
  DBG("  Host sample rate: " << hostSampleRate_ << " Hz");
  DBG("  SRC delay: " << sampleRateConversionDelaySamples_ << " samples");
  DBG("  Reported PDC: " << latencySamples << " samples");
  DBG("  Inference queue size: " << kNumInferenceBuffers << " slots");
}

#endif

void AudioPluginAudioProcessor::reset() {
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  // Hosts call AudioProcessor::reset() at processing stops and may call it for
  // seeks/discontinuities. Keep this callback bounded: invalidate the epoch
  // and clear only audio-thread-owned preallocated state. The worker observes
  // the epoch and resets all graph state before accepting another sequence.
  resetStreamingBuffersRT();
#endif
  wasPlaying.store(false, std::memory_order_release);
  hasExpectedPlayheadPosition_ = false;
  expectedPlayheadPosition_ = 0;
}

void AudioPluginAudioProcessor::resetStreamingBuffers() {
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  // fullReset() directly clears slot ownership, so stop the worker first when
  // this non-RT reset is invoked on an already prepared processor. Preserve
  // the prior running state for test/host-driven full resets; prepareToPlay()
  // and releaseResources() already enter here with the worker stopped.
  const bool restartInferenceThread = inferenceQueue_.isThreadRunning();
  if (restartInferenceThread) {
    inferenceQueue_.stopThread();
  }

  // Reset overlap-add processor (clears all buffers and indices)
  overlapAdd_.reset();
  inputSampleRateAdapter_.reset();

  // Reset output writer (crossfade state)
  outputWriter_.reset();
  // Reset inference queue state (full reset: clears ownership and indices)
  inferenceQueue_.fullReset();
  if (onnxRuntime_) {
    onnxRuntime_->resetStreamingState();
  }
  if (restartInferenceThread && onnxRuntime_ &&
      onnxRuntime_->isReadyForInference()) {
    inferenceQueue_.startThread(onnxRuntime_.get());
  }

  // Reset input sequence tracking used for recurrent-state gap detection.
  nextInputChunkSequence_ = 0;

  // Snapshot diagnostics describe the current streaming generation. Lifetime
  // totals intentionally remain cumulative across resets.
  ringFillLevel_.store(0, std::memory_order_release);
  underrunActive_.store(false, std::memory_order_release);
  lastUnderrunSamplesInLastBlock_.store(0, std::memory_order_release);

  DBG("[HS-TasNet] Streaming buffers reset");
#endif
}

void AudioPluginAudioProcessor::resetStreamingBuffersRT() {
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  // RT-safe reset for transport start/stop.
  //
  // Design principles:
  //   1. Audio thread only writes to indices it owns
  //   2. Inference thread will reset when it sees epoch change
  //   3. Fixed-latency dry history is cleared to avoid stale transport audio
  //   4. Epoch increment invalidates all in-flight inference results

  // Reset audio-thread indices and clear latency history so a seek or loop
  // cannot leak old dry audio into the new transport position.
  overlapAdd_.resetIndices();
  overlapAdd_.clearDryDelayBuffer();
  inputSampleRateAdapter_.reset();

  // Reset output writer (crossfade state)
  outputWriter_.reset();
  ringFillLevel_.store(0, std::memory_order_release);
  underrunActive_.store(false, std::memory_order_release);
  lastUnderrunSamplesInLastBlock_.store(0, std::memory_order_release);

  // Reset inference queue - increments epoch and invalidates in-flight
  // requests.
  inferenceQueue_.reset();

  // Start a new contiguous input sequence after transport resets.
  nextInputChunkSequence_ = 0;
#endif
}

void AudioPluginAudioProcessor::prepareToPlay(double sampleRate,
                                              int samplesPerBlock) {
  preparedHostBlockSize_.store(std::max(samplesPerBlock, 1),
                               std::memory_order_release);
  lastHostBlockSize_.store(0, std::memory_order_release);
  requiredLatencySamplesForLastHostBlock_.store(0, std::memory_order_release);
  realtimeCallbackTimingUnsafe_.store(false, std::memory_order_release);
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  inferenceQueue_.stopThread();
  // Every prepare attempt starts a new stream generation, including attempts
  // that later fail closed for sample rate, runtime, or model validation.
  // This prevents stale output and snapshot telemetry from surviving a host
  // reconfiguration onto the safe path.
  resetStreamingBuffers();
  const bool sampleRateCanConvertToInt =
      std::isfinite(sampleRate) && sampleRate > 0.0 &&
      sampleRate <= static_cast<double>(std::numeric_limits<int>::max());
  const int roundedSampleRate =
      sampleRateCanConvertToInt ? static_cast<int>(std::lround(sampleRate)) : 0;
  hostSampleRate_ =
      roundedSampleRate > 0 ? roundedSampleRate : kModelSampleRate;
  sampleRateConversionActive_ = false;
  sampleRateConversionDelaySamples_ = 0;
  modelSchedulingLatencySamples_ = calculateModelSchedulingLatencySamples(
      hostSampleRate_, std::max(samplesPerBlock, 1));
  const bool sampleRateSupported =
      sampleRateCanConvertToInt &&
      std::abs(sampleRate - static_cast<double>(roundedSampleRate)) < 0.5 &&
      isQualifiedCurrentChunkHostConfiguration(roundedSampleRate,
                                               samplesPerBlock);
  sampleRateSupported_.store(sampleRateSupported, std::memory_order_release);
  if (!sampleRateSupported) {
    const juce::String error =
        juce::String("Unsupported c236 current-chunk configuration ") +
        juce::String(sampleRate, 1) + " Hz / " +
        juce::String(samplesPerBlock) +
        " samples; this build requires " +
        juce::String(kCurrentChunkQualifiedHostSampleRate) + " Hz / " +
        juce::String(kCurrentChunkQualifiedHostBlockSize) + " samples";
    {
      const std::lock_guard<std::mutex> lock(statusMutex_);
      modelLoadError_ = error;
    }
    activeLatencySamples_.store(0, std::memory_order_release);
    setLatencySamples(0);
    DBG("[HS-TasNet] " << error);
    return;
  }
  {
    const std::lock_guard<std::mutex> lock(statusMutex_);
    modelLoadError_.clear();
  }
  outputWriter_.prepare(sampleRate);

  // Warn about small buffer sizes that may cause real-time issues
  constexpr int kMinRecommendedBufferSize = 128;
  if (samplesPerBlock < kMinRecommendedBufferSize) {
    DBG("[HS-TasNet] WARNING: Buffer size "
        << samplesPerBlock << " samples is below recommended "
        << kMinRecommendedBufferSize << ". May cause audio dropouts.");
  }

  // Check if OnnxRuntime is initialized
  if (!onnxRuntime_ || !onnxRuntime_->isInitialized()) {
    DBG("[ORT] ONNX Runtime not available");
    activeLatencySamples_.store(0, std::memory_order_release);
    setLatencySamples(0);
    return;
  }

  // Load the HS-TasNet model (only once)
  if (!onnxRuntime_->isModelLoaded()) {
    // Construct model path relative to the plugin binary
    juce::File pluginFile =
        juce::File::getSpecialLocation(juce::File::currentExecutableFile);
    juce::File modelFile =
        pluginFile.getParentDirectory().getParentDirectory().getChildFile(
            "Resources/model.onnx");

    DBG("[HS-TasNet] Checking model path: " << modelFile.getFullPathName());

    if (!modelFile.existsAsFile()) {
      const juce::String error =
          juce::String("Model not found: ") + modelFile.getFullPathName();
      {
        const std::lock_guard<std::mutex> lock(statusMutex_);
        modelLoadError_ = error;
      }
      activeLatencySamples_.store(0, std::memory_order_release);
      setLatencySamples(0);
      DBG("[HS-TasNet] " << error);
      return;
    }

    // Load the qualified model into the CPU ONNX Runtime session.
    juce::String loadError;
    if (!onnxRuntime_->loadModel(modelFile.getFullPathName(), loadError)) {
      {
        const std::lock_guard<std::mutex> lock(statusMutex_);
        modelLoadError_ = loadError;
      }
      activeLatencySamples_.store(0, std::memory_order_release);
      setLatencySamples(0);
      DBG("[HS-TasNet] Model load failed: " << loadError);
      return;
    }
  }

  // Always recreate bounded buffers and state for the host's current maximum
  // block size, then start the inference thread.
  // Hosts may call releaseResources() + prepareToPlay() cycles during transport
  // changes.
  if (onnxRuntime_->isModelLoaded()) {
    juce::String preparationError;
    if (!onnxRuntime_->prepareForInference(preparationError)) {
      {
        const std::lock_guard<std::mutex> lock(statusMutex_);
        modelLoadError_ = preparationError;
      }
      activeLatencySamples_.store(0, std::memory_order_release);
      setLatencySamples(0);
      DBG("[HS-TasNet] Inference preparation failed: " << preparationError);
      return;
    }

    try {
      allocateStreamingBuffers(samplesPerBlock, sampleRate);
    } catch (const std::exception& exception) {
      const juce::String error =
          juce::String("Streaming buffer allocation failed: ") +
          juce::String(exception.what());
      {
        const std::lock_guard<std::mutex> lock(statusMutex_);
        modelLoadError_ = error;
      }
      activeLatencySamples_.store(0, std::memory_order_release);
      setLatencySamples(0);
      DBG("[HS-TasNet] " << error);
      return;
    } catch (...) {
      const juce::String error = "Streaming buffer allocation failed";
      {
        const std::lock_guard<std::mutex> lock(statusMutex_);
        modelLoadError_ = error;
      }
      activeLatencySamples_.store(0, std::memory_order_release);
      setLatencySamples(0);
      DBG("[HS-TasNet] " << error);
      return;
    }

    // Reset streaming buffers to clean state for new playback session
    resetStreamingBuffers();

    // Start the background inference thread (no-op if already running)
    inferenceQueue_.startThread(onnxRuntime_.get());

    // Report latency to host for Plugin Delay Compensation (PDC)
    setLatencySamples(activeLatencySamples_.load(std::memory_order_acquire));

    // Warm up ORT: queue a dummy inference to trigger lazy initialization.
    // Use submitForWarmup() which doesn't advance write index, then reset()
    // to invalidate the warmup slot and re-synchronize queue epoch state.
    if (auto* warmup = inferenceQueue_.getWriteSlot()) {
      for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
        std::memset(warmup->inputChunk[ch].data(), 0,
                    static_cast<size_t>(kOutputChunkSize) * sizeof(float));
      }
      warmup->chunkSequence = 0;
      inferenceQueue_.submitForWarmup();

      // Wait for completion (blocking is acceptable in prepareToPlay), but
      // never indefinitely in case inference thread is stalled.
      constexpr auto kWarmupTimeout = std::chrono::seconds(2);
      const auto deadline = std::chrono::steady_clock::now() + kWarmupTimeout;
      while (!warmup->isProcessed()) {
        if (std::chrono::steady_clock::now() >= deadline) {
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      const bool warmupCompleted = warmup->isProcessed();

      if (warmupCompleted) {
        // Release without advancing consumeIdx; warmup never advances the
        // queue's write/consume timeline.
        inferenceQueue_.releaseWarmupSlot(warmup);
        DBG("[HS-TasNet] ORT warmup complete");
      } else {
        DBG("[HS-TasNet] ORT warmup timed out after "
            << static_cast<int>(kWarmupTimeout.count())
            << "s; continuing without blocking.");
      }

      // Always advance epoch after warmup attempt to invalidate any late warmup
      // result and re-sync queue indices for real-time processing.
      inferenceQueue_.reset();
    }
  }
#else
  juce::ignoreUnused(sampleRate, samplesPerBlock);
#endif
}

void AudioPluginAudioProcessor::releaseResources() {
  // When playback stops, you can use this as an opportunity to free up any
  // spare memory, etc.
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  // Stop the inference thread
  inferenceQueue_.stopThread();
  activeLatencySamples_.store(0, std::memory_order_release);
  setLatencySamples(0);
#endif
  preparedHostBlockSize_.store(0, std::memory_order_release);
  lastHostBlockSize_.store(0, std::memory_order_release);
  requiredLatencySamplesForLastHostBlock_.store(0, std::memory_order_release);
  realtimeCallbackTimingUnsafe_.store(false, std::memory_order_release);
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
void AudioPluginAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                             juce::MidiBuffer& midiMessages) {
  juce::ignoreUnused(midiMessages);

  juce::ScopedNoDenormals noDenormals;
  const int numSamples = buffer.getNumSamples();
  const bool nonRealtimeRender = isNonRealtime();
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  const int requiredLatencySamples =
      numSamples > 0
          ? calculatePluginLatencySamples(hostSampleRate_, numSamples,
                                          sampleRateConversionDelaySamples_)
          : 0;
#else
  const int requiredLatencySamples =
      numSamples > 0 ? calculatePluginLatencySamples(numSamples) : 0;
#endif
  lastHostBlockSize_.store(numSamples, std::memory_order_release);
  requiredLatencySamplesForLastHostBlock_.store(requiredLatencySamples,
                                                std::memory_order_release);
  bool unsafeRealtimeCallback = false;
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  const int activeLatency =
      activeLatencySamples_.load(std::memory_order_acquire);
  unsafeRealtimeCallback = !nonRealtimeRender && numSamples > 0 &&
                           activeLatency > 0 &&
                           requiredLatencySamples > activeLatency;
#endif
  realtimeCallbackTimingUnsafe_.store(unsafeRealtimeCallback,
                                      std::memory_order_release);
  if (unsafeRealtimeCallback) {
    unsafeRealtimeCallbackCount_.fetch_add(1, std::memory_order_relaxed);
  }
  // If the host exposes no transport state, preserve live/offline processing
  // telemetry. A known stopped transport has no playback deadline to miss.
  bool underrunTelemetryEnabled = true;
  bool resetAfterCurrentCallback = false;

  // Check for playback state change to reset streaming buffers.
  // Note: getPlayHead()->getPosition() is generally safe but not strictly
  // RT-guaranteed in all hosts (some may take locks). We only call it when we
  // need the information, and we degrade gracefully if it fails. The playhead
  // check is relatively infrequent (once per block) and essential for proper
  // transport sync. If a host's implementation is problematic, the user can
  // increase buffer size. We prioritize correct behavior over the edge case of
  // a blocking playhead implementation.
  if (juce::AudioPlayHead* currentPlayHead = getPlayHead()) {
    if (auto posInfo = currentPlayHead->getPosition()) {
      const bool isPlaying = posInfo->getIsPlaying();
      underrunTelemetryEnabled = isPlaying;
      const bool wasPlayingBefore =
          wasPlaying.exchange(isPlaying, std::memory_order_acq_rel);

      const bool playbackStarted = isPlaying && !wasPlayingBefore;
      const bool playbackStopped = !isPlaying && wasPlayingBefore;
      bool transportDiscontinuity = playbackStarted;
      resetAfterCurrentCallback = playbackStopped;
      if (const auto timeInSamples = posInfo->getTimeInSamples()) {
        if (hasExpectedPlayheadPosition_) {
          if (isPlaying && wasPlayingBefore &&
              *timeInSamples != expectedPlayheadPosition_) {
            transportDiscontinuity = true;
          }
          if (!isPlaying && !wasPlayingBefore &&
              *timeInSamples != expectedPlayheadPosition_) {
            transportDiscontinuity = true;
          }
        }

        // A stopped playhead normally reports the same position on every
        // callback. Retaining it lets us detect a stopped scrub/seek without
        // repeatedly resetting ordinary stopped callbacks.
        expectedPlayheadPosition_ =
            isPlaying ? *timeInSamples + numSamples : *timeInSamples;
        hasExpectedPlayheadPosition_ = true;
      } else {
        hasExpectedPlayheadPosition_ = false;
      }

      // Reset immediately on starts, seeks, scrubs, and loop wraps. On a
      // play-to-stop transition, first render the result already scheduled by
      // the one-hop asynchronous PDC, then reset after this callback. This is
      // queue-tail drainage; c236 itself has no graph flush call.
      if (transportDiscontinuity) {
        resetStreamingBuffersRT();
        resetAfterCurrentCallback = false;
      }
    }
  }

#if !JucePlugin_IsSynth
  // Extract input channel pointers directly (RT-safe: getBusBuffer returns a
  // view, but we avoid storing the AudioBuffer object to sidestep copy
  // ambiguity)
  const float* inputChannelPtrs[kNumChannels] = {nullptr, nullptr};
  {
    auto inputBus = getBusBuffer(buffer, true /* isInput */, 0);
    for (int ch = 0; ch < std::min(kNumChannels, inputBus.getNumChannels());
         ++ch)
      inputChannelPtrs[ch] = inputBus.getReadPointer(ch);
  }
#endif

#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  if (sampleRateSupported_.load(std::memory_order_acquire) && onnxRuntime_ &&
      onnxRuntime_->isReadyForInference() &&
      activeLatencySamples_.load(std::memory_order_acquire) > 0) {
    const size_t outRingSize = overlapAdd_.getOutputRingSize();
    auto& outputRingBuffers = overlapAdd_.getOutputRingBuffers();
    auto& delayedInputBuffer = overlapAdd_.getDelayedInputBuffer();

    // A host is allowed to exceed maximumExpectedSamplesPerBlock. The dry
    // history has generous bounded slack for normal violations; fail closed on
    // an absurdly large callback instead of overwriting unread delay samples.
    if (!overlapAdd_.canProcessHostBlock(static_cast<size_t>(numSamples))) {
      resetStreamingBuffersRT();
      for (int busIndex = 0; busIndex < getBusCount(false); ++busIndex) {
        auto outputBus = getBusBuffer(buffer, false, busIndex);
        outputBus.clear();
      }
      return;
    }

    // A non-finite input is a streaming discontinuity: sanitize it to silence
    // and reset all graph/queue state before consuming any old-epoch output.
    bool inputHadNonFiniteSample = false;
#if !JucePlugin_IsSynth
    for (int ch = 0; ch < kNumChannels && !inputHadNonFiniteSample; ++ch) {
      if (inputChannelPtrs[ch] == nullptr) {
        continue;
      }
      for (int i = 0; i < numSamples; ++i) {
        if (!std::isfinite(inputChannelPtrs[ch][i])) {
          inputHadNonFiniteSample = true;
          break;
        }
      }
    }
#endif
    if (inputHadNonFiniteSample) {
      resetStreamingBuffersRT();
    }

    // Results are scheduled by their exact input sequence, never by arrival
    // order. If inference misses a deadline, its elapsed prefix is discarded;
    // it can therefore never replay over a newer dry-fallback timeline.
    uint64_t ringOverflowEventsThisBlock = 0;
    uint64_t ringOverflowSamplesDroppedThisBlock = 0;
    uint64_t queueFullDropsThisBlock = 0;

    const auto drainReadyInferenceResults = [&]() {
      size_t consumedResults = 0;
      for (int resultIndex = 0; resultIndex < kNumInferenceBuffers;
           ++resultIndex) {
        InferenceRequest* consumeRequest =
            inferenceQueue_.getCurrentOutputSlot();
        if (consumeRequest == nullptr) {
          consumeRequest =
              inferenceQueue_.getOutputSlot(inferenceQueue_.getEpoch());
        }
        if (consumeRequest == nullptr) {
          break;
        }

        // Failed runs publish invalid markers so the exact-timeline consumer
        // can advance. Sequence zero is a valid current-chunk result.
        if (!consumeRequest->outputValid) {
          inferenceQueue_.releaseOutputSlot();
          ++consumedResults;
          continue;
        }

        if (sampleRateConversionActive_ &&
            (!consumeRequest->hostOutputValid ||
             consumeRequest->hostOutputSampleCount == 0U)) {
          ++ringOverflowEventsThisBlock;
          inferenceQueue_.releaseOutputSlot();
          ++consumedResults;
          continue;
        }

        const uint64_t outputTimelineSample =
            overlapAdd_.getOutputTimelineSample();
        const size_t convertedSampleCount =
            sampleRateConversionActive_ ? consumeRequest->hostOutputSampleCount
                                        : static_cast<size_t>(kOutputChunkSize);
        ModelOutputSchedulePlan schedulePlan;
        if (sampleRateConversionActive_) {
          const uint64_t schedulingLatency =
              static_cast<uint64_t>(modelSchedulingLatencySamples_);
          if (consumeRequest->hostOutputStartSample >
              std::numeric_limits<uint64_t>::max() - schedulingLatency) {
            schedulePlan.action =
                ModelOutputScheduleAction::kDiscardTimelineOverflow;
          } else {
            schedulePlan = planModelOutputRange(
                schedulingLatency + consumeRequest->hostOutputStartSample,
                convertedSampleCount, outputTimelineSample, outRingSize);
          }
        } else {
          const uint64_t latency = static_cast<uint64_t>(
              activeLatencySamples_.load(std::memory_order_acquire));
          schedulePlan =
              planModelOutputSchedule(consumeRequest->chunkSequence, latency,
                                      outputTimelineSample, outRingSize);
        }

        if (schedulePlan.action == ModelOutputScheduleAction::kWaitForHorizon) {
          // Keep ownership and retry against the same absolute timeline after
          // the bounded ring horizon advances.
          break;
        }
        if (schedulePlan.action != ModelOutputScheduleAction::kSchedule) {
          ++ringOverflowEventsThisBlock;
          ringOverflowSamplesDroppedThisBlock +=
              static_cast<uint64_t>(convertedSampleCount);
          inferenceQueue_.releaseOutputSlot();
          ++consumedResults;
          continue;
        }

        if (!overlapAdd_.canScheduleModelOutput(
                schedulePlan.scheduleTimelineSample,
                schedulePlan.sampleCount)) {
          // A same-timeline collision is a duplicate/corrupt result. Preserve
          // the first publication and discard this one instead of shifting
          // either.
          ++ringOverflowEventsThisBlock;
          ringOverflowSamplesDroppedThisBlock +=
              static_cast<uint64_t>(schedulePlan.sampleCount);
          inferenceQueue_.releaseOutputSlot();
          ++consumedResults;
          continue;
        }

        for (size_t i = 0; i < schedulePlan.sampleCount; ++i) {
          const size_t sourceIndex = schedulePlan.sourceOffset + i;
          const size_t destinationIndex = overlapAdd_.getOutputRingPosition(
              schedulePlan.scheduleTimelineSample + static_cast<uint64_t>(i));
          for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
            if (sampleRateConversionActive_) {
              for (const int retainedStem :
                   {kStemDrums, kStemBass, kStemVocals}) {
                const size_t stem = static_cast<size_t>(retainedStem);
                outputRingBuffers[stem][ch][destinationIndex] =
                    consumeRequest->hostOutputChunk[stem][ch][sourceIndex];
              }
            } else {
              delayedInputBuffer[ch][destinationIndex] =
                  consumeRequest->alignedInput[ch][sourceIndex];
              for (size_t stem = 0; stem < static_cast<size_t>(kNumStems);
                   ++stem) {
                outputRingBuffers[stem][ch][destinationIndex] =
                    consumeRequest->outputChunk[stem][ch][sourceIndex];
              }
            }
          }
        }
        overlapAdd_.markModelOutputScheduled(
            schedulePlan.scheduleTimelineSample, schedulePlan.sampleCount);
        if (schedulePlan.sourceOffset > 0) {
          ++ringOverflowEventsThisBlock;
          ringOverflowSamplesDroppedThisBlock +=
              static_cast<uint64_t>(schedulePlan.sourceOffset);
        }
        inferenceQueue_.releaseOutputSlot();
        ++consumedResults;
      }
      return consumedResults;
    };

    drainReadyInferenceResults();

    // Preserve every sanitized native host frame for Main/fallback before
    // advancing the independent 44.1 kHz model clock.
    for (int i = 0; i < numSamples; ++i) {
      for (int ch = 0; ch < kNumChannels; ++ch) {
#if !JucePlugin_IsSynth
        const float rawSample =
            (inputChannelPtrs[ch] != nullptr) ? inputChannelPtrs[ch][i] : 0.0f;
        const float sample = std::isfinite(rawSample) ? rawSample : 0.0f;
#else
        const float sample = 0.0f;
#endif
        sanitizedHostInputScratch_[static_cast<size_t>(ch)]
                                  [static_cast<size_t>(i)] = sample;
        overlapAdd_.pushDryInputSample(ch, sample);
      }
    }

    // The inference worker binds these exact finite samples to the graph
    // without deployment-time level normalization. Queue submission remains
    // entirely on the 44.1 kHz clock.
    const bool synchronousOfflineRender = nonRealtimeRender;
    bool offlineInferenceTimedOut = false;
    const auto submitAccumulatedModelHop = [&]() {
      if (!overlapAdd_.readyForInference()) {
        return;
      }

      if (offlineInferenceTimedOut) {
        overlapAdd_.clearInputAccum();
        return;
      }

      const uint64_t chunkSequence = nextInputChunkSequence_++;
      InferenceRequest* request = inferenceQueue_.getWriteSlot();
      if (request == nullptr && synchronousOfflineRender) {
        drainReadyInferenceResults();
        request = inferenceQueue_.getWriteSlot();
      }

      if (request) {
        request->chunkSequence = chunkSequence;
        const auto& inputAccumBuffer = overlapAdd_.getInputAccumBuffer();
        for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
          std::memcpy(request->inputChunk[ch].data(),
                      inputAccumBuffer[ch].data(),
                      static_cast<size_t>(kOutputChunkSize) * sizeof(float));
        }

        const uint32_t submittedEpoch = inferenceQueue_.getEpoch();
        inferenceQueue_.submitWriteSlot(submittedEpoch);

        if (synchronousOfflineRender) {
          constexpr auto kOfflineHopTimeout = std::chrono::seconds(5);
          const auto deadline =
              std::chrono::steady_clock::now() + kOfflineHopTimeout;
          while (!request->isProcessed() &&
                 inferenceQueue_.getEpoch() == submittedEpoch &&
                 inferenceQueue_.isThreadRunning() &&
                 std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
          }

          if (request->isProcessed() &&
              inferenceQueue_.getEpoch() == submittedEpoch) {
            drainReadyInferenceResults();
          } else {
            offlineInferenceTimedOut = true;
          }
        }
      } else {
        // Sequence still advances. The worker resets its recurrent graph and
        // output SRC at the next accepted absolute model hop.
        ++queueFullDropsThisBlock;
      }
      overlapAdd_.clearInputAccum();
    };

    size_t modelSampleCount = static_cast<size_t>(numSamples);
    const float* modelInputPointers[kNumChannels] = {
        sanitizedHostInputScratch_[0].data(),
        sanitizedHostInputScratch_[1].data()};
    if (sampleRateConversionActive_) {
      const float* hostInputPointers[kNumChannels] = {
          sanitizedHostInputScratch_[0].data(),
          sanitizedHostInputScratch_[1].data()};
      float* convertedInputPointers[kNumChannels] = {
          modelInputScratch_[0].data(), modelInputScratch_[1].data()};
      const auto conversion = inputSampleRateAdapter_.process(
          hostInputPointers, static_cast<size_t>(numSamples),
          convertedInputPointers, modelInputScratch_[0].size());
      if (!conversion.ok ||
          conversion.inputConsumed != static_cast<size_t>(numSamples)) {
        resetStreamingBuffersRT();
        modelSampleCount = 0U;
      } else {
        modelSampleCount = conversion.outputProduced;
        modelInputPointers[0] = modelInputScratch_[0].data();
        modelInputPointers[1] = modelInputScratch_[1].data();
      }
    }

    for (size_t sample = 0U; sample < modelSampleCount; ++sample) {
      for (int ch = 0; ch < kNumChannels; ++ch) {
        overlapAdd_.pushModelInputSample(ch, modelInputPointers[ch][sample]);
      }
      submitAccumulatedModelHop();
    }

    // ===== Write separated stems to output buses =====
    const int numOutputBuses = getBusCount(false /* isInput */);

    // Bus 0 (Main): dry passthrough. Do not assume in-place aliasing between
    // input bus 0 and output bus 0; some hosts/tests provide distinct buffers.
    float* mainWrite[kNumChannels] = {nullptr, nullptr};
    auto mainBus = getBusBuffer(buffer, false /* isInput */, 0);
    int mainNumCh = mainBus.getNumChannels();
    for (int ch = 0; ch < std::min(kNumChannels, mainNumCh); ++ch)
      mainWrite[ch] = mainBus.getWritePointer(ch);

    // Buses 1-4: individual stems
    float* stemWrite[4][kNumChannels] = {{nullptr, nullptr},
                                         {nullptr, nullptr},
                                         {nullptr, nullptr},
                                         {nullptr, nullptr}};
    int stemNumCh[4] = {0, 0, 0, 0};

    for (int b = 0; b < 4; ++b) {
      if (b + 1 < numOutputBuses) {
        auto stemBus = getBusBuffer(buffer, false, b + 1);
        stemNumCh[b] = stemBus.getNumChannels();
        for (int ch = 0; ch < std::min(kNumChannels, stemNumCh[b]); ++ch)
          stemWrite[b][ch] = stemBus.getWritePointer(ch);
      }
    }

    // Set up output writer and write the block
    outputWriter_.setOutputPointers(mainWrite, mainNumCh, stemWrite, stemNumCh);
    const auto writeStats = outputWriter_.writeBlock(
        overlapAdd_, outputRingBuffers, delayedInputBuffer, outRingSize,
        numSamples, underrunTelemetryEnabled,
        !unsafeRealtimeCallback,
        !sampleRateConversionActive_);

    // Report exact-timeline model samples that remain scheduled after this
    // block. Gaps are not counted as fill and never shift later output.
    ringFillLevel_.store(overlapAdd_.getOutputSamplesAvailable(),
                         std::memory_order_release);
    if (ringOverflowEventsThisBlock > 0) {
      totalRingOverflowEvents_.fetch_add(ringOverflowEventsThisBlock,
                                         std::memory_order_relaxed);
      totalRingOverflowSamplesDropped_.fetch_add(
          ringOverflowSamplesDroppedThisBlock, std::memory_order_relaxed);
    }
    if (queueFullDropsThisBlock > 0) {
      totalQueueFullChunkDrops_.fetch_add(queueFullDropsThisBlock,
                                          std::memory_order_relaxed);
    }

    underrunActive_.store(writeStats.isUnderrunNow, std::memory_order_release);
    lastUnderrunSamplesInLastBlock_.store(writeStats.underrunSamples,
                                          std::memory_order_release);
    if (writeStats.hadUnderrun) {
      totalUnderrunBlocks_.fetch_add(1, std::memory_order_acq_rel);
      totalUnderrunSamples_.fetch_add(writeStats.underrunSamples,
                                      std::memory_order_acq_rel);
    }

    if (offlineInferenceTimedOut) {
      // The current callback has already rendered from its intact aligned-dry
      // history. Now invalidate the stalled request and graph generation so
      // the next callback starts from deterministic zero state without
      // splicing a mid-callback reset onto the output timeline.
      resetStreamingBuffersRT();
      resetAfterCurrentCallback = false;
    }

    if (resetAfterCurrentCallback) {
      resetStreamingBuffersRT();
    }

    return;
  }
#endif

  // Fail-safe path when the qualified model is unavailable. Keep Main as the
  // input, route the entire residual to Other, and clear the other three stems
  // so the stem buses still sum exactly to Main.
  const int numOutputBuses = getBusCount(false /* isInput */);
  for (int busIndex = 0; busIndex < numOutputBuses; ++busIndex) {
#if !JucePlugin_IsSynth
    auto outputBus = getBusBuffer(buffer, false /* isInput */, busIndex);
    const int outNumCh = outputBus.getNumChannels();
    const int channelsToCopy = std::min(kNumChannels, outNumCh);

    const bool carriesMixture = (busIndex == 0 || busIndex == 3);
    for (int ch = 0; ch < channelsToCopy; ++ch) {
      float* outPtr = outputBus.getWritePointer(ch);
      if (carriesMixture && inputChannelPtrs[ch] != nullptr) {
        for (int i = 0; i < numSamples; ++i) {
          const float sample = inputChannelPtrs[ch][i];
          outPtr[i] = std::isfinite(sample) ? sample : 0.0f;
        }
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
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  if (resetAfterCurrentCallback) {
    resetStreamingBuffersRT();
  }
#endif
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
