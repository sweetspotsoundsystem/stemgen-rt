#include "StemgenRT/PluginProcessor.h"
#include "StemgenRT/PluginEditor.h"
#include <algorithm>
#include <vector>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <cstring>
#include <limits>
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
  // Early initialization of ONNX Runtime environment (for accurate status display).
  // The full model loading and session creation happens in prepareToPlay().
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
  if (onnxRuntime_) {
    if (modelLoadError_.isNotEmpty() && !onnxRuntime_->isModelLoaded()) {
      return juce::String("Model error: ") + modelLoadError_;
    }
    return onnxRuntime_->getStatusString();
  }
  return "ONNX Runtime: not initialized";
#else
  return "ONNX Runtime: not linked";
#endif
}

int AudioPluginAudioProcessor::getLatencySamples() const {
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  if (!onnxRuntime_ || !onnxRuntime_->isModelLoaded()) {
    return 0;
  }

  return latencySamplesForHostRate_;
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
int AudioPluginAudioProcessor::computeLatencySamplesForRate(double sampleRate) const {
  if (sampleRate <= kModelSampleRate) {
    return kOutputChunkSize;
  }

  const double ratio = sampleRate / kModelSampleRate;
  const int baseChunkLatency =
      static_cast<int>(std::ceil(static_cast<double>(kOutputChunkSize) * ratio));

  // The linear SRC path is stream-based and introduces a fixed startup offset:
  // one host sample on downsample input priming + one model sample interval on
  // upsample output priming.
  const int srcPrimingLatency =
      1 + static_cast<int>(std::ceil(sampleRate / kModelSampleRate));

  return baseChunkLatency + srcPrimingLatency;
}

void AudioPluginAudioProcessor::resetSampleRateAdapters() {
  downsamplePhase_ = 0.0;
  downsampleHasPrev_ = false;
  downsamplePrevHp_.fill(0.0f);
  downsamplePrevLp_.fill(0.0f);
  downsamplePrevFullband_.fill(0.0f);

  upsamplePhase_ = 0.0;
  upsampleHasPrev_ = false;
  upsamplePrevOrig_.fill(0.0f);
  upsamplePrevFullband_.fill(0.0f);
  upsamplePrevLow_.fill(0.0f);
  for (auto& stem : upsamplePrevStems_) {
    stem.fill(0.0f);
  }

  modelInputAccumCount_ = 0;
  for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
    std::fill(modelInputAccumBuffer_[ch].begin(), modelInputAccumBuffer_[ch].end(), 0.0f);
    std::fill(modelLowFreqAccumBuffer_[ch].begin(), modelLowFreqAccumBuffer_[ch].end(), 0.0f);
    std::fill(modelFullbandAccumBuffer_[ch].begin(), modelFullbandAccumBuffer_[ch].end(), 0.0f);
    std::fill(modelContextBuffer_[ch].begin(), modelContextBuffer_[ch].end(), 0.0f);
  }
}
#endif

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

#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
void AudioPluginAudioProcessor::allocateStreamingBuffers() {
  // Allocate overlap-add processor buffers
  overlapAdd_.allocate();

  // Reset output writer (initializes crossfade state)
  outputWriter_.reset();

  // Allocate inference queue buffers
  inferenceQueue_.allocate();

  DBG("[HS-TasNet] Streaming buffers allocated:");
  DBG("  Context size: " << kContextSize << " samples");
  DBG("  Output chunk size: " << kOutputChunkSize << " samples");
  DBG("  Internal chunk size: " << kInternalChunkSize << " samples");
  DBG("  Inference queue size: " << kNumInferenceBuffers << " slots");
}

#endif

void AudioPluginAudioProcessor::resetStreamingBuffers() {
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  // Reset overlap-add processor (clears all buffers and indices)
  overlapAdd_.reset();

  // Reset output writer (crossfade state)
  outputWriter_.reset();
  lowBandStabilizer_.reset();

  // Reset inference queue state (full reset: clears flags and indices)
  inferenceQueue_.fullReset();

  // Reset chunk sequence tracking used by chunk-boundary crossfade continuity checks.
  nextInputChunkSequence_ = 0;
  lastOutputChunkSequence_ = 0;
  hasLastOutputChunkSequence_ = false;

  // Reset startup grace period counter
  outputChunksConsumed_.store(0, std::memory_order_release);

  resetSampleRateAdapters();

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
  //   3. We clear contextBuffer here (small: 2ch × 2048 floats = 16KB) to avoid stale context
  //   4. Epoch increment invalidates all in-flight inference results

  // Reset overlap-add indices (RT-safe: no memory clearing)
  overlapAdd_.resetIndices();

  // Clear context buffer to avoid audio artifacts from stale history.
  // This is ~16KB which takes <1µs on modern CPUs.
  overlapAdd_.clearContextBuffer();

  // Reset output writer (crossfade state)
  outputWriter_.reset();
  underrunActive_.store(false, std::memory_order_release);
  lastUnderrunSamplesInLastBlock_.store(0, std::memory_order_release);

  // Reset startup grace period counter
  outputChunksConsumed_.store(0, std::memory_order_release);

  // Reset LR4 crossover filter states to avoid clicks from stale filter memory
  crossover_.reset();
  vocalsGate_.reset();
  lowBandStabilizer_.reset();

  // Reset inference queue - increments epoch and invalidates in-flight requests.
  inferenceQueue_.reset();

  // Start a new contiguous sequence after transport resets.
  nextInputChunkSequence_ = 0;
  lastOutputChunkSequence_ = 0;
  hasLastOutputChunkSequence_ = false;

  resetSampleRateAdapters();
#endif
}

void AudioPluginAudioProcessor::prepareToPlay(double sampleRate,
                                              int samplesPerBlock) {
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  hostSampleRateHz_ = (sampleRate > 0.0) ? sampleRate : kModelSampleRate;
  downsampleToModelRate_ = hostSampleRateHz_ > (kModelSampleRate + 1.0e-6);
  downsampleStep_ = hostSampleRateHz_ / kModelSampleRate;
  upsampleStep_ = kModelSampleRate / hostSampleRateHz_;
  latencySamplesForHostRate_ = computeLatencySamplesForRate(hostSampleRateHz_);

  for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
    modelInputAccumBuffer_[ch].resize(static_cast<size_t>(kOutputChunkSize), 0.0f);
    modelLowFreqAccumBuffer_[ch].resize(static_cast<size_t>(kOutputChunkSize), 0.0f);
    modelFullbandAccumBuffer_[ch].resize(static_cast<size_t>(kOutputChunkSize), 0.0f);
    modelContextBuffer_[ch].resize(static_cast<size_t>(kContextSize), 0.0f);
  }
  resetSampleRateAdapters();

  // Initialize LR4 crossover (splits input into LP + HP)
  crossover_.prepare(hostSampleRateHz_);

  // Initialize vocals gate with sample rate
  vocalsGate_.prepare(hostSampleRateHz_);
  lowBandStabilizer_.prepare(hostSampleRateHz_);

  DBG("[HS-TasNet] LR4 crossover initialized at " << kCrossoverFreqHz << " Hz (HP to model, LP to bass)");

  // Warn about small buffer sizes that may cause real-time issues
  constexpr int kMinRecommendedBufferSize = 128;
  if (samplesPerBlock < kMinRecommendedBufferSize) {
    DBG("[HS-TasNet] WARNING: Buffer size " << samplesPerBlock << " samples is below recommended "
        << kMinRecommendedBufferSize << ". May cause audio dropouts.");
  }

  // Check if OnnxRuntime is initialized
  if (!onnxRuntime_ || !onnxRuntime_->isInitialized()) {
    DBG("[ORT] ONNX Runtime not available");
    return;
  }

  // Load the HS-TasNet model (only once)
  if (!onnxRuntime_->isModelLoaded()) {
    // Construct model path relative to the plugin binary
    juce::File pluginFile = juce::File::getSpecialLocation(juce::File::currentExecutableFile);
    juce::File modelFile = pluginFile.getParentDirectory().getParentDirectory()
                                     .getChildFile("Resources/model.onnx");

    DBG("[HS-TasNet] Checking model path: " << modelFile.getFullPathName());

    if (!modelFile.existsAsFile()) {
      modelLoadError_ = juce::String("Model not found: ") + modelFile.getFullPathName();
      DBG("[HS-TasNet] " << modelLoadError_);
      return;
    }

    // Load model via OnnxRuntime (handles GPU providers, threading, etc.)
    if (!onnxRuntime_->loadModel(modelFile.getFullPathName(), modelLoadError_)) {
      DBG("[HS-TasNet] Model load failed: " << modelLoadError_);
      return;
    }

    // Prepare for inference (allocates memory info and scratch buffer)
    onnxRuntime_->prepareForInference();

    // Allocate streaming buffers for overlap-add inference
    allocateStreamingBuffers();
  }

  // Always start the inference thread if model is loaded.
  // Hosts may call releaseResources() + prepareToPlay() cycles during transport changes.
  if (onnxRuntime_->isModelLoaded()) {
    // Align dry fallback with current host-rate latency before reset.
    overlapAdd_.setDryDelaySamples(static_cast<size_t>(latencySamplesForHostRate_));

    // Reset streaming buffers to clean state for new playback session
    resetStreamingBuffers();

    // Start the background inference thread (no-op if already running)
    inferenceQueue_.startThread(onnxRuntime_.get());

    // Report latency to host for Plugin Delay Compensation (PDC)
    setLatencySamples(latencySamplesForHostRate_);

    // Warm up ORT: queue a dummy inference to trigger lazy initialization.
    // Use submitForWarmup() which doesn't advance write index, then reset()
    // to invalidate the warmup slot and re-synchronize queue epoch state.
    if (auto* warmup = inferenceQueue_.getWriteSlot()) {
      for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
        std::memset(warmup->inputChunk[ch].data(), 0,
                    static_cast<size_t>(kOutputChunkSize) * sizeof(float));
        std::memset(warmup->contextSnapshot[ch].data(), 0,
                    static_cast<size_t>(kContextSize) * sizeof(float));
        std::memset(warmup->lowFreqChunk[ch].data(), 0,
                    static_cast<size_t>(kOutputChunkSize) * sizeof(float));
      }
      warmup->normalizationGain = 1.0f;
      inferenceQueue_.submitForWarmup();

      // Wait for completion (blocking is acceptable in prepareToPlay), but
      // never indefinitely in case inference thread is stalled.
      constexpr auto kWarmupTimeout = std::chrono::seconds(2);
      const auto deadline = std::chrono::steady_clock::now() + kWarmupTimeout;
      bool warmupCompleted = false;
      while (!(warmupCompleted =
                   warmup->processed.load(std::memory_order_acquire))) {
        if (std::chrono::steady_clock::now() >= deadline) {
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      if (warmupCompleted) {
        // Clear processed flag directly (don't use releaseOutputSlot which advances consumeIdx)
        warmup->processed.store(false, std::memory_order_release);
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
  if (onnxRuntime_ && onnxRuntime_->isModelLoaded()) {
    const size_t outRingSize = overlapAdd_.getOutputRingSize();
    auto& outputRingBuffers = overlapAdd_.getOutputRingBuffers();
    auto& delayedInputBuffer = overlapAdd_.getDelayedInputBuffer();

    // Consume inference results into the ring buffer up to a target fill level.
    // For host rates above 44.1kHz, consume-side writes use model->host
    // interpolation so output buses stay in host-rate time.
    const size_t targetRingFill = std::max(
        static_cast<size_t>(kOutputChunkSize),
        static_cast<size_t>(numSamples));
    const size_t currentRingAvail = overlapAdd_.getOutputSamplesAvailable();
    size_t samplesToProcess = (currentRingAvail < targetRingFill)
        ? (targetRingFill - currentRingAvail)
        : 0;
    uint64_t ringOverflowEventsThisBlock = 0;
    uint64_t ringOverflowSamplesDroppedThisBlock = 0;
    uint64_t queueFullDropsThisBlock = 0;

    constexpr float kVocalsGateEpsilon = 1e-8f;
    constexpr float kDrumsLpShare = 0.30f;
    constexpr float kBassLpShare = 0.70f;

    auto writeProcessedHostSample = [&](
        float origLSample, float origRSample,
        float fullbandLSample, float fullbandRSample,
        float lowLSample, float lowRSample,
        float drumsL, float drumsR,
        float bassL, float bassR,
        float vocalsL, float vocalsR,
        float otherL, float otherR) {
      // Input-following soft gate (eliminates model noise floor on quiet passages)
      float gateGain = SoftGate::calculateGain(origLSample, origRSample);

      // Reinject LP bypass after chunk boundary crossfade so low-band remains continuous.
      drumsL += lowLSample * kDrumsLpShare;
      drumsR += lowRSample * kDrumsLpShare;
      bassL += lowLSample * kBassLpShare;
      bassR += lowRSample * kBassLpShare;

      // Detect spurious vocals content and transfer to "other" stem.
      float vocalsEnergy = vocalsL * vocalsL + vocalsR * vocalsR;
      float totalStemEnergy = drumsL * drumsL + drumsR * drumsR +
                              bassL * bassL + bassR * bassR +
                              vocalsEnergy +
                              otherL * otherL + otherR * otherR + kVocalsGateEpsilon;
      float vocalsPeak = std::max(std::abs(vocalsL), std::abs(vocalsR));
      float vocalsGateGain = vocalsGate_.process(vocalsEnergy, totalStemEnergy, vocalsPeak);
      float vocalsGatedL = vocalsL * vocalsGateGain;
      float vocalsGatedR = vocalsR * vocalsGateGain;
      float vocalsToOtherL = vocalsL - vocalsGatedL;
      float vocalsToOtherR = vocalsR - vocalsGatedR;

      StemPostProcessor::StemSamples stemsL{drumsL, bassL, vocalsL, otherL};
      StemPostProcessor::StemSamples stemsR{drumsR, bassR, vocalsR, otherR};
      StemPostProcessor::StemSamples outL, outR;
      StemPostProcessor::processStereo(
          stemsL, stemsR,
          vocalsGatedL, vocalsGatedR,
          vocalsToOtherL, vocalsToOtherR,
          gateGain,
          outL, outR);

      // Re-stabilize low-band distribution and suppress low-passed buzz leakage.
      lowBandStabilizer_.processStereo(origLSample, origRSample, outL, outR);

      // Keep ring bounded by dropping the oldest sample if needed.
      const size_t avail = overlapAdd_.getOutputSamplesAvailable();
      if (avail >= outRingSize) {
        overlapAdd_.setOutputReadPos((overlapAdd_.getOutputReadPos() + 1) % outRingSize);
        overlapAdd_.setOutputSamplesAvailable(avail - 1);
        ++ringOverflowEventsThisBlock;
        ++ringOverflowSamplesDroppedThisBlock;
      }

      const size_t writePos = overlapAdd_.getOutputWritePos();
      delayedInputBuffer[0][writePos] = fullbandLSample;
      delayedInputBuffer[1][writePos] = fullbandRSample;
      outputRingBuffers[0][0][writePos] = outL.drums;
      outputRingBuffers[0][1][writePos] = outR.drums;
      outputRingBuffers[1][0][writePos] = outL.bass;
      outputRingBuffers[1][1][writePos] = outR.bass;
      outputRingBuffers[2][0][writePos] = outL.vocals;
      outputRingBuffers[2][1][writePos] = outR.vocals;
      outputRingBuffers[3][0][writePos] = outL.other;
      outputRingBuffers[3][1][writePos] = outR.other;
      overlapAdd_.addOutputSamplesAvailable(1);
    };

    auto acquirePendingChunk = [&]() -> bool {
      InferenceRequest* outputSlot = inferenceQueue_.getOutputSlot(inferenceQueue_.getEpoch());
      if (!outputSlot) {
        return false;
      }

      const bool contiguousWithPreviousChunk =
          hasLastOutputChunkSequence_ &&
          outputSlot->chunkSequence == (lastOutputChunkSequence_ + 1);
      if (!contiguousWithPreviousChunk) {
        overlapAdd_.setHasPrevOverlapTail(false);
      }

      if (overlapAdd_.hasPrevOverlapTail()) {
        auto& prevTail = overlapAdd_.getPrevOverlapTail();
        constexpr float kHalfPi = 1.57079632679f;
        constexpr float kInvCrossfadeDenom =
            (kCrossfadeSamples > 1)
                ? (1.0f / static_cast<float>(kCrossfadeSamples - 1))
                : 1.0f;

        for (size_t j = 0; j < static_cast<size_t>(kCrossfadeSamples); ++j) {
          const float t = static_cast<float>(j) * kInvCrossfadeDenom;
          const float prevGain = std::cos(kHalfPi * t);
          const float currGain = std::sin(kHalfPi * t);
          for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
            for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
              outputSlot->outputChunk[stem][ch][j] =
                  prevTail[stem][ch][j] * prevGain +
                  outputSlot->outputChunk[stem][ch][j] * currGain;
            }
          }
        }
      }

      auto& prevTail = overlapAdd_.getPrevOverlapTail();
      for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
        for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
          std::memcpy(prevTail[stem][ch].data(),
                      outputSlot->overlapTail[stem][ch].data(),
                      static_cast<size_t>(kCrossfadeSamples) * sizeof(float));
        }
      }
      overlapAdd_.setHasPrevOverlapTail(true);
      lastOutputChunkSequence_ = outputSlot->chunkSequence;
      hasLastOutputChunkSequence_ = true;
      overlapAdd_.setHasPendingChunk(true);
      overlapAdd_.setPendingChunkOffset(0);
      return true;
    };

    if (downsampleToModelRate_) {
      while (samplesToProcess > 0) {
        if (!overlapAdd_.hasPendingChunk()) {
          if (!acquirePendingChunk()) {
            break;
          }
        }

        InferenceRequest* consumeRequest = inferenceQueue_.getCurrentOutputSlot();
        if (!consumeRequest) {
          overlapAdd_.setHasPendingChunk(false);
          break;
        }

        while (overlapAdd_.getPendingChunkOffset() < static_cast<size_t>(kOutputChunkSize) &&
               samplesToProcess > 0) {
          const size_t srcIndex = overlapAdd_.getPendingChunkOffset();

          std::array<float, kNumChannels> currOrig{
              consumeRequest->originalInput[0][srcIndex],
              consumeRequest->originalInput[1][srcIndex]};
          std::array<float, kNumChannels> currFullband{
              consumeRequest->fullbandInput[0][srcIndex],
              consumeRequest->fullbandInput[1][srcIndex]};
          std::array<float, kNumChannels> currLow{
              consumeRequest->lowFreqChunk[0][srcIndex],
              consumeRequest->lowFreqChunk[1][srcIndex]};
          std::array<std::array<float, kNumChannels>, kNumStems> currStems{};
          for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
            for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
              currStems[stem][ch] = consumeRequest->outputChunk[stem][ch][srcIndex];
            }
          }

          if (!upsampleHasPrev_) {
            upsamplePrevOrig_ = currOrig;
            upsamplePrevFullband_ = currFullband;
            upsamplePrevLow_ = currLow;
            upsamplePrevStems_ = currStems;
            upsampleHasPrev_ = true;
          } else {
            while (upsamplePhase_ <= 1.0) {
              const float t = static_cast<float>(upsamplePhase_);
              const float origL = upsamplePrevOrig_[0] +
                                  t * (currOrig[0] - upsamplePrevOrig_[0]);
              const float origR = upsamplePrevOrig_[1] +
                                  t * (currOrig[1] - upsamplePrevOrig_[1]);
              const float fullbandL = upsamplePrevFullband_[0] +
                                      t * (currFullband[0] - upsamplePrevFullband_[0]);
              const float fullbandR = upsamplePrevFullband_[1] +
                                      t * (currFullband[1] - upsamplePrevFullband_[1]);
              const float lowL = upsamplePrevLow_[0] +
                                 t * (currLow[0] - upsamplePrevLow_[0]);
              const float lowR = upsamplePrevLow_[1] +
                                 t * (currLow[1] - upsamplePrevLow_[1]);

              float stemsL[kNumStems];
              float stemsR[kNumStems];
              for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
                stemsL[stem] = upsamplePrevStems_[stem][0] +
                               t * (currStems[stem][0] - upsamplePrevStems_[stem][0]);
                stemsR[stem] = upsamplePrevStems_[stem][1] +
                               t * (currStems[stem][1] - upsamplePrevStems_[stem][1]);
              }

              writeProcessedHostSample(
                  origL, origR,
                  fullbandL, fullbandR,
                  lowL, lowR,
                  stemsL[0], stemsR[0],
                  stemsL[1], stemsR[1],
                  stemsL[2], stemsR[2],
                  stemsL[3], stemsR[3]);

              if (samplesToProcess > 0) {
                --samplesToProcess;
              }
              upsamplePhase_ += upsampleStep_;
            }

            upsamplePhase_ -= 1.0;
            upsamplePrevOrig_ = currOrig;
            upsamplePrevFullband_ = currFullband;
            upsamplePrevLow_ = currLow;
            upsamplePrevStems_ = currStems;
          }

          overlapAdd_.setPendingChunkOffset(srcIndex + 1);
        }

        if (overlapAdd_.getPendingChunkOffset() >= static_cast<size_t>(kOutputChunkSize)) {
          inferenceQueue_.releaseOutputSlot();
          outputChunksConsumed_.fetch_add(1, std::memory_order_relaxed);
          overlapAdd_.setHasPendingChunk(false);
          overlapAdd_.setPendingChunkOffset(0);
        }
      }

      for (int i = 0; i < numSamples; ++i) {
        std::array<float, kNumChannels> hp{};
        std::array<float, kNumChannels> lp{};
        std::array<float, kNumChannels> fullband{};

        for (int ch = 0; ch < kNumChannels; ++ch) {
#if !JucePlugin_IsSynth
          float sample = (inputChannelPtrs[ch] != nullptr) ? inputChannelPtrs[ch][i] : 0.0f;
#else
          float sample = 0.0f;
#endif
          auto filtered = crossover_.processSample(ch, sample);
          hp[static_cast<size_t>(ch)] = filtered.highPass;
          lp[static_cast<size_t>(ch)] = filtered.lowPass;
          fullband[static_cast<size_t>(ch)] = sample;
          overlapAdd_.pushInputSample(ch, filtered.highPass, filtered.lowPass, sample);
        }

        if (!downsampleHasPrev_) {
          downsamplePrevHp_ = hp;
          downsamplePrevLp_ = lp;
          downsamplePrevFullband_ = fullband;
          downsampleHasPrev_ = true;
          continue;
        }

        while (downsamplePhase_ <= 1.0) {
          if (modelInputAccumCount_ >= static_cast<size_t>(kOutputChunkSize)) {
            break;
          }

          const float t = static_cast<float>(downsamplePhase_);
          for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
            modelInputAccumBuffer_[ch][modelInputAccumCount_] =
                downsamplePrevHp_[ch] + t * (hp[ch] - downsamplePrevHp_[ch]);
            modelLowFreqAccumBuffer_[ch][modelInputAccumCount_] =
                downsamplePrevLp_[ch] + t * (lp[ch] - downsamplePrevLp_[ch]);
            modelFullbandAccumBuffer_[ch][modelInputAccumCount_] =
                downsamplePrevFullband_[ch] + t * (fullband[ch] - downsamplePrevFullband_[ch]);
          }

          ++modelInputAccumCount_;
          downsamplePhase_ += downsampleStep_;

          if (modelInputAccumCount_ >= static_cast<size_t>(kOutputChunkSize)) {
            const uint64_t chunkSequence = nextInputChunkSequence_++;
            InferenceRequest* request = inferenceQueue_.getWriteSlot();
            if (request) {
              request->chunkSequence = chunkSequence;
              const float normGain = InputNormalizer::calculateGainFromContextAndInput(
                  modelContextBuffer_, modelInputAccumBuffer_);
              request->normalizationGain = normGain;

              for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
                for (size_t j = 0; j < static_cast<size_t>(kOutputChunkSize); ++j) {
                  request->inputChunk[ch][j] = modelInputAccumBuffer_[ch][j] * normGain;
                  request->originalInput[ch][j] =
                      modelInputAccumBuffer_[ch][j] + modelLowFreqAccumBuffer_[ch][j];
                }
                for (size_t j = 0; j < static_cast<size_t>(kContextSize); ++j) {
                  request->contextSnapshot[ch][j] = modelContextBuffer_[ch][j] * normGain;
                }
                std::memcpy(request->lowFreqChunk[ch].data(), modelLowFreqAccumBuffer_[ch].data(),
                            static_cast<size_t>(kOutputChunkSize) * sizeof(float));
                std::memcpy(request->fullbandInput[ch].data(), modelFullbandAccumBuffer_[ch].data(),
                            static_cast<size_t>(kOutputChunkSize) * sizeof(float));
              }

              const size_t samplesToKeep =
                  static_cast<size_t>(kContextSize - kOutputChunkSize);
              for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
                std::memmove(modelContextBuffer_[ch].data(),
                             modelContextBuffer_[ch].data() + static_cast<size_t>(kOutputChunkSize),
                             samplesToKeep * sizeof(float));
                std::memcpy(modelContextBuffer_[ch].data() + samplesToKeep,
                            modelInputAccumBuffer_[ch].data(),
                            static_cast<size_t>(kOutputChunkSize) * sizeof(float));
              }

              inferenceQueue_.submitWriteSlot(inferenceQueue_.getEpoch());
            } else {
              ++queueFullDropsThisBlock;
#if JUCE_DEBUG
              DBG("[HS-TasNet] Queue full, dropping chunk seq=" << chunkSequence
                  << " ringAvail=" << overlapAdd_.getOutputSamplesAvailable());
#endif
            }

            modelInputAccumCount_ = 0;
          }
        }

        downsamplePhase_ -= 1.0;
        downsamplePrevHp_ = hp;
        downsamplePrevLp_ = lp;
        downsamplePrevFullband_ = fullband;
      }
    } else {
      while (samplesToProcess > 0) {
        if (!overlapAdd_.hasPendingChunk()) {
          if (!acquirePendingChunk()) {
            break;
          }
        }

        InferenceRequest* consumeRequest = inferenceQueue_.getCurrentOutputSlot();
        if (!consumeRequest) {
          overlapAdd_.setHasPendingChunk(false);
          break;
        }

        const size_t srcBase = overlapAdd_.getPendingChunkOffset();
        const size_t remainingInChunk =
            static_cast<size_t>(kOutputChunkSize) - srcBase;
        const size_t samplesToCopy = std::min(samplesToProcess, remainingInChunk);

        const float* origL = consumeRequest->originalInput[0].data() + srcBase;
        const float* origR = consumeRequest->originalInput[1].data() + srcBase;
        const float* fullbandL = consumeRequest->fullbandInput[0].data() + srcBase;
        const float* fullbandR = consumeRequest->fullbandInput[1].data() + srcBase;
        const float* lowL = consumeRequest->lowFreqChunk[0].data() + srcBase;
        const float* lowR = consumeRequest->lowFreqChunk[1].data() + srcBase;
        const float* stemData[kNumStems][kNumChannels];
        for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
          stemData[stem][0] = consumeRequest->outputChunk[stem][0].data() + srcBase;
          stemData[stem][1] = consumeRequest->outputChunk[stem][1].data() + srcBase;
        }

        for (size_t i = 0; i < samplesToCopy; ++i) {
          writeProcessedHostSample(
              origL[i], origR[i],
              fullbandL[i], fullbandR[i],
              lowL[i], lowR[i],
              stemData[0][0][i], stemData[0][1][i],
              stemData[1][0][i], stemData[1][1][i],
              stemData[2][0][i], stemData[2][1][i],
              stemData[3][0][i], stemData[3][1][i]);
        }

        overlapAdd_.setPendingChunkOffset(srcBase + samplesToCopy);
        samplesToProcess -= samplesToCopy;

        if (overlapAdd_.getPendingChunkOffset() >= static_cast<size_t>(kOutputChunkSize)) {
          inferenceQueue_.releaseOutputSlot();
          outputChunksConsumed_.fetch_add(1, std::memory_order_relaxed);
          overlapAdd_.setHasPendingChunk(false);
          overlapAdd_.setPendingChunkOffset(0);
        }
      }

      // Accumulate input samples until we have kOutputChunkSize
      // Split input into HP + LP. Feed HP to the model and carry LP separately
      // for low-band reconstruction after inference.
      for (int i = 0; i < numSamples; ++i) {
        for (int ch = 0; ch < kNumChannels; ++ch) {
#if !JucePlugin_IsSynth
          float sample = (inputChannelPtrs[ch] != nullptr) ? inputChannelPtrs[ch][i] : 0.0f;
#else
          float sample = 0.0f;
#endif
          auto filtered = crossover_.processSample(ch, sample);
          overlapAdd_.pushInputSample(ch, filtered.highPass, filtered.lowPass, sample);
        }

        if (overlapAdd_.readyForInference()) {
          const uint64_t chunkSequence = nextInputChunkSequence_++;
          InferenceRequest* request = inferenceQueue_.getWriteSlot();

          if (request) {
            request->chunkSequence = chunkSequence;
            const auto& inputAccumBuffer = overlapAdd_.getInputAccumBuffer();
            const auto& lowFreqAccumBuffer = overlapAdd_.getLowFreqAccumBuffer();
            const auto& contextBuffer = overlapAdd_.getContextBuffer();
            float normGain = InputNormalizer::calculateGainFromContextAndInput(
                contextBuffer, inputAccumBuffer);
            request->normalizationGain = normGain;

            const auto& fullbandAccumBuffer = overlapAdd_.getFullbandAccumBuffer();
            for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
              for (size_t j = 0; j < static_cast<size_t>(kOutputChunkSize); ++j) {
                request->inputChunk[ch][j] = inputAccumBuffer[ch][j] * normGain;
                request->originalInput[ch][j] = inputAccumBuffer[ch][j] + lowFreqAccumBuffer[ch][j];
              }
              for (size_t j = 0; j < static_cast<size_t>(kContextSize); ++j) {
                request->contextSnapshot[ch][j] = contextBuffer[ch][j] * normGain;
              }
              std::memcpy(request->lowFreqChunk[ch].data(), lowFreqAccumBuffer[ch].data(),
                          static_cast<size_t>(kOutputChunkSize) * sizeof(float));
              std::memcpy(request->fullbandInput[ch].data(), fullbandAccumBuffer[ch].data(),
                          static_cast<size_t>(kOutputChunkSize) * sizeof(float));
            }

            overlapAdd_.updateContextBuffer();
            inferenceQueue_.submitWriteSlot(inferenceQueue_.getEpoch());
          } else {
            ++queueFullDropsThisBlock;
#if JUCE_DEBUG
            DBG("[HS-TasNet] Queue full, dropping chunk seq=" << chunkSequence
                << " ringAvail=" << overlapAdd_.getOutputSamplesAvailable());
#endif
          }

          overlapAdd_.clearInputAccum();
        }
      }
    }

    // Prime dry delay line before main output takes a sample so the first block
    // isn't silent due to the initial zeroed buffer.
    if (!overlapAdd_.isDryDelayPrimed()) {
      overlapAdd_.primeDryDelayFromInput(inputChannelPtrs, numSamples);
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
    float* stemWrite[4][kNumChannels] = {{nullptr, nullptr}, {nullptr, nullptr},
                                         {nullptr, nullptr}, {nullptr, nullptr}};
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
    const auto writeStats =
        outputWriter_.writeBlock(overlapAdd_, outputRingBuffers, delayedInputBuffer, outRingSize, numSamples);

    // Grace period: don't count underruns until the first inference result has
    // been consumed. Before that, the pipeline is still filling and underruns
    // are expected (not a performance problem).
    const bool pastGracePeriod =
        outputChunksConsumed_.load(std::memory_order_relaxed) > 0;

    // Report ring fill AFTER the read — this reflects actual excess buffering
    // beyond PDC. Samples consumed in the same block don't add latency.
    ringFillLevel_.store(overlapAdd_.getOutputSamplesAvailable(), std::memory_order_release);
    if (ringOverflowEventsThisBlock > 0) {
      totalRingOverflowEvents_.fetch_add(ringOverflowEventsThisBlock, std::memory_order_relaxed);
      totalRingOverflowSamplesDropped_.fetch_add(ringOverflowSamplesDroppedThisBlock,
                                                 std::memory_order_relaxed);
    }
    if (queueFullDropsThisBlock > 0) {
      totalQueueFullChunkDrops_.fetch_add(queueFullDropsThisBlock, std::memory_order_relaxed);
    }

    if (pastGracePeriod) {
      underrunActive_.store(writeStats.isUnderrunNow,
                            std::memory_order_release);
      lastUnderrunSamplesInLastBlock_.store(writeStats.underrunSamples,
                                            std::memory_order_release);
      if (writeStats.hadUnderrun) {
        totalUnderrunBlocks_.fetch_add(1, std::memory_order_acq_rel);
        totalUnderrunSamples_.fetch_add(writeStats.underrunSamples,
                                       std::memory_order_acq_rel);
      }
    }

#if JUCE_DEBUG
    if (writeStats.underrunTransition) {
      DBG("[HS-TasNet] Underrun transition: ringAvail=" << writeStats.ringAvailAtStart
          << " xfadeGain=" << writeStats.crossfadeGainAtStart
          << " gracePeriod=" << (pastGracePeriod ? "no" : "yes"));
    }
#endif

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
