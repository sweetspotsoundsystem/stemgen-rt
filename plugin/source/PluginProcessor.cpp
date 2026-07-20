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
  // A partial final hop plus the graph's required zero-hop flush can require
  // up to two chunks before the final separated samples emerge.
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
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  if (onnxRuntime_) {
    if (!sampleRateSupported_) {
      return "Model requires a 44.1 kHz host sample rate";
    }
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
  if (!sampleRateSupported_ || !onnxRuntime_ ||
      !onnxRuntime_->isModelLoaded()) {
    return 0;
  }
  
  // One hop is needed to accumulate/queue audio and the stateful graph emits
  // the preceding hop. This fixed value is reported for host PDC.
  return kPluginLatencySamples;
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
void AudioPluginAudioProcessor::allocateStreamingBuffers(
    int maximumHostBlockSize) {
  overlapAdd_.allocate(static_cast<size_t>(
      std::max(maximumHostBlockSize, kOutputChunkSize)));

  // Reset output writer (initializes crossfade state)
  outputWriter_.reset();

  // Allocate inference queue buffers
  inferenceQueue_.allocate();

  DBG("[HS-TasNet] Streaming buffers allocated:");
  DBG("  Model chunk size: " << kOutputChunkSize << " samples");
  DBG("  Model analysis window: " << kAnalysisWindowSize << " samples");
  DBG("  Reported PDC: " << kPluginLatencySamples << " samples");
  DBG("  Inference queue size: " << kNumInferenceBuffers << " slots");
}

#endif

void AudioPluginAudioProcessor::resetStreamingBuffers() {
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  // Reset overlap-add processor (clears all buffers and indices)
  overlapAdd_.reset();

  // Reset output writer (crossfade state)
  outputWriter_.reset();
  // Reset inference queue state (full reset: clears flags and indices)
  inferenceQueue_.fullReset();
  if (onnxRuntime_) {
    onnxRuntime_->resetStreamingState();
  }

  // Reset input sequence tracking used for recurrent-state gap detection.
  nextInputChunkSequence_ = 0;

  // Reset startup grace period counter
  outputChunksConsumed_.store(0, std::memory_order_release);

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

  // Reset output writer (crossfade state)
  outputWriter_.reset();
  underrunActive_.store(false, std::memory_order_release);
  lastUnderrunSamplesInLastBlock_.store(0, std::memory_order_release);

  // Reset startup grace period counter
  outputChunksConsumed_.store(0, std::memory_order_release);

  // Reset inference queue - increments epoch and invalidates in-flight requests.
  inferenceQueue_.reset();

  // Start a new contiguous input sequence after transport resets.
  nextInputChunkSequence_ = 0;
#endif
}

void AudioPluginAudioProcessor::prepareToPlay(double sampleRate,
                                              int samplesPerBlock) {
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  inferenceQueue_.stopThread();
  sampleRateSupported_ =
      std::abs(sampleRate - static_cast<double>(kModelSampleRate)) < 0.5;
  if (!sampleRateSupported_) {
    modelLoadError_ = juce::String("Unsupported sample rate ")
                      + juce::String(sampleRate, 1)
                      + " Hz; this model requires 44100 Hz";
    setLatencySamples(0);
    DBG("[HS-TasNet] " << modelLoadError_);
    return;
  }
  modelLoadError_.clear();

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

    // Load the qualified model into the CPU ONNX Runtime session.
    if (!onnxRuntime_->loadModel(modelFile.getFullPathName(), modelLoadError_)) {
      DBG("[HS-TasNet] Model load failed: " << modelLoadError_);
      return;
    }

  }

  // Always recreate bounded buffers and state for the host's current maximum
  // block size, then start the inference thread.
  // Hosts may call releaseResources() + prepareToPlay() cycles during transport changes.
  if (onnxRuntime_->isModelLoaded()) {
    onnxRuntime_->prepareForInference();
    allocateStreamingBuffers(samplesPerBlock);

    // Reset streaming buffers to clean state for new playback session
    resetStreamingBuffers();

    // Start the background inference thread (no-op if already running)
    inferenceQueue_.startThread(onnxRuntime_.get());

    // Report latency to host for Plugin Delay Compensation (PDC)
    setLatencySamples(kPluginLatencySamples);

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
      while (!warmup->processed.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() >= deadline) {
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      const bool warmupCompleted =
          warmup->processed.load(std::memory_order_acquire);

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

      bool transportDiscontinuity = isPlaying && !wasPlayingBefore;
      if (isPlaying) {
        if (const auto timeInSamples = posInfo->getTimeInSamples()) {
          if (wasPlayingBefore && hasExpectedPlayheadPosition_ &&
              *timeInSamples != expectedPlayheadPosition_) {
            transportDiscontinuity = true;
          }
          expectedPlayheadPosition_ = *timeInSamples + numSamples;
          hasExpectedPlayheadPosition_ = true;
        } else {
          hasExpectedPlayheadPosition_ = false;
        }
      } else {
        hasExpectedPlayheadPosition_ = false;
      }

      // Reset on starts, seeks, scrubs, and loop wraps. The queue epoch makes
      // the recurrent-state reset happen safely on the inference worker.
      if (transportDiscontinuity) {
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
  if (sampleRateSupported_ && onnxRuntime_ &&
      onnxRuntime_->isModelLoaded()) {
    const size_t outRingSize = overlapAdd_.getOutputRingSize();
    auto& outputRingBuffers = overlapAdd_.getOutputRingBuffers();
    auto& delayedInputBuffer = overlapAdd_.getDelayedInputBuffer();

    // Consume inference results into the ring buffer up to a target fill level.
    // Unlike consuming ALL results, this limits the ring to just enough for the
    // current host block read. Excess results stay in the inference queue (16 slots)
    // and are consumed in subsequent blocks. This avoids:
    //   - Persistent ring buildup from startup bursts (was causing ~60ms latency)
    //   - Cascading underruns from discarding ring data (cap approach)
    //
    // The algorithm:
    //   1. If no pending chunk, check if a processed result is ready
    //   2. Copy from the pending chunk to the ring buffer
    //   3. When chunk is fully copied, move to next processed result
    //   4. Stop when ring reaches target fill level

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

    while (samplesToProcess > 0) {
      // If no pending chunk, try to acquire one
      if (!overlapAdd_.hasPendingChunk()) {
        InferenceRequest* outputSlot = inferenceQueue_.getOutputSlot(inferenceQueue_.getEpoch());
        if (!outputSlot) {
          break;  // No more results ready (stale ones are auto-discarded)
        }

        // Every reset produces one pre-roll marker. Consume it without adding
        // samples so the next output remains aligned with the first real hop.
        if (!outputSlot->outputValid) {
          inferenceQueue_.releaseOutputSlot();
          continue;
        }

        // Check ring buffer capacity before writing
        size_t avail = overlapAdd_.getOutputSamplesAvailable();
        if (avail + static_cast<size_t>(kOutputChunkSize) > outRingSize) {
          // Ring buffer would overflow - drop oldest samples to make room
          size_t overflow = (avail + static_cast<size_t>(kOutputChunkSize)) - outRingSize;
          overlapAdd_.setOutputReadPos((overlapAdd_.getOutputReadPos() + overflow) % outRingSize);
          overlapAdd_.setOutputSamplesAvailable(avail - overflow);
          ++ringOverflowEventsThisBlock;
          ringOverflowSamplesDroppedThisBlock += static_cast<uint64_t>(overflow);
#if JUCE_DEBUG
          DBG("[HS-TasNet] Output ring overflow: dropped " << overflow
              << " oldest samples (ringAvail=" << avail
              << ", ringSize=" << outRingSize << ")");
#endif
        }

        // We have a valid pending chunk to copy
        overlapAdd_.setHasPendingChunk(true);
        overlapAdd_.setPendingChunkOffset(0);
      }

      // Get the current output slot (already validated when we acquired the pending chunk)
      InferenceRequest* consumeRequest = inferenceQueue_.getCurrentOutputSlot();

      size_t remainingInChunk = static_cast<size_t>(kOutputChunkSize) - overlapAdd_.getPendingChunkOffset();
      size_t samplesToCopy = std::min(samplesToProcess, remainingInChunk);
      
      // Copy the qualified raw model output and its aligned mixture reference.
      // The graph owns overlap-add; no external context or boundary crossfade
      // is applied here.
      size_t writePos = overlapAdd_.getOutputWritePos();
      const size_t srcBase = overlapAdd_.getPendingChunkOffset();

      const float* stemData[kNumStems][kNumChannels];
      for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
        stemData[stem][0] = consumeRequest->outputChunk[stem][0].data() + srcBase;
        stemData[stem][1] = consumeRequest->outputChunk[stem][1].data() + srcBase;
      }

      for (size_t i = 0; i < samplesToCopy; ++i) {
        for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
          delayedInputBuffer[ch][writePos] =
              consumeRequest->alignedInput[ch][srcBase + i];
          for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
            outputRingBuffers[stem][ch][writePos] = stemData[stem][ch][i];
          }
        }
        
        // Advance write position with branch instead of modulo
        ++writePos;
        if (writePos == outRingSize) writePos = 0;
      }
      overlapAdd_.addOutputSamplesAvailable(samplesToCopy);
      overlapAdd_.setPendingChunkOffset(overlapAdd_.getPendingChunkOffset() + samplesToCopy);
      samplesToProcess -= samplesToCopy;

      // Check if chunk is fully copied
      if (overlapAdd_.getPendingChunkOffset() >= static_cast<size_t>(kOutputChunkSize)) {
        // Release the slot and move to next
        inferenceQueue_.releaseOutputSlot();
        outputChunksConsumed_.fetch_add(1, std::memory_order_relaxed);
        overlapAdd_.setHasPendingChunk(false);
        overlapAdd_.setPendingChunkOffset(0);
      }
    }

    // Accumulate raw fullband samples exactly as used by the qualified c91
    // validation path.
    for (int i = 0; i < numSamples; ++i) {
      for (int ch = 0; ch < kNumChannels; ++ch) {
#if !JucePlugin_IsSynth
        float sample = (inputChannelPtrs[ch] != nullptr) ? inputChannelPtrs[ch][i] : 0.0f;
#else
        float sample = 0.0f;
#endif
        overlapAdd_.pushInputSample(ch, sample);
      }

      // When we have enough samples, queue for inference
      if (overlapAdd_.readyForInference()) {
        const uint64_t chunkSequence = nextInputChunkSequence_++;

        // Get the next write slot (nullptr if queue is full)
        InferenceRequest* request = inferenceQueue_.getWriteSlot();

        if (request) {
          request->chunkSequence = chunkSequence;

          const auto& inputAccumBuffer = overlapAdd_.getInputAccumBuffer();
          for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
            std::memcpy(request->inputChunk[ch].data(), inputAccumBuffer[ch].data(),
                        static_cast<size_t>(kOutputChunkSize) * sizeof(float));
          }

          // Submit the request (handles epoch stamping and index advancement)
          inferenceQueue_.submitWriteSlot(inferenceQueue_.getEpoch());
        }
        else {
          // Queue is full, so the worker will detect the sequence gap and reset
          // recurrent/OLA state before processing the next accepted chunk.
          ++queueFullDropsThisBlock;
#if JUCE_DEBUG
          DBG("[HS-TasNet] Queue full, dropping chunk seq=" << chunkSequence
              << " ringAvail=" << overlapAdd_.getOutputSamplesAvailable());
#endif
        }

        overlapAdd_.clearInputAccum();
      }
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
