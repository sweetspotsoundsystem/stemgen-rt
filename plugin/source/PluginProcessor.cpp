#include "StemgenRT/PluginProcessor.h"
#include "StemgenRT/PluginEditor.h"
#include <algorithm>
#include <vector>
#include <cstdint>
#include <cmath>
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

  // Reset inference queue state (full reset: clears flags and indices)
  inferenceQueue_.fullReset();

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

  // Reset LR4 crossover filter states to avoid clicks from stale filter memory
  crossover_.reset();
  vocalsGate_.reset();

  // Reset inference queue - increments epoch and invalidates in-flight requests.
  inferenceQueue_.reset();
#endif
}

void AudioPluginAudioProcessor::prepareToPlay(double sampleRate,
                                              int samplesPerBlock) {
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  // Initialize LR4 crossover (splits input into LP + HP)
  crossover_.prepare(sampleRate);

  // Initialize vocals gate with sample rate
  vocalsGate_.prepare(sampleRate);

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
    // Reset streaming buffers to clean state for new playback session
    resetStreamingBuffers();

    // Start the background inference thread (no-op if already running)
    inferenceQueue_.startThread(onnxRuntime_.get());

    // Report latency to host for Plugin Delay Compensation (PDC)
    setLatencySamples(kOutputChunkSize);

    // Warm up ORT: queue a dummy inference to trigger lazy initialization
    // Use submitForWarmup() which doesn't advance indices, then reset() to
    // increment epoch and cause inference thread to reset its readIdx back to 0.
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

      // Wait for completion (blocking OK in prepareToPlay)
      // Check processed flag directly since we're doing a special warmup flow
      while (!warmup->processed.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      // Clear processed flag directly (don't use releaseOutputSlot which advances consumeIdx)
      warmup->processed.store(false, std::memory_order_release);

      // Increment epoch to reset inference thread's readIdx back to 0.
      // Warmup advanced inferenceReadIdx to 1, but writeIdx stayed at 0.
      // The epoch change causes inference thread to reset readIdx on next wakeup.
      inferenceQueue_.reset();
      DBG("[HS-TasNet] ORT warmup complete");
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
      if (!overlapAdd_.hasPendingChunk()) {
        InferenceRequest* outputSlot = inferenceQueue_.getOutputSlot(inferenceQueue_.getEpoch());
        if (!outputSlot) {
          break;  // No more results ready (stale ones are auto-discarded)
        }

        // Check ring buffer capacity before writing
        size_t avail = overlapAdd_.getOutputSamplesAvailable();
        if (avail + static_cast<size_t>(kOutputChunkSize) > outRingSize) {
          // Ring buffer would overflow - drop oldest samples to make room
          size_t overflow = (avail + static_cast<size_t>(kOutputChunkSize)) - outRingSize;
          overlapAdd_.setOutputReadPos((overlapAdd_.getOutputReadPos() + overflow) % outRingSize);
          overlapAdd_.setOutputSamplesAvailable(avail - overflow);
        }

        // We have a valid pending chunk to copy
        overlapAdd_.setHasPendingChunk(true);
        overlapAdd_.setPendingChunkOffset(0);
      }

      // Get the current output slot (already validated when we acquired the pending chunk)
      InferenceRequest* consumeRequest = inferenceQueue_.getCurrentOutputSlot();

      size_t remainingInChunk = static_cast<size_t>(kOutputChunkSize) - overlapAdd_.getPendingChunkOffset();
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
      constexpr float kEnergyEpsilon = ResidualProcessor::kResidualEpsilon;  // Avoid division by zero

      // Compute base write position once, then increment with branch
      size_t writePos = overlapAdd_.getOutputWritePos();
      const size_t srcBase = overlapAdd_.getPendingChunkOffset();

      // Get raw pointers for source data (avoid repeated operator[] on unique_ptr)
      const float* origL = consumeRequest->originalInput[0].data() + srcBase;
      const float* origR = consumeRequest->originalInput[1].data() + srcBase;
      const float* stemData[kNumStems][kNumChannels];
      for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
        stemData[stem][0] = consumeRequest->outputChunk[stem][0].data() + srcBase;
        stemData[stem][1] = consumeRequest->outputChunk[stem][1].data() + srcBase;
      }

      for (size_t i = 0; i < samplesToCopy; ++i) {
        // Input-following soft gate (eliminates model noise floor on quiet passages)
        float origLSample = origL[i];
        float origRSample = origR[i];
        float gateGain = SoftGate::calculateGain(origLSample, origRSample);

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
        // Detect spurious vocals content and transfer to "other" stem
        float vocalsEnergy = vocals_L * vocals_L + vocals_R * vocals_R;
        float totalStemEnergy = drums_L * drums_L + drums_R * drums_R +
                                bass_L * bass_L + bass_R * bass_R +
                                vocalsEnergy +
                                other_L * other_L + other_R * other_R + kEnergyEpsilon;
        float vocalsPeak = std::max(std::abs(vocals_L), std::abs(vocals_R));

        // Process through vocals gate (handles smoothing internally)
        float vocalsGateGain = vocalsGate_.process(vocalsEnergy, totalStemEnergy, vocalsPeak);

        // Apply vocals gate: transfer gated vocals to "other"
        float vocalsGated_L = vocals_L * vocalsGateGain;
        float vocalsGated_R = vocals_R * vocalsGateGain;
        float vocalsToOther_L = vocals_L - vocalsGated_L;
        float vocalsToOther_R = vocals_R - vocalsGated_R;
        // ===== end vocals gate =====

        // Store original samples in delayed input buffer
        delayedInputBuffer[0][writePos] = origLSample;
        delayedInputBuffer[1][writePos] = origRSample;

        // Process residual calculation and distribution using ResidualProcessor
        ResidualProcessor::StemSamples stemsL{drums_L, bass_L, vocals_L, other_L};
        ResidualProcessor::StemSamples stemsR{drums_R, bass_R, vocals_R, other_R};
        ResidualProcessor::StemSamples outL, outR;

        ResidualProcessor::processStereo(
            origLSample, origRSample,
            stemsL, stemsR,
            vocalsGated_L, vocalsGated_R,
            vocalsToOther_L, vocalsToOther_R,
            gateGain,
            outL, outR);

        // Write processed stems to output ring buffers
        outputRingBuffers[0][0][writePos] = outL.drums;
        outputRingBuffers[0][1][writePos] = outR.drums;
        outputRingBuffers[1][0][writePos] = outL.bass;
        outputRingBuffers[1][1][writePos] = outR.bass;
        outputRingBuffers[2][0][writePos] = outL.vocals;
        outputRingBuffers[2][1][writePos] = outR.vocals;
        outputRingBuffers[3][0][writePos] = outL.other;
        outputRingBuffers[3][1][writePos] = outR.other;
        
        // Advance write position with branch instead of modulo
        ++writePos;
        if (writePos == outRingSize) writePos = 0;
      }
      // ===== end optimized ring buffer write =====
      
      overlapAdd_.addOutputSamplesAvailable(samplesToCopy);
      overlapAdd_.setPendingChunkOffset(overlapAdd_.getPendingChunkOffset() + samplesToCopy);
      samplesToProcess -= samplesToCopy;

      // Check if chunk is fully copied
      if (overlapAdd_.getPendingChunkOffset() >= static_cast<size_t>(kOutputChunkSize)) {
        // Release the slot and move to next
        inferenceQueue_.releaseOutputSlot();
        overlapAdd_.setHasPendingChunk(false);
        overlapAdd_.setPendingChunkOffset(0);
      }
    }

    // Accumulate input samples until we have kOutputChunkSize
    // Strategy: Split input into LP + HP using LR4 crossover.
    //   - HP goes to model (model struggles with low frequencies in chunked inference)
    //   - LP is stored and added to bass stem after inference
    // This keeps the crossover coherent: LP + HP = original (LR4 property)
    for (int i = 0; i < numSamples; ++i) {
      for (int ch = 0; ch < kNumChannels; ++ch) {
#if !JucePlugin_IsSynth
        float sample = (inputChannelPtrs[ch] != nullptr) ? inputChannelPtrs[ch][i] : 0.0f;
#else
        float sample = 0.0f;
#endif
        // Process through LR4 crossover (splits into LP + HP)
        auto filtered = crossover_.processSample(ch, sample);

        // Push HP, LP, and dry samples to overlap-add processor
        // - HP goes to model (model struggles with low frequencies in chunked inference)
        // - LP is stored and added to bass stem after inference
        // - Dry is for underrun fallback (latency-aligned input)
        overlapAdd_.pushInputSample(ch, filtered.highPass, filtered.lowPass, sample);
      }

      // When we have enough samples, queue for inference
      if (overlapAdd_.readyForInference()) {
        // Get the next write slot (nullptr if queue is full)
        InferenceRequest* request = inferenceQueue_.getWriteSlot();

        if (request) {
          // Get references to accumulated buffers
          const auto& inputAccumBuffer = overlapAdd_.getInputAccumBuffer();
          const auto& lowFreqAccumBuffer = overlapAdd_.getLowFreqAccumBuffer();
          const auto& contextBuffer = overlapAdd_.getContextBuffer();

          // Calculate normalization gain from HP input
          const float* inputPtrs[kNumChannels] = {inputAccumBuffer[0].data(), inputAccumBuffer[1].data()};
          float normGain = InputNormalizer::calculateGain(inputPtrs, kNumChannels,
                                                          static_cast<size_t>(kOutputChunkSize));
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

          // Update context buffer for next chunk (stores HP-filtered samples, NOT normalized)
          overlapAdd_.updateContextBuffer();

          // Submit the request (handles epoch stamping and index advancement)
          inferenceQueue_.submitWriteSlot(inferenceQueue_.getEpoch());
        }
        // else: queue is full, drop this chunk (will cause audio dropout)

        overlapAdd_.clearInputAccum();
      }
    }

    // ===== Write separated stems to output buses =====
    const int numOutputBuses = getBusCount(false /* isInput */);

    // Bus 0 (Main): sum of all stems
    float* mainWrite[kNumChannels] = {nullptr, nullptr};
    int mainNumCh = 0;
    {
        auto mainBus = getBusBuffer(buffer, false /* isInput */, 0);
        mainNumCh = mainBus.getNumChannels();
        for (int ch = 0; ch < std::min(kNumChannels, mainNumCh); ++ch)
            mainWrite[ch] = mainBus.getWritePointer(ch);
    }

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
    outputWriter_.writeBlock(overlapAdd_, outputRingBuffers, outRingSize, numSamples);

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
