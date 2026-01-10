#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace audio_plugin {

// HS-TasNet model constants
constexpr int kNumStems = 4;             // drums, bass, other, vocals
constexpr int kNumChannels = 2;          // stereo

// Overlap-add streaming constants for real-time inference
// Latency: kOutputChunkSize / 44100 * 1000 = ~11.6ms at 44.1kHz
// Inference runs in a background thread to avoid blocking the audio callback
constexpr int kOutputChunkSize = 512;   // New samples per streaming frame (~11.6ms at 44.1kHz)
constexpr int kContextSize = 1024;       // Context on each side to avoid edge artifacts
constexpr int kInternalChunkSize = kContextSize + kOutputChunkSize + kContextSize;  // 2560 samples

// Low-frequency crossover constants
// Low frequencies are poorly captured by chunk-based inference (e.g., 100 Hz = ~20ms period = 882 samples at 44.1kHz).
// We use a Linkwitz-Riley 4th order (LR4) crossover to split the mixture before inference:
//   1. Split input into LP + HP using LR4 (sums flat: LP + HP = original with phase shift only)
//   2. Feed only HP to the model (avoids sub-bass artifacts from chunked processing)
//   3. Add LP to bass stem (and optionally drums) to reconstruct full-spectrum output
// This ensures the crossover is coherent: both LP and HP are derived from the same signal.
constexpr float kCrossoverFreqHz = 80.0f;  // Crossover frequency in Hz

// Input-following soft gate constants
// Neural network models have a noise floor - they output small values even for silent input.
// Worse, these errors are often correlated across stems such that sum(stems)≈0, but each
// individual stem has noise. This gate attenuates stem outputs when input is very quiet,
// eliminating audible noise when soloing stems on silent passages.
// Conservative thresholds to avoid affecting quiet musical content (reverb tails, etc.)
constexpr float kSoftGateThresholdDb = -72.0f;  // Threshold below which gate starts closing
// Precomputed linear values for efficiency (10^(dB/20))
constexpr float kSoftGateThreshold = 0.00025f;  // -72dB in linear
constexpr float kSoftGateFloor = 0.000016f;     // -96dB in linear (16-bit noise floor)

// Vocals gate constants
// On instrumental tracks, the model often outputs spurious low-level content in the vocals stem.
// This gate detects when vocals energy is very low relative to the mix and transfers it to "other".
// Two criteria: (1) ratio of vocals to total energy, (2) absolute vocals level.
// Real vocals are typically above -25dB; be aggressive about gating quiet content.
//
// Ratio-based gating: when vocals are a tiny fraction of the mix, they're likely noise
constexpr float kVocalsGateRatioThreshold = 0.01f;   // Below 1% of mix energy, start gating
constexpr float kVocalsGateRatioFloor = 0.003f;      // Below 0.3%, fully gate (transfer to other)
//
// Level-based gating: absolute vocals level threshold (real vocals are rarely this quiet)
// Uses peak amplitude (max of L/R) rather than RMS for faster response
constexpr float kVocalsGateLevelThresholdDb = -28.0f;  // Above this, vocals pass through
constexpr float kVocalsGateLevelFloorDb = -32.0f;      // Below this, fully gate
// Precomputed linear values: 10^(dB/20)
constexpr float kVocalsGateLevelThreshold = 0.04f;     // -28dB in linear
constexpr float kVocalsGateLevelFloor = 0.025f;        // -32dB in linear
//
// Asymmetric attack/release time constants for vocals gate (in seconds)
// Fast attack so vocals come in quickly, slow release to avoid pumping on gaps
// Actual coefficients are calculated in prepareToPlay based on sample rate
constexpr float kVocalsGateAttackTimeSec = 0.015f;   // ~15ms attack
constexpr float kVocalsGateReleaseTimeSec = 0.4f;    // ~400ms release

// Precomputed inverse ranges for gate calculations to avoid division in the audio thread
constexpr float kSoftGateInvRange = 1.0f / (kSoftGateThreshold - kSoftGateFloor);
constexpr float kVocalsGateRatioInvRange = 1.0f / (kVocalsGateRatioThreshold - kVocalsGateRatioFloor);
constexpr float kVocalsGateLevelInvRange = 1.0f / (kVocalsGateLevelThreshold - kVocalsGateLevelFloor);
static_assert(kSoftGateThreshold > kSoftGateFloor, "Soft gate threshold must be greater than floor.");
static_assert(kVocalsGateRatioThreshold > kVocalsGateRatioFloor, "Vocals gate ratio threshold must be greater than floor.");
static_assert(kVocalsGateLevelThreshold > kVocalsGateLevelFloor, "Vocals gate level threshold must be greater than floor.");

// Input normalization constants
// Neural networks have a noise floor - by normalizing input to a consistent level,
// quiet signals get amplified before inference and the output gets scaled back down,
// effectively pushing the noise floor below the signal.
constexpr float kNormTargetRmsDb = -12.0f;      // Target RMS level for model input
constexpr float kNormTargetRms = 0.251f;        // -12dB in linear (10^(-12/20))
constexpr float kNormMaxGainDb = 40.0f;         // Maximum gain to apply (avoid amplifying silence)
constexpr float kNormMaxGain = 100.0f;          // 40dB in linear
constexpr float kNormMinInputRms = 0.000251f;   // -72dB RMS - below this, don't normalize (too quiet)

// Number of inference buffers for double-buffering
constexpr int kNumInferenceBuffers = 8;  // Allow some buffering for timing variations

class AudioPluginAudioProcessor : public juce::AudioProcessor {
public:
  AudioPluginAudioProcessor();
  ~AudioPluginAudioProcessor() override;

  // Returns a short user-facing status string for ONNX Runtime availability
  // and initialization state, suitable for the editor to display.
  juce::String getOrtStatusString() const;

  // Returns the current plugin latency in samples.
  // This accounts for input accumulation, inference queue depth, and output buffering.
  int getLatencySamples() const;
  
  // Returns the current plugin latency in milliseconds based on sample rate.
  double getLatencyMs() const;

  void prepareToPlay(double sampleRate, int samplesPerBlock) override;
  void releaseResources() override;

  bool isBusesLayoutSupported(const BusesLayout& layouts) const override;

  void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
  using AudioProcessor::processBlock;

  juce::AudioProcessorEditor* createEditor() override;
  bool hasEditor() const override;

  const juce::String getName() const override;

  bool acceptsMidi() const override;
  bool producesMidi() const override;
  bool isMidiEffect() const override;
  double getTailLengthSeconds() const override;

  int getNumPrograms() override;
  int getCurrentProgram() override;
  void setCurrentProgram(int index) override;
  const juce::String getProgramName(int index) override;
  void changeProgramName(int index, const juce::String& newName) override;

  void getStateInformation(juce::MemoryBlock& destData) override;
  void setStateInformation(const void* data, int sizeInBytes) override;

  // Reset streaming buffers to zeros (call when playback stops/restarts)
  void resetStreamingBuffers();

private:
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  // ONNX Runtime environment deleter
  struct OrtEnvDeleter {
    void operator()(void* p) const noexcept;
  };

  // ONNX Runtime session deleter
  struct OrtSessionDeleter {
    void operator()(void* p) const noexcept;
  };

  // ONNX Runtime opaque handles (stored as void* to avoid header exposure)
  std::unique_ptr<void, OrtEnvDeleter> ortEnv;
  std::unique_ptr<void, OrtSessionDeleter> ortSession;
  std::string ortRuntimeVersion;
  std::string ortExecutionProvider;  // "CUDA" or "CPU"
  bool ortInitialized{false};
  bool modelLoaded{false};
  bool usingGPU{false};  // True if CUDA execution provider is active
  juce::String modelLoadError;  // Stores the last model loading error for display

  // ========== Overlap-add streaming buffers ==========
  // Context buffer: holds kContextSize samples of history for left context
  std::array<std::vector<float>, kNumChannels> contextBuffer;  // [channel][kContextSize]
  
  // Input accumulation buffer: accumulates kOutputChunkSize samples before inference
  std::array<std::vector<float>, kNumChannels> inputAccumBuffer;  // [channel][samples]
  size_t inputAccumCount{0};  // How many samples accumulated so far

  // Output ring buffer: stores separated audio for stems
  // [stem][channel][samples]
  std::array<std::array<std::vector<float>, kNumChannels>, kNumStems> outputRingBuffers;
  size_t outputReadPos{0};
  size_t outputSamplesAvailable{0};
  
  // Amortized chunk consumption: instead of copying full chunks in bursts,
  // we copy incrementally across callbacks proportional to numSamples.
  // This tracks how many samples of the current pending chunk we've copied.
  size_t pendingChunkCopyOffset{0};
  bool hasPendingChunk{false};

  // Delayed input ring buffer: stores original input aligned with output for residual calculation
  // This ensures sum(stems) = original input for lossless reconstruction
  std::array<std::vector<float>, kNumChannels> delayedInputBuffer;
  size_t delayedInputWritePos{0};

  // ========== Underrun fallback: dry signal with crossfade ==========
  // When separated output is unavailable (underrun), we output the latency-aligned
  // dry signal instead of silence. This turns "dropout" into "temporary bypass."
  // A crossfade smooths transitions between separated and dry.
  
  // Dry delay line: holds raw input delayed by kOutputChunkSize samples
  // Written continuously with every incoming sample (not just at inference time)
  std::array<std::vector<float>, kNumChannels> dryDelayLine;
  size_t dryDelayWritePos{0};
  size_t dryDelayReadPos{0};  // Lags behind writePos by kOutputChunkSize
  
  // Crossfade state for smooth transitions between separated and dry
  static constexpr int kCrossfadeSamples = 64;  // ~1.5ms at 44.1kHz
  float crossfadeGain{1.0f};  // 1.0 = full separated, 0.0 = full dry
  bool wasInUnderrun{false};  // Previous block's underrun state

  // Vocals gate smoothing state (one-pole lowpass filter)
  float vocalsGateGainSmoothed{1.0f};  // Smoothed gate gain to avoid pumping
  float vocalsGateAttackCoeff{0.9985f};   // Calculated in prepareToPlay based on sample rate
  float vocalsGateReleaseCoeff{0.99994f}; // Calculated in prepareToPlay based on sample rate

  // ========== Background inference thread ==========
  std::unique_ptr<std::thread> inferenceThread;
  std::atomic<bool> shouldStopInference{false};
  
  // Lock-free queue for inference requests (simple ring buffer)
  // Each slot contains: [channel][samples] for one chunk
  struct InferenceRequest {
    std::array<std::vector<float>, kNumChannels> inputChunk;  // kOutputChunkSize HP-filtered samples (model input)
    std::array<std::vector<float>, kNumChannels> contextSnapshot;  // kContextSize HP-filtered samples (model context)
    std::array<std::vector<float>, kNumChannels> originalInput;  // kOutputChunkSize fullband samples (for residual calc)
    std::array<std::vector<float>, kNumChannels> lowFreqChunk;  // kOutputChunkSize LP-filtered samples (add to bass)
    float normalizationGain{1.0f};  // Gain applied to input; inverse applied to output
    std::atomic<bool> ready{false};  // True when data is ready for inference
    std::atomic<bool> processed{false};  // True when inference is complete
    uint32_t epoch{0};  // Epoch when request was created (for stale detection)
    
    // Output data (filled by inference thread)
    std::array<std::array<std::vector<float>, kNumChannels>, kNumStems> outputChunk;
  };
  std::array<std::unique_ptr<InferenceRequest>, kNumInferenceBuffers> inferenceQueue;
  std::atomic<size_t> inferenceWriteIdx{0};  // Next slot for audio thread to write
  std::atomic<size_t> inferenceReadIdx{0};   // Next slot for inference thread to process
  std::atomic<size_t> outputConsumeIdx{0};   // Next slot for audio thread to consume output
  
  // Mutex for inference thread wait (only used by inference thread, never locked by audio thread)
  std::mutex inferenceMutex;
  std::condition_variable inferenceCV;
  
  // Epoch counter for stale inference detection (incremented on reset).
  // When audio thread resets, it increments epoch. Inference thread detects this
  // and discards any in-flight or stale results. This avoids data races by not
  // requiring cross-thread buffer clearing.
  std::atomic<uint32_t> resetEpoch{0};
  
  // Pre-allocated ORT memory info (created once, avoids per-inference allocation)
  void* ortMemoryInfo{nullptr};  // OrtMemoryInfo*
  
  // Pre-allocated scratch buffer for inference (avoids per-inference heap allocation)
  std::vector<float> inferenceScratchBuffer;
  
  // ========== LR4 IIR crossover for low-frequency bypass ==========
  // Linkwitz-Riley 4th order crossover (2x cascaded Butterworth, Q=0.707107)
  // Sums flat: LP + HP = original (with phase shift only, no amplitude ripple)
  //
  // Strategy: Split input into LP + HP before inference.
  //   1. LP-filter input -> lowFreqAccumBuffer (stored for bass reconstruction)
  //   2. HP-filter input -> inputAccumBuffer (fed to model)
  //   3. Model processes HP content only (avoids sub-bass artifacts)
  //   4. Add LP to bass stem after inference (reconstructs full-spectrum bass)
  // This keeps the crossover coherent: LP and HP are derived from the same signal.
  
  // Low-pass filters on input: capture low frequencies to add to bass stem
  // (runs on audio thread in processBlock)
  std::array<juce::dsp::IIR::Filter<float>, kNumChannels> lpFilter1;
  std::array<juce::dsp::IIR::Filter<float>, kNumChannels> lpFilter2;
  
  // High-pass filters on input: remove low frequencies before model inference
  // (runs on audio thread in processBlock)
  std::array<juce::dsp::IIR::Filter<float>, kNumChannels> inputHpFilter1;
  std::array<juce::dsp::IIR::Filter<float>, kNumChannels> inputHpFilter2;
  
  // Low-frequency accumulation buffer: stores LP-filtered samples in parallel with inputAccumBuffer
  std::array<std::vector<float>, kNumChannels> lowFreqAccumBuffer;

  // Internal methods
  void runOverlapAddInference(InferenceRequest& request); // Process one inference request
  void inferenceThreadFunc();  // Background thread function
  void allocateStreamingBuffers();  // Allocate streaming buffers
  void startInferenceThread();  // Start background thread
  void stopInferenceThread();   // Stop background thread
  void resetStreamingBuffersRT();  // RT-safe reset (O(1), no memory operations)
#endif

  // Track playback state for hidden state reset
  std::atomic<bool> wasPlaying{false};

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioPluginAudioProcessor)
};
}  // namespace audio_plugin
