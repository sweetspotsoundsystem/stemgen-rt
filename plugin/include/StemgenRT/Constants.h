#pragma once

namespace audio_plugin {

// ========== Core model constants ==========
constexpr int kNumStems = 4;             // drums, bass, other, vocals
constexpr int kNumChannels = 2;          // stereo

// Stem indices in model output
constexpr int kStemDrums = 0;
constexpr int kStemBass = 1;
constexpr int kStemVocals = 2;
constexpr int kStemOther = 3;

// ========== Overlap-add streaming constants ==========
// Latency: kOutputChunkSize / 44100 * 1000 = ~11.6ms at 44.1kHz
// Inference runs in a background thread to avoid blocking the audio callback
constexpr int kOutputChunkSize = 512;   // New samples per streaming frame (~11.6ms at 44.1kHz)
constexpr int kContextSize = 1024;       // Context on each side to avoid edge artifacts
constexpr int kInternalChunkSize = kContextSize + kOutputChunkSize + kContextSize;  // 2560 samples

// ========== Low-frequency crossover constants ==========
// Low frequencies are poorly captured by chunk-based inference (e.g., 100 Hz = ~20ms period = 882 samples at 44.1kHz).
// We use a Linkwitz-Riley 4th order (LR4) crossover to split the mixture before inference:
//   1. Split input into LP + HP using LR4 (sums flat: LP + HP = original with phase shift only)
//   2. Feed only HP to the model (avoids sub-bass artifacts from chunked processing)
//   3. Add LP to bass stem (and optionally drums) to reconstruct full-spectrum output
// This ensures the crossover is coherent: both LP and HP are derived from the same signal.
constexpr float kCrossoverFreqHz = 250.0f;  // Crossover frequency in Hz

// ========== Input-following soft gate constants ==========
// Neural network models have a noise floor - they output small values even for silent input.
// Worse, these errors are often correlated across stems such that sum(stems)~=0, but each
// individual stem has noise. This gate attenuates stem outputs when input is very quiet,
// eliminating audible noise when soloing stems on silent passages.
// Conservative thresholds to avoid affecting quiet musical content (reverb tails, etc.)
constexpr float kSoftGateThresholdDb = -72.0f;  // Threshold below which gate starts closing
// Precomputed linear values for efficiency (10^(dB/20))
constexpr float kSoftGateThreshold = 0.00025f;  // -72dB in linear
constexpr float kSoftGateFloor = 0.000016f;     // -96dB in linear (16-bit noise floor)

// ========== Vocals gate constants ==========
// On instrumental tracks, the model often outputs spurious low-level content in the vocals stem.
// This gate detects when vocals energy is very low relative to the mix and transfers it to "other".
// Two criteria: (1) ratio of vocals to total energy, (2) absolute vocals level.
// Keep gating conservative enough to avoid suppressing quiet but valid vocals.
//
// Ratio-based gating: when vocals are a tiny fraction of the mix, they're likely noise
constexpr float kVocalsGateRatioThreshold = 0.0040f;  // Below 0.40% of mix energy, start gating
constexpr float kVocalsGateRatioFloor = 0.0008f;      // Below 0.08%, fully gate (transfer to other)
//
// Level-based gating: absolute vocals level threshold (real vocals are rarely this quiet)
// Uses peak amplitude (max of L/R) rather than RMS for faster response
constexpr float kVocalsGateLevelThresholdDb = -39.0f;  // Above this, vocals pass through
constexpr float kVocalsGateLevelFloorDb = -50.0f;      // Below this, fully gate
// Precomputed linear values: 10^(dB/20)
constexpr float kVocalsGateLevelThreshold = 0.0112f;   // -39dB in linear
constexpr float kVocalsGateLevelFloor = 0.0032f;       // -50dB in linear
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

// Static assertions for gate parameter validity
static_assert(kSoftGateThreshold > kSoftGateFloor, "Soft gate threshold must be greater than floor.");
static_assert(kVocalsGateRatioThreshold > kVocalsGateRatioFloor, "Vocals gate ratio threshold must be greater than floor.");
static_assert(kVocalsGateLevelThreshold > kVocalsGateLevelFloor, "Vocals gate level threshold must be greater than floor.");

// ========== Input normalization constants ==========
// Neural networks have a noise floor - by normalizing input to a consistent level,
// quiet signals get amplified before inference and the output gets scaled back down,
// effectively pushing the noise floor below the signal.
constexpr float kNormTargetRmsDb = -12.0f;      // Target RMS level for model input
constexpr float kNormTargetRms = 0.251f;        // -12dB in linear (10^(-12/20))
constexpr float kNormMaxGainDb = 40.0f;         // Maximum gain to apply (avoid amplifying silence)
constexpr float kNormMaxGain = 100.0f;          // 40dB in linear
constexpr float kNormMinInputRms = 0.000251f;   // -72dB RMS - below this, don't normalize (too quiet)

// ========== Inference queue constants ==========
constexpr int kNumInferenceBuffers = 16;  // Allow more buffering for timing variations

// Ring buffer sizing: keep enough capacity for the inference queue plus a few extra
// host blocks so that the audio thread can keep reading even when inference is late.
constexpr int kOutputRingBufferSlackChunks = 8;  // Extra chunks beyond full queue depth
constexpr int kOutputRingBufferChunks = kNumInferenceBuffers + kOutputRingBufferSlackChunks;

// ========== Crossfade constants ==========
// Chunk-boundary overlap-add smoothing window.
constexpr int kCrossfadeSamples = 256;  // ~5.8ms at 44.1kHz
// Dry/separated fallback transition window during underruns.
constexpr int kUnderrunCrossfadeSamples = 64;  // ~1.5ms at 44.1kHz

// ========== Low-band stabilization ==========
// Keep model fullband, but stabilize stem low-end by redistributing low frequencies
// from the dry signal based on each stem's low-band energy.
constexpr float kLowBandStabilizerCutoffHz = 250.0f;

}  // namespace audio_plugin
