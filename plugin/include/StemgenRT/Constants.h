#pragma once

namespace audio_plugin {

// The qualified c91 deployment emits [drums, bass, vocals, other].
constexpr int kNumStems = 4;
constexpr int kNumChannels = 2;
constexpr int kStemDrums = 0;
constexpr int kStemBass = 1;
constexpr int kStemVocals = 2;
constexpr int kStemOther = 3;

// Fixed model contract. The graph consumes one 512-sample hop and emits the
// preceding hop while carrying its analysis overlap and fusion-GRU state.
constexpr int kModelSampleRate = 44100;
constexpr int kOutputChunkSize = 512;
constexpr int kAnalysisWindowSize = 1024;
constexpr int kFusionHiddenLayers = 2;
constexpr int kFusionHiddenSize = 1000;

// The background design needs one hop to collect/queue audio and the graph has
// one hop of output delay. Report both to the host for honest PDC.
constexpr int kModelOutputDelayChunks = 1;
constexpr int kAsyncQueueDelayChunks = 1;
constexpr int kPluginLatencyChunks =
    kModelOutputDelayChunks + kAsyncQueueDelayChunks;
constexpr int kPluginLatencySamples = kPluginLatencyChunks * kOutputChunkSize;

static_assert(kAnalysisWindowSize == 2 * kOutputChunkSize);
static_assert(kPluginLatencySamples == 1024);

// Allow bounded timing variation without blocking the real-time audio thread.
constexpr int kNumInferenceBuffers = 16;
constexpr int kOutputRingBufferSlackChunks = 8;
constexpr int kOutputRingBufferChunks =
    kNumInferenceBuffers + kOutputRingBufferSlackChunks;

// Smooth transitions between model output and the latency-aligned dry fallback.
constexpr int kUnderrunCrossfadeSamples = 64;

}  // namespace audio_plugin
