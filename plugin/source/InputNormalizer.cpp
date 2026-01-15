#include "StemgenRT/InputNormalizer.h"

namespace audio_plugin {

float InputNormalizer::calculateRms(const float* const* inputChunks, size_t numChannels, size_t numSamples) {
    float sumSquares = 0.0f;
    for (size_t ch = 0; ch < numChannels; ++ch) {
        for (size_t j = 0; j < numSamples; ++j) {
            float sample = inputChunks[ch][j];
            sumSquares += sample * sample;
        }
    }
    return std::sqrt(sumSquares / static_cast<float>(numChannels * numSamples));
}

float InputNormalizer::calculateGain(const float* const* inputChunks, size_t numChannels, size_t numSamples) {
    float rms = calculateRms(inputChunks, numChannels, numSamples);

    // Don't normalize if input is too quiet (avoid amplifying silence)
    if (rms < kNormMinInputRms) {
        return 1.0f;
    }

    // Calculate gain needed to reach target RMS, clamped to max
    return std::min(kNormTargetRms / rms, kNormMaxGain);
}

void InputNormalizer::applyGain(float* buffer, size_t numSamples, float gain) {
    for (size_t i = 0; i < numSamples; ++i) {
        buffer[i] *= gain;
    }
}

}  // namespace audio_plugin
