#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>
#include "Constants.h"

namespace audio_plugin {

// RMS-based input normalization for neural network inference.
// By normalizing input to a consistent level, quiet signals get amplified
// before inference and the output gets scaled back down, effectively pushing
// the noise floor below the signal.
//
// This is a stateless utility - it calculates normalization gains and applies them.
class InputNormalizer {
public:
    // Calculate the normalization gain for a chunk of stereo samples.
    // Returns a gain value that, when applied, normalizes the input to kNormTargetRms.
    // Returns 1.0 if input is too quiet (below kNormMinInputRms).
    //
    // inputChunks: array of [channel][sample] pointers
    // numSamples: number of samples per channel
    static float calculateGain(const float* const* inputChunks, size_t numChannels, size_t numSamples);

    // Overload for std::vector-based storage
    template<size_t N>
    static float calculateGain(const std::array<std::vector<float>, N>& inputChunks, size_t numSamples) {
        std::array<const float*, N> ptrs;
        for (size_t i = 0; i < N; ++i) {
            ptrs[i] = inputChunks[i].data();
        }
        return calculateGain(ptrs.data(), N, numSamples);
    }

    // Calculate normalization gain from context + input combined.
    // Prevents extreme normGain when levels differ between context and input
    // (e.g., loud kick tail in context, silence in input → normGain would
    // amplify context to insane levels if computed from input alone).
    static float calculateGainFromContextAndInput(
        const std::array<std::vector<float>, kNumChannels>& contextBuffer,
        const std::array<std::vector<float>, kNumChannels>& inputAccumBuffer);

    // Apply normalization gain to a buffer (in-place)
    static void applyGain(float* buffer, size_t numSamples, float gain);

    // Apply inverse gain to restore original levels
    static void applyInverseGain(float* buffer, size_t numSamples, float gain) {
        if (gain > 0.0f) {
            applyGain(buffer, numSamples, 1.0f / gain);
        }
    }

    // Calculate RMS of a stereo chunk
    static float calculateRms(const float* const* inputChunks, size_t numChannels, size_t numSamples);
};

}  // namespace audio_plugin
