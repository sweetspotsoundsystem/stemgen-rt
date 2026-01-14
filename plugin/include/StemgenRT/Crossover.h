#pragma once

#include <juce_dsp/juce_dsp.h>
#include <array>
#include "Constants.h"

namespace audio_plugin {

// LR4 (Linkwitz-Riley 4th order) crossover filter for stereo audio.
// Splits input into low-pass and high-pass outputs that sum flat (LP + HP = original).
// Used to bypass low frequencies around the neural network, which struggles with
// sub-bass in chunk-based processing.
class Crossover {
public:
    Crossover() = default;

    // Prepare the crossover with sample rate and cutoff frequency
    void prepare(double sampleRate, float cutoffHz = kCrossoverFreqHz);

    // Reset filter states (call on playback start/stop)
    void reset();

    // Process a single sample for one channel
    // Returns both LP and HP outputs
    struct FilterOutput {
        float lowPass;
        float highPass;
    };
    FilterOutput processSample(int channel, float input);

    // Convenience: get only LP output for a sample
    float processLowPass(int channel, float input);

    // Convenience: get only HP output for a sample
    float processHighPass(int channel, float input);

private:
    // Butterworth Q for LR4 (1/sqrt(2))
    static constexpr float kButterworthQ = 0.7071067811865476f;

    // Two cascaded Butterworth filters = LR4 response
    std::array<juce::dsp::IIR::Filter<float>, kNumChannels> lpFilter1_;
    std::array<juce::dsp::IIR::Filter<float>, kNumChannels> lpFilter2_;
    std::array<juce::dsp::IIR::Filter<float>, kNumChannels> hpFilter1_;
    std::array<juce::dsp::IIR::Filter<float>, kNumChannels> hpFilter2_;
};

}  // namespace audio_plugin
