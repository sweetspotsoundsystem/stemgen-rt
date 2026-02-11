#pragma once

#include <array>
#include <juce_dsp/juce_dsp.h>
#include "Constants.h"
#include "StemPostProcessor.h"

namespace audio_plugin {

// Stabilizes low-frequency stem output without changing plugin latency.
// We keep each stem's high-frequency content, while replacing the low band
// with a dry-signal-constrained redistribution based on stem low-band energy.
class LowBandStabilizer {
public:
    void prepare(double sampleRate);
    void reset();

    void processStereo(float dryL, float dryR,
                       StemPostProcessor::StemSamples& stemsL,
                       StemPostProcessor::StemSamples& stemsR);

private:
    float processOnePole(float input, float& state) const;
    void processChannel(int channel, float drySample,
                        StemPostProcessor::StemSamples& stems);

    float alpha_{0.0f};
    float energyAlpha_{0.0f};
    float highEnvAlpha_{0.0f};
    float highGateAlpha_{0.0f};
    std::array<float, kNumChannels> dryLowState_{};
    std::array<float, kNumChannels> dryLowEnvState_{};
    std::array<float, kNumChannels> dryHighEnvState_{};
    std::array<float, kNumChannels> leakHighEnvState_{};
    std::array<float, kNumChannels> highLeakGainState_{};
    std::array<float, kNumChannels> drumsLeakHighEnvState_{};
    std::array<float, kNumChannels> drumsHighLeakGainState_{};
    std::array<float, kNumChannels> lowDominanceGateState_{};
    std::array<std::array<float, kNumChannels>, kNumStems> stemLowState_{};
    std::array<std::array<float, kNumChannels>, kNumStems> stemEnergyState_{};

    // Steep high-band detector for gating other/vocals on low-passed input.
    std::array<juce::dsp::IIR::Filter<float>, kNumChannels> highSenseHp1_;
    std::array<juce::dsp::IIR::Filter<float>, kNumChannels> highSenseHp2_;
};

}  // namespace audio_plugin
