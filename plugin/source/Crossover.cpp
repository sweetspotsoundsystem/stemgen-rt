#include "StemgenRT/Crossover.h"

namespace audio_plugin {

void Crossover::prepare(double sampleRate, float cutoffHz) {
    auto lpCoeffs = juce::dsp::IIR::Coefficients<float>::makeLowPass(
        sampleRate, cutoffHz, kButterworthQ);
    auto hpCoeffs = juce::dsp::IIR::Coefficients<float>::makeHighPass(
        sampleRate, cutoffHz, kButterworthQ);

    for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
        lpFilter1_[ch].coefficients = lpCoeffs;
        lpFilter2_[ch].coefficients = lpCoeffs;
        hpFilter1_[ch].coefficients = hpCoeffs;
        hpFilter2_[ch].coefficients = hpCoeffs;
    }

    reset();
}

void Crossover::reset() {
    for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
        lpFilter1_[ch].reset();
        lpFilter2_[ch].reset();
        hpFilter1_[ch].reset();
        hpFilter2_[ch].reset();
    }
}

Crossover::FilterOutput Crossover::processSample(int channel, float input) {
    // Cascaded Butterworth for LR4 response
    auto ch = static_cast<size_t>(channel);
    float lp = lpFilter2_[ch].processSample(lpFilter1_[ch].processSample(input));
    float hp = hpFilter2_[ch].processSample(hpFilter1_[ch].processSample(input));
    return {lp, hp};
}

float Crossover::processLowPass(int channel, float input) {
    auto ch = static_cast<size_t>(channel);
    return lpFilter2_[ch].processSample(lpFilter1_[ch].processSample(input));
}

float Crossover::processHighPass(int channel, float input) {
    auto ch = static_cast<size_t>(channel);
    return hpFilter2_[ch].processSample(hpFilter1_[ch].processSample(input));
}

}  // namespace audio_plugin
