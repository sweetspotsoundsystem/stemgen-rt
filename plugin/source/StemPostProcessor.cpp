#include "StemgenRT/StemPostProcessor.h"

namespace audio_plugin {

StemPostProcessor::StemSamples StemPostProcessor::process(
    const StemSamples& stems,
    float gatedVocals,
    float vocalsToOther,
    float softGateGain) {

    float otherAdj = stems.other + vocalsToOther;

    StemSamples result;
    result.drums = stems.drums * softGateGain;
    result.bass = stems.bass * softGateGain;
    result.vocals = gatedVocals * softGateGain;
    result.other = otherAdj * softGateGain;

    return result;
}

void StemPostProcessor::processStereo(
    const StemSamples& stemsL, const StemSamples& stemsR,
    float gatedVocalsL, float gatedVocalsR,
    float vocalsToOtherL, float vocalsToOtherR,
    float softGateGain,
    StemSamples& outL, StemSamples& outR) {

    outL = process(stemsL, gatedVocalsL, vocalsToOtherL, softGateGain);
    outR = process(stemsR, gatedVocalsR, vocalsToOtherR, softGateGain);
}

}  // namespace audio_plugin
