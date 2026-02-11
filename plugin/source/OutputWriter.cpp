#include "StemgenRT/OutputWriter.h"
#include "StemgenRT/OverlapAddProcessor.h"
#include <algorithm>
#include <cstring>

namespace audio_plugin {

void OutputWriter::reset() {
    crossfadeGain_ = 1.0f;
}

void OutputWriter::setOutputPointers(float* mainWrite[kNumChannels], int mainNumCh,
                                     float* stemWrite[4][kNumChannels], int stemNumCh[4]) {
    for (int ch = 0; ch < kNumChannels; ++ch) {
        mainWrite_[ch] = mainWrite[ch];
    }
    mainNumCh_ = mainNumCh;

    for (int b = 0; b < 4; ++b) {
        for (int ch = 0; ch < kNumChannels; ++ch) {
            stemWrite_[b][ch] = stemWrite[b][ch];
        }
        stemNumCh_[b] = stemNumCh[b];
    }
}

void OutputWriter::writeBlock(
    OverlapAddProcessor& overlapAdd,
    const std::array<std::array<std::vector<float>, kNumChannels>, kNumStems>& outputRingBuffers,
    size_t ringSize,
    int numSamples) {

    // Use locals for ring state (audio thread owns these, so no atomics needed)
    size_t readPos = overlapAdd.getOutputReadPos();
    size_t avail = overlapAdd.getOutputSamplesAvailable();

    // Local crossfade gain for smooth transitions
    float xfadeGain = crossfadeGain_;
    constexpr float xfadeDelta =
        1.0f / static_cast<float>(kUnderrunCrossfadeSamples);

    for (int i = 0; i < numSamples; ++i) {
        const bool have = (avail > 0);

        // Update crossfade gain: ramp toward 1.0 (separated) or 0.0 (dry)
        // When separated audio is available (have=true), ramp up toward 1.0
        // When underrun (have=false), ramp down toward 0.0
        if (have) {
            xfadeGain = std::min(1.0f, xfadeGain + xfadeDelta);
        } else {
            xfadeGain = std::max(0.0f, xfadeGain - xfadeDelta);
        }

        // Get dry signal (latency-aligned input) for fallback/crossfade
        float dry[kNumChannels];
        for (int ch = 0; ch < kNumChannels; ++ch) {
            dry[ch] = overlapAdd.readDryDelaySample(ch);
        }

        // Main bus: don't write — input passes through unmodified via
        // JUCE in-place buffer sharing.

        // Stem buses (if enabled)
        // During underrun, output dry/4 to each stem (approximate equal split)
        // During normal operation, output separated stems with crossfade
        for (int busIdx = 0; busIdx < 4; ++busIdx) {
            if (stemNumCh_[busIdx] <= 0)
                continue;

            const size_t stemIndex = kBusToStemMap[busIdx];
            for (int ch = 0; ch < std::min(kNumChannels, stemNumCh_[busIdx]); ++ch) {
                float stemSample = 0.0f;
                if (have) {
                    stemSample = outputRingBuffers[stemIndex][static_cast<size_t>(ch)][readPos];
                }
                // Crossfade: separated stem when available, dry/4 as fallback
                // The /4 distributes dry signal equally across 4 stems
                float dryStem = dry[ch] * 0.25f;
                float output = xfadeGain * stemSample + (1.0f - xfadeGain) * dryStem;
                stemWrite_[busIdx][ch][i] = output;
            }
        }

        // Advance ring read position only if we consumed a sample
        if (have) {
            ++readPos;
            if (readPos == ringSize) readPos = 0;
            --avail;
        }

        // Always advance dry delay read position (dry input flows continuously)
        overlapAdd.advanceDryDelayPos();
    }

    // Commit ring state back to overlap-add processor
    overlapAdd.setOutputReadPos(readPos);
    overlapAdd.setOutputSamplesAvailable(avail);
    crossfadeGain_ = xfadeGain;
}

}  // namespace audio_plugin
