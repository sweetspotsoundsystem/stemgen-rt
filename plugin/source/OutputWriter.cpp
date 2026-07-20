#include "StemgenRT/OutputWriter.h"
#include "StemgenRT/OverlapAddProcessor.h"
#include <algorithm>
#include <cstring>

namespace audio_plugin {

void OutputWriter::reset() {
    crossfadeGain_ = 0.0f;
    wasUnderrun_ = false;
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

OutputWriter::WriteResult OutputWriter::writeBlock(
    OverlapAddProcessor& overlapAdd,
    const std::array<std::array<std::vector<float>, kNumChannels>, kNumStems>& outputRingBuffers,
    const std::array<std::vector<float>, kNumChannels>& delayedInputBuffer,
    size_t ringSize,
    int numSamples) {

    WriteResult result;

    // Use locals for ring state (audio thread owns these, so no atomics needed)
    size_t readPos = overlapAdd.getOutputReadPos();
    size_t avail = overlapAdd.getOutputSamplesAvailable();

    // Capture diagnostic state at block start
    result.ringAvailAtStart = avail;
    result.crossfadeGainAtStart = crossfadeGain_;

    // Local crossfade gain for smooth transitions
    float xfadeGain = crossfadeGain_;
    constexpr float xfadeDelta =
        1.0f / static_cast<float>(kUnderrunCrossfadeSamples);

    for (int i = 0; i < numSamples; ++i) {
        const bool have = (avail > 0);
        if (!have) {
            result.hadUnderrun = true;
            ++result.underrunSamples;
        }

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

        // Main bus: read the exact mixture aligned with the model output. During
        // underruns, crossfade to the fixed-latency dry signal.
        float mainOutput[kNumChannels];
        for (int ch = 0; ch < kNumChannels; ++ch) {
            const float delayedSample =
                have ? delayedInputBuffer[static_cast<size_t>(ch)][readPos] : 0.0f;
            mainOutput[ch] =
                xfadeGain * delayedSample + (1.0f - xfadeGain) * dry[ch];
            if (ch < mainNumCh_ && mainWrite_[ch] != nullptr) {
                mainWrite_[ch][i] = mainOutput[ch];
            }
        }

        // Crossfade every internal stem against an equal dry split, then route
        // the final floating-point residual to Other. This preserves
        // Main == Drums + Bass + Vocals + Other through provider differences,
        // startup, and underrun transitions.
        for (int ch = 0; ch < kNumChannels; ++ch) {
            float stems[kNumStems];
            const float dryStem = dry[ch] * 0.25f;
            for (int stem = 0; stem < kNumStems; ++stem) {
                const float separated = have
                    ? outputRingBuffers[static_cast<size_t>(stem)]
                                       [static_cast<size_t>(ch)][readPos]
                    : 0.0f;
                stems[stem] =
                    xfadeGain * separated + (1.0f - xfadeGain) * dryStem;
            }
            stems[kStemOther] = mainOutput[ch] - stems[kStemDrums]
                                - stems[kStemBass] - stems[kStemVocals];

            for (int busIdx = 0; busIdx < 4; ++busIdx) {
                if (ch >= stemNumCh_[busIdx] || stemWrite_[busIdx][ch] == nullptr) {
                    continue;
                }
                stemWrite_[busIdx][ch][i] =
                    stems[kBusToStemMap[busIdx]];
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

    // If we are in/near fallback, mark underrun as active for UI visibility.
    result.isUnderrunNow = (result.hadUnderrun || xfadeGain < 1.0f);

    // Detect underrun transition (entering underrun state)
    result.underrunTransition = (result.hadUnderrun && !wasUnderrun_);
    wasUnderrun_ = result.hadUnderrun;

    return result;
}

}  // namespace audio_plugin
