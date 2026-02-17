#include "StemgenRT/OverlapAddProcessor.h"
#include <algorithm>
#include <cstring>

namespace audio_plugin {

OverlapAddProcessor::OverlapAddProcessor() = default;

void OverlapAddProcessor::allocate() {
    // Size output ring buffer to match kOutputRingBufferChunks from Constants.h
    const size_t ringSize = static_cast<size_t>(kOutputChunkSize) * kOutputRingBufferChunks;
    const size_t dryDelaySize = dryDelaySamples_ * 2;

    for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
        inputAccumBuffer_[ch].resize(static_cast<size_t>(kOutputChunkSize), 0.0f);
        lowFreqAccumBuffer_[ch].resize(static_cast<size_t>(kOutputChunkSize), 0.0f);
        fullbandAccumBuffer_[ch].resize(static_cast<size_t>(kOutputChunkSize), 0.0f);
        contextBuffer_[ch].resize(static_cast<size_t>(kContextSize), 0.0f);
        delayedInputBuffer_[ch].resize(ringSize, 0.0f);
        // Dry delay line: 2x configured delay for safety margin.
        dryDelayLine_[ch].assign(dryDelaySize, 0.0f);
    }

    for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
        for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
            outputRingBuffers_[stem][ch].resize(ringSize, 0.0f);
            prevOverlapTail_[stem][ch].resize(static_cast<size_t>(kCrossfadeSamples), 0.0f);
        }
    }

    // Initialize dry delay positions so readPos lags behind writePos by configured dry delay.
    // This provides latency-aligned dry signal from the start
    dryDelayWritePos_ = dryDelaySamples_;
    dryDelayReadPos_ = 0;
}

void OverlapAddProcessor::reset() {
    for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
        std::fill(inputAccumBuffer_[ch].begin(), inputAccumBuffer_[ch].end(), 0.0f);
        std::fill(lowFreqAccumBuffer_[ch].begin(), lowFreqAccumBuffer_[ch].end(), 0.0f);
        std::fill(fullbandAccumBuffer_[ch].begin(), fullbandAccumBuffer_[ch].end(), 0.0f);
        std::fill(contextBuffer_[ch].begin(), contextBuffer_[ch].end(), 0.0f);
        std::fill(delayedInputBuffer_[ch].begin(), delayedInputBuffer_[ch].end(), 0.0f);
        std::fill(dryDelayLine_[ch].begin(), dryDelayLine_[ch].end(), 0.0f);
    }

    for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
        for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
            std::fill(outputRingBuffers_[stem][ch].begin(), outputRingBuffers_[stem][ch].end(), 0.0f);
        }
    }

    inputAccumCount_ = 0;
    outputReadPos_ = 0;
    outputSamplesAvailable_ = 0;
    delayedInputWritePos_ = 0;
    hasPendingChunk_ = false;
    pendingChunkCopyOffset_ = 0;
    hasPrevOverlapTail_ = false;

    for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
        for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
            std::fill(prevOverlapTail_[stem][ch].begin(), prevOverlapTail_[stem][ch].end(), 0.0f);
        }
    }

    // Initialize dry delay positions so readPos lags behind writePos by configured dry delay.
    dryDelayWritePos_ = dryDelaySamples_;
    dryDelayReadPos_ = 0;
    dryDelayPrimed_ = false;
}

void OverlapAddProcessor::resetIndices() {
    // RT-safe reset: only reset indices
    inputAccumCount_ = 0;
    outputReadPos_ = 0;
    outputSamplesAvailable_ = 0;
    delayedInputWritePos_ = 0;
    hasPendingChunk_ = false;
    pendingChunkCopyOffset_ = 0;
    hasPrevOverlapTail_ = false;

    // Initialize dry delay positions so readPos lags behind writePos by configured dry delay.
    dryDelayWritePos_ = dryDelaySamples_;
    dryDelayReadPos_ = 0;
    dryDelayPrimed_ = false;
}

void OverlapAddProcessor::pushInputSample(int channel, float hpSample, float lpSample, float drySample) {
    auto ch = static_cast<size_t>(channel);
    if (inputAccumCount_ < static_cast<size_t>(kOutputChunkSize)) {
        inputAccumBuffer_[ch][inputAccumCount_] = hpSample;
        lowFreqAccumBuffer_[ch][inputAccumCount_] = lpSample;
        fullbandAccumBuffer_[ch][inputAccumCount_] = drySample;
    }

    // Write to dry delay line (size is 2x kOutputChunkSize)
    const size_t dryDelaySize = dryDelayLine_[0].size();
    dryDelayLine_[ch][dryDelayWritePos_] = drySample;

    // Only increment once per sample (for last channel)
    if (channel == kNumChannels - 1) {
        if (inputAccumCount_ < static_cast<size_t>(kOutputChunkSize)) {
            inputAccumCount_++;
        }
        dryDelayWritePos_ = (dryDelayWritePos_ + 1) % dryDelaySize;
    }
}

void OverlapAddProcessor::clearInputAccum() {
    inputAccumCount_ = 0;
}

void OverlapAddProcessor::clearContextBuffer() {
    for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
        std::memset(contextBuffer_[ch].data(), 0, contextBuffer_[ch].size() * sizeof(float));
    }
}

void OverlapAddProcessor::updateContextBuffer() {
    // Copy last kContextSize samples from input to context
    if constexpr (kOutputChunkSize >= kContextSize) {
        const size_t srcOffset = static_cast<size_t>(kOutputChunkSize - kContextSize);
        for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
            std::memcpy(contextBuffer_[ch].data(),
                        inputAccumBuffer_[ch].data() + srcOffset,
                        static_cast<size_t>(kContextSize) * sizeof(float));
        }
    } else {
        const size_t samplesToKeep = static_cast<size_t>(kContextSize - kOutputChunkSize);
        for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
            std::memmove(contextBuffer_[ch].data(),
                         contextBuffer_[ch].data() + static_cast<size_t>(kOutputChunkSize),
                         samplesToKeep * sizeof(float));
            std::memcpy(contextBuffer_[ch].data() + samplesToKeep,
                        inputAccumBuffer_[ch].data(),
                        static_cast<size_t>(kOutputChunkSize) * sizeof(float));
        }
    }
}

bool OverlapAddProcessor::readOutputSample(int stem, int channel, float& sample) {
    if (outputSamplesAvailable_ == 0) {
        return false;
    }
    sample = outputRingBuffers_[static_cast<size_t>(stem)][static_cast<size_t>(channel)][outputReadPos_];
    return true;
}

void OverlapAddProcessor::advanceOutputReadPos() {
    if (outputSamplesAvailable_ > 0) {
        outputReadPos_ = (outputReadPos_ + 1) % getOutputRingSize();
        outputSamplesAvailable_--;
    }
}

void OverlapAddProcessor::writeDryDelaySample(int channel, float sample) {
    dryDelayLine_[static_cast<size_t>(channel)][dryDelayWritePos_] = sample;
}

float OverlapAddProcessor::readDryDelaySample(int channel) const {
    return dryDelayLine_[static_cast<size_t>(channel)][dryDelayReadPos_];
}

void OverlapAddProcessor::advanceDryDelayPos() {
    const size_t dryDelaySize = dryDelayLine_[0].size();
    dryDelayReadPos_ = (dryDelayReadPos_ + 1) % dryDelaySize;
}

void OverlapAddProcessor::primeDryDelayFromInput(
    const float* inputPointers[kNumChannels], int numSamples) {
    if (numSamples <= 0)
        return;

    const size_t targetFill = dryDelaySamples_;
    const size_t inputCount = static_cast<size_t>(numSamples);
    const size_t copyCount = std::min(targetFill, inputCount);
    const size_t srcOffset = inputCount - copyCount;

    for (int ch = 0; ch < kNumChannels; ++ch) {
        auto& dryDelay = dryDelayLine_[static_cast<size_t>(ch)];
        std::fill_n(dryDelay.begin(), targetFill, 0.0f);

        if (inputPointers[ch] != nullptr) {
            std::memcpy(dryDelay.data(),
                        inputPointers[ch] + srcOffset,
                        copyCount * sizeof(float));
        }
    }

    dryDelayPrimed_ = true;
}

void OverlapAddProcessor::setDryDelaySamples(size_t delaySamples) {
    dryDelaySamples_ = std::max<size_t>(1, delaySamples);
    const size_t dryDelaySize = dryDelaySamples_ * 2;

    for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
        dryDelayLine_[ch].assign(dryDelaySize, 0.0f);
    }

    dryDelayWritePos_ = dryDelaySamples_;
    dryDelayReadPos_ = 0;
    dryDelayPrimed_ = false;
}

}  // namespace audio_plugin
