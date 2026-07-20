#include "StemgenRT/OverlapAddProcessor.h"

#include <algorithm>

namespace audio_plugin {

OverlapAddProcessor::OverlapAddProcessor() = default;

void OverlapAddProcessor::allocate(size_t maximumHostBlockSize) {
    const size_t outputRingSize =
        static_cast<size_t>(kOutputChunkSize * kOutputRingBufferChunks);
    const size_t hostBlockCapacity = std::max(
        maximumHostBlockSize, static_cast<size_t>(kOutputChunkSize));
    // processBlock accumulates the complete host block before reading fallback
    // samples, so retain latency plus one declared maximum host block.
    const size_t dryDelaySize =
        static_cast<size_t>(kPluginLatencySamples) + hostBlockCapacity + 1;

    for (auto& channel : inputAccumBuffer_) {
        channel.assign(static_cast<size_t>(kOutputChunkSize), 0.0f);
    }
    for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
        delayedInputBuffer_[ch].assign(outputRingSize, 0.0f);
        dryDelayLine_[ch].assign(dryDelaySize, 0.0f);
    }
    for (auto& stem : outputRingBuffers_) {
        for (auto& channel : stem) {
            channel.assign(outputRingSize, 0.0f);
        }
    }

    resetIndices();
}

void OverlapAddProcessor::reset() {
    for (auto& channel : inputAccumBuffer_) {
        std::fill(channel.begin(), channel.end(), 0.0f);
    }
    for (auto& channel : delayedInputBuffer_) {
        std::fill(channel.begin(), channel.end(), 0.0f);
    }
    for (auto& stem : outputRingBuffers_) {
        for (auto& channel : stem) {
            std::fill(channel.begin(), channel.end(), 0.0f);
        }
    }
    clearDryDelayBuffer();
    resetIndices();
}

void OverlapAddProcessor::resetIndices() {
    inputAccumCount_ = 0;
    outputReadPos_ = 0;
    outputSamplesAvailable_ = 0;
    hasPendingChunk_ = false;
    pendingChunkCopyOffset_ = 0;
    dryDelayReadPos_ = 0;
    dryDelayWritePos_ = static_cast<size_t>(kPluginLatencySamples);
}

void OverlapAddProcessor::clearDryDelayBuffer() {
    for (auto& channel : dryDelayLine_) {
        std::fill(channel.begin(), channel.end(), 0.0f);
    }
}

void OverlapAddProcessor::pushInputSample(int channel, float sample) {
    const auto channelIndex = static_cast<size_t>(channel);
    if (inputAccumCount_ < static_cast<size_t>(kOutputChunkSize)) {
        inputAccumBuffer_[channelIndex][inputAccumCount_] = sample;
    }

    dryDelayLine_[channelIndex][dryDelayWritePos_] = sample;
    if (channel == kNumChannels - 1) {
        if (inputAccumCount_ < static_cast<size_t>(kOutputChunkSize)) {
            ++inputAccumCount_;
        }
        ++dryDelayWritePos_;
        if (dryDelayWritePos_ == dryDelayLine_[0].size()) {
            dryDelayWritePos_ = 0;
        }
    }
}

void OverlapAddProcessor::clearInputAccum() {
    inputAccumCount_ = 0;
}

float OverlapAddProcessor::readDryDelaySample(int channel) const {
    return dryDelayLine_[static_cast<size_t>(channel)][dryDelayReadPos_];
}

void OverlapAddProcessor::advanceDryDelayPos() {
    ++dryDelayReadPos_;
    if (dryDelayReadPos_ == dryDelayLine_[0].size()) {
        dryDelayReadPos_ = 0;
    }
}

}  // namespace audio_plugin
