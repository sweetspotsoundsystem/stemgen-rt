#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "Constants.h"

namespace audio_plugin {

// Owns the audio-thread accumulation, output-ring, and latency-aligned dry
// fallback buffers. Model overlap-add now lives inside the ONNX graph; the
// historical class name is retained to avoid unnecessary API churn.
class OverlapAddProcessor {
public:
    OverlapAddProcessor();

    void allocate(size_t maximumHostBlockSize =
                      static_cast<size_t>(kOutputChunkSize));
    void reset();
    void resetIndices();
    void clearDryDelayBuffer();

    size_t getInputAccumCount() const { return inputAccumCount_; }
    void pushInputSample(int channel, float sample);
    bool readyForInference() const {
        return inputAccumCount_ >= static_cast<size_t>(kOutputChunkSize);
    }
    const std::array<std::vector<float>, kNumChannels>& getInputAccumBuffer() const {
        return inputAccumBuffer_;
    }
    void clearInputAccum();

    std::array<std::array<std::vector<float>, kNumChannels>, kNumStems>&
    getOutputRingBuffers() {
        return outputRingBuffers_;
    }
    std::array<std::vector<float>, kNumChannels>& getDelayedInputBuffer() {
        return delayedInputBuffer_;
    }

    size_t getOutputReadPos() const { return outputReadPos_; }
    void setOutputReadPos(size_t pos) { outputReadPos_ = pos; }
    size_t getOutputSamplesAvailable() const { return outputSamplesAvailable_; }
    void setOutputSamplesAvailable(size_t count) {
        outputSamplesAvailable_ = count;
    }
    void addOutputSamplesAvailable(size_t count) {
        outputSamplesAvailable_ += count;
    }
    size_t getOutputRingSize() const { return outputRingBuffers_[0][0].size(); }
    size_t getOutputWritePos() const {
        return (outputReadPos_ + outputSamplesAvailable_) % getOutputRingSize();
    }

    float readDryDelaySample(int channel) const;
    void advanceDryDelayPos();

    bool hasPendingChunk() const { return hasPendingChunk_; }
    void setHasPendingChunk(bool value) { hasPendingChunk_ = value; }
    size_t getPendingChunkOffset() const { return pendingChunkCopyOffset_; }
    void setPendingChunkOffset(size_t value) { pendingChunkCopyOffset_ = value; }

private:
    std::array<std::vector<float>, kNumChannels> inputAccumBuffer_;
    size_t inputAccumCount_{0};

    std::array<std::array<std::vector<float>, kNumChannels>, kNumStems>
        outputRingBuffers_;
    std::array<std::vector<float>, kNumChannels> delayedInputBuffer_;
    size_t outputReadPos_{0};
    size_t outputSamplesAvailable_{0};

    std::array<std::vector<float>, kNumChannels> dryDelayLine_;
    size_t dryDelayWritePos_{0};
    size_t dryDelayReadPos_{0};

    bool hasPendingChunk_{false};
    size_t pendingChunkCopyOffset_{0};
};

}  // namespace audio_plugin
