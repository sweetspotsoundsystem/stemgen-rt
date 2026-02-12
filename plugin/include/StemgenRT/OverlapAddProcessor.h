#pragma once

#include <array>
#include <vector>
#include <cstddef>
#include "Constants.h"

namespace audio_plugin {

// Manages the streaming buffers for overlap-add inference.
// Handles input accumulation, context management, output ring buffers,
// and dry fallback delay line.
class OverlapAddProcessor {
public:
    OverlapAddProcessor();

    // Allocate all buffers
    void allocate();

    // Reset all buffers to zero
    void reset();

    // RT-safe reset: only reset indices, defer clearing to background
    void resetIndices();

    // === Input accumulation ===

    // Get the number of samples accumulated so far
    size_t getInputAccumCount() const { return inputAccumCount_; }

    // Add a sample to the input accumulation buffer (called per-sample)
    void pushInputSample(int channel, float hpSample, float lpSample, float drySample);

    // Check if we have enough samples for inference
    bool readyForInference() const { return inputAccumCount_ >= static_cast<size_t>(kOutputChunkSize); }

    // Get accumulated HP input buffer for normalization calculation
    const std::array<std::vector<float>, kNumChannels>& getInputAccumBuffer() const { return inputAccumBuffer_; }

    // Get context buffer
    const std::array<std::vector<float>, kNumChannels>& getContextBuffer() const { return contextBuffer_; }

    // Get accumulated LP buffer
    const std::array<std::vector<float>, kNumChannels>& getLowFreqAccumBuffer() const { return lowFreqAccumBuffer_; }

    // Clear input accumulation after queueing inference
    void clearInputAccum();

    // Update context buffer with new samples (called after queueing inference)
    void updateContextBuffer();

    // Clear context buffer (RT-safe, called on transport start)
    void clearContextBuffer();

    // === Output ring buffer ===

    // Get output ring buffer for writing (from inference results)
    std::array<std::array<std::vector<float>, kNumChannels>, kNumStems>& getOutputRingBuffers() {
        return outputRingBuffers_;
    }

    // Get delayed input buffer for residual calculation
    std::array<std::vector<float>, kNumChannels>& getDelayedInputBuffer() { return delayedInputBuffer_; }

    // Ring buffer state
    size_t getOutputReadPos() const { return outputReadPos_; }
    void setOutputReadPos(size_t pos) { outputReadPos_ = pos; }
    size_t getOutputSamplesAvailable() const { return outputSamplesAvailable_; }
    void setOutputSamplesAvailable(size_t n) { outputSamplesAvailable_ = n; }
    void addOutputSamplesAvailable(size_t n) { outputSamplesAvailable_ += n; }
    size_t getOutputRingSize() const { return outputRingBuffers_[0][0].size(); }

    // Delayed input buffer position
    size_t getDelayedInputWritePos() const { return delayedInputWritePos_; }

    // Calculate write position in ring buffer
    size_t getOutputWritePos() const {
        return (outputReadPos_ + outputSamplesAvailable_) % getOutputRingSize();
    }

    // Read from output ring buffer and advance read position
    // Returns false if no samples available
    bool readOutputSample(int stem, int channel, float& sample);

    // Advance read position (call after reading all stems for one sample)
    void advanceOutputReadPos();

    // === Dry delay line for underrun fallback ===

    // Write to dry delay line (called per-sample)
    void writeDryDelaySample(int channel, float sample);

    // Read from dry delay line (delayed by kOutputChunkSize)
    float readDryDelaySample(int channel) const;

    // Advance dry delay positions (call once per sample after reading)
    void advanceDryDelayPos();

    // Dry delay priming helpers
    bool isDryDelayPrimed() const { return dryDelayPrimed_; }
    void primeDryDelayFromInput(const float* inputPointers[kNumChannels], int numSamples);

    // === Chunk boundary crossfade state ===

    // Previous chunk's overlap tail for crossfading at chunk boundaries.
    // Eliminates discontinuities between adjacent model output chunks.
    bool hasPrevOverlapTail() const { return hasPrevOverlapTail_; }
    void setHasPrevOverlapTail(bool v) { hasPrevOverlapTail_ = v; }
    std::array<std::array<std::vector<float>, kNumChannels>, kNumStems>& getPrevOverlapTail() {
        return prevOverlapTail_;
    }

    // === Pending chunk state (for amortized copying) ===

    bool hasPendingChunk() const { return hasPendingChunk_; }
    void setHasPendingChunk(bool v) { hasPendingChunk_ = v; }
    size_t getPendingChunkOffset() const { return pendingChunkCopyOffset_; }
    void setPendingChunkOffset(size_t v) { pendingChunkCopyOffset_ = v; }

private:
    // Input accumulation
    std::array<std::vector<float>, kNumChannels> inputAccumBuffer_;   // HP-filtered
    std::array<std::vector<float>, kNumChannels> lowFreqAccumBuffer_; // LP-filtered
    size_t inputAccumCount_{0};

    // Context buffer for model (HP-filtered history)
    std::array<std::vector<float>, kNumChannels> contextBuffer_;

    // Output ring buffers [stem][channel][samples]
    std::array<std::array<std::vector<float>, kNumChannels>, kNumStems> outputRingBuffers_;
    size_t outputReadPos_{0};
    size_t outputSamplesAvailable_{0};

    // Delayed input for residual calculation
    std::array<std::vector<float>, kNumChannels> delayedInputBuffer_;
    size_t delayedInputWritePos_{0};

    // Dry delay line for underrun fallback
    std::array<std::vector<float>, kNumChannels> dryDelayLine_;
    size_t dryDelayWritePos_{0};
    size_t dryDelayReadPos_{0};
    bool dryDelayPrimed_{false};

    // Chunk boundary crossfade state
    std::array<std::array<std::vector<float>, kNumChannels>, kNumStems> prevOverlapTail_;
    bool hasPrevOverlapTail_{false};

    // Amortized chunk copying state
    bool hasPendingChunk_{false};
    size_t pendingChunkCopyOffset_{0};
};

}  // namespace audio_plugin
