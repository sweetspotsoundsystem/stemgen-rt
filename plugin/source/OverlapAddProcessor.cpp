#include "StemgenRT/OverlapAddProcessor.h"

#include <algorithm>

namespace audio_plugin {

OverlapAddProcessor::OverlapAddProcessor() = default;

void OverlapAddProcessor::allocate(size_t maximumHostBlockSize,
                                   size_t latencySamples,
                                   size_t maximumModelOutputSamples) {
  const size_t hostBlockCapacity = std::max(
      maximumHostBlockSize, static_cast<size_t>(kMinimumHostBlockCapacity));
  const size_t modelOutputCapacity =
      std::max(maximumModelOutputSamples, static_cast<size_t>(1));
  const size_t fixedOutputRingSize =
      modelOutputCapacity * static_cast<size_t>(kOutputRingBufferChunks);
  // Offline processing may synchronously schedule every result for a complete
  // large host callback before OutputWriter advances the timeline. Retain that
  // callback, the fixed PDC horizon, and one complete graph hop.
  const size_t callbackOutputRingSize =
      hostBlockCapacity + latencySamples + modelOutputCapacity;
  const size_t outputRingSize =
      std::max(fixedOutputRingSize, callbackOutputRingSize);
  latencySamples_ = latencySamples;
  dryDelayHostBlockCapacity_ = hostBlockCapacity;
  // processBlock accumulates the complete host block before reading fallback
  // samples, so retain latency plus one declared maximum host block.
  const size_t dryDelaySize = latencySamples_ + hostBlockCapacity + 1;

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
  outputTimelineTags_.assign(outputRingSize, kInvalidOutputTimelineTag);
  outputGenerationTags_.assign(outputRingSize, 0);
  outputGeneration_ = 0;

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
  std::fill(outputTimelineTags_.begin(), outputTimelineTags_.end(),
            kInvalidOutputTimelineTag);
  std::fill(outputGenerationTags_.begin(), outputGenerationTags_.end(), 0);
  outputGeneration_ = 0;
  clearDryDelayStorage();
  resetIndices();
}

void OverlapAddProcessor::resetIndices() {
  inputAccumCount_ = 0;
  outputTimelineSample_ = 0;
  scheduledOutputSamples_ = 0;
  // A generation change makes every existing timeline tag stale in O(1).
  // Wrapping would require more than 2^64 transport resets in one allocation;
  // keep zero reserved for storage that has never been published.
  ++outputGeneration_;
  if (outputGeneration_ == 0) {
    outputGeneration_ = 1;
  }
  dryDelayReadPos_ = 0;
  dryDelayWritePos_ = latencySamples_;
  dryInputSamplesWritten_ = 0;
  dryOutputSamplesRead_ = 0;
}

void OverlapAddProcessor::clearDryDelayBuffer() {
  // The audio-thread transport path calls this immediately after resetIndices.
  // Logical invalidation is sufficient: readDryDelaySample() returns zero until
  // the corresponding post-reset input sample has actually been written.
  dryInputSamplesWritten_ = 0;
  dryOutputSamplesRead_ = 0;
}

void OverlapAddProcessor::clearDryDelayStorage() {
  for (auto& channel : dryDelayLine_) {
    std::fill(channel.begin(), channel.end(), 0.0f);
  }
}

void OverlapAddProcessor::pushDryInputSample(int channel, float sample) {
  const auto channelIndex = static_cast<size_t>(channel);
  dryDelayLine_[channelIndex][dryDelayWritePos_] = sample;
  if (channel == kNumChannels - 1) {
    ++dryDelayWritePos_;
    if (dryDelayWritePos_ == dryDelayLine_[0].size()) {
      dryDelayWritePos_ = 0;
    }
    ++dryInputSamplesWritten_;
  }
}

void OverlapAddProcessor::pushModelInputSample(int channel, float sample) {
  const auto channelIndex = static_cast<size_t>(channel);
  if (inputAccumCount_ < static_cast<size_t>(kOutputChunkSize)) {
    inputAccumBuffer_[channelIndex][inputAccumCount_] = sample;
  }
  if (channel == kNumChannels - 1 &&
      inputAccumCount_ < static_cast<size_t>(kOutputChunkSize)) {
    ++inputAccumCount_;
  }
}

void OverlapAddProcessor::pushInputSample(int channel, float sample) {
  pushModelInputSample(channel, sample);
  pushDryInputSample(channel, sample);
}

void OverlapAddProcessor::clearInputAccum() {
  inputAccumCount_ = 0;
}

bool OverlapAddProcessor::canScheduleModelOutput(uint64_t firstTimelineSample,
                                                 size_t sampleCount) const {
  if (sampleCount == 0 || outputTimelineTags_.empty() ||
      firstTimelineSample < outputTimelineSample_) {
    return false;
  }
  const uint64_t ringSize = static_cast<uint64_t>(getOutputRingSize());
  const uint64_t count = static_cast<uint64_t>(sampleCount);
  if (firstTimelineSample > std::numeric_limits<uint64_t>::max() - count ||
      firstTimelineSample + count > outputTimelineSample_ + ringSize) {
    return false;
  }
  for (size_t i = 0; i < sampleCount; ++i) {
    const uint64_t timelineSample =
        firstTimelineSample + static_cast<uint64_t>(i);
    const size_t ringPosition = getOutputRingPosition(timelineSample);
    if (outputGenerationTags_[ringPosition] == outputGeneration_ &&
        outputTimelineTags_[ringPosition] != kInvalidOutputTimelineTag) {
      return false;
    }
  }
  return true;
}

void OverlapAddProcessor::markModelOutputScheduled(uint64_t firstTimelineSample,
                                                   size_t sampleCount) {
  for (size_t i = 0; i < sampleCount; ++i) {
    const uint64_t timelineSample =
        firstTimelineSample + static_cast<uint64_t>(i);
    const size_t ringPosition = getOutputRingPosition(timelineSample);
    outputTimelineTags_[ringPosition] = timelineSample;
    outputGenerationTags_[ringPosition] = outputGeneration_;
  }
  scheduledOutputSamples_ += sampleCount;
}

bool OverlapAddProcessor::hasModelOutputForCurrentSample() const {
  if (outputTimelineTags_.empty()) {
    return false;
  }
  const size_t readPosition = getOutputReadPos();
  return outputGenerationTags_[readPosition] == outputGeneration_ &&
         outputTimelineTags_[readPosition] == outputTimelineSample_;
}

void OverlapAddProcessor::advanceOutputTimeline() {
  if (hasModelOutputForCurrentSample()) {
    outputTimelineTags_[getOutputReadPos()] = kInvalidOutputTimelineTag;
    if (scheduledOutputSamples_ > 0) {
      --scheduledOutputSamples_;
    }
  }
  ++outputTimelineSample_;
}

float OverlapAddProcessor::readDryDelaySample(int channel) const {
  if (dryOutputSamplesRead_ < static_cast<uint64_t>(latencySamples_)) {
    return 0.0f;
  }
  const uint64_t alignedInputSample =
      dryOutputSamplesRead_ - static_cast<uint64_t>(latencySamples_);
  if (alignedInputSample >= dryInputSamplesWritten_) {
    return 0.0f;
  }
  return dryDelayLine_[static_cast<size_t>(channel)][dryDelayReadPos_];
}

void OverlapAddProcessor::advanceDryDelayPos() {
  ++dryOutputSamplesRead_;
  ++dryDelayReadPos_;
  if (dryDelayReadPos_ == dryDelayLine_[0].size()) {
    dryDelayReadPos_ = 0;
  }
}

}  // namespace audio_plugin
