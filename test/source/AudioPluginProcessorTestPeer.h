#pragma once

#include <StemgenRT/PluginProcessor.h>

namespace audio_plugin {

class AudioPluginProcessorTestPeer {
 public:
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  static void stopWorker(AudioPluginAudioProcessor& processor) {
    processor.inferenceQueue_.stopThread();
  }

  static bool allSubmissionsCompleted(AudioPluginAudioProcessor& processor) {
    const auto& queue = processor.inferenceQueue_;
    for (const auto& request : queue.queue_) {
      const auto state = request->getState();
      if (request->getEpoch() == queue.getEpoch() &&
          state != InferenceRequest::SlotState::Empty &&
          state != InferenceRequest::SlotState::Processed &&
          state != InferenceRequest::SlotState::Reading) {
        return false;
      }
    }
    return true;
  }

  // Observe publication without claiming the slot. The next real-time
  // callback must still consume the result through the production scheduler.
  static bool submissionCompleted(AudioPluginAudioProcessor& processor) {
    const auto& queue = processor.inferenceQueue_;
    const size_t index = (queue.writeIdx_.load(std::memory_order_acquire) +
                          queue.queue_.size() - 1U) %
                         queue.queue_.size();
    const auto& request = *queue.queue_[index];
    return request.getEpoch() == queue.getEpoch() && request.isProcessed();
  }
#endif
};

}  // namespace audio_plugin
