#pragma once

#include <StemgenRT/PluginProcessor.h>

namespace audio_plugin {

class AudioPluginProcessorTestPeer {
public:
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  static void stopWorker(AudioPluginAudioProcessor& processor) {
    processor.inferenceQueue_.stopThread();
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
