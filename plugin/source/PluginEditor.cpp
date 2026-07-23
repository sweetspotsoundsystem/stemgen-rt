#include "StemgenRT/PluginEditor.h"
#include "StemgenRT/PluginProcessor.h"

#if HAS_LOGO_ASSET
#include "BinaryData.h"
#endif

namespace audio_plugin {

#if !STEMGENRT_DEBUG_UI
// =============================================================================
// Release build - logo plus lightweight streaming health
// =============================================================================

AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor(
    AudioPluginAudioProcessor& p)
    : AudioProcessorEditor(&p), processorRef(p) {
  setSize(380, 420);

#if HAS_LOGO_ASSET
  logoImage = juce::ImageCache::getFromMemory(BinaryData::logo_png,
                                              BinaryData::logo_pngSize);
#endif
  startTimerHz(4);
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor() {
  stopTimer();
}

void AudioPluginAudioProcessorEditor::paint(juce::Graphics& g) {
  g.fillAll(juce::Colours::black);

#if HAS_LOGO_ASSET
  if (logoImage.isValid()) {
    const auto logoBounds =
        getLocalBounds().reduced(20).removeFromTop(170).toFloat();
    const auto imageWidth = static_cast<float>(logoImage.getWidth());
    const auto imageHeight = static_cast<float>(logoImage.getHeight());
    const float scale = juce::jmin(logoBounds.getWidth() / imageWidth,
                                   logoBounds.getHeight() / imageHeight);
    g.drawImage(logoImage, logoBounds.withSizeKeepingCentre(
                               imageWidth * scale, imageHeight * scale));
  }
#else
  g.setColour(juce::Colours::white);
  g.setFont(24.0f);
  g.drawFittedText("StemgenRT", getLocalBounds().removeFromTop(170),
                   juce::Justification::centred, 1);
#endif

  auto area = getLocalBounds().reduced(16);
  area.removeFromTop(170);

  const bool unsafeTiming = processorRef.isRealtimeCallbackTimingUnsafe();
  const bool fallbackActive = processorRef.isUnderrunActive();
  const bool deadlineMissed = processorRef.getSameCallbackTimeoutCount() > 0U;
  const bool modelReady = processorRef.getLatencySamples() > 0;
  const auto priorityStatus = processorRef.getInferenceWorkerPriorityStatus();
  const bool priorityFailed =
      priorityStatus == InferenceQueue::WorkerPriorityStatus::Failed;
  const juce::String health = unsafeTiming     ? "PDC timing warning"
                              : fallbackActive ? "Dry fallback active"
                              : deadlineMissed ? "Deadline miss recorded"
                              : modelReady     ? "Streaming normally"
                                               : "Model unavailable";
  g.setColour(
      unsafeTiming || !modelReady
          ? juce::Colours::orangered
          : (fallbackActive || deadlineMissed || priorityFailed
                 ? juce::Colours::orange
                 : juce::Colours::limegreen));
  g.setFont(16.0f);
  g.drawFittedText("Health: " + health, area.removeFromTop(26),
                   juce::Justification::centred, 1);

  g.setColour(juce::Colours::white);
  g.setFont(12.5f);
  g.drawFittedText(processorRef.getOrtStatusString(), area.removeFromTop(42),
                   juce::Justification::centred, 2);

  g.drawFittedText(juce::String::formatted("PDC: %.1f ms (%d samples)",
                                           processorRef.getLatencyMs(),
                                           processorRef.getLatencySamples()),
                   area.removeFromTop(24), juce::Justification::centred, 1);
  g.drawFittedText(
      juce::String::formatted(
          "Host block: prepared %d, current %d (needs %d PDC)",
          processorRef.getPreparedHostBlockSize(),
          processorRef.getLastHostBlockSize(),
          processorRef.getRequiredLatencySamplesForLastHostBlock()),
      area.removeFromTop(24), juce::Justification::centred, 1);

  g.setColour(fallbackActive ? juce::Colours::orange : juce::Colours::white);
  g.drawFittedText(juce::String("Fallback: ") +
                       (fallbackActive ? "active" : "inactive") + " | " +
                       juce::String(static_cast<juce::int64>(
                           processorRef.getUnderrunSampleCount())) +
                       " samples",
                   area.removeFromTop(24), juce::Justification::centred, 1);

  g.setColour(unsafeTiming ? juce::Colours::orangered : juce::Colours::white);
  g.drawFittedText("Unsafe callback timing: " +
                       juce::String(static_cast<juce::int64>(
                           processorRef.getUnsafeRealtimeCallbackCount())) +
                       " callbacks",
                   area.removeFromTop(24), juce::Justification::centred, 1);

  g.setColour(deadlineMissed ? juce::Colours::orange : juce::Colours::white);
  g.drawFittedText(
      juce::String::formatted(
          "Deadline misses: %lld | wait: %d us (max %d)",
          static_cast<long long>(processorRef.getSameCallbackTimeoutCount()),
          processorRef.getLastSameCallbackWaitMicroseconds(),
          processorRef.getMaximumSameCallbackWaitMicroseconds()),
      area.removeFromTop(24), juce::Justification::centred, 1);

  juce::String priorityText;
  switch (priorityStatus) {
    case InferenceQueue::WorkerPriorityStatus::NotAttempted:
      priorityText = "pending";
      break;
    case InferenceQueue::WorkerPriorityStatus::Applied:
      priorityText = "applied";
      break;
    case InferenceQueue::WorkerPriorityStatus::Failed:
      priorityText = "failed";
      break;
    case InferenceQueue::WorkerPriorityStatus::Unsupported:
      priorityText = "unsupported";
      break;
  }
  g.setColour(priorityFailed ? juce::Colours::orange : juce::Colours::white);
  g.drawFittedText("Worker priority: " + priorityText, area.removeFromTop(24),
                   juce::Justification::centred, 1);
}

void AudioPluginAudioProcessorEditor::resized() {}

void AudioPluginAudioProcessorEditor::timerCallback() {
  repaint();
}

#else
// =============================================================================
// Debug build - display status info
// =============================================================================

AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor(
    AudioPluginAudioProcessor& p)
    : AudioProcessorEditor(&p), processorRef(p) {
  setSize(360, 420);
  startTimer(100);
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor() {
  stopTimer();
}

void AudioPluginAudioProcessorEditor::paint(juce::Graphics& g) {
  g.fillAll(
      getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));

  g.setColour(juce::Colours::white);
  g.setFont(15.0f);
  juce::Rectangle<int> area = getLocalBounds().reduced(10);

  // Title
  g.drawFittedText("StemgenRT", area.removeFromTop(30),
                   juce::Justification::centred, 1);

  area.removeFromTop(10);  // Spacing

  // Status
  const auto status = processorRef.getOrtStatusString();
  g.drawFittedText(status, area.removeFromTop(24), juce::Justification::centred,
                   1);

  area.removeFromTop(20);  // Spacing

  // Latency display
  int latencySamples = processorRef.getLatencySamples();
  double latencyMs = processorRef.getLatencyMs();

  juce::String latencyText = juce::String::formatted(
      "PDC: %.1f ms (%d samples)", latencyMs, latencySamples);
  g.drawFittedText(latencyText, area.removeFromTop(24),
                   juce::Justification::centred, 1);

  size_t ringFill = processorRef.getRingFillLevel();
  double sampleRate = processorRef.getSampleRate();
  if (sampleRate <= 0.0)
    sampleRate = 44100.0;
  double ringFillMs = (static_cast<double>(ringFill) / sampleRate) * 1000.0;
  g.drawFittedText(
      juce::String::formatted("Scheduled model: %zu samples (%.1f ms)",
                              ringFill, ringFillMs),
      area.removeFromTop(24), juce::Justification::centred, 1);

  const bool fallbackBlendActive = processorRef.isUnderrunActive();
  const uint64_t underrunBlocks = processorRef.getUnderrunBlockCount();
  const uint64_t underrunSamples = processorRef.getUnderrunSampleCount();
  const size_t lastUnderrunSamples =
      processorRef.getUnderrunSamplesInLastBlock();
  const bool underrunThisBlock = (lastUnderrunSamples > 0);
  const uint64_t queueFullDrops = processorRef.getQueueFullChunkDropCount();
  const uint64_t ringOverflowEvents = processorRef.getRingOverflowEventCount();
  const uint64_t ringOverflowSamples =
      processorRef.getRingOverflowSampleDropCount();
  const uint64_t deadlineMisses = processorRef.getSameCallbackTimeoutCount();

  g.setColour(fallbackBlendActive ? juce::Colours::orange
                                  : juce::Colours::white);
  g.drawFittedText(juce::String("Dry fallback: ") +
                       (fallbackBlendActive ? "active" : "inactive"),
                   area.removeFromTop(24), juce::Justification::centred, 1);
  g.setColour(underrunThisBlock ? juce::Colours::orange : juce::Colours::white);
  g.drawFittedText(juce::String("Underrun this block: ") +
                       (underrunThisBlock ? "yes" : "no"),
                   area.removeFromTop(24), juce::Justification::centred, 1);
  g.setColour(juce::Colours::white);
  g.drawFittedText("Underrun events: " + juce::String(underrunBlocks),
                   area.removeFromTop(24), juce::Justification::centred, 1);
  g.drawFittedText("Underrun samples: " + juce::String(underrunSamples) +
                       " (last: " + juce::String(lastUnderrunSamples) + ")",
                   area.removeFromTop(24), juce::Justification::centred, 1);
  g.drawFittedText("Queue-full drops: " +
                       juce::String(static_cast<juce::int64>(queueFullDrops)) +
                       " chunks",
                   area.removeFromTop(24), juce::Justification::centred, 1);
  g.drawFittedText(
      "Dropped model output: " +
          juce::String(static_cast<juce::int64>(ringOverflowSamples)) +
          " samples (" +
          juce::String(static_cast<juce::int64>(ringOverflowEvents)) +
          " events)",
      area.removeFromTop(24), juce::Justification::centred, 1);
  g.setColour(deadlineMisses > 0U ? juce::Colours::orange
                                 : juce::Colours::white);
  g.drawFittedText(
      "Deadline misses: " +
          juce::String(static_cast<juce::int64>(deadlineMisses)) +
          " | wait: " +
          juce::String(processorRef.getLastSameCallbackWaitMicroseconds()) +
          " us (max " +
          juce::String(processorRef.getMaximumSameCallbackWaitMicroseconds()) +
          ")",
      area.removeFromTop(24), juce::Justification::centred, 1);
}

void AudioPluginAudioProcessorEditor::resized() {}

void AudioPluginAudioProcessorEditor::timerCallback() {
  repaint();
}

#endif

}  // namespace audio_plugin
