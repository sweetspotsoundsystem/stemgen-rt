#include "StemgenRT/PluginEditor.h"
#include "StemgenRT/PluginProcessor.h"

#if HAS_LOGO_ASSET
#include "BinaryData.h"
#endif

namespace audio_plugin {

#if !STEMGENRT_DEBUG_UI
// =============================================================================
// Release build - just display logo
// =============================================================================

AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor(
    AudioPluginAudioProcessor& p)
    : AudioProcessorEditor(&p), processorRef(p) {
  setSize(300, 300);

#if HAS_LOGO_ASSET
  logoImage = juce::ImageCache::getFromMemory(BinaryData::logo_png,
                                              BinaryData::logo_pngSize);
#endif
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor() = default;

void AudioPluginAudioProcessorEditor::paint(juce::Graphics& g) {
  g.fillAll(juce::Colours::black);

#if HAS_LOGO_ASSET
  if (logoImage.isValid()) {
    auto bounds = getLocalBounds().reduced(20).toFloat();
    float scale = juce::jmin(bounds.getWidth() / logoImage.getWidth(),
                             bounds.getHeight() / logoImage.getHeight());
    g.drawImage(logoImage,
                bounds.withSizeKeepingCentre(logoImage.getWidth() * scale,
                                             logoImage.getHeight() * scale));
  }
#else
  g.setColour(juce::Colours::white);
  g.setFont(24.0f);
  g.drawFittedText("StemgenRT", getLocalBounds(), juce::Justification::centred,
                   1);
#endif
}

void AudioPluginAudioProcessorEditor::resized() {}

#else
// =============================================================================
// Debug build - display status info
// =============================================================================

AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor(
    AudioPluginAudioProcessor& p)
    : AudioProcessorEditor(&p), processorRef(p) {
  setSize(360, 320);
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
  g.drawFittedText(status, area.removeFromTop(24),
                   juce::Justification::centred, 1);

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
  if (sampleRate <= 0.0) sampleRate = 44100.0;
  double ringFillMs = (static_cast<double>(ringFill) / sampleRate) * 1000.0;
  // Ring fill up to kOutputChunkSize is normal pipeline buffering already
  // covered by PDC. Only excess above one chunk adds real latency.
  size_t ringExcess = (ringFill > static_cast<size_t>(kOutputChunkSize))
      ? (ringFill - static_cast<size_t>(kOutputChunkSize))
      : 0;
  double excessMs = (static_cast<double>(ringExcess) / sampleRate) * 1000.0;
  double totalDelayMs = latencyMs + excessMs;
  g.drawFittedText(
      juce::String::formatted("Ring: %zu samples (%.1f ms) | Total: %.1f ms",
                              ringFill, ringFillMs, totalDelayMs),
      area.removeFromTop(24), juce::Justification::centred, 1);

  const bool fallbackBlendActive = processorRef.isUnderrunActive();
  const uint64_t underrunBlocks = processorRef.getUnderrunBlockCount();
  const uint64_t underrunSamples = processorRef.getUnderrunSampleCount();
  const size_t lastUnderrunSamples = processorRef.getUnderrunSamplesInLastBlock();
  const bool underrunThisBlock = (lastUnderrunSamples > 0);
  const uint64_t queueFullDrops = processorRef.getQueueFullChunkDropCount();
  const uint64_t ringOverflowEvents = processorRef.getRingOverflowEventCount();
  const uint64_t ringOverflowSamples = processorRef.getRingOverflowSampleDropCount();

  g.setColour(fallbackBlendActive ? juce::Colours::orange : juce::Colours::white);
  g.drawFittedText(
      juce::String("Fallback blend: ") +
          (fallbackBlendActive ? "active" : "inactive"),
      area.removeFromTop(24), juce::Justification::centred, 1);
  g.setColour(underrunThisBlock ? juce::Colours::orange : juce::Colours::white);
  g.drawFittedText(
      juce::String("Underrun this block: ") +
          (underrunThisBlock ? "yes" : "no"),
      area.removeFromTop(24), juce::Justification::centred, 1);
  g.setColour(juce::Colours::white);
  g.drawFittedText("Underrun events: " + juce::String(underrunBlocks),
                   area.removeFromTop(24), juce::Justification::centred, 1);
  g.drawFittedText("Underrun samples: " + juce::String(underrunSamples) +
                       " (last: " + juce::String(lastUnderrunSamples) + ")",
                   area.removeFromTop(24), juce::Justification::centred, 1);
  g.drawFittedText("Queue-full drops: " +
                       juce::String(static_cast<juce::int64>(queueFullDrops)) + " chunks",
                   area.removeFromTop(24), juce::Justification::centred, 1);
  g.drawFittedText("Ring overflow drops: " +
                       juce::String(static_cast<juce::int64>(ringOverflowSamples)) +
                       " samples (" +
                       juce::String(static_cast<juce::int64>(ringOverflowEvents)) + " events)",
                   area.removeFromTop(24), juce::Justification::centred, 1);
}

void AudioPluginAudioProcessorEditor::resized() {}

void AudioPluginAudioProcessorEditor::timerCallback() { repaint(); }

#endif

}  // namespace audio_plugin
