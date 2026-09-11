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
  setSize(440, 440);

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
  const juce::Colour background(0xff101419);
  const juce::Colour panel(0xff1c232b);
  const juce::Colour muted(0xffa8b5c3);
  const juce::Colour accent(0xff7ee2b8);
  g.fillAll(background);
  auto area = getLocalBounds().reduced(24);

  g.setColour(juce::Colours::white);
  g.setFont(juce::FontOptions(27.0f, juce::Font::bold));
  g.drawText("StemgenRT", area.removeFromTop(34),
             juce::Justification::centredLeft);
  g.setColour(muted);
  g.setFont(juce::FontOptions(14.0f));
  g.drawText("Live music separation", area.removeFromTop(23),
             juce::Justification::centredLeft);
  area.removeFromTop(18);

  const bool ready = processorRef.getLatencySamples() > 0;
  const bool unsafe = processorRef.isRealtimeCallbackTimingUnsafe();
  const bool fallback = processorRef.isUnderrunActive();
  const juce::String health = !ready     ? "Separation unavailable"
                              : unsafe   ? "Buffer changed: restart audio"
                              : fallback ? "Catching up: mix routed to Other"
                                         : "Ready to separate";
  g.setColour(!ready || unsafe ? juce::Colours::orangered
              : fallback       ? juce::Colours::orange
                               : accent);
  g.setFont(juce::FontOptions(16.0f, juce::Font::bold));
  g.drawFittedText(health, area.removeFromTop(27),
                   juce::Justification::centredLeft, 1);
  g.setColour(muted);
  g.setFont(juce::FontOptions(12.5f));
  const juce::String detail =
      !ready ? processorRef.getOrtStatusString()
             : "44.1 kHz | 128-sample buffer for lowest latency";
  g.drawFittedText(detail, area.removeFromTop(40),
                   juce::Justification::centredLeft, 2);
  area.removeFromTop(10);

  auto latency = area.removeFromTop(57);
  g.setColour(panel);
  g.fillRoundedRectangle(latency.toFloat(), 8.0f);
  latency.reduce(14, 0);
  g.setColour(juce::Colours::white);
  g.setFont(juce::FontOptions(22.0f, juce::Font::bold));
  g.drawText(juce::String(processorRef.getLatencyMs(), 2) + " ms",
             latency.removeFromLeft(140), juce::Justification::centredLeft);
  g.setColour(muted);
  g.setFont(juce::FontOptions(12.0f));
  g.drawFittedText("Host-compensated delay\n" +
                       juce::String(processorRef.getLatencySamples()) +
                       " samples",
                   latency, juce::Justification::centredRight, 2);
  area.removeFromTop(18);

  constexpr std::array<const char*, 4> stems = {"Drums", "Bass", "Other",
                                                "Vocals"};
  for (int row = 0; row < 2; ++row) {
    auto line = area.removeFromTop(39);
    for (int column = 0; column < 2; ++column) {
      auto tile = line.removeFromLeft((getWidth() - 56) / 2);
      tile.removeFromRight(8);
      g.setColour(panel);
      g.fillRoundedRectangle(tile.toFloat(), 6.0f);
      g.setColour(juce::Colours::white);
      g.setFont(juce::FontOptions(14.0f));
      g.drawText(stems[static_cast<size_t>(row * 2 + column)],
                 tile.reduced(12, 0), juce::Justification::centredLeft);
    }
    area.removeFromTop(8);
  }
  g.setColour(muted);
  g.setFont(juce::FontOptions(12.0f));
  g.drawFittedText(
      "Route the four stereo stem outputs in your host.\nMain carries the "
      "complete delayed mix.",
      area.removeFromTop(35), juce::Justification::centredLeft, 2);
  const auto missed = processorRef.getUnderrunSampleCount();
  if (missed > 0U) {
    g.setColour(fallback ? juce::Colours::orange : muted);
    g.drawText("Fallback used: " +
                   juce::String(static_cast<juce::int64>(missed)) + " samples",
               area.removeFromTop(22), juce::Justification::centredLeft);
  }
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
  const uint64_t dueBoundaryMisses = processorRef.getSameCallbackTimeoutCount();

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
  g.setColour(dueBoundaryMisses > 0U ? juce::Colours::orange
                                     : juce::Colours::white);
  g.drawFittedText(
      "Due-boundary misses: " +
          juce::String(static_cast<juce::int64>(dueBoundaryMisses)) +
          " | nonblocking",
      area.removeFromTop(24), juce::Justification::centred, 1);
}

void AudioPluginAudioProcessorEditor::resized() {}

void AudioPluginAudioProcessorEditor::timerCallback() {
  repaint();
}

#endif

}  // namespace audio_plugin
