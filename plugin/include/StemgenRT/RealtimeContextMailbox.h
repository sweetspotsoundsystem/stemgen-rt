#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace audio_plugin {

// One serialized audio producer, one worker consumer. Context must have a
// bounded, allocation-free copy constructor (JUCE's AudioWorkgroup retains an
// OS handle in inline storage). Only the consumer destroys retained contexts.
// Lifecycle destruction is allowed after both threads have stopped.
template <typename Context, size_t Capacity = 8U>
class RealtimeContextMailbox {
 public:
  static_assert(Capacity > 0U && Capacity < 256U);
  static_assert(std::atomic<uint64_t>::is_always_lock_free);
  static_assert(std::atomic<bool>::is_always_lock_free);

  // Audio thread. Never replaces an owned context, even when saturated.
  // An overflow still publishes a new generation: the consumer must leave
  // the obsolete group, rather than keep using a previous host's workgroup.
  bool publish(const Context& context) noexcept {
    const uint64_t generation = ++producerGeneration_;
    for (size_t index = 0; index < Capacity; ++index) {
      auto& slot = slots_[index];
      if (!slot.ready.load(std::memory_order_acquire)) {
        slot.context.emplace(context);
        slot.generation = generation;
        slot.ready.store(true, std::memory_order_release);
        desired_.store((generation << 8U) | (index + 1U),
                       std::memory_order_release);
        return true;
      }
    }
    overflowCount_.fetch_add(1U, std::memory_order_relaxed);
    desired_.store(generation << 8U, std::memory_order_release);
    return false;
  }

  // Worker thread. Delivers the latest completed publication and retires all
  // older handles, including updates superseded while inference was running.
  // The callback takes ownership before retirement; no raw handle escapes.
  template <typename Consumer>
  bool consumeLatest(Consumer&& consume) {
    const uint64_t desired = desired_.load(std::memory_order_acquire);
    if (desired == consumed_) {
      return false;
    }
    const uint64_t generation = desired >> 8U;
    const size_t encodedIndex = static_cast<size_t>(desired & 0xffU);
    std::optional<Context> next;
    if (encodedIndex != 0U) {
      auto& slot = slots_[encodedIndex - 1U];
      next.emplace(std::move(*slot.context));
    }
    consume(std::move(next), encodedIndex == 0U);
    consumed_ = desired;
    for (auto& slot : slots_) {
      if (slot.ready.load(std::memory_order_acquire) &&
          slot.generation <= generation) {
        slot.context.reset();
        slot.ready.store(false, std::memory_order_release);
      }
    }
    return true;
  }

  uint64_t getOverflowCount() const noexcept {
    return overflowCount_.load(std::memory_order_acquire);
  }
  // Worker only; observers use the owning controller's atomic snapshot.
  uint64_t consumedGeneration() const noexcept { return consumed_ >> 8U; }

 private:
  struct Slot {
    std::optional<Context> context;
    uint64_t generation{0};
    std::atomic<bool> ready{false};
  };
  std::array<Slot, Capacity> slots_;
  uint64_t producerGeneration_{0}, consumed_{0};
  std::atomic<uint64_t> desired_{0}, overflowCount_{0};
};

}  // namespace audio_plugin
