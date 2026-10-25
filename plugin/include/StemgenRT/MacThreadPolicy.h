#pragma once

#include <cstdint>
#include <limits>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <pthread.h>
#endif

namespace audio_plugin {

// Worker-owned policy guard. Construct, start, finish and destroy on the same
// thread. The caller establishes QoS before start. No audio callback calls
// this guard. Synthetic host diagnostics also use it outside the measured loop.
// A passing readback does not establish deadlines or host-workgroup membership.
class MacThreadPolicy {
 public:
  MacThreadPolicy() = default;
  MacThreadPolicy(const MacThreadPolicy&) = delete;
  MacThreadPolicy& operator=(const MacThreadPolicy&) = delete;
  ~MacThreadPolicy() { static_cast<void>(restore()); }

  bool start(bool realtime,
             uint64_t computationNanoseconds,
             uint64_t constraintNanoseconds = 128ULL * 1000000000 /
                                              44100) noexcept {
#if defined(__APPLE__)
    if (started_ || computationNanoseconds == 0 ||
        computationNanoseconds > constraintNanoseconds ||
        constraintNanoseconds > kPeriodNanoseconds ||
        computationNanoseconds < constraintNanoseconds / 2) {
      return false;
    }
    mach_timebase_info_data_t timebase{};
    if (mach_timebase_info(&timebase) != KERN_SUCCESS || timebase.numer == 0 ||
        timebase.denom == 0) {
      return false;
    }
    // Both bounded nanosecond values times a uint32 denominator fit uint64.
    const uint64_t period =
        kPeriodNanoseconds * timebase.denom / timebase.numer;
    const uint64_t computation =
        computationNanoseconds * timebase.denom / timebase.numer;
    const uint64_t constraint =
        constraintNanoseconds * timebase.denom / timebase.numer;
    if (period > std::numeric_limits<uint32_t>::max() || computation == 0) {
      return false;
    }
    period_ = static_cast<uint32_t>(period);
    computation_ = static_cast<uint32_t>(computation);
    constraint_ = static_cast<uint32_t>(constraint);
    realtime_ = realtime;
    if (realtime_) {
      thread_time_constraint_policy_data_t policy{};
      policy.period = period_;
      policy.computation = computation_;
      policy.constraint = constraint_;
      policy.preemptible = TRUE;
      if (thread_policy_set(pthread_mach_thread_np(pthread_self()),
                            THREAD_TIME_CONSTRAINT_POLICY,
                            reinterpret_cast<thread_policy_t>(&policy),
                            THREAD_TIME_CONSTRAINT_POLICY_COUNT) !=
          KERN_SUCCESS) {
        return false;
      }
      needsRestore_ = true;
    }
    started_ = true;
    return verify();
#else
    static_cast<void>(realtime);
    static_cast<void>(computationNanoseconds);
    static_cast<void>(constraintNanoseconds);
    return false;
#endif
  }

  bool finish() noexcept {
#if defined(__APPLE__)
    const bool observed = started_ && verify();
    started_ = false;
    const bool restored = restore();
    return observed && restored;
#else
    return false;
#endif
  }

 private:
  bool restore() noexcept {
#if defined(__APPLE__)
    if (needsRestore_) {
      if (thread_policy_set(pthread_mach_thread_np(pthread_self()),
                            THREAD_STANDARD_POLICY, nullptr,
                            THREAD_STANDARD_POLICY_COUNT) != KERN_SUCCESS) {
        return false;
      }
      needsRestore_ = false;
    }
#endif
    return true;
  }

#if defined(__APPLE__)
  bool verify() const noexcept {
    thread_time_constraint_policy_data_t policy{};
    mach_msg_type_number_t count = THREAD_TIME_CONSTRAINT_POLICY_COUNT;
    boolean_t getDefault = FALSE;
    if (thread_policy_get(pthread_mach_thread_np(pthread_self()),
                          THREAD_TIME_CONSTRAINT_POLICY,
                          reinterpret_cast<thread_policy_t>(&policy), &count,
                          &getDefault) != KERN_SUCCESS ||
        count != THREAD_TIME_CONSTRAINT_POLICY_COUNT) {
      return false;
    }
    if (!realtime_) {
      return getDefault != FALSE;
    }
    return getDefault == FALSE && policy.period == period_ &&
           policy.computation == computation_ &&
           policy.constraint == constraint_ && policy.preemptible != FALSE;
  }

  static constexpr uint64_t kPeriodNanoseconds = 128ULL * 1000000000 / 44100;
  uint32_t period_{0}, computation_{0}, constraint_{0};
  bool realtime_{false}, started_{false}, needsRestore_{false};
#endif
};

}  // namespace audio_plugin
