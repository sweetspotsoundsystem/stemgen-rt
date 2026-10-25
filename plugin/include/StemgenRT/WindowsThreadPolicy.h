#pragma once

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <avrt.h>

namespace audio_plugin {

// Owned by one worker thread, including destruction. MMCSS status records API
// acceptance and successful cleanup, not a native policy readback. Hosts own
// their audio callback's policy; this class is never used from processBlock.
class WindowsThreadPolicy {
 public:
  enum class Status {
    NotAttempted = 0,
    Applied = 1,
    Unsupported = 2,
    Failed = 3
  };
  struct Api {
    int (*getPriority)();
    bool (*setPriority)(int);
    HANDLE (*join)(DWORD*);
    bool (*setTaskPriority)(HANDLE, AVRT_PRIORITY);
    bool (*leave)(HANDLE);
  };

  explicit WindowsThreadPolicy(const Api& api = nativeApi()) noexcept
      : api_(api) {}
  ~WindowsThreadPolicy() { static_cast<void>(finish()); }
  WindowsThreadPolicy(const WindowsThreadPolicy&) = delete;
  WindowsThreadPolicy& operator=(const WindowsThreadPolicy&) = delete;

  bool start(bool realtime,
             AVRT_PRIORITY priority = AVRT_PRIORITY_HIGH) noexcept {
    if (changed_ || task_)
      return false;
    status_ = Status::Failed;
    original_ = api_.getPriority();
    if (original_ == THREAD_PRIORITY_ERROR_RETURN ||
        !api_.setPriority(THREAD_PRIORITY_HIGHEST))
      return false;
    changed_ = true;
    if (api_.getPriority() != THREAD_PRIORITY_HIGHEST)
      return false;
    status_ = Status::Applied;
    if (!realtime)
      return true;
    DWORD index = 0;
    task_ = api_.join(&index);
    if (task_ && api_.setTaskPriority(task_, priority))
      return true;
    // Unavailable MMCSS keeps the verified ordinary-priority fallback. Do not
    // retain a registration whose requested relative priority was rejected.
    if (task_) {
      if (!api_.leave(task_)) {
        status_ = Status::Failed;
        return false;
      }
      task_ = nullptr;
    }
    status_ = api_.getPriority() == THREAD_PRIORITY_HIGHEST
                  ? Status::Unsupported
                  : Status::Failed;
    return status_ != Status::Failed;
  }

  bool finish() noexcept {
    if (task_) {
      if (!api_.leave(task_)) {
        status_ = Status::Failed;
        return false;
      }
      task_ = nullptr;
    }
    if (changed_) {
      // Revert MMCSS before restoring the thread's original ordinary priority.
      if (!api_.setPriority(original_) || api_.getPriority() != original_) {
        status_ = Status::Failed;
        return false;
      }
      changed_ = false;
    }
    return status_ != Status::Failed;
  }
  Status status() const noexcept { return status_; }

 private:
  static const Api& nativeApi() noexcept {
    static const Api api{
        [] { return GetThreadPriority(GetCurrentThread()); },
        [](int value) {
          return SetThreadPriority(GetCurrentThread(), value) != 0;
        },
        [](DWORD* index) {
          return AvSetMmThreadCharacteristicsW(L"Pro Audio", index);
        },
        [](HANDLE task, AVRT_PRIORITY value) {
          return AvSetMmThreadPriority(task, value) != 0;
        },
        [](HANDLE task) { return AvRevertMmThreadCharacteristics(task) != 0; }};
    return api;
  }
  const Api& api_;
  HANDLE task_{};
  int original_{};
  bool changed_{};
  Status status_{Status::NotAttempted};
};
}  // namespace audio_plugin
#endif
