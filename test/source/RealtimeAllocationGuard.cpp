#include "RealtimeAllocationGuard.h"

#include <cstdlib>
#include <new>
#if defined(_WIN32)
#include <malloc.h>
#endif

namespace {
thread_local audio_plugin_test::HeapTraffic* activeTraffic = nullptr;

void* allocate(size_t size, size_t alignment = 0U) {
  for (;;) {
    void* result = nullptr;
    const size_t bytes = size == 0U ? 1U : size;
    if (alignment == 0U) {
      result = std::malloc(bytes);
    } else {
#if defined(_WIN32)
      result = _aligned_malloc(bytes, alignment);
#else
      if (posix_memalign(&result, alignment, bytes) != 0) {
        result = nullptr;
      }
#endif
    }
    if (result != nullptr) {
      if (activeTraffic != nullptr) {
        ++activeTraffic->allocations;
      }
      return result;
    }
    if (const auto handler = std::get_new_handler()) {
      handler();
    } else {
      throw std::bad_alloc();
    }
  }
}

void deallocate(void* memory, bool aligned = false) noexcept {
  if (memory != nullptr && activeTraffic != nullptr) {
    ++activeTraffic->deallocations;
  }
#if defined(_WIN32)
  if (aligned) {
    _aligned_free(memory);
    return;
  }
#else
  static_cast<void>(aligned);
#endif
  std::free(memory);
}
}  // namespace

namespace audio_plugin_test {
RealtimeAllocationGuard::RealtimeAllocationGuard() noexcept
    : previous_(activeTraffic) {
  activeTraffic = &traffic_;
}
RealtimeAllocationGuard::~RealtimeAllocationGuard() {
  finish();
}
HeapTraffic RealtimeAllocationGuard::finish() noexcept {
  if (activeTraffic == &traffic_) {
    activeTraffic = previous_;
  }
  return traffic_;
}
}  // namespace audio_plugin_test

void* operator new(size_t size) {
  return allocate(size);
}
void* operator new[](size_t size) {
  return allocate(size);
}
void operator delete(void* memory) noexcept {
  deallocate(memory);
}
void operator delete[](void* memory) noexcept {
  deallocate(memory);
}
void operator delete(void* memory, size_t) noexcept {
  deallocate(memory);
}
void operator delete[](void* memory, size_t) noexcept {
  deallocate(memory);
}
void* operator new(size_t size, std::align_val_t alignment) {
  return allocate(size, static_cast<size_t>(alignment));
}
void* operator new[](size_t size, std::align_val_t alignment) {
  return allocate(size, static_cast<size_t>(alignment));
}
void operator delete(void* memory, std::align_val_t) noexcept {
  deallocate(memory, true);
}
void operator delete[](void* memory, std::align_val_t) noexcept {
  deallocate(memory, true);
}
void operator delete(void* memory, size_t, std::align_val_t) noexcept {
  deallocate(memory, true);
}
void operator delete[](void* memory, size_t, std::align_val_t) noexcept {
  deallocate(memory, true);
}
void* operator new(size_t size, const std::nothrow_t&) noexcept {
  try {
    return allocate(size);
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](size_t size, const std::nothrow_t&) noexcept {
  try {
    return allocate(size);
  } catch (...) {
    return nullptr;
  }
}
void operator delete(void* memory, const std::nothrow_t&) noexcept {
  deallocate(memory);
}
void operator delete[](void* memory, const std::nothrow_t&) noexcept {
  deallocate(memory);
}
void* operator new(size_t size,
                   std::align_val_t alignment,
                   const std::nothrow_t&) noexcept {
  try {
    return allocate(size, static_cast<size_t>(alignment));
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](size_t size,
                     std::align_val_t alignment,
                     const std::nothrow_t&) noexcept {
  try {
    return allocate(size, static_cast<size_t>(alignment));
  } catch (...) {
    return nullptr;
  }
}
void operator delete(void* memory,
                     std::align_val_t,
                     const std::nothrow_t&) noexcept {
  deallocate(memory, true);
}
void operator delete[](void* memory,
                       std::align_val_t,
                       const std::nothrow_t&) noexcept {
  deallocate(memory, true);
}
