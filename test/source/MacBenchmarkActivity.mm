#include "MacBenchmarkActivity.h"

#import <Foundation/Foundation.h>

namespace audio_plugin_test {

struct MacBenchmarkActivity::Impl {
  id activity;
};

MacBenchmarkActivity::MacBenchmarkActivity() : impl_(std::make_unique<Impl>()) {
  @autoreleasepool {
    impl_->activity = [[NSProcessInfo processInfo]
        beginActivityWithOptions:NSActivityUserInitiated |
                                 NSActivityLatencyCritical
                          reason:@"StemgenRT standalone real-time comparison"];
  }
}

MacBenchmarkActivity::~MacBenchmarkActivity() {
  @autoreleasepool {
    if (impl_->activity != nil) {
      [[NSProcessInfo processInfo] endActivity:impl_->activity];
      impl_->activity = nil;
    }
  }
}

bool MacBenchmarkActivity::isActive() const noexcept {
  return impl_->activity != nil;
}

}  // namespace audio_plugin_test
