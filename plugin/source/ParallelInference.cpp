#include <juce_cryptography/juce_cryptography.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include "StemgenRT/MacThreadPolicy.h"
#include "StemgenRT/ParallelInference.h"
#include "StemgenRT/ParallelModelContract.h"
#include "StemgenRT/QualifiedModelContract.h"
#include "StemgenRT/WindowsThreadPolicy.h"
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
#include <onnxruntime_c_api.h>
#endif

namespace audio_plugin {
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
struct ParallelInference::Impl {
  struct Branch {
    OrtSession* session{};
    OrtValue* audioOutput{};
    std::array<const char*, 6> inputNames{}, outputNames{};
    std::array<const OrtValue*, 6> inputs{};
    std::array<OrtValue*, 6> outputs{};
    size_t count{};
  };
  struct Helper {
    std::thread thread;
    std::atomic<bool> pending{false}, done{true};
    std::atomic<int> initial{0}, final{0};
    WorkerWorkgroup* workgroup{};
    bool ok{};
  };
  const OrtApi* api{};
  std::array<Branch, 3> branches;
  std::array<Helper, 2> helpers;
  std::atomic<bool> stopping{false};
  bool running{};

  bool status(OrtStatus* value, juce::String& error) const {
    if (!value)
      return true;
    error = juce::String(api->GetErrorMessage(value));
    api->ReleaseStatus(value);
    return false;
  }
  bool validTensor(OrtSession* session,
                   size_t index,
                   bool output,
                   const char* expectedName,
                   const int64_t* shape,
                   size_t rank,
                   juce::String& error) const {
    OrtAllocator* allocator{};
    if (!status(api->GetAllocatorWithDefaultOptions(&allocator), error))
      return false;
    char* name{};
    if (!status(
            output ? api->SessionGetOutputName(session, index, allocator, &name)
                   : api->SessionGetInputName(session, index, allocator, &name),
            error))
      return false;
    const bool nameOk = juce::String(name) == expectedName;
    allocator->Free(allocator, name);
    if (!nameOk) {
      error = "Partition tensor name differs";
      return false;
    }
    OrtTypeInfo* type{};
    if (!status(output ? api->SessionGetOutputTypeInfo(session, index, &type)
                       : api->SessionGetInputTypeInfo(session, index, &type),
                error))
      return false;
    const OrtTensorTypeAndShapeInfo* tensor{};
    ONNXTensorElementDataType dtype{};
    size_t dimensions{};
    bool ok = status(api->CastTypeInfoToTensorInfo(type, &tensor), error) &&
              tensor &&
              status(api->GetTensorElementType(tensor, &dtype), error) &&
              dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
              status(api->GetDimensionsCount(tensor, &dimensions), error) &&
              dimensions == rank;
    std::array<int64_t, 4> actual{};
    if (ok)
      ok =
          rank <= actual.size() &&
          status(api->GetDimensions(tensor, actual.data(), rank), error) &&
          std::equal(actual.begin(),
                     actual.begin() + static_cast<std::ptrdiff_t>(rank), shape);
    api->ReleaseTypeInfo(type);
    if (!ok && error.isEmpty())
      error = "Partition tensor shape/type differs";
    return ok;
  }
  bool runBranch(size_t index) {
    auto& branch = branches[index];
    auto outputs = branch.outputs;
    auto* result = api->Run(
        branch.session, nullptr, branch.inputNames.data(), branch.inputs.data(),
        branch.count, branch.outputNames.data(), branch.count, outputs.data());
    bool ok = result == nullptr;
    if (result)
      api->ReleaseStatus(result);
    for (size_t i = 0; i < branch.count; ++i) {
      if (outputs[i] != branch.outputs[i]) {
        ok = false;
        if (outputs[i] &&
            std::find(branch.outputs.begin(), branch.outputs.end(),
                      outputs[i]) == branch.outputs.end())
          api->ReleaseValue(outputs[i]);
      }
    }
    return ok;
  }
  void worker(size_t helperIndex, bool realtime) {
    auto& helper = helpers[helperIndex];
    const size_t branchIndex = helperIndex == 0 ? 0U : 2U;
    juce::Thread::setCurrentThreadName(helperIndex == 0 ? "StemgenRT drums"
                                                        : "StemgenRT vocals");
    const juce::ScopedNoDenormals noDenormals;
    MacThreadPolicy policy;
#if defined(__APPLE__)
    const bool applied =
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) == 0 &&
        policy.start(realtime, 1500000);
    helper.initial.store(applied ? 1 : 3, std::memory_order_release);
    if (!applied) {
      helper.final.store(3, std::memory_order_release);
      return;
    }
#elif defined(_WIN32)
    WindowsThreadPolicy windowsPolicy;
    const bool applied = windowsPolicy.start(realtime);
    helper.initial.store(static_cast<int>(windowsPolicy.status()),
                         std::memory_order_release);
    if (!applied) {
      helper.final.store(3, std::memory_order_release);
      return;
    }
#else
    helper.initial.store(2, std::memory_order_release);
#endif
    if (helper.workgroup)
      helper.workgroup->workerStarted(realtime);
    while (!stopping.load(std::memory_order_acquire)) {
#if defined(_WIN32)
      // Only the inference worker publishes/notifies tasks. Value-based waits
      // retain an early publication; stop also changes the value before waking.
      helper.pending.wait(false, std::memory_order_acquire);
      if (stopping.load(std::memory_order_acquire))
        break;
#endif
      if (helper.workgroup)
        helper.workgroup->poll();
      if (helper.pending.exchange(false, std::memory_order_acquire)) {
        helper.ok = runBranch(branchIndex);
        helper.done.store(true, std::memory_order_release);
#if defined(_WIN32)
        helper.done.notify_one();
#endif
      } else {
#if !defined(_WIN32)
        std::this_thread::sleep_for(std::chrono::microseconds(50));
#endif
      }
    }
    if (helper.workgroup)
      helper.workgroup->workerStopped();
#if defined(__APPLE__)
    helper.final.store(policy.finish() ? 1 : 3, std::memory_order_release);
#elif defined(_WIN32)
    const bool restored = windowsPolicy.finish();
    helper.final.store(restored ? static_cast<int>(windowsPolicy.status()) : 3,
                       std::memory_order_release);
#else
    helper.final.store(2, std::memory_order_release);
#endif
  }
};

ParallelInference::ParallelInference() : impl_(std::make_unique<Impl>()) {}
ParallelInference::~ParallelInference() {
  stop();
  if (impl_->api)
    for (auto& b : impl_->branches) {
      if (b.audioOutput)
        impl_->api->ReleaseValue(b.audioOutput);
      if (b.session)
        impl_->api->ReleaseSession(b.session);
    }
}

bool ParallelInference::load(const OrtApi* api,
                             OrtEnv* env,
                             const juce::File& directory,
                             juce::String& error) {
  impl_->api = api;
  OrtSessionOptions* options{};
  if (!impl_->status(api->CreateSessionOptions(&options), error))
    return false;
  const bool configured =
      impl_->status(
          api->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL),
          error) &&
      impl_->status(api->SetSessionExecutionMode(options, ORT_SEQUENTIAL),
                    error) &&
      impl_->status(api->SetIntraOpNumThreads(options, 1), error) &&
      impl_->status(api->SetInterOpNumThreads(options, 1), error) &&
      impl_->status(
          api->AddSessionConfigEntry(options, "mlas.disable_kleidiai", "1"),
          error) &&
      impl_->status(api->AddSessionConfigEntry(
                        options, "session.intra_op.allow_spinning", "0"),
                    error) &&
      impl_->status(api->AddSessionConfigEntry(
                        options, "session.inter_op.allow_spinning", "0"),
                    error);
  if (!configured) {
    api->ReleaseSessionOptions(options);
    return false;
  }
  bool ok = true;
  for (size_t index = 0; index < 3 && ok; ++index) {
    const auto& contract = parallel_model::kBranches[index];
    auto& b = impl_->branches[index];
    const auto file =
        directory.getChildFile(juce::String(contract.name.data()) + ".onnx");
    if (!file.existsAsFile() || file.getSize() != contract.bytes ||
        juce::SHA256(file).toHexString() !=
            juce::String(contract.sha256.data())) {
      error =
          "Experimental partition identity differs: " + file.getFullPathName();
      ok = false;
      break;
    }
#ifdef _WIN32
    const std::wstring path(file.getFullPathName().toWideCharPointer());
    ok = impl_->status(
        api->CreateSession(env, path.c_str(), options, &b.session), error);
#else
    ok = impl_->status(
        api->CreateSession(env, file.getFullPathName().toRawUTF8(), options,
                           &b.session),
        error);
#endif
    if (!ok)
      break;
    size_t inputs{}, outputs{};
    b.count = contract.count + 1;
    ok =
        impl_->status(api->SessionGetInputCount(b.session, &inputs), error) &&
        impl_->status(api->SessionGetOutputCount(b.session, &outputs), error) &&
        inputs == b.count && outputs == b.count;
    constexpr std::array<int64_t, 4> waveShape{1, 1, 2, 128};
    b.inputNames[0] = qualified_model::kInputNames[0].data();
    b.outputNames[0] = qualified_model::kOutputNames[0].data();
    if (ok)
      ok = impl_->validTensor(b.session, 0, false, b.inputNames[0],
                              qualified_model::kInputAudioShape.data(), 3,
                              error) &&
           impl_->validTensor(b.session, 0, true, b.outputNames[0],
                              waveShape.data(), 4, error);
    for (size_t state = 0; state < contract.count && ok; ++state) {
      const size_t position = contract.offset + state;
      b.inputNames[state + 1] =
          qualified_model::kInputNames[position + 1].data();
      b.outputNames[state + 1] =
          qualified_model::kOutputNames[position + 1].data();
      const auto& shape = qualified_model::kStateShapes[position];
      ok = impl_->validTensor(b.session, state + 1, false,
                              b.inputNames[state + 1], shape.data(),
                              shape.size(), error) &&
           impl_->validTensor(b.session, state + 1, true,
                              b.outputNames[state + 1], shape.data(),
                              shape.size(), error);
    }
  }
  api->ReleaseSessionOptions(options);
  if (!ok && error.isEmpty())
    error = "Experimental partition interface differs";
  return ok;
}

bool ParallelInference::prepare(OrtMemoryInfo* memory,
                                const std::array<OrtValue*, 14>& inputs,
                                const std::array<OrtValue*, 14>& outputs,
                                float* separated,
                                juce::String& error) {
  if (impl_->running) {
    error = "Stop parallel workers before preparing";
    return false;
  }
  constexpr std::array<int64_t, 4> shape{1, 1, 2, 128};
  for (size_t index = 0; index < 3; ++index) {
    auto& b = impl_->branches[index];
    const auto& contract = parallel_model::kBranches[index];
    if (b.audioOutput)
      impl_->api->ReleaseValue(b.audioOutput);
    b.audioOutput = nullptr;
    if (!impl_->status(impl_->api->CreateTensorWithDataAsOrtValue(
                           memory, separated + index * 256, 256 * sizeof(float),
                           shape.data(), shape.size(),
                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &b.audioOutput),
                       error))
      return false;
    b.inputs[0] = inputs[0];
    b.outputs[0] = b.audioOutput;
    for (size_t state = 0; state < contract.count; ++state) {
      b.inputs[state + 1] = inputs[contract.offset + state + 1];
      b.outputs[state + 1] = outputs[contract.offset + state + 1];
    }
  }
  return true;
}
bool ParallelInference::start(bool realtime,
                              std::array<WorkerWorkgroup*, 2> workgroups) {
  if (impl_->running)
    return true;
  if (!impl_->branches[0].audioOutput)
    return false;
  impl_->stopping.store(false, std::memory_order_release);
  try {
    for (size_t i = 0; i < 2; ++i) {
      auto& h = impl_->helpers[i];
      h.workgroup = workgroups[i];
      h.initial.store(0);
      h.final.store(0);
      h.pending.store(false);
      h.done.store(true);
      h.ok = false;
      h.thread =
          std::thread([this, i, realtime] { impl_->worker(i, realtime); });
    }
  } catch (...) {
    stop();
    return false;
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  for (auto& h : impl_->helpers) {
    while (h.initial.load(std::memory_order_acquire) == 0 &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const int status = h.initial.load(std::memory_order_acquire);
    if (status == 0 || status == 3) {
      stop();
      return false;
    }
  }
  impl_->running = true;
  return true;
}
void ParallelInference::stop() {
  impl_->stopping.store(true, std::memory_order_release);
#if defined(_WIN32)
  // Lifecycle caller only, after the primary worker has gathered its last hop.
  // A notification without a value change would not release atomic::wait.
  for (auto& h : impl_->helpers) {
    h.pending.store(true, std::memory_order_release);
    h.pending.notify_one();
  }
#endif
  for (auto& h : impl_->helpers)
    if (h.thread.joinable())
      h.thread.join();
  impl_->running = false;
}
bool ParallelInference::run() {
  if (!impl_->running)
    return false;
  for (auto& h : impl_->helpers) {
    h.done.store(false, std::memory_order_relaxed);
    h.pending.store(true, std::memory_order_release);
#if defined(_WIN32)
    h.pending.notify_one();
#endif
  }
  const bool bassOk = impl_->runBranch(1);
  for (auto& h : impl_->helpers)
    while (!h.done.load(std::memory_order_acquire)) {
#if defined(_WIN32)
      h.done.wait(false, std::memory_order_acquire);
#else
      std::this_thread::sleep_for(std::chrono::microseconds(10));
#endif
    }
  return bassOk && impl_->helpers[0].ok && impl_->helpers[1].ok;
}
int ParallelInference::initialPolicy(size_t i) const noexcept {
  return i < 2 ? impl_->helpers[i].initial.load(std::memory_order_acquire) : 3;
}
int ParallelInference::finalPolicy(size_t i) const noexcept {
  return i < 2 ? impl_->helpers[i].final.load(std::memory_order_acquire) : 3;
}
#else
struct ParallelInference::Impl {};
ParallelInference::ParallelInference() : impl_(std::make_unique<Impl>()) {}
ParallelInference::~ParallelInference() = default;
bool ParallelInference::load(const OrtApi*,
                             OrtEnv*,
                             const juce::File&,
                             juce::String& e) {
  e = "ORT unavailable";
  return false;
}
bool ParallelInference::prepare(OrtMemoryInfo*,
                                const std::array<OrtValue*, 14>&,
                                const std::array<OrtValue*, 14>&,
                                float*,
                                juce::String&) {
  return false;
}
bool ParallelInference::start(bool, std::array<WorkerWorkgroup*, 2>) {
  return false;
}
void ParallelInference::stop() {}
bool ParallelInference::run() {
  return false;
}
int ParallelInference::initialPolicy(size_t) const noexcept {
  return 2;
}
int ParallelInference::finalPolicy(size_t) const noexcept {
  return 2;
}
#endif
}  // namespace audio_plugin
