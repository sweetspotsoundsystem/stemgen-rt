#include <dlfcn.h>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}

static uint32_t readU32(std::ifstream& stream) {
  std::array<unsigned char, 4> b{};
  stream.read(reinterpret_cast<char*>(b.data()), 4);
  require(static_cast<bool>(stream), "Truncated fixture");
  return uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) |
         (uint32_t(b[3]) << 24);
}

struct Case {
  size_t frames;
  std::vector<float> input, expected;
};

static std::vector<Case> readCases(const char* path) {
  std::ifstream stream(path, std::ios::binary);
  require(stream.is_open(), "Cannot open fixture");
  std::array<char, 8> magic{};
  stream.read(magic.data(), 8);
  require(std::string(magic.data(), 8) == "SGRTG001", "Wrong fixture format");
  require(readU32(stream) == 8, "Wrong case count");
  std::vector<Case> cases;
  for (size_t index = 0; index < 8; ++index) {
    const size_t frames = readU32(stream);
    require(frames > 0 && frames <= 20000, "Invalid case length");
    Case c{frames, std::vector<float>(2 * frames),
           std::vector<float>(8 * frames)};
    for (auto* values : {&c.input, &c.expected}) {
      for (auto& value : *values) {
        const uint32_t bits = readU32(stream);
        std::memcpy(&value, &bits, 4);
        require(std::isfinite(value), "Nonfinite fixture sample");
      }
    }
    cases.push_back(std::move(c));
  }
  require(stream.peek() == std::char_traits<char>::eof(),
          "Trailing fixture bytes");
  return cases;
}

int main(int argc, char** argv) {
  try {
    require(argc == 3, "Usage: diagnose MODEL FIXTURE");
    require(std::string(OrtGetApiBase()->GetVersionString()) == "1.26.0",
            "Require ORT 1.26.0");
    Dl_info loaded{};
    require(dladdr(reinterpret_cast<void*>(&OrtGetApiBase), &loaded) != 0 &&
                loaded.dli_fname,
            "Cannot identify loaded runtime");
    std::cerr << "Loaded runtime: " << loaded.dli_fname << '\n';
    std::cerr << "Runtime build: " << Ort::GetApi().GetBuildInfoString()
              << '\n';
    const auto cases = readCases(argv[2]);
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "stemgen-parity-diagnostic");
    const auto memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::cout << std::setprecision(12);
    for (const std::string mode :
         {"default", "kleidiai_disabled", "optimizations_disabled"}) {
      Ort::SessionOptions options;
      options.SetExecutionMode(ORT_SEQUENTIAL);
      options.SetIntraOpNumThreads(1);
      options.SetInterOpNumThreads(1);
      options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
      options.AddConfigEntry("session.intra_op.allow_spinning", "0");
      options.AddConfigEntry("session.inter_op.allow_spinning", "0");
      if (mode == "kleidiai_disabled")
        options.AddConfigEntry("mlas.disable_kleidiai", "1");
      if (mode == "optimizations_disabled")
        options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
      Ort::Session session(env, argv[1], options);
      require(session.GetInputCount() == 9 && session.GetOutputCount() == 9,
              "Wrong model ABI");
      Ort::AllocatorWithDefaultOptions allocator;
      std::vector<std::string> inputNames, outputNames;
      std::vector<std::vector<int64_t>> shapes;
      std::vector<std::vector<float>> buffers;
      for (size_t i = 0; i < 9; ++i) {
        inputNames.emplace_back(
            session.GetInputNameAllocated(i, allocator).get());
        outputNames.emplace_back(
            session.GetOutputNameAllocated(i, allocator).get());
        auto type = session.GetInputTypeInfo(i);
        auto info = type.GetTensorTypeAndShapeInfo();
        require(info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                "Wrong input precision");
        shapes.push_back(info.GetShape());
        size_t count = 1;
        for (const auto dim : shapes.back()) {
          require(dim > 0 && dim < 10000, "Invalid static shape");
          count *= static_cast<size_t>(dim);
        }
        buffers.emplace_back(count);
      }
      require(buffers[0].size() == 256, "Wrong audio shape");
      std::vector<const char*> inputs, outputs;
      std::vector<Ort::Value> tensors;
      for (size_t i = 0; i < 9; ++i) {
        inputs.push_back(inputNames[i].c_str());
        outputs.push_back(outputNames[i].c_str());
        tensors.push_back(Ort::Value::CreateTensor<float>(
            memory, buffers[i].data(), buffers[i].size(), shapes[i].data(),
            shapes[i].size()));
      }
      for (const auto& c : cases) {
        for (auto& b : buffers)
          std::fill(b.begin(), b.end(), 0.0f);
        std::array<double, 4> maximum{}, squared{};
        double elapsed = 0;
        const size_t hops = (c.frames + 127) / 128;
        for (size_t hop = 0; hop <= hops; ++hop) {
          std::fill(buffers[0].begin(), buffers[0].end(), 0.0f);
          for (size_t ch = 0; ch < 2; ++ch)
            for (size_t i = 0; i < 128 && hop * 128 + i < c.frames; ++i)
              buffers[0][ch * 128 + i] = c.input[ch * c.frames + hop * 128 + i];
          const auto start = std::chrono::steady_clock::now();
          auto result = session.Run(Ort::RunOptions{nullptr}, inputs.data(),
                                    tensors.data(), 9, outputs.data(), 9);
          elapsed += std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - start)
                         .count();
          require(
              result[0].GetTensorTypeAndShapeInfo().GetElementCount() == 1024,
              "Wrong output shape");
          for (size_t s = 0; s < 9; ++s) {
            const auto info = result[s].GetTensorTypeAndShapeInfo();
            require(
                info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                "Wrong output precision");
            const auto* data = result[s].GetTensorData<float>();
            for (size_t i = 0; i < info.GetElementCount(); ++i)
              require(std::isfinite(data[i]), "Nonfinite output/state");
            if (s) {
              require(info.GetShape() == shapes[s], "Wrong state shape");
              std::copy(data, data + buffers[s].size(), buffers[s].begin());
            }
          }
          if (!hop)
            continue;
          const auto* data = result[0].GetTensorData<float>();
          for (size_t stem = 0; stem < 4; ++stem)
            for (size_t ch = 0; ch < 2; ++ch)
              for (size_t i = 0; i < 128 && (hop - 1) * 128 + i < c.frames;
                   ++i) {
                const size_t sample = (hop - 1) * 128 + i;
                const double error =
                    double(data[(stem * 2 + ch) * 128 + i]) -
                    c.expected[(stem * 2 + ch) * c.frames + sample];
                maximum[stem] = std::max(maximum[stem], std::abs(error));
                squared[stem] += error * error;
              }
        }
        const double maximumError =
            *std::max_element(maximum.begin(), maximum.end());
        std::cout << "{\"mode\":\"" << mode << "\",\"frames\":" << c.frames
                  << ",\"max_error\":" << maximumError << ",\"parity_pass\":"
                  << (maximumError <= 1e-5 ? "true" : "false")
                  << ",\"mean_run_ms\":" << elapsed * 1000 / double(hops + 1)
                  << ",\"stems\":[";
        for (size_t stem = 0; stem < 4; ++stem) {
          if (stem)
            std::cout << ',';
          std::cout << "{\"max_error\":" << maximum[stem] << ",\"rmse\":"
                    << std::sqrt(squared[stem] / double(2 * c.frames)) << '}';
        }
        std::cout << "]}" << std::endl;
      }
    }
    // A completed diagnostic may contain failed parity cases. Preserve each
    // verdict above; this exit reports completion, not platform qualification.
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Diagnostic error: " << e.what() << '\n';
    return 2;
  }
}
