#ifndef PROTON_PROFILER_GPU_PROFILER_H_
#define PROTON_PROFILER_GPU_PROFILER_H_

#include "Context/Context.h"
#include "Profiler.h"
#include "Utility/Atomic.h"
#include <thread>

#include <atomic>
#include <map>

namespace proton {

template <typename ConcreteProfilerT>
class GPUProfiler : public Profiler,
                    public OpInterface,
                    public Singleton<ConcreteProfilerT> {
public:
  GPUProfiler();
  virtual ~GPUProfiler();

protected:
  // OpInterface
  void startOp(const Scope &scope) override { pImpl->startOp(scope); }
  void stopOp(const Scope &scope) override { pImpl->stopOp(scope); }

  void setOpInProgress(bool value) override {
    profilerState.isRecording = value;
  }

  bool isOpInProgress() override { return profilerState.isRecording; }

  // Profiler
  virtual void doStart() override { pImpl->doStart(); }
  virtual void doFlush() override { pImpl->doFlush(); }
  virtual void doStop() override { pImpl->doStop(); }

  struct ProfilerState {
    ConcreteProfilerT &profiler;
    std::set<Data *> dataSet;
    size_t level{0};
    bool isRecording{false};
    Scope scope{};

    ProfilerState(ConcreteProfilerT &profiler) : profiler(profiler) {}

    void record(const Scope &scope) {
      this->scope = scope;
      // Take a snapshot of the current dataset
      this->dataSet = profiler.getDataSet();
    }

    void enterOp() {
      if (level == 0 && isRecording) {
        profiler.enterOp(scope);
        for (auto data : dataSet)
          data->enterOp(scope);
      }
      level++;
    }

    void exitOp() {
      level--;
      if (level == 0 && isRecording) {
        profiler.exitOp(scope);
        for (auto data : dataSet)
          data->exitOp(this->scope);
      }
    }
  };

  struct Correlation {
    // Max wait time = MAX_RETRIES * SLEEP_MS = 3s
    static inline const uint64_t MAX_RETRIES = 3000;
    static inline const uint64_t SLEEP_MS = 1;
    std::mutex mutex;
    std::atomic<uint64_t> maxSubmittedCorrelationId{0};
    std::atomic<uint64_t> maxCompletedCorrelationId{0};

    Correlation() = default;

    void submit(const uint64_t correlationId) {
      atomicMax(maxSubmittedCorrelationId, correlationId);
    }

    void complete(const uint64_t correlationId) {
      atomicMax(maxCompletedCorrelationId, correlationId);
    }

    template <typename FlushFnT> void flush(FlushFnT &&flushFn) {
      // We get the current max correlationId and wait until all the
      // operations with correlationId less than the current max
      // correlationId are completed.
      // If there's any threads that are not completing the operations,
      // we don't wait for them to submit operations to the GPU.
      // Although this is not ideal, it's better than waiting infinitely or
      // explicitly call cuda device synchronize when we have multiple devices.
      auto submittedId = maxSubmittedCorrelationId.load();
      auto completedId = maxCompletedCorrelationId.load();
      auto retries = MAX_RETRIES;
      while ((completedId < submittedId) && --retries > 0) {
        flushFn();
        std::this_thread::sleep_for(std::chrono::microseconds(SLEEP_MS));
        completedId = maxCompletedCorrelationId.load();
      }
    }
  };

  static thread_local ProfilerState profilerState;
  Correlation correlation;

  // Use the pimpl idiom to hide the implementation details. This lets us avoid
  // including the cupti header from this header. The cupti header and the
  // equivalent header from AMD define conflicting macros, so we want to use
  // those headers only within cpp files.
  class GPUProfilerPimplInterface {
  public:
    GPUProfilerPimplInterface(ConcreteProfilerT *profiler)
        : profiler(profiler) {}
    virtual ~GPUProfilerPimplInterface() = default;

    virtual void startOp(const Scope &scope) = 0;
    virtual void stopOp(const Scope &scope) = 0;
    virtual void doStart() = 0;
    virtual void doFlush() = 0;
    virtual void doStop() = 0;

  protected:
    ConcreteProfilerT *profiler;
  };
  std::unique_ptr<GPUProfilerPimplInterface> pImpl;
};

} // namespace proton

#endif // PROTON_PROFILER_CUPTI_PROFILER_H_
