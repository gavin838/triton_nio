#include "Profiler/RoctracerProfiler.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Driver/GPU/Hip.h"
#include "Driver/GPU/Roctracer.h"
#include <hip/amd_detail/hip_runtime_prof.h>

#include <roctracer/roctracer.h>
#include <roctracer/roctracer_ext.h>
#include <roctracer/roctracer_hip.h>

#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>

#include <cxxabi.h>
#include <unistd.h>

namespace proton {

template <>
thread_local GPUProfiler<RoctracerProfiler>::ProfilerState
    GPUProfiler<RoctracerProfiler>::profilerState(
        RoctracerProfiler::instance());

namespace {

std::shared_ptr<Metric>
convertActivityToMetric(const roctracer_record_t *activity) {
  std::shared_ptr<Metric> metric;
  switch (activity->kind) {
  case kHipVdiCommandKernel: {
    metric = std::make_shared<KernelMetric>(
        static_cast<uint64_t>(activity->begin_ns),
        static_cast<uint64_t>(activity->end_ns), 1,
        static_cast<uint64_t>(activity->device_id),
        static_cast<uint64_t>(DeviceType::HIP));
    break;
  }
  default:
    break;
  }
  return metric;
}

void addMetric(size_t scopeId, std::set<Data *> &dataSet,
               const roctracer_record_t *activity) {
  for (auto *data : dataSet) {
    data->addMetric(scopeId, convertActivityToMetric(activity));
  }
}

void processActivityKernel(std::map<uint32_t, size_t> &correlation,
                           std::set<Data *> &dataSet,
                           const roctracer_record_t *activity) {
  auto correlationId = activity->correlation_id;
  // TODO: non-triton kernels
  if (correlation.find(correlationId) == correlation.end()) {
    return;
  }
  auto externalId = correlation[correlationId];
  addMetric(externalId, dataSet, activity);
  // Track correlation ids from the same stream and erase those < correlationId
  correlation.erase(correlationId);
}

void processActivity(std::map<uint32_t, size_t> &externalCorrelation,
                     std::set<Data *> &dataSet,
                     const roctracer_record_t *record) {
  switch (record->kind) {
  case 0x11F1: // Task - kernel enqueued by graph launch
  case kHipVdiCommandKernel: {
    processActivityKernel(externalCorrelation, dataSet, record);
    break;
  }
  default:
    break;
  }
}

} // namespace

namespace {

std::pair<bool, bool> matchKernelCbId(uint32_t cbId) {
  bool isRuntimeApi = false;
  bool isDriverApi = false;
  switch (cbId) {
  // TODO: switch to directly subscribe the APIs
  case HIP_API_ID_hipExtLaunchKernel:
  case HIP_API_ID_hipExtLaunchMultiKernelMultiDevice:
  case HIP_API_ID_hipExtModuleLaunchKernel:
  case HIP_API_ID_hipHccModuleLaunchKernel:
  case HIP_API_ID_hipLaunchCooperativeKernel:
  case HIP_API_ID_hipLaunchCooperativeKernelMultiDevice:
  case HIP_API_ID_hipLaunchKernel:
  case HIP_API_ID_hipModuleLaunchKernel:
  case HIP_API_ID_hipGraphLaunch:
  case HIP_API_ID_hipModuleLaunchCooperativeKernel:
  case HIP_API_ID_hipModuleLaunchCooperativeKernelMultiDevice: {
    isRuntimeApi = true;
    break;
  }
  default:
    break;
  }
  return std::make_pair(isRuntimeApi, isDriverApi);
}
// C++ symbol demangle
static inline const char *cxxDemangle(const char *symbol) {
  size_t funcnamesize;
  int status;
  const char *ret =
      (symbol != NULL)
          ? abi::__cxa_demangle(symbol, NULL, &funcnamesize, &status)
          : symbol;
  return (ret != NULL) ? ret : symbol;
}

const char *kernelName(uint32_t domain, uint32_t cid,
                       const void *callback_data) {
  const char *name = "";
  if (domain == ACTIVITY_DOMAIN_HIP_API) {
    const hip_api_data_t *data = (const hip_api_data_t *)(callback_data);
    switch (cid) {
    case HIP_API_ID_hipExtLaunchKernel: {
      auto &params = data->args.hipExtLaunchKernel;
      name = cxxDemangle(
          hip::getKernelNameRefByPtr(params.function_address, params.stream));
    } break;
    case HIP_API_ID_hipExtLaunchMultiKernelMultiDevice: {
      auto &params =
          data->args.hipExtLaunchMultiKernelMultiDevice.launchParamsList__val;
      name =
          cxxDemangle(hip::getKernelNameRefByPtr(params.func, params.stream));
    } break;
    case HIP_API_ID_hipExtModuleLaunchKernel: {
      auto &params = data->args.hipExtModuleLaunchKernel;
      name = cxxDemangle(hip::getKernelNameRef(params.f));
    } break;
    case HIP_API_ID_hipHccModuleLaunchKernel: {
      auto &params = data->args.hipHccModuleLaunchKernel;
      name = cxxDemangle(hip::getKernelNameRef(params.f));
    } break;
    case HIP_API_ID_hipLaunchCooperativeKernel: {
      auto &params = data->args.hipLaunchCooperativeKernel;
      name = cxxDemangle(hip::getKernelNameRefByPtr(params.f, params.stream));
    } break;
    case HIP_API_ID_hipLaunchCooperativeKernelMultiDevice: {
      auto &params = data->args.hipLaunchCooperativeKernelMultiDevice
                         .launchParamsList__val;
      name =
          cxxDemangle(hip::getKernelNameRefByPtr(params.func, params.stream));
    } break;
    case HIP_API_ID_hipLaunchKernel: {
      auto &params = data->args.hipLaunchKernel;
      name = cxxDemangle(
          hip::getKernelNameRefByPtr(params.function_address, params.stream));
    } break;
    case HIP_API_ID_hipModuleLaunchKernel: {
      auto &params = data->args.hipModuleLaunchKernel;
      name = cxxDemangle(hip::getKernelNameRef(params.f));
    } break;
    case HIP_API_ID_hipGraphLaunch: {
      name = "graphLaunch";
    } break;
    default:;
    }
  }
  return name;
}

} // namespace

struct RoctracerProfiler::RoctracerProfilerPimpl
    : public GPUProfiler<RoctracerProfiler>::GPUProfilerPimplInterface {
  RoctracerProfilerPimpl(RoctracerProfiler *profiler)
      : GPUProfiler<RoctracerProfiler>::GPUProfilerPimplInterface(profiler) {}
  virtual ~RoctracerProfilerPimpl() = default;

  void startOp(const Scope &scope);
  void stopOp(const Scope &scope);

  void doStart();
  void doFlush();
  void doStop();

  static void apiCallback(uint32_t domain, uint32_t cid,
                          const void *callbackData, void *arg);
  static void activityCallback(const char *begin, const char *end, void *arg);

  const inline static size_t BufferSize = 64 * 1024 * 1024;

  std::map<uint32_t, size_t> externalCorrelation;
};

void RoctracerProfiler::RoctracerProfilerPimpl::apiCallback(
    uint32_t domain, uint32_t cid, const void *callback_data, void *arg) {
  auto [isRuntimeAPI, isDriverAPI] = matchKernelCbId(cid);
  if (!(isRuntimeAPI || isDriverAPI)) {
    return;
  }
  auto &profiler =
      dynamic_cast<RoctracerProfiler &>(RoctracerProfiler::instance());
  auto &correlation = profiler.correlation;
  if (domain == ACTIVITY_DOMAIN_HIP_API) {
    const hip_api_data_t *data = (const hip_api_data_t *)(callback_data);
    if (data->phase == ACTIVITY_API_PHASE_ENTER) {
      // Valid context and outermost level of the kernel launch
      const char *name = kernelName(domain, cid, callback_data);
      // roctracer::getOpString(ACTIVITY_DOMAIN_HIP_API, cid, 0);	//
      // proper api name
      auto scopeId = Scope::getNewScopeId();
      auto scope = Scope(scopeId, name);
      profilerState.record(scope);
      profilerState.enterOp();
    } else if (data->phase == ACTIVITY_API_PHASE_EXIT) {
      profilerState.exitOp();
      // Track outstanding op for flush
      correlation.submit(data->correlation_id);
    }
  }
}

void RoctracerProfiler::RoctracerProfilerPimpl::activityCallback(
    const char *begin, const char *end, void *arg) {
  RoctracerProfiler &profiler =
      dynamic_cast<RoctracerProfiler &>(RoctracerProfiler::instance());
  auto &pImpl = dynamic_cast<RoctracerProfilerPimpl &>(*profiler.pImpl.get());
  auto &externalCorrelation = pImpl.externalCorrelation;
  auto &dataSet = profiler.dataSet;
  auto &correlation = profiler.correlation;

  const roctracer_record_t *record =
      reinterpret_cast<const roctracer_record_t *>(begin);
  const roctracer_record_t *endRecord =
      reinterpret_cast<const roctracer_record_t *>(end);
  uint64_t maxCorrelationId = 0;

  while (record < endRecord) {
    // Log latest completed correlation id.  Used to ensure we have flushed all
    // data on stop
    maxCorrelationId =
        std::max<uint64_t>(maxCorrelationId, record->correlation_id);
    processActivity(externalCorrelation, dataSet, record);
    roctracer::getNextRecord<true>(record, &record);
  }
  correlation.complete(maxCorrelationId);
}

void RoctracerProfiler::RoctracerProfilerPimpl::startOp(const Scope &scope) {}

void RoctracerProfiler::RoctracerProfilerPimpl::stopOp(const Scope &scope) {}

void RoctracerProfiler::RoctracerProfilerPimpl::doStart() {
  roctracer::enableDomainCallback<true>(ACTIVITY_DOMAIN_HIP_API, apiCallback,
                                        nullptr);
  // Activity Records
  roctracer_properties_t properties{0};
  properties.buffer_size = BufferSize;
  properties.buffer_callback_fun = activityCallback;
  roctracer::openPool<true>(&properties);
  roctracer::enableDomainActivity<true>(ACTIVITY_DOMAIN_HIP_OPS);
  roctracer::start();
}

void RoctracerProfiler::RoctracerProfilerPimpl::doFlush() {
  // Implement reliable flushing.
  // Wait for all dispatched ops to be reported
  roctracer::flushActivity<true>();
  profiler->correlation.flush([]() { roctracer::flushActivity<true>(); });
}

void RoctracerProfiler::RoctracerProfilerPimpl::doStop() {
  roctracer::stop();
  roctracer::disableDomainCallback<true>(ACTIVITY_DOMAIN_HIP_API);
  roctracer::disableDomainActivity<true>(ACTIVITY_DOMAIN_HIP_OPS);
  roctracer::closePool<true>();
}

RoctracerProfiler::RoctracerProfiler() {
  pImpl = std::make_unique<RoctracerProfilerPimpl>(this);
}

RoctracerProfiler::~RoctracerProfiler() = default;

} // namespace proton
