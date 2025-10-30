// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <level_zero/ze_intel_gpu.h>

namespace torch {
namespace comms {

// Use Level Zero native types directly
using xpuStream_t = ze_command_queue_handle_t;

// Device properties structure - simplified mapping
struct xpuDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    int major;
    int minor;
    int multiProcessorCount;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
};

// Graph-related types (placeholder - unsupported in Level Zero)
using xpuGraph_t = void*;
using xpuGraphNode_t = void*;
using xpuUserObject_t = void*;
using xpuHostFn_t = void(*)(void*);

#define XPU_CHECK(xpu_api, call, err_str)                               \
  do {                                                                    \
    ze_result_t status = call;                                            \
    if (status != ZE_RESULT_SUCCESS) {                                          \
      std::stringstream ss;                                               \
      ss << err_str << ": " << xpu_api->getErrorString(status) << " at " \
         << __FILE__ << ":" << __LINE__;                                  \
      throw std::runtime_error(ss.str());                                 \
    }                                                                     \
  } while (0)

/**
 * Abstract interface for XPU API operations.
 * This allows for dependency injection and testing by providing
 * a way to override XPU API calls.
 */
class XpuApi {
 public:
  virtual ~XpuApi() = default;

  // Device management
  virtual ze_result_t setDevice(int device) = 0;
  virtual ze_result_t getDeviceProperties(xpuDeviceProp* prop, int device) = 0;
  virtual ze_result_t memGetInfo(size_t* free, size_t* total) = 0;
  virtual ze_result_t getDeviceCount(int* count) = 0;

  // Stream management
  virtual ze_result_t streamCreateWithPriority(
      xpuStream_t* pStream,
      unsigned int flags,
      int priority) = 0;
  virtual ze_result_t streamDestroy(xpuStream_t stream) = 0;
  virtual ze_result_t streamWaitEvent(
      xpuStream_t stream,
      ze_event_handle_t event,
      unsigned int flags) = 0;
  virtual xpuStream_t getCurrentXPUStream(int device_index) = 0;
  virtual ze_result_t streamSynchronize(xpuStream_t stream) = 0;

  virtual ze_result_t malloc(void** devPtr, size_t size) = 0;
  virtual ze_result_t free(void* devPtr) = 0;
  virtual ze_result_t memcpyAsync(
      void* dst,
      const void* src,
      size_t count,
      xpuStream_t stream) = 0;

  // Event management
  virtual ze_result_t eventCreate(ze_event_handle_t* event) = 0;
  virtual ze_result_t eventCreateWithFlags(
      ze_event_handle_t* event,
      unsigned int flags) = 0;
  virtual ze_result_t eventDestroy(ze_event_handle_t event) = 0;
  virtual ze_result_t eventRecord(ze_event_handle_t event, xpuStream_t stream) = 0;
  virtual ze_result_t eventQuery(ze_event_handle_t event) = 0;

  // Error handling
  virtual const char* getErrorString(ze_result_t error) = 0;
};

/**
 * Default implementation that calls the underlying XPU APIs directly.
 */
class DefaultXpuApi : public XpuApi {
 public:
  ~DefaultXpuApi() override = default;

  // Device management
  ze_result_t setDevice(int device) override;
  ze_result_t getDeviceProperties(xpuDeviceProp* prop, int device) override;
  ze_result_t memGetInfo(size_t* free, size_t* total) override;
  ze_result_t getDeviceCount(int* count) override;

  // Stream management
  virtual ze_result_t streamCreateWithPriority(
      xpuStream_t* pStream,
      unsigned int flags,
      int priority) override;
  ze_result_t streamDestroy(xpuStream_t stream) override;
  ze_result_t streamWaitEvent(
      xpuStream_t stream,
      ze_event_handle_t event,
      unsigned int flags) override;
  xpuStream_t getCurrentXPUStream(int device_index) override;
  ze_result_t streamSynchronize(xpuStream_t stream) override;

  // Memory management
  ze_result_t malloc(void** devPtr, size_t size) override;
  ze_result_t free(void* devPtr) override;
  ze_result_t memcpyAsync(
      void* dst,
      const void* src,
      size_t count,
      xpuStream_t stream) override;

  // Event management
  ze_result_t eventCreate(ze_event_handle_t* event) override;
  ze_result_t eventCreateWithFlags(ze_event_handle_t* event, unsigned int flags)
      override;
  ze_result_t eventDestroy(ze_event_handle_t event) override;
  ze_result_t eventRecord(ze_event_handle_t event, xpuStream_t stream) override;
  ze_result_t eventQuery(ze_event_handle_t event) override;

  // Error handling
  const char* getErrorString(ze_result_t error) override;
};

} // namespace comms
} // namespace torch
