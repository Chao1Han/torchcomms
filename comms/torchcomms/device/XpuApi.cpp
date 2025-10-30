// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/device/XpuApi.hpp"
#include <ATen/xpu/XPUContext.h>
#include <level_zero/ze_intel_gpu.h>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace torch {
namespace comms {

// Global Level Zero context management
static ze_driver_handle_t g_driver = nullptr;
static ze_device_handle_t* g_devices = nullptr;
static uint32_t g_device_count = 0;
static ze_context_handle_t g_context = nullptr;

static ze_result_t initLevelZero() {
    static bool initialized = false;
    if (initialized) return ZE_RESULT_SUCCESS;
    
    uint32_t driver_count = 0;
    ze_init_driver_type_desc_t desc = {};
    desc.flags = ZE_INIT_DRIVER_TYPE_FLAG_GPU;
    
    ze_result_t result = zeInitDrivers(&driver_count, nullptr, &desc);
    if (result != ZE_RESULT_SUCCESS || driver_count == 0) {
        return ZE_RESULT_ERROR_UNINITIALIZED;
    }
    
    std::vector<ze_driver_handle_t> drivers(driver_count);
    result = zeInitDrivers(&driver_count, drivers.data(), &desc);
    if (result != ZE_RESULT_SUCCESS) return result;
    
    // Find the first driver with devices
    std::vector<ze_device_handle_t> all_devices;
    for (const auto& driver : drivers) {
        uint32_t device_count = 0;
        ze_result_t dev_result = zeDeviceGet(driver, &device_count, nullptr);
        if (dev_result != ZE_RESULT_SUCCESS || device_count == 0) {
            continue; // No devices found for this driver
        }
        
        g_driver = driver;
        all_devices.resize(device_count);
        dev_result = zeDeviceGet(driver, &device_count, all_devices.data());
        if (dev_result == ZE_RESULT_SUCCESS) {
            break;
        }
    }
    
    if (all_devices.empty()) {
        return ZE_RESULT_ERROR_UNINITIALIZED;
    }
    
    // Filter out iGPUs, keep only dGPUs
    auto is_igpu = [](ze_device_handle_t device) -> bool {
        ze_device_properties_t device_prop = {};
        device_prop.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        ze_result_t prop_result = zeDeviceGetProperties(device, &device_prop);
        if (prop_result != ZE_RESULT_SUCCESS) {
            return false; // Assume not iGPU if we can't get properties
        }
        return (device_prop.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) != 0;
    };
    
    // Case 1: No GPUs found - should not happen as we already checked
    if (all_devices.empty()) {
        return ZE_RESULT_ERROR_UNINITIALIZED;
    }
    
    // Case 2: No dGPU found, but iGPU is available - return error (we want dGPU only)
    if (std::all_of(all_devices.begin(), all_devices.end(), is_igpu)) {
        return ZE_RESULT_ERROR_UNINITIALIZED; // Only iGPUs found, not suitable
    }
    
    // Case 3: dGPU found - remove all iGPUs, keep only dGPUs
    all_devices.erase(
        std::remove_if(all_devices.begin(), all_devices.end(), is_igpu), 
        all_devices.end()
    );
    
    // Copy filtered devices to global array
    g_device_count = all_devices.size();
    g_devices = new ze_device_handle_t[g_device_count];
    for (uint32_t i = 0; i < g_device_count; ++i) {
        g_devices[i] = all_devices[i];
    }
    
    // Create context with the driver
    ze_context_desc_t context_desc = {};
    context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    result = zeContextCreate(g_driver, &context_desc, &g_context);
    
    initialized = (result == ZE_RESULT_SUCCESS);
    return result;
}

// DefaultXpuApi implementation

ze_result_t DefaultXpuApi::setDevice(int device) {
    ze_result_t result = initLevelZero();
    if (result != ZE_RESULT_SUCCESS) return result;
    
    if (device < 0 || device >= (int)g_device_count) {
        return ZE_RESULT_ERROR_INVALID_ARGUMENT;
    }
    
    // Level Zero doesn't have a global "set device" concept like CUDA
    // Device selection happens at command queue creation time
    return ZE_RESULT_SUCCESS;
}

ze_result_t DefaultXpuApi::getDeviceProperties(
    xpuDeviceProp* prop,
    int device) {
    ze_result_t result = initLevelZero();
    if (result != ZE_RESULT_SUCCESS) return result;
    
    if (device < 0 || device >= (int)g_device_count || !prop) {
        return ZE_RESULT_ERROR_INVALID_ARGUMENT;
    }
    
    ze_device_properties_t ze_props = {};
    ze_props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    result = zeDeviceGetProperties(g_devices[device], &ze_props);
    if (result != ZE_RESULT_SUCCESS) return result;
    
    // Map Level Zero properties to XPU format
    strncpy(prop->name, ze_props.name, 255);
    prop->name[255] = '\0';
    prop->multiProcessorCount = ze_props.numSlices * ze_props.numSubslicesPerSlice;
    prop->major = ze_props.deviceId >> 16;
    prop->minor = ze_props.deviceId & 0xFFFF;
    
    // Get memory properties
    ze_device_memory_properties_t mem_props = {};
    mem_props.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
    uint32_t mem_count = 1;
    zeDeviceGetMemoryProperties(g_devices[device], &mem_count, &mem_props);
    prop->totalGlobalMem = mem_props.totalSize;
    
    return ZE_RESULT_SUCCESS;
}

ze_result_t DefaultXpuApi::memGetInfo(size_t* free, size_t* total) {
    ze_result_t result = initLevelZero();
    if (result != ZE_RESULT_SUCCESS) return result;
    
    if (!free || !total) return ZE_RESULT_ERROR_INVALID_NULL_POINTER;
    
    // Level Zero doesn't provide direct memory usage info
    // This would require querying device memory properties
    if (g_device_count > 0) {
        ze_device_memory_properties_t mem_props = {};
        mem_props.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
        uint32_t mem_count = 1;
        result = zeDeviceGetMemoryProperties(g_devices[0], &mem_count, &mem_props);
        if (result == ZE_RESULT_SUCCESS) {
            *total = mem_props.totalSize;
            *free = mem_props.totalSize; // Approximation - Level Zero doesn't track usage
            return ZE_RESULT_SUCCESS;
        }
    }
    
    return ZE_RESULT_ERROR_UNINITIALIZED;
}

ze_result_t DefaultXpuApi::getDeviceCount(int* count) {
    ze_result_t result = initLevelZero();
    if (result != ZE_RESULT_SUCCESS) return result;
    
    if (!count) return ZE_RESULT_ERROR_INVALID_NULL_POINTER;
    
    *count = (int)g_device_count;
    return ZE_RESULT_SUCCESS;
}

ze_result_t DefaultXpuApi::streamCreateWithPriority(
    xpuStream_t* pStream,
    unsigned int flags,
    int priority) {
    ze_result_t result = initLevelZero();
    if (result != ZE_RESULT_SUCCESS) return result;
    
    if (!pStream) return ZE_RESULT_ERROR_INVALID_NULL_POINTER;
    if (g_device_count == 0) return ZE_RESULT_ERROR_UNINITIALIZED;
    
    ze_command_queue_desc_t queue_desc = {};
    queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    queue_desc.flags = ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY;
    queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    
    // Map priority (Level Zero uses different priority enum)
    if (priority < 0) {
        queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH;
    } else if (priority == 0) {
        queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    } else {
        queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW;
    }
    
    return zeCommandQueueCreate(g_context, g_devices[0], &queue_desc, pStream);
}

ze_result_t DefaultXpuApi::streamDestroy(xpuStream_t stream) {
    if (!stream) return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    return zeCommandQueueDestroy(stream);
}

ze_result_t DefaultXpuApi::streamWaitEvent(
    xpuStream_t stream,
    ze_event_handle_t event,
    unsigned int flags) {
    if (!stream || !event) return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    
    // Create a command list to add the wait command
    ze_command_list_desc_t cmd_desc = {};
    cmd_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    cmd_desc.flags = ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING;
    
    ze_command_list_handle_t cmd_list;
    ze_result_t result = zeCommandListCreate(g_context, g_devices[0], &cmd_desc, &cmd_list);
    if (result != ZE_RESULT_SUCCESS) return result;
    
    result = zeCommandListAppendWaitOnEvents(cmd_list, 1, &event);
    if (result != ZE_RESULT_SUCCESS) {
        zeCommandListDestroy(cmd_list);
        return result;
    }
    
    result = zeCommandListClose(cmd_list);
    if (result != ZE_RESULT_SUCCESS) {
        zeCommandListDestroy(cmd_list);
        return result;
    }
    
    result = zeCommandQueueExecuteCommandLists(stream, 1, &cmd_list, nullptr);
    zeCommandListDestroy(cmd_list);
    return result;
}

xpuStream_t DefaultXpuApi::getCurrentXPUStream(int device_index) {
    // This should integrate with PyTorch's XPU stream management
    // For now, return a placeholder - this needs PyTorch integration
    return at::xpu::getCurrentXPUStream(device_index).stream();
}

ze_result_t DefaultXpuApi::streamSynchronize(xpuStream_t stream) {
    if (!stream) return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    return zeCommandQueueSynchronize(stream, UINT64_MAX);
}

ze_result_t DefaultXpuApi::streamIsCapturing(
    xpuStream_t stream,
    xpuStreamCaptureStatus* pCaptureStatus) {
    if (!pCaptureStatus) return ZE_RESULT_ERROR_INVALID_NULL_POINTER;
    
    // Level Zero doesn't have graph capture - always return "not capturing"
    *pCaptureStatus = xpuStreamCaptureStatusNone;
    return ZE_RESULT_SUCCESS;
}

ze_result_t DefaultXpuApi::streamGetCaptureInfo(
    xpuStream_t stream,
    xpuStreamCaptureStatus* pCaptureStatus,
    unsigned long long* pId) {
    if (!pCaptureStatus) return ZE_RESULT_ERROR_INVALID_NULL_POINTER;
    
    *pCaptureStatus = xpuStreamCaptureStatusNone;
    if (pId) *pId = 0;
    return ZE_RESULT_SUCCESS;
}

ze_result_t DefaultXpuApi::userObjectCreate(
    xpuUserObject_t* object_out,
    void* ptr,
    xpuHostFn_t destroy,
    unsigned int initialRefcount,
    unsigned int flags) {
    // Level Zero doesn't have user objects - return unsupported
    return ZE_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ze_result_t DefaultXpuApi::graphRetainUserObject(
    xpuGraph_t graph,
    xpuUserObject_t object,
    unsigned int count,
    unsigned int flags) {
    // Level Zero doesn't have graphs - return unsupported
    return ZE_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ze_result_t DefaultXpuApi::streamGetCaptureInfo_v2(
    xpuStream_t stream,
    xpuStreamCaptureStatus* captureStatus_out,
    unsigned long long* id_out,
    xpuGraph_t* graph_out,
    const xpuGraphNode_t** dependencies_out,
    size_t* numDependencies_out) {
    if (captureStatus_out) *captureStatus_out = xpuStreamCaptureStatusNone;
    if (id_out) *id_out = 0;
    if (graph_out) *graph_out = nullptr;
    if (dependencies_out) *dependencies_out = nullptr;
    if (numDependencies_out) *numDependencies_out = 0;
    
    return ZE_RESULT_SUCCESS;
}

ze_result_t DefaultXpuApi::malloc(void** devPtr, size_t size) {
    ze_result_t result = initLevelZero();
    if (result != ZE_RESULT_SUCCESS) return result;
    
    if (!devPtr) return ZE_RESULT_ERROR_INVALID_NULL_POINTER;
    if (size == 0) return ZE_RESULT_ERROR_INVALID_SIZE;
    
    ze_device_mem_alloc_desc_t mem_desc = {};
    mem_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    mem_desc.flags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED;
    
    return zeMemAllocDevice(g_context, &mem_desc, size, 0, g_devices[0], devPtr);
}

ze_result_t DefaultXpuApi::free(void* devPtr) {
    if (!devPtr) return ZE_RESULT_ERROR_INVALID_NULL_POINTER;
    return zeMemFree(g_context, devPtr);
}

ze_result_t DefaultXpuApi::memcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    xpuStream_t stream) {
    if (!dst || !src || !stream) return ZE_RESULT_ERROR_INVALID_NULL_POINTER;
    if (count == 0) return ZE_RESULT_SUCCESS;
    
    ze_command_list_desc_t cmd_desc = {};
    cmd_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    cmd_desc.flags = ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING;
    
    ze_command_list_handle_t cmd_list;
    ze_result_t result = zeCommandListCreate(g_context, g_devices[0], &cmd_desc, &cmd_list);
    if (result != ZE_RESULT_SUCCESS) return result;
    
    result = zeCommandListAppendMemoryCopy(cmd_list, dst, src, count, nullptr, 0, nullptr);
    if (result != ZE_RESULT_SUCCESS) {
        zeCommandListDestroy(cmd_list);
        return result;
    }
    
    result = zeCommandListClose(cmd_list);
    if (result != ZE_RESULT_SUCCESS) {
        zeCommandListDestroy(cmd_list);
        return result;
    }
    
    result = zeCommandQueueExecuteCommandLists(stream, 1, &cmd_list, nullptr);
    zeCommandListDestroy(cmd_list);
    return result;
}

ze_result_t DefaultXpuApi::eventCreate(ze_event_handle_t* event) {
    ze_result_t result = initLevelZero();
    if (result != ZE_RESULT_SUCCESS) return result;
    
    if (!event) return ZE_RESULT_ERROR_INVALID_NULL_POINTER;
    
    // Create basic event pool (without timing)
    ze_event_pool_desc_t pool_desc = {};
    pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    pool_desc.count = 1;
    
    ze_event_pool_handle_t event_pool;
    result = zeEventPoolCreate(g_context, &pool_desc, 1, &g_devices[0], &event_pool);
    if (result != ZE_RESULT_SUCCESS) return result;
    
    ze_event_desc_t event_desc = {};
    event_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
    event_desc.index = 0;
    event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
    
    return zeEventCreate(event_pool, &event_desc, event);
}

ze_result_t DefaultXpuApi::eventCreateWithFlags(
    ze_event_handle_t* event,
    unsigned int flags) {
    ze_result_t result = initLevelZero();
    if (result != ZE_RESULT_SUCCESS) return result;
    
    if (!event) return ZE_RESULT_ERROR_INVALID_NULL_POINTER;
    
    // Check if timing is enabled (flags=1)
    bool enable_timing = (flags & 0x1) != 0;
    
    if (enable_timing) {
        // Try to use Intel extension for counter-based events with timing
        // This requires the Intel Level Zero extensions
        #ifdef ZEX_STRUCTURE_COUNTER_BASED_EVENT_DESC
        // Use counter-based event for timing support
        ze_event_pool_desc_t pool_desc = {};
        pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
        pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
        pool_desc.count = 1;
        
        ze_event_pool_handle_t event_pool;
        result = zeEventPoolCreate(g_context, &pool_desc, 1, &g_devices[0], &event_pool);
        if (result != ZE_RESULT_SUCCESS) return result;
        
        ze_event_desc_t event_desc = {};
        event_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
        event_desc.index = 0;
        event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
        event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
        
        return zeEventCreate(event_pool, &event_desc, event);
        #else
        // Fallback: create standard event with kernel timestamp flag
        ze_event_pool_desc_t pool_desc = {};
        pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
        pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
        pool_desc.count = 1;
        
        ze_event_pool_handle_t event_pool;
        result = zeEventPoolCreate(g_context, &pool_desc, 1, &g_devices[0], &event_pool);
        if (result != ZE_RESULT_SUCCESS) return result;
        
        ze_event_desc_t event_desc = {};
        event_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
        event_desc.index = 0;
        event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
        event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
        
        return zeEventCreate(event_pool, &event_desc, event);
        #endif
    } else {
        // Create standard event without timing
        return eventCreate(event);
    }
}

ze_result_t DefaultXpuApi::eventDestroy(ze_event_handle_t event) {
    if (!event) return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    return zeEventDestroy(event);
}

ze_result_t DefaultXpuApi::eventRecord(ze_event_handle_t event, xpuStream_t stream) {
    if (!event || !stream) return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    
    ze_command_list_desc_t cmd_desc = {};
    cmd_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    cmd_desc.flags = ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING;
    
    ze_command_list_handle_t cmd_list;
    ze_result_t result = zeCommandListCreate(g_context, g_devices[0], &cmd_desc, &cmd_list);
    if (result != ZE_RESULT_SUCCESS) return result;
    
    result = zeCommandListAppendSignalEvent(cmd_list, event);
    if (result != ZE_RESULT_SUCCESS) {
        zeCommandListDestroy(cmd_list);
        return result;
    }
    
    result = zeCommandListClose(cmd_list);
    if (result != ZE_RESULT_SUCCESS) {
        zeCommandListDestroy(cmd_list);
        return result;
    }
    
    result = zeCommandQueueExecuteCommandLists(stream, 1, &cmd_list, nullptr);
    zeCommandListDestroy(cmd_list);
    return result;
}

ze_result_t DefaultXpuApi::eventQuery(ze_event_handle_t event) {
    if (!event) return ZE_RESULT_ERROR_INVALID_NULL_HANDLE;
    
    ze_result_t status = zeEventQueryStatus(event);
    if (status == ZE_RESULT_SUCCESS) {
        return ZE_RESULT_SUCCESS;
    } else if (status == ZE_RESULT_NOT_READY) {
        return ZE_RESULT_NOT_READY;
    }
    return status; // Error case
}

const char* DefaultXpuApi::getErrorString(ze_result_t error) {
    switch (error) {
        case ZE_RESULT_SUCCESS: return "ZE_RESULT_SUCCESS";
        case ZE_RESULT_NOT_READY: return "ZE_RESULT_NOT_READY";
        case ZE_RESULT_ERROR_DEVICE_LOST: return "ZE_RESULT_ERROR_DEVICE_LOST";
        case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY: return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
        case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY: return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
        case ZE_RESULT_ERROR_UNINITIALIZED: return "ZE_RESULT_ERROR_UNINITIALIZED";
        case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE: return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
        case ZE_RESULT_ERROR_INVALID_ARGUMENT: return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
        case ZE_RESULT_ERROR_INVALID_NULL_HANDLE: return "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
        case ZE_RESULT_ERROR_INVALID_NULL_POINTER: return "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
        case ZE_RESULT_ERROR_INVALID_SIZE: return "ZE_RESULT_ERROR_INVALID_SIZE";
        case ZE_RESULT_ERROR_UNSUPPORTED_SIZE: return "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
        case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT: return "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
        default: return "Unknown Level Zero error";
    }
}

} // namespace comms
} // namespace torch
