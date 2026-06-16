#include "gtpin_shared_library.hpp"

#include <iostream>

#ifdef _WIN32
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

namespace ov::intel_gpu::gtpin {

SharedLibrary::SharedLibrary(void* handle, bool unload_on_destroy)
    : m_handle(handle),
      m_unload_on_destroy(unload_on_destroy) {
    std::cerr << "[OV][GTPIN] SharedLibrary constructor. handle="
              << m_handle
              << ", unload_on_destroy="
              << (m_unload_on_destroy ? "true" : "false")
              << std::endl;
}

SharedLibrary::~SharedLibrary() {
    std::cerr << "[OV][GTPIN] SharedLibrary destructor. handle="
              << m_handle
              << ", unload_on_destroy="
              << (m_unload_on_destroy ? "true" : "false")
              << std::endl;

    if (!m_handle) {
        std::cerr << "[OV][GTPIN] SharedLibrary destructor skipped unload: null handle "
                  << std::endl;
        return;
    }

    if (!m_unload_on_destroy) {
        std::cerr << "[OV][GTPIN] SharedLibrary destructor keeping library loaded"
                  << std::endl;
        return;
    }

#ifdef _WIN32
    if (!FreeLibrary(static_cast<HMODULE>(m_handle))) {
        std::cerr << "[OV][GTPIN] FreeLibrary failed. error="
                  << GetLastError()
                  << std::endl;
    } else {
        std::cerr << "[OV][GTPIN] FreeLibrary succeeded" << std::endl;
    }
#else
    if (dlclose(m_handle) != 0) {
        std::cerr << "[OV][GTPIN] dlclose failed. error="
                  << dlerror()
                  << std::endl;
    } else {
        std::cerr << "[OV][GTPIN] dlclose succeeded" << std::endl;
    }
#endif

    m_handle = nullptr;
}

std::pair<std::unique_ptr<SharedLibrary>, bool>
SharedLibrary::load(const std::string& path,
                    bool global_symbols,
                    bool unload_on_destroy) {
    std::cerr << "[OV][GTPIN] SharedLibrary::load begin. path="
              << path
              << ", global_symbols="
              << (global_symbols ? "true" : "false")
              << ", unload_on_destroy="
              << (unload_on_destroy ? "true" : "false")
              << std::endl;

#ifdef _WIN32
    (void)global_symbols;

    HMODULE handle = LoadLibraryExA(
        path.c_str(),
        nullptr,
        LOAD_WITH_ALTERED_SEARCH_PATH);

    if (!handle) {
        std::cerr << "[OV][GTPIN] Failed to load library: "
                  << path
                  << ", error="
                  << GetLastError()
                  << std::endl;
        return {nullptr, false};
    }

    std::cerr << "[OV][GTPIN] SharedLibrary::load succeeded. path="
              << path
              << ", handle="
              << static_cast<void*>(handle)
              << std::endl;

    return {
        std::unique_ptr<SharedLibrary>(
            new SharedLibrary(static_cast<void*>(handle), unload_on_destroy)),
        true
    };
#else
    int flags = RTLD_NOW | (global_symbols ? RTLD_GLOBAL : RTLD_LOCAL);

    void* handle = dlopen(path.c_str(), flags);
    if (!handle) {
        std::cerr << "[OV][GTPIN] Failed to load library: "
                  << path
                  << ", error="
                  << dlerror()
                  << std::endl;
        return {nullptr, false};
    }

    std::cerr << "[OV][GTPIN] SharedLibrary::load succeeded. path="
              << path
              << ", handle="
              << handle
              << std::endl;

    return {
        std::unique_ptr<SharedLibrary>(
            new SharedLibrary(handle, unload_on_destroy)),
        true
    };
#endif
}

void* SharedLibrary::symbol(const std::string& name) const {
    std::cerr << "[OV][GTPIN] Resolving symbol: "
              << name
              << std::endl;

    if (!m_handle) {
        std::cerr << "[OV][GTPIN] Failed to resolve symbol: null library handle"
                  << std::endl;
        return nullptr;
    }

#ifdef _WIN32
    auto symbol = GetProcAddress(static_cast<HMODULE>(m_handle), name.c_str());
    if (!symbol) {
        std::cerr << "[OV][GTPIN] Failed to resolve symbol: "
                  << name
                  << ", error="
                  << GetLastError()
                  << std::endl;
        return nullptr;
    }

    std::cerr << "[OV][GTPIN] Resolved symbol: "
              << name
              << ", address="
              << reinterpret_cast<void*>(symbol)
              << std::endl;

    return reinterpret_cast<void*>(symbol);
#else
    dlerror();
    void* symbol = dlsym(m_handle, name.c_str());
    const char* error = dlerror();

    if (error) {
        std::cerr << "[OV][GTPIN] Failed to resolve symbol: "
                  << name
                  << ", error="
                  << error
                  << std::endl;
        return nullptr;
    }

    std::cerr << "[OV][GTPIN] Resolved symbol: "
              << name
              << ", address="
              << symbol
              << std::endl;

    return symbol;
#endif
}

}  // namespace ov::intel_gpu::gtpin
