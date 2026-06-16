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
      m_unload_on_destroy(unload_on_destroy) {}

SharedLibrary::~SharedLibrary() {
    if (!m_handle || !m_unload_on_destroy) {
        return;
    }

#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(m_handle));
#else
    dlclose(m_handle);
#endif

    m_handle = nullptr;
}

std::pair<std::unique_ptr<SharedLibrary>, bool>
SharedLibrary::load(const std::string& path,
                    bool global_symbols,
                    bool unload_on_destroy) {
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

    return {
        std::unique_ptr<SharedLibrary>(
            new SharedLibrary(handle, unload_on_destroy)),
        true
    };
#endif
}

void* SharedLibrary::symbol(const std::string& name) const {
    if (!m_handle) {
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
    }
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

    return symbol;
#endif
}

}  // namespace ov::intel_gpu::gtpin