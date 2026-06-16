#pragma once

#include <memory>
#include <string>
#include <utility>

namespace ov::intel_gpu::gtpin {

class SharedLibrary {
public:
    ~SharedLibrary();

    static std::pair<std::unique_ptr<SharedLibrary>, bool>
    load(const std::string& path,
         bool global_symbols = false,
         bool unload_on_destroy = true);

    void* symbol(const std::string& name) const;

private:
    SharedLibrary(void* handle,
                  bool unload_on_destroy);

private:
    void* m_handle = nullptr;
    bool m_unload_on_destroy = true;
};

}  // namespace ov::intel_gpu::gtpin