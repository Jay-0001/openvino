#pragma once

#include <memory>

//new namespace for gtpin -- understanding it's necessity
namespace ov::intel_gpu::gtpin {

class GtpinSession;

class GtpinProfiler {
public:
    GtpinProfiler();
    ~GtpinProfiler();
    
    bool initialize();

    bool enabled() const;

private:
    std::unique_ptr<GtpinSession> m_session;
};

// New orchestration entry point
void initialize_once();

}