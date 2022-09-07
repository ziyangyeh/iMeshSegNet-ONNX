#ifndef __ENGINE_H__
#define __ENGINE_H__

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <string>

class Logger : public nvinfer1::ILogger           
{
public:
    void log(Severity severity, const char* msg) noexcept override;
};

class InferEngine
{
private:
    Logger logger;
    uint32_t flag;
    std::string path;
public:
    cudaStream_t stream;
    InferEngine(Logger, uint32_t, std::string);
    nvinfer1::ICudaEngine* create_engine();
    ~InferEngine();
};

#endif
