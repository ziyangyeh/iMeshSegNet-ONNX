#ifndef __ENGINE_H__
#define __ENGINE_H__

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>

std::unique_ptr<char[]> load_engine_file(const std::string&, int*);

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
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::shared_ptr<nvinfer1::IExecutionContext> context;
    InferEngine(Logger, uint32_t, std::string);
    ~InferEngine();
};

#endif
