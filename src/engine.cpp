#include "engine.h"
#include <fstream>
#include <iostream>
#include <sstream>

void Logger::log(Severity severity, const char* msg) noexcept
{
    if (severity <= Severity::kWARNING)
        std::cout << msg << std::endl;
}

InferEngine::InferEngine(Logger tmp_logger, uint32_t tmp_flag, std::string tmp_path)
{
    flag = tmp_flag;
    path = tmp_path;
    stream = nullptr;
}

nvinfer1::ICudaEngine* InferEngine::create_engine()
{
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);

    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);

    std::string str(path);
    auto new_str = str.substr(0,str.length()-4)+"trt";
    std::ifstream f(new_str.c_str());
    bool ok = f.good();

    if(!ok){
        std::cout<<"Didn't find TRT file, start creating."<<std::endl;
        nvonnxparser::IParser*  parser = nvonnxparser::createParser(*network, logger);

        parser->parseFromFile(path.c_str(), 1);

        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();

        profile->setDimensions(network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1,15,1000));
        profile->setDimensions(network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(2,15,5000));
        profile->setDimensions(network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(2,15,10000));
        profile->setDimensions(network->getInput(1)->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1,1000,1000));
        profile->setDimensions(network->getInput(1)->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(2,5000,5000));
        profile->setDimensions(network->getInput(1)->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(2,10000,10000));
        profile->setDimensions(network->getInput(2)->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1,1000,1000));
        profile->setDimensions(network->getInput(2)->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(2,5000,5000));
        profile->setDimensions(network->getInput(2)->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(2,10000,10000));

        nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

        config->addOptimizationProfile(profile);

        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30);

        nvinfer1::IHostMemory*  serializedModel = builder->buildSerializedNetwork(*network, *config);

        std::ofstream ofs(new_str.c_str(), std::ios::out | std::ios::binary);
        ofs.write((char*)(serializedModel ->data()), serializedModel ->size());
        ofs.close();

        std::cout<<"TRT file is created."<<std::endl;

        delete parser;
        delete network;
        delete config;
        delete builder;

        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
        nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size());
        delete serializedModel;
        return engine;
    }
    else
    {
        std::cout<<"Found TRT file."<<std::endl;
        std::ifstream planFile(new_str.c_str());

        std::stringstream planBuffer;
        planBuffer << planFile.rdbuf();
        std::string plan = planBuffer.str();
        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
        nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine((void*)plan.data(), plan.size());
        return engine;
    }
}

InferEngine::~InferEngine()
{
}