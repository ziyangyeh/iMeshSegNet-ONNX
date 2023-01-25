#include "engine.h"
#include <fstream>
#include <iostream>
#include <sstream>

std::unique_ptr<char[]> load_engine_file(const std::string& engineFile, int* data_length)
{
    std::fstream file;

    file.open(engineFile, std::ios::binary | std::ios::in);
    if (!file.is_open())
    {
    std::cout << "read engine file" << engineFile << " failed" << std::endl;
    return nullptr;
    }
    file.seekg(0, std::ios::end);
    int length = file.tellg();
    file.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> data(new char[length]);
    file.read(data.get(), length);
    *data_length = length;

    file.close();

    return data;
}

void Logger::log(Severity severity, const char* msg) noexcept
{
    if (severity <= Severity::kWARNING)
        std::cout << msg << std::endl;
}

InferEngine::InferEngine(Logger tmp_logger, uint32_t tmp_flag, std::string tmp_path)
{
    flag = tmp_flag;
    path = tmp_path;

    std::unique_ptr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};

    std::unique_ptr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(flag)};

    std::string str(path);
    auto new_str = str.substr(0,str.length()-4)+"trt";
    std::ifstream f(new_str.c_str());
    bool ok = f.good();
    f.close();

    if(!ok){
        std::cout<<"Didn't find TRT file, start creating."<<std::endl;
        std::unique_ptr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, logger)};

        parser->parseFromFile(path.c_str(), 1);

        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();

        profile->setDimensions(network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1,15,1000));
        profile->setDimensions(network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1,15,10000));
        profile->setDimensions(network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(2,15,10000));
        profile->setDimensions(network->getInput(1)->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1,1000,1000));
        profile->setDimensions(network->getInput(1)->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1,10000,10000));
        profile->setDimensions(network->getInput(1)->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(2,10000,10000));
        profile->setDimensions(network->getInput(2)->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1,1000,1000));
        profile->setDimensions(network->getInput(2)->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1,10000,10000));
        profile->setDimensions(network->getInput(2)->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(2,10000,10000));

        std::unique_ptr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};

        config->addOptimizationProfile(profile);

        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30);

        std::unique_ptr<nvinfer1::IHostMemory> serializedModel{builder->buildSerializedNetwork(*network, *config)};

        std::ofstream ofs(new_str, std::ios::out | std::ios::binary);
        ofs.write((char*)(serializedModel ->data()), serializedModel ->size());
        ofs.close();

        std::cout<<"TRT file is created."<<std::endl;

        auto runtime{nvinfer1::createInferRuntime(logger)};
        engine.reset(runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size()));
        context.reset(engine->createExecutionContext()); 
    }
    else
    {
        std::cout<<"Found TRT file."<<std::endl;

        std::ifstream planFile(new_str);
        std::stringstream planBuffer;
        planBuffer << planFile.rdbuf();
        std::string plan = planBuffer.str();
        // int engine_data_length = 0;
        // std::unique_ptr<char[]> plan = load_engine_file(new_str, &engine_data_length);

        auto runtime{nvinfer1::createInferRuntime(logger)};
        // engine.reset(runtime->deserializeCudaEngine(plan.get(), engine_data_length));
        engine.reset(runtime->deserializeCudaEngine(plan.data(), plan.size()));
        context.reset(engine->createExecutionContext());
    }
}

InferEngine::~InferEngine()
{
}