#include "teeth_labeler.h"

TeethLabeler::TeethLabeler(std::string model)
{
    model_path = model;;
    label_num = -1;
}

void TeethLabeler::do_infer(std::shared_ptr<open3d::geometry::TriangleMesh> mesh, std::vector<int> &labels)
{
    torch::NoGradGuard no_grad;
    auto logger = new Logger();
    uint32_t flag = 1U <<static_cast<uint32_t> (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
    auto infer = new InferEngine(*logger, flag, model_path);
    cudaStream_t stream = infer->stream;
    nvinfer1::ICudaEngine* engine = infer->create_engine();
    std::cout << "Engine is created." << std::endl;
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    std::cout<<"Context is created."<<std::endl;

    auto mwf = getTensors(mesh);

    auto batch_size = 1;
    int points_num = mwf.size;

    std::cout<<"Inference is started."<<std::endl;
    auto out_tmp = do_inference(1, points_num, mwf, context, engine, stream);
    std::cout<<"Inference is ended."<<std::endl;

    std::cout << "Refining is started." << std::endl;
    auto label = cut_with_graph(out_tmp, mwf);
    std::cout << "Refining is ended." << std::endl;

    label_num = *torch::from_blob(label.data(), label.size(), torch::TensorOptions().dtype(torch::kInt32).requires_grad(false)).max().data_ptr<int>();

    labels.insert(labels.begin(), label.begin(), label.end());

    
}

TeethLabeler::~TeethLabeler()
{
}