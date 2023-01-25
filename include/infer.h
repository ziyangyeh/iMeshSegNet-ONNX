#ifndef __INFER_H__
#define __INFER_H__

#include "engine.h"

#include <torch/torch.h>
#include "open3d/Open3D.h"
#include "open3d/3rdparty/Eigen/Eigen"
#include "open3d/3rdparty/Eigen/Core"

#include <vector>

struct MeshWithFeature{
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh;
    std::vector<at::Tensor> tensors;
    std::vector<Eigen::Vector3i> sim_tri;
    Eigen::MatrixX3d mesh_normals;
    Eigen::MatrixXd barycenters;
    int size;
};

torch::Tensor EigenMatrixToTorchTensor(Eigen::MatrixXd);
std::shared_ptr<MeshWithFeature> getTensors(std::shared_ptr<open3d::geometry::TriangleMesh>);
torch::Tensor do_inference(int, int, std::shared_ptr<MeshWithFeature>, std::shared_ptr<nvinfer1::IExecutionContext>, std::shared_ptr<nvinfer1::ICudaEngine>);

#endif