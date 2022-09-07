
#ifndef __GCO_H__
#define __GCO_H__

#include "infer.h"

#include "GCoptimization.h"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXi_rm;

std::vector<int> cut_with_graph(torch::Tensor, MeshWithFeature);

#endif