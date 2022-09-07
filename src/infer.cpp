#include "infer.h"

torch::Tensor EigenMatrixToTorchTensor(Eigen::MatrixXd e){
    auto t = torch::rand({e.cols(),e.rows()}, torch::TensorOptions().requires_grad(false));
    float* data = t.data_ptr<float>();

    Eigen::Map<Eigen::MatrixXf> ef(data,t.size(1),t.size(0));
    ef = e.cast<float>();
    // t.requires_grad_(false);
    return t.transpose(0,1);
}

MeshWithFeature getTensors(std::shared_ptr<open3d::geometry::TriangleMesh> origin_mesh){ 
    // mesh->Translate(-mesh->GetCenter());
    auto mesh = origin_mesh->SimplifyQuadricDecimation(10000,std::numeric_limits<double>::infinity(),1.0);
    mesh->ComputeTriangleNormals(true);

    auto tmp_pts = mesh->vertices_;

    auto tmp_tri = mesh->triangles_;

    auto tmp_nor = mesh->triangle_normals_;

    Eigen::MatrixXd cells = Eigen::MatrixXd::Zero((int)tmp_tri.size(), 9);
    Eigen::MatrixX3d barycenters = Eigen::MatrixX3d::Zero((int)tmp_tri.size(), 3);
    #pragma omp parallel for
    for(auto &iter : tmp_tri) {
        auto index = &iter - &tmp_tri[0];
        Eigen::Matrix<double, 1, 9> sub_cord;
        sub_cord[0] = tmp_pts.at(iter[0])[0];
        sub_cord[1] = tmp_pts.at(iter[0])[1];
        sub_cord[2] = tmp_pts.at(iter[0])[2];
        sub_cord[3] = tmp_pts.at(iter[1])[0];
        sub_cord[4] = tmp_pts.at(iter[1])[1];
        sub_cord[5] = tmp_pts.at(iter[1])[2];
        sub_cord[6] = tmp_pts.at(iter[2])[0];
        sub_cord[7] = tmp_pts.at(iter[2])[1];
        sub_cord[8] = tmp_pts.at(iter[2])[2];

        Eigen::Matrix<double, 1, 3> sub_cent;
        sub_cent[0] = (sub_cord[0] + sub_cord[3] + sub_cord[6]) / 3;
        sub_cent[1] = (sub_cord[1] + sub_cord[4] + sub_cord[7]) / 3;
        sub_cent[2] = (sub_cord[2] + sub_cord[5] + sub_cord[8]) / 3;

        cells.row(index) = sub_cord;
        barycenters.row(index) = sub_cent;
    }

    Eigen::MatrixX3d mesh_normals = Eigen::MatrixX3d::Zero((int)tmp_nor.size(), 3);
    #pragma omp parallel for
    for(auto &iter : tmp_nor){mesh_normals.row(&iter - &tmp_nor[0]) = tmp_nor[&iter - &tmp_nor[0]];}

    Eigen::MatrixX3d points = Eigen::MatrixX3d::Zero((int)tmp_pts.size(), 3);
    #pragma omp parallel for
    for(auto &iter : tmp_pts){points.row(&iter - &tmp_pts[0]) = tmp_pts[&iter - &tmp_pts[0]];}

    auto maxs=points.colwise().maxCoeff();
    auto mins=points.colwise().minCoeff();
    auto means=points.colwise().mean();
    Eigen::Array3d stds;
    #pragma omp parallel for
    for(int i = 0; i < stds.size(); i++){stds[i]=sqrt(((points.col(i).array() - points.col(i).mean()).square().sum() / (points.col(i).size() - 1)));}

    auto normals=mesh_normals;
    auto nor_means=mesh_normals.colwise().mean();
    Eigen::Array3d nor_stds;
    #pragma omp parallel for
    for(int i = 0; i < nor_stds.size(); i++){nor_stds[i]=sqrt(((mesh_normals.col(i).array() - mesh_normals.col(i).mean()).square().sum() / (mesh_normals.col(i).size() - 1)));}

    Eigen::MatrixXd barycenters_copy = barycenters;

    #pragma omp parallel for
    for (int i=0; i<3; i++){
        cells.col(i) = (cells.col(i).array() - means[i]) / stds[i];
        cells.col(i+3) = (cells.col(i+3).array() - means[i]) / stds[i];
        cells.col(i+6) = (cells.col(i+6).array() - means[i]) / stds[i];
        barycenters.col(i) = (barycenters.col(i).array() - mins[i]) / (maxs[i]-mins[i]);
        normals.col(i) = (normals.col(i).array() - nor_means[i]) / nor_stds[i];
    }

    Eigen::MatrixXd X = Eigen::MatrixXd::Zero((int)tmp_tri.size(), cells.cols()+barycenters.cols()+mesh_normals.cols());
    X << cells, barycenters, normals;

    auto X_T = EigenMatrixToTorchTensor(barycenters);

    auto D_T = torch::cdist(X_T, X_T);
    D_T.requires_grad_(false);

    // Eigen::Map<MatrixXf_rm> D(D_T.data_ptr<float>(), D_T.size(0), D_T.size(1));

    auto D_T_a = D_T.accessor<float,2>();

    auto abc = std::chrono::high_resolution_clock::now();
    auto A_S = torch::zeros({X.rows(), X.rows()}, torch::TensorOptions().requires_grad(false));
    auto A_L = torch::zeros({X.rows(), X.rows()}, torch::TensorOptions().requires_grad(false));
    auto A_S_a = A_S.accessor<float,2>();
    auto A_L_a = A_L.accessor<float,2>();


    #pragma omp parallel for
    for (int i=0; i<D_T_a.size(0); i++){
        for (int j=0; j<D_T_a.size(1); j++){
            if (D_T_a[i][j] < 0.1){A_S_a[i][j]=1.0;}
            if (D_T_a[i][j] < 0.2){A_L_a[i][j]=1.0;}
        }
    }

    auto A_S_col_sum = torch::zeros({A_S_a.size(1),1}, torch::TensorOptions().requires_grad(false));
    auto A_S_cs_a = A_S_col_sum.accessor<float, 2>();
    #pragma omp parallel for
    for(int i=0; i<A_S_a.size(1); i++){
        A_S_col_sum[i] = A_S.index({"...", i}).sum();
    }

    auto A_L_col_sum = torch::zeros({A_L_a.size(1),1}, torch::TensorOptions().requires_grad(false));
    auto A_L_cs_a = A_L_col_sum.accessor<float, 2>();
    #pragma omp parallel for
    for(int i=0; i<A_L_a.size(1); i++){
        A_L_col_sum[i] = A_L.index({"...", i}).sum();
    }

    auto A_S_dot = torch::matmul(A_S_col_sum, torch::ones({1, A_S_a.size(1)}));
    auto A_L_dot = torch::matmul(A_L_col_sum, torch::ones({1, A_L_a.size(1)}));

    std::vector<at::Tensor> output_tensor;
    output_tensor.push_back((A_S / A_S_dot).ravel());
    output_tensor.push_back((A_L / A_L_dot).ravel());
    output_tensor.push_back(EigenMatrixToTorchTensor(X).transpose(0,1).ravel());

    struct MeshWithFeature mwf;
    mwf.mesh = origin_mesh;
    mwf.tensors = output_tensor;
    mwf.size = tmp_tri.size();
    mwf.sim_tri = mesh->triangles_;
    mwf.mesh_normals = mesh_normals;
    mwf.barycenters = barycenters_copy;
    return mwf;
}

torch::Tensor do_inference(int batch_size, int points_num, MeshWithFeature meshwithfeature, nvinfer1::IExecutionContext* context, nvinfer1::ICudaEngine* engine, cudaStream_t stream){
    context->setBindingDimensions(0, nvinfer1::Dims3(batch_size, 15, points_num));
    context->setBindingDimensions(1, nvinfer1::Dims3(batch_size, points_num, points_num));
    context->setBindingDimensions(2, nvinfer1::Dims3(batch_size, points_num, points_num));

    auto output = torch::zeros({batch_size, points_num, 17}, torch::TensorOptions().requires_grad(false)).ravel();

    const int inputIndex = engine->getBindingIndex("input");
    const int asIndex = engine->getBindingIndex("a_s");
    const int alIndex = engine->getBindingIndex("a_l");
    const int outputIndex = engine->getBindingIndex("output");

    void* buffers[4];

    auto A_S = meshwithfeature.tensors[0];
    auto A_L = meshwithfeature.tensors[1];
    auto X = meshwithfeature.tensors[2];

	cudaMalloc(&buffers[inputIndex], X.numel() * sizeof(float));
	cudaMalloc(&buffers[asIndex], A_S.numel() * sizeof(float));
	cudaMalloc(&buffers[alIndex], A_L.numel() * sizeof(float));
	cudaMalloc(&buffers[outputIndex], output.numel() * sizeof(float));

    std::vector<float> input_v(X.data_ptr<float>(), X.data_ptr<float>()+X.numel());
    float* input_f = &input_v[0];
    std::vector<float> a_s_v(A_S.data_ptr<float>(), A_S.data_ptr<float>()+A_S.numel());
    float* a_s_f = &a_s_v[0];
    std::vector<float> a_l_v(A_L.data_ptr<float>(), A_L.data_ptr<float>()+A_L.numel());
    float* a_l_f = &a_l_v[0];
    std::vector<float> output_v(output.data_ptr<float>(), output.data_ptr<float>()+output.numel());

    float* output_f = &output_v[0];

    cudaMemcpyAsync(buffers[0], input_f, X.numel()*sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buffers[1], a_s_f, A_S.numel()*sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buffers[2], a_l_f, A_L.numel()*sizeof(float), cudaMemcpyHostToDevice, stream);
    context->enqueueV2(buffers, stream, nullptr);
    cudaMemcpyAsync(output_f, buffers[3], output.numel()*sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return torch::from_blob(output_f, {batch_size, points_num, 17}, torch::TensorOptions().requires_grad(false));
}