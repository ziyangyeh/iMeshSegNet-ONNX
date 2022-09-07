#include "gco.h"

std::vector<int> cut_with_graph(torch::Tensor out_tmp, MeshWithFeature mesh){
    auto batch_size = 1;
    auto round_factor = 100;

    auto ota = out_tmp.accessor<float,3>();
    auto ota_thresh = 1.0e-6;
    #pragma omp parallel for
    for(int b = 0; b < batch_size; b++){
        for(int i = 0; i < ota.size(1); i++) {
            for(int j = 0; j < ota.size(2); j++) {
                if(ota[0][i][j]<ota_thresh){ota[0][i][j]=ota_thresh;}
            }
        }
    }
    auto unaries = (torch::log10(out_tmp) * -round_factor).reshape({-1, 17}).to(torch::kInt32);
    auto pairwise = (1 - torch::eye(17)).to(torch::kInt32);;

    auto cell_ids = torch::zeros({(signed long)mesh.size, 3}, torch::TensorOptions().dtype(torch::kInt32));
    Eigen::Map<MatrixXi_rm> et(cell_ids.data_ptr<int>(),cell_ids.size(0),cell_ids.size(1));
    et = Eigen::Map<MatrixXi_rm>(reinterpret_cast<int*>(mesh.sim_tri.data()),mesh.size,3);

    auto lambda_c = 30;
    auto edges = torch::empty({1, 3});

    auto normals_t = EigenMatrixToTorchTensor(mesh.mesh_normals);
    auto barycenters_t = EigenMatrixToTorchTensor(mesh.barycenters);

    auto pi = 3.1415926;

    #pragma omp parallel for ordered
    for(int i = 0; i < cell_ids.size(0); i++){
        auto nei = torch::sum(torch::isin(cell_ids, cell_ids.index({i})), 1);
        auto nei_id = torch::where(nei==2)[0].to(torch::kInt32);
        auto nei_id_ptr = nei_id.data_ptr<int>();
        for(int j = 0; j < nei_id.size(0); j++){
            #pragma omp ordered
            if(i < nei_id_ptr[j]){
                auto cos_theta = torch::dot(normals_t[i], normals_t[nei_id_ptr[j]])/torch::norm(normals_t[i])/torch::norm(normals_t[nei_id_ptr[j]]);
                if(*cos_theta.data_ptr<float>() >= 1.0){*cos_theta.data_ptr<float>() = 0.9999;}
                auto theta = torch::arccos(cos_theta);
                auto phi = torch::norm(barycenters_t[i] - barycenters_t[nei_id_ptr[j]]);
                if(*theta.data_ptr<float>() > pi/2.0){
                    float tmp_t[3] = {(float)i, (float)nei_id_ptr[j], -*(torch::log10(theta/pi)*phi).data_ptr<float>()};
                    auto tharray = torch::zeros({1,3},torch::kFloat32); //or use kF64
                    std::memcpy(tharray.data_ptr(),tmp_t,sizeof(double)*tharray.numel());
                    tharray.reshape({1,3});
                    at::Tensor tensors[2] = {edges, tharray};
                    edges = torch::concat(tensors, 0);
                }
                else{
                    auto beta = 1 + torch::norm(torch::dot(normals_t[i], normals_t[nei_id_ptr[j]]));
                    float tmp_t[3] = {(float)i, (float)nei_id_ptr[j], -*(beta*torch::log10(theta/pi)*phi).data_ptr<float>()};
                    auto tharray = torch::zeros({1,3},torch::kFloat32); //or use kF64
                    std::memcpy(tharray.data_ptr(),tmp_t,sizeof(double)*tharray.numel());
                    tharray.reshape({1,3});
                    at::Tensor tensors[2] = {edges, tharray};
                    edges = torch::concat(tensors, 0);
                }
            }
        }
    }
    edges = edges.index({torch::indexing::Slice({1, torch::indexing::None})});
    auto e_a = edges.accessor<float, 2>();
    #pragma omp parallel for
    for(int i = 0; i < e_a.size(0); i++){e_a[i][2] *= lambda_c*round_factor;}
    edges = edges.to(torch::kI32);

    GCoptimizationGeneralGraph* gc = new GCoptimizationGeneralGraph(unaries.size(0), pairwise.size(0));
    #pragma omp parallel for
    for(int i=0; i<edges.size(0); i++){
        if(edges[i].size(0) == 3){gc->setNeighbors(*edges[i][0].data_ptr<int>(), *edges[i][1].data_ptr<int>(), *edges[i][2].data_ptr<int>());}
        else{gc->setNeighbors(*edges[i][0].data_ptr<int>(), *edges[i][1].data_ptr<int>());}
    }
    gc->setDataCost(unaries.data_ptr<int>());
    gc->setSmoothCost(pairwise.data_ptr<int>());
    gc->expansion(5);
    int result[unaries.size(0)];
    #pragma omp parallel for
    for(int i = 0; i<unaries.size(0); i++){result[i]=gc->whatLabel(i);}

    auto final_t = torch::from_blob(result, {unaries.size(0)}, torch::TensorOptions().dtype(torch::kInt32));

    auto origin_mesh = mesh.mesh;
    Eigen::MatrixXd origin_barycenters = Eigen::MatrixXd::Zero((int)origin_mesh->triangles_.size(), 3);
    #pragma omp parallel for
    for(auto &iter : origin_mesh->triangles_) {
        auto index = &iter - &origin_mesh->triangles_[0];
        Eigen::Matrix<double, 1, 9> sub_cord;
        sub_cord[0] = origin_mesh->vertices_.at(iter[0])[0];
        sub_cord[1] = origin_mesh->vertices_.at(iter[0])[1];
        sub_cord[2] = origin_mesh->vertices_.at(iter[0])[2];
        sub_cord[3] = origin_mesh->vertices_.at(iter[1])[0];
        sub_cord[4] = origin_mesh->vertices_.at(iter[1])[1];
        sub_cord[5] = origin_mesh->vertices_.at(iter[1])[2];
        sub_cord[6] = origin_mesh->vertices_.at(iter[2])[0];
        sub_cord[7] = origin_mesh->vertices_.at(iter[2])[1];
        sub_cord[8] = origin_mesh->vertices_.at(iter[2])[2];

        Eigen::Matrix<double, 1, 3> sub_cent;
        sub_cent[0] = (sub_cord[0] + sub_cord[3] + sub_cord[6]) / 3;
        sub_cent[1] = (sub_cord[1] + sub_cord[4] + sub_cord[7]) / 3;
        sub_cent[2] = (sub_cord[2] + sub_cord[5] + sub_cord[8]) / 3;

        origin_barycenters.row(index) = sub_cent;
    }

    std::vector<Eigen::Vector3d> src_barycenters(mesh.barycenters.rowwise().begin(), mesh.barycenters.rowwise().end());
    auto src_pcd = new open3d::geometry::PointCloud();
    src_pcd->points_ = src_barycenters;
    auto kdtree = new open3d::geometry::KDTreeFlann(*src_pcd);

    std::vector<Eigen::Vector3d> center_cells_copy(origin_barycenters.rowwise().begin(), origin_barycenters.rowwise().end());

    std::vector<int> ori_res(center_cells_copy.size());
    int num = 1;
    #pragma omp parallel for
    for(int i=0; i<center_cells_copy.size();i++){
        std::vector<int> new_indices_vec(num);
        std::vector<double> new_dists_vec(num);
        kdtree->SearchKNN(center_cells_copy[i], num, new_indices_vec, new_dists_vec);
        ori_res[i] = *final_t[new_indices_vec[0]].data_ptr<int>();
    }

    return ori_res;
}