#include "teeth_labeler.h"
#include "open3d/Open3D.h"

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

int main(){
    std::string mesh_path = "/home/ziyang/Desktop/iMeshSegNet-ONNX/mesh/input/arch_upper_1.ply";
    std::string model_path = "/home/ziyang/Desktop/iMeshSegNet-ONNX/onnx/model_sim.onnx";

    std::vector<int> labels;

    auto mesh = open3d::io::CreateMeshFromFile(mesh_path.c_str());

    auto tl = new TeethLabeler(model_path);

    tl->do_infer(mesh, labels);

    auto label_num = tl->label_num;

    return 0;
}

/**
    std::cout<<label_num<<std::endl;
    std::cout<<labels[0]<<std::endl;
    std::cout<<labels[1]<<std::endl;
    std::cout<<labels[2]<<std::endl;
    std::cout<<labels[3]<<std::endl;
    int* result = &label[0];

    std::ofstream myfile ("example.txt");
    if (myfile.is_open())
    {
        for(int count = 0; count < label.size(); count ++){
            myfile << result[count] << "\n" ;
        }
        myfile.close();
    }
**/