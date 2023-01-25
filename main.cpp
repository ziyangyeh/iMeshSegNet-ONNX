#include "teeth_labeler.h"
#include "open3d/Open3D.h"

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include <chrono>   

int main(){
    std::string model_path = "/home/ziyang/Desktop/iMeshSegNet-ONNX/MeshSegNet-sim.onnx";

    auto tl = new TeethLabeler(model_path);

    std::string input_mesh_path;

    while(std::cin>>input_mesh_path)
    {
        // cudaDeviceReset();
        std::vector<int> labels;

        auto mesh = open3d::io::CreateMeshFromFile(input_mesh_path.c_str());

        auto start = std::chrono::system_clock::now();

        tl->do_infer(mesh, labels);

        auto label_num = tl->label_num;

        auto end = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout <<  "Spent " << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << " seconds." << std::endl;
        
        int* result = &labels[0];
        std::ofstream myfile ("example.txt");
        if (myfile.is_open())
        {
            for(int count = 0; count < labels.size(); count ++){
                myfile << result[count] << "\n" ;
            }
            myfile.close();
        }
    }
    return 0;
}

/**
    std::cout<<label_num<<std::endl;
    std::cout<<labels[0]<<std::endl;
    std::cout<<labels[1]<<std::endl;
    std::cout<<labels[2]<<std::endl;
    std::cout<<labels[3]<<std::endl;
    int* result = &labels[0];

    std::ofstream myfile ("example.txt");
    if (myfile.is_open())
    {
        for(int count = 0; count < labels.size(); count ++){
            myfile << result[count] << "\n" ;
        }
        myfile.close();
    }
**/