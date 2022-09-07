#ifndef __TEETH_LABELER_H__
#define __TEETH_LABELER_H__

#include "engine.h"
#include "gco.h"
#include "infer.h"

class TeethLabeler
{
private:
    std::string model_path;
public:
    int label_num;
    TeethLabeler(std::string);
    void do_infer(std::shared_ptr<open3d::geometry::TriangleMesh>, std::vector<int>&);
    ~TeethLabeler();
};

#endif