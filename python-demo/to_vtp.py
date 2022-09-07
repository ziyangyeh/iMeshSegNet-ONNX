import vedo
import numpy as np

mesh_path = "/home/ziyang/Desktop/iMeshSegNet-ONNX/mesh/input/arch_upper_1.ply";

mesh = vedo.Mesh(mesh_path)

label = np.loadtxt("/home/ziyang/Desktop/iMeshSegNet-ONNX/build/example.txt")

print(mesh.NCells())
print(label.shape)

mesh.celldata['Label'] = label

vedo.write(mesh, "out.vtp")

# input_mesh_path = "/home/ziyang/Desktop/iMeshSegNet-ONNX/mesh/output/arch_upper_1_sim.ply";
# origin_mesh_path = "/home/ziyang/Desktop/iMeshSegNet-ONNX/mesh/input/arch_upper_1.ply";

# mesh = vedo.Mesh(input_mesh_path)

# label = np.loadtxt("/home/ziyang/Desktop/iMeshSegNet-ONNX/build/example.txt")

# print(mesh.NCells())
# print(label.shape)

# mesh.celldata['Label'] = label

# vedo.write(mesh, "out_1.vtp")

# mesh = vedo.Mesh(origin_mesh_path)

# label = np.loadtxt("/home/ziyang/Desktop/iMeshSegNet-ONNX/build/example_com.txt")

# print(mesh.NCells())
# print(label.shape)

# mesh.celldata['Label'] = label

# vedo.write(mesh, "out_t_1.vtp")

# import open3d as o3d
# mesh = o3d.io.read_triangle_mesh("/home/ziyang/Desktop/iMeshSegNet-ONNX/mesh/input/arch_lower_1.ply")
# print(mesh.triangles)
# mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=10000)
# print(mesh.triangles)
# o3d.io.write_triangle_mesh("/home/ziyang/Desktop/iMeshSegNet-ONNX/mesh/output/arch_lower_1_sim.ply", mesh)


# mesh = o3d.io.read_triangle_mesh("/home/ziyang/Desktop/iMeshSegNet-ONNX/mesh/input/arch_upper_1.ply")
# print(mesh.triangles)
# mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=10000)
# print(mesh.triangles)
# o3d.io.write_triangle_mesh("/home/ziyang/Desktop/iMeshSegNet-ONNX/mesh/output/arch_upper_1_sim.ply", mesh)
