# easy_mesh_vtk

# several examples:

    # create a new mesh by loading a VTP file
    mesh = Easy_Mesh('Sample_010.vtp')
    mesh.get_cell_edges()
    mesh.get_cell_normals()
    mesh.get_point_curvatures()
    mesh.to_vtp('example.vtp')
    
    # create a new mesh by loading a STL/OBJ file
    mesh = Easy_Mesh('Test5.stl')
    mesh.set_cell_labels(np.ones([mesh.cells.shape[0], 1]))
    mesh.get_cell_edges()
    mesh.get_cell_normals()
    mesh.to_vtp('example2.vtp')
    
    # create a new mesh by loading a main STL file and label it with other STL files
    mesh = Easy_Mesh('Sample_01_d.stl')
    mesh1 = Easy_Mesh('Sample_01_T2_d.stl')
    mesh2 = Easy_Mesh('Sample_01_T3_d.stl')   
    ## make a label dict in which key=str(label_ID), value=cells (i.e., [n, 9] array)
    label_dict = {'1': mesh1.cells, '2': mesh2.cells} 
    mesh.set_cell_labels(label_dict)
    mesh.to_vtp('example_with_labels.vtp')
    mesh.mesh_reflection('x')
    mesh.to_vtp('example_with_labels_fliped.vtp')
    
    # decimation
    mesh_d = Easy_Mesh('A0_Sample_01.vtp')
    mesh_d.mesh_decimation(0.5)
    print(mesh_d.cells.shape)
    print(mesh_d.points.shape)
    mesh_d.get_cell_edges()
    mesh_d.get_cell_normals()
    mesh_d.compute_cell_attributes_by_svm(mesh.cells, mesh.cell_attributes['Label'], 'Label')
    mesh_d.to_vtp('decimation_example.vtp')
    
    # subdivision
    mesh_s = Easy_Mesh('A0_Sample_01.vtp')
    mesh_s.mesh_subdivision(2, method='butterfly')
    print(mesh_s.cells.shape)
    print(mesh_s.points.shape)
    mesh_s.get_cell_edges()
    mesh_s.get_cell_normals()
    mesh_s.compute_cell_attributes_by_svm(mesh.cells, mesh.cell_attributes['Label'], 'Label')
    mesh_s.to_vtp('subdivision_example.vtp')
    
    # flip mesh for augmentation
    mesh_f = Easy_Mesh('A0_Sample_01.vtp')
    mesh_f.mesh_reflection(ref_axis='x')
    mesh_f.to_vtp('flipped_example.vtp')

    # create a new mesh from cells
    mesh2 = Easy_Mesh()
    mesh2.cells = mesh.cells[np.where(mesh.cell_attributes['Label']==1)[0]]
    mesh2.update_cell_ids_and_points()
    mesh2.set_cell_labels(mesh.cell_attributes['Label'][np.where(mesh.cell_attributes['Label']==1)[0]])
    mesh2.to_vtp('part_example.vtp')
    
    # downsampled UR3 (label==5) and compute heatmap
    tooth_idx = np.where(mesh.cell_attributes['Label']==5)[0]
    print(len(tooth_idx))
    mesh2 = Easy_Mesh()
    mesh2.cells = mesh.cells[tooth_idx]
    mesh2.update_cell_ids_and_points()
    target_cells = 400
    rate = 1.0 - target_cells/len(tooth_idx) - 0.005
    mesh2.mesh_decimation(rate)
    mesh2.get_cell_normals()
    mesh2.compute_guassian_heatmap(mesh2.points[3])
    mesh2.to_vtp('Canine_d.vtp')
    
    # downsampled UR3 (label==5) and compute heatmap
    tooth_idx = np.where(mesh.cell_attributes['Label']==5)[0]
    mesh2 = Easy_Mesh()
    mesh2.cells = mesh.cells[tooth_idx]
    mesh2.update_cell_ids_and_points()
    mesh2.mesh_subdivision(2, method='butterfly')
    mesh2.get_cell_normals()
    mesh2.compute_displacement_map(np.array([0, 0, 0]))
    mesh2.to_vtp('Canine_s.vtp')
    
    # trnasform a mesh
    matrix = GetVTKTransformationMatrix()
    mesh.mesh_transform(matrix)
    mesh.to_vtp('example_t.vtp')
	
# easy_landmark_vtk

# several examples:

    # create a new set of landmarks by loading a VTP file
    landmark = Easy_Landmark('A0_Sample_1_10_landmarks.vtp')
    landmark.to_vtp('example_ld.vtp')
    
    # create a new set of landmarks by giving a numpy array
    landmark2 = Easy_Landmark()
    landmark2.points = np.array([[3, 10, 2], [0, 0, 5]])
    landmark2.to_vtp('example_ld2.vtp')
    
    # transform a set of landmarks
    matrix = GetVTKTransformationMatrix()
    landmark = Easy_Landmark('A0_Sample_1_10_landmarks.vtp')
    landmark.landmark_transform(matrix)
    landmark.to_vtp('example_ld2.vtp')
    
    # flip landmarks based on a mesh
    mesh = Easy_Mesh('A0_Sample_01.vtp')
    landmark = Easy_Landmark('A0_Sample_1_10_landmarks.vtp')
    landmark.landmark_reflection(mesh, ref_axis='x')
    mesh.mesh_reflection(ref_axis='x')
    mesh.to_vtp('flipped_example.vtp')
    landmark.to_vtp('flipped_example_landmarks.vtp')
