import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
import math
from easy_mesh_vtk import *

class Easy_Landmark(object):
    def __init__(self, filename = None, warning=False):
        self.warning = warning
        self.reader = None
        self.vtkPolyData = None
        self.points = np.array([])
        self.point_attributes = dict()
        self.filename = filename
        if self.filename != None:
            if self.filename[-3:].lower() == 'vtp':
                self.read_vtp(self.filename)
        
        
    def get_landmark_data_from_vtkPolyData(self):
        data = self.vtkPolyData
        
        n_points = data.GetNumberOfPoints()
        mesh_points = np.zeros([n_points, 3], dtype='float32')
    
        for i in range(n_points):
            mesh_points[i][0], mesh_points[i][1], mesh_points[i][2] = data.GetPoint(i)
        
        self.points = mesh_points
        
        #read point arrays
        for i_attribute in range(self.vtkPolyData.GetPointData().GetNumberOfArrays()):
            self.load_point_attributes(self.vtkPolyData.GetPointData().GetArrayName(i_attribute), self.vtkPolyData.GetPointData().GetArray(i_attribute).GetNumberOfComponents())
        
        
    def read_vtp(self, vtp_filename):
        '''
        update
            self.filename
            self.reader
            self.vtkPolyData
            self.points
            self.point_attributes
        '''
        self.filename = vtp_filename
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(vtp_filename)
        reader.Update()
        self.reader = reader
    
        data = reader.GetOutput()
        self.vtkPolyData = data        
        self.get_landmark_data_from_vtkPolyData()
        
    
    def load_point_attributes(self, attribute_name, dim):
        self.point_attributes[attribute_name] = np.zeros([self.points.shape[0], dim])
        try:
            if dim == 1:
                for i in range(self.points.shape[0]):
                    self.point_attributes[attribute_name][i, 0] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetValue(i)
            elif dim == 2:
                for i in range(self.points.shape[0]):
                    self.point_attributes[attribute_name][i, 0] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetComponent(i, 0)
                    self.point_attributes[attribute_name][i, 1] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetComponent(i, 1)
            elif dim == 3:
                for i in range(self.points.shape[0]):
                    self.point_attributes[attribute_name][i, 0] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetComponent(i, 0)
                    self.point_attributes[attribute_name][i, 1] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetComponent(i, 1)
                    self.point_attributes[attribute_name][i, 2] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetComponent(i, 2)
        except:
            if self.warning:
                print('No cell attribute named "{0}" in file: {1}'.format(attribute_name, self.filename))
        
        
    def update_vtkPolyData(self):
        '''
        call this function when manipulating self.points
        '''
        vtkPolyData = vtk.vtkPolyData()
        points = vtk.vtkPoints()
    
        points.SetData(numpy_to_vtk(self.points))
        vtkPolyData.SetPoints(points)
        
        #update point_attributes
        for i_key in self.point_attributes.keys():
            point_attribute = vtk.vtkDoubleArray()
            point_attribute.SetName(i_key);
            if self.point_attributes[i_key].shape[1] == 1:
                point_attribute.SetNumberOfComponents(self.point_attributes[i_key].shape[1])
                for i_attribute in self.point_attributes[i_key]:
                    point_attribute.InsertNextTuple1(i_attribute)
                vtkPolyData.GetPointData().AddArray(point_attribute)
#                vtkPolyData.GetPointData().SetScalars(cell_attribute)
            elif self.point_attributes[i_key].shape[1] == 2:
                point_attribute.SetNumberOfComponents(self.point_attributes[i_key].shape[1])
                for i_attribute in self.point_attributes[i_key]:
                    point_attribute.InsertNextTuple2(i_attribute[0], i_attribute[1])
                vtkPolyData.GetPointData().AddArray(point_attribute)
#                vtkPolyData.GetPointData().SetVectors(cell_attribute)
            elif self.point_attributes[i_key].shape[1] == 3:
                point_attribute.SetNumberOfComponents(self.point_attributes[i_key].shape[1])
                for i_attribute in self.point_attributes[i_key]:
                    point_attribute.InsertNextTuple3(i_attribute[0], i_attribute[1], i_attribute[2])
                vtkPolyData.GetPointData().AddArray(point_attribute)
#                vtkPolyData.GetPointData().SetVectors(cell_attribute)
            else:
                if self.warning:
                    print('Check attribute dimension, only support 1D, 2D, and 3D now')
        
        vtkPolyData.Modified()
        self.vtkPolyData = vtkPolyData
       
    
    def landmark_transform(self, vtk_matrix):
        Trans = vtk.vtkTransform()
        Trans.SetMatrix(vtk_matrix)
        
        TransFilter = vtk.vtkTransformPolyDataFilter()
        TransFilter.SetTransform(Trans)
        TransFilter.SetInputData(self.vtkPolyData)
        TransFilter.Update()
        
        self.vtkPolyData = TransFilter.GetOutput()
        self.get_landmark_data_from_vtkPolyData()
    
    
    def landmark_reflection(self, easy_mesh, ref_axis='x'):
        xmin = np.min(easy_mesh.points[:, 0])
        xmax = np.max(easy_mesh.points[:, 0])
        ymin = np.min(easy_mesh.points[:, 1])
        ymax = np.max(easy_mesh.points[:, 1])
        zmin = np.min(easy_mesh.points[:, 2])
        zmax = np.max(easy_mesh.points[:, 2])
        center = np.array([np.mean(easy_mesh.points[:, 0]), np.mean(easy_mesh.points[:, 1]), np.mean(easy_mesh.points[:, 2])])
        
        if ref_axis == 'x':
            point1 = [xmin, ymin, zmin]
            point2 = [xmin, ymax, zmin]
            point3 = [xmin, ymin, zmax]
        elif ref_axis == 'y':
            point1 = [xmin, ymin, zmin]
            point2 = [xmax, ymin, zmin]
            point3 = [xmin, ymin, zmax]
        elif ref_axis == 'z':
            point1 = [xmin, ymin, zmin]
            point2 = [xmin, ymax, zmin]
            point3 = [xmax, ymin, zmin]
        else:
            if self.warning:
                print('Invalid ref_axis!')
            
        #get equation of the plane by three points
        v1 = np.zeros([3,])
        v2 = np.zeros([3,])

        for i in range(3):
            v1[i] = point1[i] - point2[i]
            v2[i] = point1[i] - point3[i]

        normal_vec = np.cross(v1, v2)/np.linalg.norm(np.cross(v1, v2))

        flipped_mesh_points = np.copy(easy_mesh.points)
    
        #flip mesh points
        for idx in range(len(easy_mesh.points)):
            tmp_p1 = easy_mesh.points[idx, 0:3]
            
            tmp_v1 = tmp_p1 - point1
            dis_v1 = np.dot(tmp_v1, normal_vec)*normal_vec
                    
            flipped_p1 = tmp_p1 - 2*dis_v1
            flipped_mesh_points[idx, 0:3] = flipped_p1
            
        for idx in range(len(self.points)):
            tmp_p1 = self.points[idx, 0:3]
            
            tmp_v1 = tmp_p1 - point1
            dis_v1 = np.dot(tmp_v1, normal_vec)*normal_vec
                    
            flipped_p1 = tmp_p1 - 2*dis_v1
            self.points[idx, 0:3] = flipped_p1

        #move flipped_mesh_points back to the center
        flipped_center = np.array([np.mean(flipped_mesh_points[:, 0]), np.mean(flipped_mesh_points[:, 1]), np.mean(flipped_mesh_points[:, 2])])
        displacement = center - flipped_center
        
        self.points[:, 0:3] += displacement
        
        
    def to_vtp(self, vtp_filename):
        self.update_vtkPolyData()
        
        if vtk.VTK_MAJOR_VERSION <= 5:
            self.vtkPolyData.Update()
     
        writer = vtk.vtkXMLPolyDataWriter();
        writer.SetFileName("{0}".format(vtp_filename));
        if vtk.VTK_MAJOR_VERSION <= 5:
            writer.SetInput(self.vtkPolyData)
        else:
            writer.SetInputData(self.vtkPolyData)
        writer.Write()
        
        
#if __name__ == '__main__':
    
#    # create a new set of landmarks by loading a VTP file
#    landmark = Easy_Landmark('A0_Sample_1_10_landmarks.vtp')
#    landmark.to_vtp('example_ld.vtp')
#    
#    # create a new set of landmarks by giving a numpy array
#    landmark2 = Easy_Landmark()
#    landmark2.points = np.array([[3, 10, 2], [0, 0, 5]])
#    landmark2.to_vtp('example_ld2.vtp')
#    
#    # transform a set of landmarks
#    matrix = GetVTKTransformationMatrix()
#    landmark = Easy_Landmark('A0_Sample_1_10_landmarks.vtp')
#    landmark.landmark_transform(matrix)
#    landmark.to_vtp('example_ld2.vtp')
#    
#    # flip landmarks based on a mesh
#    mesh = Easy_Mesh('A0_Sample_01.vtp')
#    landmark = Easy_Landmark('A0_Sample_1_10_landmarks.vtp')
#    landmark.landmark_reflection(mesh, ref_axis='x')
#    mesh.mesh_reflection(ref_axis='x')
#    mesh.to_vtp('flipped_example.vtp')
#    landmark.to_vtp('flipped_example_landmarks.vtp')