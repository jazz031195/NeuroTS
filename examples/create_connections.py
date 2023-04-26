from pathlib import Path

import numpy as np

import neurots

class Sphere:

    def __init__(self, neurite_type : str, position : dict, neurite_id: int, sphere_id :int, radius : float):
        self.neurite_type = neurite_type
        self.position = position
        self.neurite_id = neurite_id
        self.sphere_id = sphere_id 
        self.radius = radius
        self.collision_spheres = []
    
    def set_collision_spheres(self, collision_spheres):
        self.collision_spheres = collision_spheres


class Sphere_Projection :

    def __init__(self, position : float, neurite_id: int, sphere_id :int, axis : str, neurite_type : str):
        self.position = position
        self.neurite_id = neurite_id
        self.sphere_id = sphere_id 
        self.axis = axis
        self.neurite_type = neurite_type

    def __eq__(self, other):
        if self.sphere_id == other.sphere_id and self.neurite_id == other.neurite_id and self.neurite_type == other.neurite_type:
            return True
        else:
            return False

    
def add_sphere_to_projections(s : Sphere, projections_x : list, projections_y : list, projections_z : list):
    proj_x_1 = Sphere_Projection (s.position["x"] + s.radius, s.neurite_id, s.sphere_id, "x", s.neurite_type)
    projections_x.append(proj_x_1)
    proj_x_2 = Sphere_Projection (s.position["x"] - s.radius, s.neurite_id, s.sphere_id, "x", s.neurite_type)
    projections_x.append(proj_x_2)
    # sort based on position
    projections_x = sorted(projections_x, key=lambda x: x.position, reverse=True)

    proj_y_1 = Sphere_Projection (s.position["y"] + s.radius, s.neurite_id, s.sphere_id, "y", s.neurite_type)
    projections_y.append(proj_y_1)
    proj_y_2 = Sphere_Projection (s.position["y"] - s.radius, s.neurite_id, s.sphere_id, "y", s.neurite_type)
    projections_y.append(proj_y_2)
    # sort based on position
    projections_y = sorted(projections_y, key=lambda x: x.position, reverse=True)

    proj_z_1 = Sphere_Projection (s.position["z"] + s.radius, s.neurite_id, s.sphere_id, "z", s.neurite_type)
    projections_z.append(proj_z_1)
    proj_z_2 = Sphere_Projection (s.position["z"] - s.radius, s.neurite_id, s.sphere_id, "z", s.neurite_type)
    projections_z.append(proj_z_2)
    # sort based on position
    projections_z = sorted(projections_z, key=lambda x: x.position, reverse=True)


def create_projections(list_path_to_neurons, neurite_type):
    projections_x = []
    projections_y = []
    projections_z = []
    neurite_id = 0
    neurites = []
    for path in list_path_to_neurons:
        spheres = []
        sphere_id = 0
        with open(path) as f:
            for line in f.readlines():
                position = {}
                position["x"] = line[0]
                position["y"] = line[1]
                position["z"] = line[2]
                radius = line[3]
                sphere = Sphere (neurite_type, position, neurite_id, sphere_id, radius)
                add_sphere_to_projections(sphere, projections_x, projections_y, projections_z)
                spheres.append(sphere)
                sphere_id += 1
        neurites.append(spheres)
        neurite_id += 1

    return spheres, neurites, projections_x, projections_y, projections_z

def projections_inbetween(projections: list, s : Sphere, axis = str):

    proj_1 = Sphere_Projection (s.position[axis] + s.radius, s.neurite_id, s.sphere_id, axis, s.neurite_type)
    proj_2 = Sphere_Projection (s.position[axis] - s.radius, s.neurite_id, s.sphere_id, axis, s.neurite_type)

    proj_1_index = next((p for p in projections if proj_1 == p), None)
    
    proj_2_index = next((p for p in projections if proj_2 == p), None) 

    projections_in_between = []

    if (proj_1_index != None and proj_2_index != None):
        for i in range(proj_1_index+1, proj_2_index):
            projections_in_between.append(projections[i])
    
        return projections_in_between
    else :
        print("Sphere is not in projections environment")
        raise ValueError
    

def create_connectome(projections_x :list, projections_y: list, projections_z: list, neurites : list):

    connectome = np.zeros((len(neurites), len(neurites)))
    for neurite in neurites:
        for sphere in neurite :
            if sphere.neurite_type == "axon":
                proj_inbetween_x = projections_inbetween(projections_x, sphere, "x")
                proj_inbetween_y = projections_inbetween(projections_y, sphere, "y")
                proj_inbetween_z = projections_inbetween(projections_z, sphere, "z")

                colliding_spheres = []
            
                if (len(proj_inbetween_x)>0 and len(proj_inbetween_y)>0 and len(proj_inbetween_z)>0):
                    for p in proj_inbetween_x:
                        if p.neurite_type == "dendrite" and p in proj_inbetween_y and p in proj_inbetween_z :
                            colliding_sphere = neurites[p.neurite_id][p.sphere_id]
                            colliding_spheres.append(colliding_sphere)
                            connectome[sphere.neurite_id][p.neurite_id] = 1
                            connectome[p.neurite_id][sphere.neurite_id] = 1
                    
    return connectome
