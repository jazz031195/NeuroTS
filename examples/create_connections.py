from pathlib import Path
import pandas as pd
import numpy as np
import os

class Sphere:

    def __init__(self, neurite_type : int, position : dict, neurite_id: int, sphere_id :int, radius : float):
        self.neurite_type = neurite_type # 3: basal, 4: apical,
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
        """
        Defines equality between two sphere projections
        """
        if self.sphere_id == other.sphere_id and self.neurite_id == other.neurite_id and self.neurite_type == other.neurite_type:
            return True
        else:
            return False
        
    def collide(self, other):
        """
        Check for collision
        """
        # calculate the difference between the two arrays
        diff = self.position - other.position
        # calculate the Euclidean distance using the linalg.norm function
        distance = np.linalg.norm(diff)
        if distance > self.radius + other.radius:
            return False
        else:
            return True

    
def add_sphere_to_projections(s : Sphere, projections_x : list, projections_y : list, projections_z : list):
    # x axis
    proj_x_1 = Sphere_Projection (s.position["x"] + s.radius, s.neurite_id, s.sphere_id, "x", s.neurite_type)
    projections_x.append(proj_x_1)
    proj_x_2 = Sphere_Projection (s.position["x"] - s.radius, s.neurite_id, s.sphere_id, "x", s.neurite_type)
    projections_x.append(proj_x_2)
    # sort based on position
    projections_x = sorted(projections_x, key=lambda x: x.position, reverse=True)

    # y axis 
    proj_y_1 = Sphere_Projection (s.position["y"] + s.radius, s.neurite_id, s.sphere_id, "y", s.neurite_type)
    projections_y.append(proj_y_1)
    proj_y_2 = Sphere_Projection (s.position["y"] - s.radius, s.neurite_id, s.sphere_id, "y", s.neurite_type)
    projections_y.append(proj_y_2)
    # sort based on position
    projections_y = sorted(projections_y, key=lambda x: x.position, reverse=True)

    # z axis
    proj_z_1 = Sphere_Projection (s.position["z"] + s.radius, s.neurite_id, s.sphere_id, "z", s.neurite_type)
    projections_z.append(proj_z_1)
    proj_z_2 = Sphere_Projection (s.position["z"] - s.radius, s.neurite_id, s.sphere_id, "z", s.neurite_type)
    projections_z.append(proj_z_2)
    # sort based on position
    projections_z = sorted(projections_z, key=lambda x: x.position, reverse=True)

    return projections_x, projections_y, projections_z


def create_projections(paths):
    """
    Read spheres from swc files and save them in the environment as projections
    paths : list of str, list of the paths to the swc files
    neurites : list of a list of spheres objects, so list of neurites
    projections : list of sphere projection objects, in x , y and z
    """
    projections_x = []
    projections_y = []
    projections_z = []
    neurite_id = 0
    points = np.random.rand(len(paths),3)*10
    neurites = []
    for e,path in enumerate(paths):
        print(f"Creating projection for Neurite {e}")
        # reads from swc file
        neurons_data = pd.read_csv(path, sep="\s+", header=1)
        # coordinates incremented with a random point
        xs = [i+ points[e,0] for i in list(neurons_data["X"])]
        ys = [i+ points[e,1] for i in list(neurons_data["Y"])]
        zs = [i+ points[e,2] for i in  list(neurons_data["Z"])]
        types = list(neurons_data["type"])
        radii = list(neurons_data["radius"])
        sphere_id = 0
        spheres = []
        for x,y,z,t,r in zip(xs,ys,zs,types,radii):
            position = {}
            position["x"] = x 
            position["y"] = y
            position["z"] = z
            radius = r
            neurite_type = t
            sphere = Sphere (neurite_type, position, neurite_id, sphere_id, radius)
            projections_x, projections_y, projections_z = add_sphere_to_projections(sphere, projections_x, projections_y, projections_z)
            spheres.append(sphere)
            sphere_id += 1
        neurites.append(spheres)
        neurite_id += 1

    return neurites, projections_x, projections_y, projections_z

def projections_inbetween(projections: list, s : Sphere, axis = str):
    """
    Find projections of spheres that might collide with sphere s 
    projections : list of sphere projections objects, can be inx,y or z axis
    s : sphere to check collision with
    axis : str, x, y or z
    """

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
    """
    Creates connectome from all intersection between apical and basal dendrites
    projections_x,projections_y,projections_z : list of sphere projections objects
    neurites : list of a list of spheres objects, so list of neurites
    """
    connectome = np.zeros((len(neurites), len(neurites)))
    
    for e, neurite in enumerate(neurites):
        print(f"Creating connectome : Neurite {e}")
        for sphere in neurite :
            # if apical
            if sphere.neurite_type == 4:
                proj_inbetween_x = projections_inbetween(projections_x, sphere, "x")
                proj_inbetween_y = projections_inbetween(projections_y, sphere, "y")
                proj_inbetween_z = projections_inbetween(projections_z, sphere, "z")

                colliding_spheres = []
            
                if (len(proj_inbetween_x)>0 and len(proj_inbetween_y)>0 and len(proj_inbetween_z)>0):
                    for p in proj_inbetween_x:
                        # if basal
                        if p.neurite_type == 3 and p in proj_inbetween_y and p in proj_inbetween_z :
                            colliding_sphere = neurites[p.neurite_id][p.sphere_id]
                            if sphere.collide(colliding_sphere):
                                colliding_spheres.append(colliding_sphere)
                                connectome[sphere.neurite_id][p.neurite_id] = 1
                                connectome[p.neurite_id][sphere.neurite_id] = 1
                    
    return connectome

def main():
    files = os.listdir("results_neurons")
    paths = [f"results_neurons/{f}" for f in files]
    print("Creating projections")
    neurites, projections_x, projections_y, projections_z = create_projections(paths)
    connectome = create_connectome(projections_x, projections_y, projections_z, neurites)
    print(connectome)

main()