from pathlib import Path
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import bisect


class Sphere:

    def __init__(self, neurite_type : int, position : dict, neurite_id: int, sphere_id :int, radius : float, parent : int, index : int):
        self.neurite_type = neurite_type # 3: basal, 4: apical , 2 : axons
        self.position = position
        self.neurite_id = neurite_id
        self.sphere_id = sphere_id
        self.radius = radius
        self.parent = parent
        self.index  = index
    
    def collide(self, other, distance_to_be_inside):
        """
        Check for collision
        """
        # calculate the difference between the two arrays
        pos1 = np.array([self.position["x"],self.position["y"],self.position["z"]])
        pos2 = np.array([other.position["x"],other.position["y"],other.position["z"]])
        diff = pos1 - pos2
        # calculate the Euclidean distance using the linalg.norm function
        distance = np.linalg.norm(diff)
        if distance > self.radius + other.radius + distance_to_be_inside:
            return False
        else:
            return True

    def distance(self, other):
        """
        Check for collision
        """
        # calculate the difference between the two arrays
        pos1 = np.array([self.position["x"],self.position["y"],self.position["z"]])
        pos2 = np.array([other.position["x"],other.position["y"],other.position["z"]])
        diff = pos1 - pos2
        # calculate the Euclidean distance using the linalg.norm function
        distance = np.linalg.norm(diff)
        return distance


class Sphere_Projection :

    def __init__(self, position : float, neurite_id: int, sphere_id :int, axis : str, neurite_type : str):
        self.position = position
        self.neurite_id = neurite_id # 3: basal, 4: apical
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
    
    def isin(self, other : list):
        for p in other:
            if self.neurite_id == p.neurite_id and self.sphere_id == p.sphere_id:
                return True
        return False


    
def add_sphere_to_projections(s : Sphere, projections_x : list, projections_y : list, projections_z : list):
    # x axis
    proj_x_1 = Sphere_Projection (s.position["x"] + s.radius, s.neurite_id, s.sphere_id, "x", s.neurite_type)
    projections_x.append(proj_x_1)
    proj_x_2 = Sphere_Projection (s.position["x"] - s.radius, s.neurite_id, s.sphere_id, "x", s.neurite_type)
    projections_x.append(proj_x_2)
    # sort based on position
    projections_x = sorted(projections_x, key=lambda x: x.position, reverse=False)

    # y axis 
    proj_y_1 = Sphere_Projection (s.position["y"] + s.radius, s.neurite_id, s.sphere_id, "y", s.neurite_type)
    projections_y.append(proj_y_1)
    proj_y_2 = Sphere_Projection (s.position["y"] - s.radius, s.neurite_id, s.sphere_id, "y", s.neurite_type)
    projections_y.append(proj_y_2)
    # sort based on position
    projections_y = sorted(projections_y, key=lambda x: x.position, reverse=False)

    # z axis
    proj_z_1 = Sphere_Projection (s.position["z"] + s.radius, s.neurite_id, s.sphere_id, "z", s.neurite_type)
    projections_z.append(proj_z_1)
    proj_z_2 = Sphere_Projection (s.position["z"] - s.radius, s.neurite_id, s.sphere_id, "z", s.neurite_type)
    projections_z.append(proj_z_2)
    # sort based on position
    projections_z = sorted(projections_z, key=lambda x: x.position, reverse=False)

    return projections_x, projections_y, projections_z

def interpolate_df(neurons_data):
    neurons_data = neurons_data.loc[neurons_data["type"]!= 1]
    parent_indices = neurons_data.copy()["parent"].values
    parent_pos = neurons_data.copy().set_index("index").reindex(parent_indices)[["X", "Y", "Z"]].values
    sphere_pos = neurons_data.copy()[["X", "Y", "Z"]].values
    diff = sphere_pos - parent_pos
    norm_diff = np.linalg.norm(diff, axis=1)
    mask = norm_diff > 1 
    interpolated_rows = neurons_data[mask].copy()
    interpolated_rows[["X", "Y", "Z"]] = (sphere_pos[mask] + parent_pos[mask]) / 2
    interpolated_rows["radius"] = 0.5
    interpolated_rows = interpolated_rows.astype({"index": int, "type": int, "parent": int})
    new_neurons_data = pd.concat([neurons_data, interpolated_rows], axis=0)
    return new_neurons_data

def create_projections(repetitions, type, paths, max_area,points = None):
    """
    Read spheres from swc files and save them in the environment as projections. Only the basal dendrites are saved in the projections.
    paths : list of str, list of the paths to the swc files
    neurites : list of a list of spheres objects, so list of neurites
    projections : list of sphere projection objects, in x , y and z
    """
    projections_x = []
    projections_y = []
    projections_z = []
    neurite_id = 0
    if points is None:
        points = np.random.rand(int(len(paths)*repetitions),3)*max_area
        np.save(f"points_L4_{type}_rep_{repetitions}", points)
    neurites = []
    for path in paths:
        for n in range(repetitions):
            if n == 0:
                print(f"Creating projection for Neurite {neurite_id}")
                print(f"Path : {path}")
                # reads from swc file
                neurons_data = pd.read_csv(path, sep="\s+", skiprows=2, header=0)
                neurons_data.columns=["index", "type","X", "Y","Z","radius","parent"]
                # coordinates incremented with a random point
                neurons_data  = interpolate_df(neurons_data)
            new_neurons_data = neurons_data.copy()

            new_neurons_data["X"] = [i+ points[neurite_id,0] for i in list(new_neurons_data["X"])]
            new_neurons_data["Y"] = [i+ points[neurite_id,1] for i in list(new_neurons_data["Y"])]
            new_neurons_data["Z"] = [i+ points[neurite_id,2] for i in  list(new_neurons_data["Z"])]
            print(f"Number of spheres : {len(new_neurons_data)}")
            

            sphere_id = 0
            spheres = []
            for x,y,z,type,ind,parent in zip(list(new_neurons_data["X"]),list(new_neurons_data["Y"]),list(new_neurons_data["Z"]), list(new_neurons_data["type"]), list(new_neurons_data["index"]), list(new_neurons_data["parent"])):
                    
                position = {}
                position["x"] = float(x)
                position["y"] = float(y)
                position["z"] = float(z)
                neurite_type = int(type)
                        
                #print(f"Added sphere neurite type: {neurite_type}")
                sphere = Sphere (neurite_type, position, neurite_id, sphere_id, 0.5, parent, ind)
                if neurite_type == 3 or neurite_type == 4:
                    projections_x, projections_y, projections_z = add_sphere_to_projections(sphere, projections_x, projections_y, projections_z)
                spheres.append(sphere)
                sphere_id +=1

            neurites.append(spheres)
            neurite_id += 1
            

    return neurites, projections_x, projections_y, projections_z, points

def projections_inbetween(projections: list, s: Sphere, axis: str, distance_to_be_inside: float):
    """
    Find projections of spheres that might collide with sphere s 
    projections: list of sphere projection objects, can be in x, y, or z axis
    s: sphere to check collision with
    axis: str, x, y, or z
    distance_to_be_inside: distance from which two spheres are considered to be colliding
    """

    proj_1_pos = s.position[axis] + s.radius + distance_to_be_inside
    proj_2_pos = s.position[axis] - s.radius - distance_to_be_inside

    start_index = bisect.bisect_left([p.position for p in projections], proj_2_pos)
    end_index = bisect.bisect_right([p.position for p in projections], proj_1_pos)


    return projections[start_index:end_index]

    

def create_connectome(projections_x :list, projections_y: list, projections_z: list, neurites : list):
    """
    Creates connectome from all intersection between apical and basal dendrites
    projections_x,projections_y,projections_z : list of sphere projections objects
    neurites : list of a list of spheres objects, so list of neurites
    """
    connectome = np.zeros((len(neurites), len(neurites)))
    distance_to_be_inside = 1e-5
    colliding_spheres = []
    
    for e, neurite in enumerate(neurites):
        print(f"Creating connectome : Neuron {e}")
        #print(f"Number of spheres in neurite : {len(neurite)}")
        for sphere in neurite :
            # if apical
            #print(f"Progress : {sphere.sphere_id}/{len(neurite)}, neuron {e}/{len(neurites)}")
            
            if sphere.neurite_type == 2:
                proj_inbetween_x = projections_inbetween(projections_x, sphere, "x", distance_to_be_inside)
                proj_inbetween_y = projections_inbetween(projections_y, sphere, "y", distance_to_be_inside)
                proj_inbetween_z = projections_inbetween(projections_z, sphere, "z", distance_to_be_inside)

                if (len(proj_inbetween_x)>0 and len(proj_inbetween_y)>0 and len(proj_inbetween_z)>0):
                    for p in proj_inbetween_x:
                        # if basal
                        if p.isin(proj_inbetween_y) and p.isin(proj_inbetween_z) :
                            colliding_sphere = neurites[p.neurite_id][p.sphere_id]
                            if sphere.collide(colliding_sphere, distance_to_be_inside):
                                colliding_spheres.append(colliding_sphere)
                                connectome[sphere.neurite_id][p.neurite_id] = 1
  
    return connectome, colliding_spheres

def main_shrunk(repetitions, max_area, type):
    files = os.listdir(f"synthesized_neurons/shrunk_75_synthetic_L4_{type}/")
    paths = [f"synthesized_neurons/shrunk_75_synthetic_L4_{type}/{f}" for f in files if ".swc" in f]
    print("Creating projections")
    points = np.load(f"points_L4_{type}_rep_{repetitions}.npy")
    neurites, projections_x, projections_y, projections_z, points = create_projections(repetitions,type,paths,points=points, max_area=max_area)
    connectome, colliding_spheres = create_connectome(projections_x, projections_y, projections_z, neurites)
    np.save(f"connectome_shrunk_75_L4_{type}_rep_{repetitions}", connectome)
    np.save(f"colliding_spheres_shrunk_75_L4_{type}_rep_{repetitions}", colliding_spheres)
    np.save(f"points_shrunk_75_L4_{type}_rep_{repetitions}", points)
    print("Finished !")

def main(repetitions, max_area, type):
    files = os.listdir(f"synthesized_neurons/synthetic_L4_{type}/")
    paths = [f"synthesized_neurons/synthetic_L4_{type}/{f}" for f in files if ".swc" in f]
    print("Creating projections")
    neurites, projections_x, projections_y, projections_z, points = create_projections(repetitions,type, paths, max_area=max_area)
    connectome, colliding_spheres = create_connectome(projections_x, projections_y, projections_z, neurites)
    np.save(f"connectome_L4_{type}_rep_{repetitions}", connectome)
    np.save(f"colliding_spheres_L4_{type}_rep_{repetitions}", colliding_spheres)
    print("Finished !")

def create_all(type, shrunk = False):
    repetitions = 4
    max_area = 100
    if not shrunk:
        main(repetitions, max_area, type)
    else:
        main_shrunk(repetitions, max_area, type)

def find_radii(file):
    radii =[] 
    with open(file) as f:
        for e,line in enumerate(f.readlines()):
            if e > 1:
                line_list =list(dict.fromkeys(line.split(" ")))
                x = float(line_list[2] )
                y = float(line_list[3] )
                z = float(line_list[4] )
                a = np.array([x,y,z])
                if e >2:
                    radii.append((np.linalg.norm(a-b)))
                b = a

    sns.histplot(radii)
    plt.show()


#create_all("SSC", shrunk = True)
# find_radii("/home/localadmin/Documents/math_class/project/NeuroTS/examples/synthesized_neurons/synthetic_L4_TPC/synthetic_sm100429a1-5_INT_idD.swc")
