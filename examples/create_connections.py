from pathlib import Path
import pandas as pd
import numpy as np
import os


class Sphere:

    def __init__(self, neurite_type : int, position : dict, neurite_id: int, sphere_id :int, radius : float):
        self.neurite_type = neurite_type # 3: basal, 4: apical , 2 : axons
        self.position = position
        self.neurite_id = neurite_id
        self.sphere_id = sphere_id 
        self.radius = radius
    
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

def interpolate_coordinates(xs,ys,zs,types, step_size):
    # Initialize the output list
    interpolated_coords = []
    interpolated_types = []

    # Iterate over the input coordinates
    for i in range(len(xs) - 1):
        # Get the current and next coordinate
        curr_coord = np.array([xs[i],ys[i], zs[i]])
        next_coord = np.array([xs[i+1],ys[i+1], zs[i+1]])

        # Calculate the distance between the current and next coordinate
        distance = np.linalg.norm(next_coord - curr_coord)

        # Calculate the number of steps needed to reach the desired distance
        num_steps = int(np.ceil(distance / step_size))
        if num_steps == 0:
            num_steps = 1

        # Calculate the step size for each dimension
        step = (next_coord - curr_coord) / num_steps

        # Perform interpolation between the current and next coordinate
        for j in range(num_steps):
            interpolated_coord = curr_coord + j * step
            interpolated_coords.append(interpolated_coord)
            interpolated_types.append(types[i])

    # Add the last coordinate
    interpolated_coords.append(np.array([xs[i-1],ys[i-1], zs[i-1]]))
    interpolated_types.append(types[i-1])

    return interpolated_coords, interpolated_types

def create_projections(repetitions, paths, max_area,points = None):
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
    neurites = []
    for path in paths:
        for n in range(repetitions):
            print(f"Creating projection for Neurite {neurite_id}")
            print(f"Path : {path}")
            # reads from swc file
            neurons_data = pd.read_csv(path, sep="\s+", skiprows=2, header=0)
            neurons_data.columns=["index", "type","X", "Y","Z","radius","parent"]
            # coordinates incremented with a random point
            xs = [i+ points[neurite_id,0] for i in list(neurons_data["X"])]
            ys = [i+ points[neurite_id,1] for i in list(neurons_data["Y"])]
            zs = [i+ points[neurite_id,2] for i in  list(neurons_data["Z"])]
            types = list(neurons_data["type"])
            sphere_id = 0
            spheres = []

            interpolated_coords,types = interpolate_coordinates(xs,ys,zs, types, 1.0)
            print(f"Number of interpolated coordinates : {len(interpolated_coords)}")
            for interpolated_coord,type in zip(interpolated_coords, types):

                position = {}
                position["x"] = float(interpolated_coord[0])
                position["y"] = float(interpolated_coord[1])
                position["z"] = float(interpolated_coord[2])
                neurite_type = int(type)
                sphere = Sphere (neurite_type, position, neurite_id, sphere_id, 0.5)
                #print(f"Added sphere neurote type: {sphere.neurite_type}")
                if neurite_type == 3 or neurite_type == 4:
                    projections_x, projections_y, projections_z = add_sphere_to_projections(sphere, projections_x, projections_y, projections_z)
                spheres.append(sphere)
                sphere_id += 1

            neurites.append(spheres)
            neurite_id += 1

    return neurites, projections_x, projections_y, projections_z, points

def projections_inbetween(projections: list, s : Sphere, axis : str, distance_to_be_inside : float):
    """
    Find projections of spheres that might collide with sphere s 
    projections : list of sphere projections objects, can be inx,y or z axis
    s : sphere to check collision with
    axis : str, x, y or z
    distance_to_be_inside : distance from which tow spheres are considered to be collisiding
    """

    proj_1 = Sphere_Projection (s.position[axis] + s.radius + distance_to_be_inside, s.neurite_id, s.sphere_id, axis, s.neurite_type)
    proj_2 = Sphere_Projection (s.position[axis] - s.radius - distance_to_be_inside, s.neurite_id, s.sphere_id, axis, s.neurite_type)

    projections_in_between = [p for p in projections if proj_1.position > p.position and proj_2.position < p.position]


    return projections_in_between

    

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
        print(f"Creating connectome : Neurite {e}")
        #print(f"Number of spheres in neurite : {len(neurite)}")
        for sphere in neurite :
            # if apical
            print(f"Progress : {sphere.sphere_id}/{len(neurite)}")

            if sphere.neurite_type == 2:
                proj_inbetween_x = projections_inbetween(projections_x, sphere, "x", distance_to_be_inside)
                proj_inbetween_y = projections_inbetween(projections_y, sphere, "y", distance_to_be_inside)
                proj_inbetween_z = projections_inbetween(projections_z, sphere, "z", distance_to_be_inside)

                if (len(proj_inbetween_x)>0 and len(proj_inbetween_y)>0 and len(proj_inbetween_z)>0):
                    for p in proj_inbetween_x:
                        # if basal
                        if (p.neurite_type == 3 or p.neurite_type == 4) and p.isin(proj_inbetween_y) and p.isin(proj_inbetween_z) :
                            colliding_sphere = neurites[p.neurite_id][p.sphere_id]
                            if sphere.collide(colliding_sphere, distance_to_be_inside):
                                colliding_spheres.append(colliding_sphere)
                                connectome[sphere.neurite_id][p.neurite_id] = 1
  
    return connectome, colliding_spheres

def main_shrunk(repetitions, max_area, type):
    files = os.listdir(f"synthesized_neurons/shrunk_75_synthetic_L4_{type}/")
    paths = [f"synthesized_neurons/shrunk_75_synthetic_L4_{type}/{f}" for f in files if ".swc" in f]
    print("Creating projections")
    points = np.load(f"points_L4_{type}_rep_{repetitions}_interp.npy")
    neurites, projections_x, projections_y, projections_z, points = create_projections(repetitions,paths,points=points, max_area=max_area)
    connectome, colliding_spheres = create_connectome(projections_x, projections_y, projections_z, neurites)
    np.save(f"connectome_shrunk_75_L4_{type}_rep_{repetitions}_interp", connectome)
    np.save(f"colliding_spheres_shrunk_75_L4_{type}_rep_{repetitions}_interp", colliding_spheres)
    np.save(f"points_shrunk_75_L4_{type}_rep_{repetitions}_interp", points)
    print("Finished !")

def main(repetitions, max_area, type):
    files = os.listdir(f"synthesized_neurons/synthetic_L4_{type}/")
    paths = [f"synthesized_neurons/synthetic_L4_{type}/{f}" for f in files if ".swc" in f]
    print("Creating projections")
    neurites, projections_x, projections_y, projections_z, points = create_projections(repetitions,paths, max_area=max_area)
    connectome, colliding_spheres = create_connectome(projections_x, projections_y, projections_z, neurites)
    np.save(f"connectome_L4_{type}_rep_{repetitions}_interp", connectome)
    np.save(f"colliding_spheres_L4_{type}_rep_{repetitions}_interp", colliding_spheres)
    np.save(f"points_L4_{type}_rep_{repetitions}_interp", points)
    print("Finished !")

repetitions = 1
max_area = 100
type = "SSC"
# main(repetitions, max_area, type)
main_shrunk(repetitions, max_area, type)
