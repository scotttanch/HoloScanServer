import os

import numpy as np
from matplotlib.pyplot import imsave
from numpy import sqrt
from pandas import read_csv
from readgssi.dzt import readdzt

from flaskr.create_texture import create_textures

"""
    HoloScan object generation code for use with the HoloScan Server
    
    Author: Scott Tanch

####### --------------------------------- Reference Frames --------------------------------- #####
        
        Input data is saved in the RealSense Reference Frame (right-handed, x-forward, z-up)
        while Output data is in the Unity Reference Frame (left-handed, x-forward, y-up). 
        
####### --------------------------------------- Units -------------------------------------- #####   

        All units are assumed to be in meters, which is the natural output of the Realsense
        and the input for Unity.
        
"""


def parallel_curve(x_points, z_points, distance):
    """
    Generated by ChatGPT 3.5

    Generates a parallel curve in the xz plane at some specified distance.

    Args:
        x_points (list[float]): x coordinates defining the original curve
        z_points (list[float]): z coordinates defining the original curve
        distance (float): Distance from the original curve to the parallel curve

    Returns:
        (tuple[list[float], list[float]]): x and z cordinates of the parallel curve
    """

    # Convert x and y points to numpy arrays for vectorized operations
    x_array, z_array = np.array(x_points), np.array(z_points)

    # Compute differences between consecutive points to get tangent vectors
    dx, dz = np.diff(x_array), np.diff(z_array)

    # Compute normalized tangent vectors
    norm = np.sqrt(dx ** 2 + dz ** 2)
    norm[norm == 0] = np.inf
    normalized_tangents_x = dx / norm
    normalized_tangents_z = dz / norm

    # Rotate tangent vectors by 90 degrees to get normal vectors
    normals_x = normalized_tangents_z
    normals_z = -normalized_tangents_x

    # Scale normal vectors by the distance
    scaled_normals_x = normals_x * -distance
    scaled_normals_z = normals_z * -distance

    # Extend arrays for the last point
    scaled_normals_x = np.append(scaled_normals_x, scaled_normals_x[-1])
    scaled_normals_z = np.append(scaled_normals_z, scaled_normals_z[-1])

    # Generate points for parallel curve
    parallel_x_points = x_array + scaled_normals_x
    parallel_z_points = z_array + scaled_normals_z

    return parallel_x_points.tolist(), parallel_z_points.tolist()


def reduce_resolution(x_points, y_points, z_points, resolution=0.0, mean=False, endpoint=True):
    """
    Given a set of arbitrarily spaced points, reduce the minimum spacing between points.

    When the mean argument is set to true, points within the resolution radius are averaged together. This has the consequence of the
    set of points returned not being a subset of the orignal set.

    Args:
        x_points (list[float]): original x cordinates
        y_points (list[float]): original y cordinates
        z_points (list[float]): original z cordinates
        resolution (float): target for minimum spacing between points
        mean (bool): when False, all points within the resoltuion radius are discarded
        endpoint (bool): when True the last point in the original set is included regardless of its proximity to the previous point

    Returns:
        (tuple[list[float], list[float], list[float]]): x, y, and z corindates of the reduced set of points
    """

    # Raise a value error is our lists arent of equal length
    if not (len(x_points) == len(y_points) == len(z_points)):
        raise ValueError("Lists must be of equal length")

    new_x = []
    new_y = []
    new_z = []

    # This is just so pycharm will shut up about using a variable before I made it
    x_end = 0
    y_end = 0
    z_end = 0

    if endpoint:
        x_end = x_points[-1]
        y_end = y_points[-1]
        z_end = z_points[-1]

    # In mean mode, we average points within the resolution
    if mean:
        while x_points:
            x0 = x_points.pop(0)  # Pull the first point from the stacks
            y0 = y_points.pop(0)
            z0 = z_points.pop(0)
            x_in = [x0]
            y_in = [y0]
            z_in = [z0]
            while True and x_points:  # Make sure we dont try and pop more than we can
                d = sqrt((x_points[0] - x0) ** 2 + (y_points[0] - y0) ** 2 + (z_points[0] - z0) ** 2)
                if d < resolution:
                    x_in.append(x_points.pop(0))
                    y_in.append(y_points.pop(0))
                    z_in.append(z_points.pop(0))
                else:
                    break

            new_x.append(np.mean(x_in))
            new_y.append(np.mean(y_in))
            new_z.append(np.mean(z_in))

    # In non-mean mode we discard points within the resolutution radius, this ensures that the returned points
    # are a subset of the original points and not a new set entirely
    if not mean:
        while x_points:
            x0 = x_points.pop(0)  # Pull the first point from the stacks
            y0 = y_points.pop(0)
            z0 = z_points.pop(0)
            new_x.append(x0)
            new_y.append(y0)
            new_z.append(z0)
            while x_points:  # Make sure we dont try and pop more than we can
                d = sqrt((x_points[0] - x0) ** 2 + (y_points[0] - y0) ** 2 + (z_points[0] - z0) ** 2)
                if d < resolution:
                    x_points.pop(0)  # Pop and discard the point at the head
                    y_points.pop(0)  # Since it is in the radius we dont care about it
                    z_points.pop(0)  # except we might care about it if, if its the end point
                else:
                    break

    # If we care about the end point and thus saved it earlier, add the end point if needed
    if endpoint:
        if (new_x[-1] != x_end) and (new_y[-1] != y_end) and (new_z != z_end):
            new_x.append(x_end)
            new_y.append(y_end)
            new_z.append(z_end)

    return new_x, new_y, new_z


def read_position_data(file):
    """
    Reads a csv file whose first columns are x, y, z relative to the RealSense reference frame (right-handed z-up) and
    swaps to the Unity refernce frame (left-handed y-up).

    Args:
        file (str): name of the file being read

    Returns:
        (tuple[list[float], list[float], list[float]]): x, y, z cordinates in the Unity refernce frame

    """
    # TODO: This needs to be updated everytime I change how data is saved by the HardPack itself
    # csvs on the server are currently just x, y, z (RS Frame) -> x, z, y (Unity Frame)
    df = read_csv(file, header=None)
    xs, ys, zs = reduce_resolution(df[0].to_list(), df[2].to_list(), df[1].to_list(), resolution=0.001, endpoint=True)
    return xs, ys, zs


def create_geometry(x_points, y_points, z_points, depth):
    """
    Generates the nessicary mesh data for unity to render a surface composed of triangles. In SolidWorks terms, a line is extruded
    in the negative y direction to create a surface.

    Notes:
        For practical reasons, the mesh is actually comprised of two discojointed surfaces with flipped normal vectors,
        allowing it to be visable from both side. Suffice to say you can only see a triangle if its normal vector is
        pointing towards you.

        As another side note, the choice of format for the verts and uvs makes sense, each row a cordinate. One might notice
        that the tris are one value per row. This is a holdover from a processing choice in HoloScan3D and has no importance
        other than it is the way it is.

    Args:
        x_points (list[float]): original x points
        y_points (list[float]): original y points
        z_points (list[float]): original z points
        depth (float): depth to extrude the surface in the negative y-direction

    Returns:
        (tuple[list[str], list[str], list[str]]): vertexs, triangles, and uvs. These are formatted to be used by the writelines
        file operation
    """

    # From the original file, create two parallel paths at some offset
    pos_shift_x, pos_shift_z = parallel_curve(x_points, z_points, 0.001)
    neg_shift_x, neg_shift_z = parallel_curve(x_points, z_points, -0.001)

    pos_segments = [0]
    neg_segments = [0]

    # calculate the fraction along the file length of each point
    for i in range(1, len(pos_shift_x)):
        new_segment = np.sqrt((pos_shift_x[i] - pos_shift_x[i - 1]) ** 2 +
                              (pos_shift_z[i] - pos_shift_z[i - 1]) ** 2 +
                              (y_points[i] - y_points[i - 1]) ** 2)
        pos_segments.append(pos_segments[-1] + new_segment)

        new_segment = np.sqrt((neg_shift_x[i] - neg_shift_x[i - 1]) ** 2 +
                              (neg_shift_z[i] - neg_shift_z[i - 1]) ** 2 +
                              (y_points[i] - y_points[i - 1]) ** 2)
        neg_segments.append(neg_segments[-1] + new_segment)

    pos_tot = pos_segments[-1]
    neg_tot = neg_segments[-1]

    norm_pos_seg = [x / (2 * pos_tot) for x in pos_segments]
    norm_neg_seg = [x / (2 * neg_tot) for x in neg_segments]

    n = np.shape(x_points)[0]

    uvs = []
    for i in range(n):
        uvs.append(f"{0.5 + norm_pos_seg[i]},1\n")
    for i in range(n):
        uvs.append(f"{0.5 + norm_pos_seg[i]},0\n")
    for i in range(n):
        uvs.append(f"{0.5 - norm_neg_seg[i]},1\n")
    for i in range(n):
        uvs.append(f"{0.5 - norm_neg_seg[i]},0\n")

    verts = []
    for x, z, y in zip(pos_shift_x, pos_shift_z, y_points):
        verts.append(f"{x},{y},{z}\n")
    for x, z, y in zip(pos_shift_x, pos_shift_z, y_points):
        verts.append(f"{x},{y-depth},{z}\n")
    for x, z, y in zip(neg_shift_x, neg_shift_z, y_points):
        verts.append(f"{x},{y},{z}\n")
    for x, z, y in zip(neg_shift_x, neg_shift_z, y_points):
        verts.append(f"{x},{y-depth},{z}\n")

    tris = []
    for i in range(0, n - 2):
        tris.append(f"{i}\n")
        tris.append(f"{i + 1}\n")
        tris.append(f"{i + n + 1}\n")
        tris.append(f"{i}\n")
        tris.append(f"{i + n + 1}\n")
        tris.append(f"{i + n}\n")
    for i in range(2 * n, 3 * n - 2):
        tris.append(f"{i + n}\n")
        tris.append(f"{i + n + 1}\n")
        tris.append(f"{i}\n")
        tris.append(f"{i + n + 1}\n")
        tris.append(f"{i + 1}\n")
        tris.append(f"{i}\n")

    return verts, tris, uvs


def create_resources(folder_path):
    """
    Processes a single scan on the HoloScan server into mesh and texture files.

    Notes:
        folder path must be of the form './Survyes/PerkinsYard/090', and the folder must contain the files
        FILE____090.dzt and PATH____090.csv

    Args:
        folder_path (str): path to the folder being processed

    """

    # extract the scan number from folder_path
    scan_number = folder_path.rsplit(os.sep, maxsplit=1)[1]

    # construct the relevant file paths
    dzt_path = f"{folder_path}{os.sep}FILE____{scan_number}.DZT"        # Path to radar data
    positon_path = f"{folder_path}{os.sep}PATH____{scan_number}.csv"    # Path to position data
    st_texture_path = f"{folder_path}{os.sep}st.png"                    # Path for the standard texture
    rtt_texture_path = f"{folder_path}{os.sep}rtt.png"                  # Path for the reduced texture
    vert_path = f"{folder_path}{os.sep}verts.csv"                       # Path for mesh verticies
    tri_path = f"{folder_path}{os.sep}tris.csv"                         # Path for mesh triangles
    uv_path = f"{folder_path}{os.sep}uvs.csv"                           # Path for mesh uvs

    # It should not be possible to raise these errors but its always good to be safe
    if not os.path.exists(dzt_path):
        raise FileNotFoundError("DZT File Not Found")

    if not os.path.exists(positon_path):
        raise FileNotFoundError("Position File Not Found")

    # Extract radar and position data
    header, data, _ = readdzt(dzt_path)
    positions = read_position_data(positon_path)

    # Create the standard and reduced texture (only radar data dependant)
    standard, reduced = create_textures(header, data)

    # Create the mesh data (dependant on position data and max scan depth)
    verts, tris, uvs = create_geometry(*positions, header['rhf_depth'])

    # Write out mesh files
    with open(vert_path, "w") as f:
        f.writelines(verts)

    with open(tri_path, "w") as f:
        f.writelines(tris)

    with open(uv_path, "w") as f:
        f.writelines(uvs)

    # Save both images
    imsave(st_texture_path, standard)
    imsave(rtt_texture_path, reduced)

    return
