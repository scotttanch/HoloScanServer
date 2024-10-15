import os

import numpy as np
import pandas as pd
from readgssi.dzt import readdzt
import matplotlib.pyplot as plt
import mesh_generation

# This script should take in a dzt file and a moasure path file and generate a mesh and texture

MAIN_PARENT = "./Surveys/"


def read_moasure(survey_folder):
    """
    Read a moasure project file and seperate it into csv based on the layer name. This function modified the directory
    structure by adding subfolders to the survey folder for each layer. It is also assumed that the csv and survey
    folder share a name, up to the csv extension of course
    :param survey_folder:
    :return:
    """
    survey_csv = "./Surveys/" + survey_folder + "/" + survey_folder + ".csv"
    # this will raise a file not found exception if the file is named improperly, I cant seem to catch that
    # OSError doesn't derive from BaseException?
    df = pd.read_csv(survey_csv)

    # Each layer will corespond to a seperate dzt file so we want to split them into seperate csvs
    layers = df['Layer-Name'].unique()
    sub_folders = []
    for layer in layers:
        sub_frame = df.loc[df['Layer-Name'] == layer]
        # I'm going to do the recodinating here
        # unity uses a left-handed system and the moasure is right, so I need to swap y and z
        xs = sub_frame['X:ft'].to_list()
        zs = sub_frame['Y:ft'].to_list()
        ys = sub_frame['Z:ft'].to_list()
        # all DZT names are 3 numbers, so we need to pad the string with zeros
        scan_name = str(layer).zfill(3)
        sub_folders.append(scan_name)
        # each new csv will be written to a unique folder for that scan
        os.mkdir("./Surveys/" + survey_folder + "/" + scan_name)
        with open("./Surveys/" + survey_folder + "/" + scan_name + "/FILE____" + scan_name + ".csv", "w+") as f:
            for (x, y, z) in zip(xs, ys, zs):
                # I'm also going to convert to meters here
                line = str(x * 0.3048) + "," + str(y * 0.3048) + "," + str(z * 0.3048) + "\n"
                f.write(line)

    return sub_folders


def read_path_csv(csv_file):
    df = pd.read_csv(csv_file, header=None)
    xs = df[0].to_list()
    ys = df[2].to_list()
    zs = df[1].to_list()
    # TODO: Implement some kind of smoothing on points being read in. Maybe step through and ignore any points less that some critical separation
    return xs, ys, zs

def texture_from_dzt(scan_path, scan_name, model_folder):

    # Scan path is the path from the main dir to the dzt file
    # Scan name is of the form FILE____XXX.DZT
    # model_folder is the path to the folder where model data will be stored

    header, array, _ = readdzt(scan_path)
    # I'm pretty sure array is actually a dictionary of channels, we can assume that our data is single channel, but
    # it's still a dict or a list so the first channel being 0
    channel_0 = array[0]
    depth = header['rhf_depth']
    # Presumably scan name includes the .DZT extension, so we need to strip that off
    raw_name = scan_name.split(".")[0]
    texture_name = raw_name + ".png"
    scan_folder = raw_name.split("_")[-1]
    # the texture should be saved to the folder under the parent directory that corresponds to the scan number
    # The scan number needs to extracted from
    flipped = np.fliplr(channel_0)
    full_texture = np.hstack((flipped, channel_0))
    plt.imsave(model_folder + "/" + texture_name, full_texture)

    # TODO: Implement a second texture set. This one should remove the background in a more literal sense by making it transparent

    return depth


# This function will take in a parent directory and scan number and create mesh, uv, and texture data
def create_resource(parent_directory, scan_number):

    # build the gssi format file name from a 3 digit scan number
    dzt_filename = "FILE____" + scan_number + ".DZT"

    # Use the previous file name and the parent directory to locate the actual dzt file
    dzt_location = parent_directory + "/" + scan_number + "/" + dzt_filename

    # Set the path for the model data folder based on the parent folder and scan number
    model_data_folder = parent_directory + "/" + scan_number

    # Set the scan line file
    scan_line_file = model_data_folder + "/" + "PATH____" + scan_number + ".csv"

    # Need to read in the x,y,z cordinates of the survey from the model_folder
    scan_x, scan_y, scan_z = read_path_csv(scan_line_file)

    # I think I want to delete the scan_line_file from the directory after it's been used to create then mesh
    #os.remove(scan_line_file)

    # find the dzt and write the image texture to the model data folder
    depth = texture_from_dzt(dzt_location, dzt_filename, model_data_folder)

    # use the scan points and the depth from reading the dzt to create the mesh data and save it to the model folder
    mesh_generation.generate_mesh(scan_x, scan_y, scan_z, depth, model_data_folder)

    return None


# This function should take a survey directory path
def process_survey(survey_folder):
    # read the moasure csv file, create a new sub folder for each layer, and put the csv for each layer in the right
    # sub folder
    parent_folder_path = MAIN_PARENT + survey_folder
    scan_folders = [sub_folder for sub_folder in os.listdir(parent_folder_path) if os.path.isdir(os.path.join(parent_folder_path, sub_folder))]
    for sub_folder in scan_folders:
        #print(sub_folder)
        create_resource(parent_folder_path, sub_folder)
    return None


process_survey("PerkinsYard2")
#create_resource("./Surveys/Perkins Yard", "068")
#create_resource("./Surveys/Perkins Yard", "069")
#create_resource("./Surveys/Perkins Yard", "070")
#texture_from_dzt("./Surveys/Perkins Yard", "FILE____068.DZT")
