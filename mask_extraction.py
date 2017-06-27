from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import csv
import os
from glob import glob
import pandas as pd
import settings
try:
    from tqdm import tqdm # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x


############
#
# Getting list of image files
DATA_DIR = settings.BASE_DIR
TRAIN_DIR = settings.TRAIN_RAW_DIR
VAL_DIR = settings.VAL_RAW_DIR
#TEST_DIR = settings.TEST_RAW_DIR
CSV_DIR = settings.CSV_DIR
OUTPUT_DIR = settings.OUTPUT_DIR

#luna_subset_path = settings.TRAIN_RAW_DIR + 'train_subset00/' #luna_path+"subset1/"
#output_path = settings.BASE_DIR + 'tutorial_out/' #"/home/jonathan/tutorial/"
#file_list=glob(luna_subset_path+"*.mhd")

#Some helper functions

def make_mask(center,diam,z,width,height,spacing,origin):
    '''
Center : centers of circles px -- list of coordinates x,y,z
diam : diameters of circles px -- diameter
widthXheight : pixel dim of image
spacing = mm/px conversion rate np array x,y,z
origin = x,y,z mm np.array
z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height,width]) # 0's everywhere except nodule swapping x,y to match img
    #convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center-origin)/spacing
    v_diam = int(diam/spacing[0]+5)
    v_xmin = np.max([0,int(v_center[0]-v_diam)-5])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)+5])
    v_ymin = np.max([0,int(v_center[1]-v_diam)-5]) 
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)+5])

    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)

    # Convert back to world coordinates for distance calculation
    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
    return(mask)

def matrix2int16(matrix):
    ''' 
matrix must be a numpy array NXN
Returns uint16 version
    '''
    m_min= np.min(matrix)
    m_max= np.max(matrix)
    matrix = matrix-m_min
    return(np.array(np.rint( (matrix-m_min)/float(m_max-m_min) * 65535.0),dtype=np.uint16))




#####
#
# Looping over the image files
#
def extract_masks(df_node, file_list, output_path):
    for fcount, img_file in enumerate(tqdm(file_list)):
        seriesuid = img_file.split('/')[-1].split('.')[0]
        print(seriesuid)
        print(img_file)
        mini_df = df_node[df_node["seriesuid"]==seriesuid] #get all nodules associate with file
        if mini_df.shape[0] > 0: # some files may not have a nodule--skipping those 
            # load the data once
            itk_img = sitk.ReadImage(img_file) 
            img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
            num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
            origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
            spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
            # go through all nodes (why just the biggest?)
            for node_idx, cur_row in mini_df.iterrows():       
                node_x = cur_row["coordX"]
                node_y = cur_row["coordY"]
                node_z = cur_row["coordZ"]
                diam = cur_row["diameter_mm"]
                # just keep 3 slices
                imgs = np.ndarray([3,height,width],dtype=np.float32)
                masks = np.ndarray([3,height,width],dtype=np.uint8)
                center = np.array([node_x, node_y, node_z])   # nodule center
                v_center = np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)
                for i, i_z in enumerate(np.arange(int(v_center[2])-1,
                                int(v_center[2])+2).clip(0, num_z-1)): # clip prevents going out of bounds in Z
                    mask = make_mask(center, diam, i_z*spacing[2]+origin[2],
                                    width, height, spacing, origin)
                    masks[i] = mask
                    imgs[i] = img_array[i_z]
                np.save(os.path.join(output_path,"images_{}_{}.npy".format(seriesuid, str(node_idx).zfill(4))),imgs)
                np.save(os.path.join(output_path,"masks_{}_{}.npy".format(seriesuid, str(node_idx).zfill(4))),masks)



df_node = pd.read_csv(CSV_DIR+"train/annotations.csv")
df_val = pd.read_csv(CSV_DIR+"val/annotations.csv")
#df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
df_node = df_node.dropna()
df_val = df_val.dropna()

train_file_list = glob(TRAIN_DIR+'*/*.mhd')
val_file_list = glob(VAL_DIR+'*/*.mhd')

#file_list.extend(val_file_list)
print(len(train_file_list))
print(train_file_list[:5])

print(len(val_file_list))
print(val_file_list[:5])

TRAIN_MASK_OUT_DIR = OUTPUT_DIR + 'masks/train/'
VAL_MASK_OUT_DIR = OUTPUT_DIR + 'masks/val/'

if not os.path.exists(TRAIN_MASK_OUT_DIR):
    os.mkdir(TRAIN_MASK_OUT_DIR)
if not os.path.exists(VAL_MASK_OUT_DIR):
    os.mkdir(VAL_MASK_OUT_DIR)

#extract_masks(df_node, train_file_list, TRAIN_MASK_OUT_DIR)
extract_masks(df_val, val_file_list, VAL_MASK_OUT_DIR)

#####################
#
# Helper function to get rows in data frame associated 
# with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)
#
# The locations of the nodes
#df_node = pd.read_csv(CSV_DIR+"train/annotations.csv")
#df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
#df_node = df_node.dropna()
