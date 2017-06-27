import settings
import helpers
import SimpleITK  # conda install -c https://conda.anaconda.org/simpleitk SimpleITK
import numpy
import pandas
#import ntpath
import cv2  # conda install -c https://conda.anaconda.org/menpo opencv3
import shutil
import random
import math
import multiprocessing
from bs4 import BeautifulSoup #  conda install beautifulsoup4, coda install lxml
import os
import glob

random.seed(1321)
numpy.random.seed(1321)


def find_mhd_file(patient_id):
    for subject_no in range(settings.LUNA_SUBSET_START_INDEX, 10):
        src_dir = settings.LUNA16_RAW_SRC_DIR + "subset" + str(subject_no) + "/"
        for src_path in glob.glob(src_dir + "*.mhd"):
            if patient_id in src_path:
                return src_path
    return None

def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def process_image(src_path):
    patient_id = os.path.basename(src_path).replace(".mhd", "")
    print("Patient: ", patient_id)

    dst_dir = settings.EXTRACTED_IMAGE_DIR + patient_id + "/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)

    origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    print("Origin (x,y,z): ", origin)

    direction = numpy.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
    print("Direction: ", direction)


    spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing / settings.TARGET_VOXEL_MM
    print("Rescale: ", rescale)

    img_array = helpers.rescale_patient_images(img_array, spacing, settings.TARGET_VOXEL_MM)

    img_list = []
    for i in range(img_array.shape[0]):
        img = img_array[i]
        seg_img, mask = helpers.get_segmented_lungs(img.copy())
        img_list.append(seg_img)
        img = normalize(img)
        cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_i.png", img * 255)
        cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_m.png", mask * 255)

def process_images(delete_existing=False, only_process_patient=None):
    if delete_existing and os.path.exists(settings.LUNA16_EXTRACTED_IMAGE_DIR):
        print("Removing old stuff..")
        if os.path.exists(settings.EXTRACTED_IMAGE_DIR):
            shutil.rmtree(settings.EXTRACTED_IMAGE_DIR)

    if not os.path.exists(settings.EXTRACTED_IMAGE_DIR):
        os.mkdir(settings.EXTRACTED_IMAGE_DIR)
        os.mkdir(settings.EXTRACTED_IMAGE_DIR + "_labels/")

    #for subject_no in range(settings.LUNA_SUBSET_START_INDEX, 10):
    for subject_no in range(6,15):
        src_dir = settings.TRAIN_RAW_DIR + "train_subset" + str(subject_no).zfill(2) + "/"
        #src_dir = settings.TEST_RAW_DIR + "test2_subset" + str(subject_no).zfill(2) + "/"
        
        src_paths = glob.glob(src_dir + "*.mhd")
        print("src_dir: {}". format(src_dir))
        print("src_paths: {}". format(src_paths))
        if only_process_patient is None and True:
            pool = multiprocessing.Pool(4)
            pool.map(process_image, src_paths)
            del pool
        else:
            for src_path in src_paths:
                print(src_path)
                if only_process_patient is not None:
                    if only_process_patient not in src_path:
                        continue
                process_image(src_path)

if __name__ == "__main__":
    if True:
        only_process_patient = None #'LKDS-00132'
        process_images(delete_existing=False, only_process_patient=only_process_patient)

