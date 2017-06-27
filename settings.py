import os
#COMPUTER_NAME = os.environ['hostname']
#print("Computer: ", COMPUTER_NAME)

TARGET_VOXEL_MM = 1.00
MEAN_PIXEL_VALUE_NODULE = 41
LUNA_SUBSET_START_INDEX = 0
SEGMENTER_IMG_SIZE = 320


BASE_DIR = '/home/chicm/ml/kgdata/tianchi/'
TRAIN_RAW_DIR = BASE_DIR + 'train/'
VAL_RAW_DIR = BASE_DIR + 'val/'
TEST_RAW_DIR = BASE_DIR + 'test2/'
CSV_DIR = BASE_DIR + 'csv/'
OUTPUT_DIR = BASE_DIR + 'tutorial_out/'

PRED_CSV_DIR = BASE_DIR + 'pred_csv/'

EXTRACTED_IMAGE_DIR = BASE_DIR + "extracted_images/"

