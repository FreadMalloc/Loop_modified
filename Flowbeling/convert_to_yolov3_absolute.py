import flowbel
import cv2
import os
import math
import argparse
import sys
import numpy as np
import json
import glob

ap = argparse.ArgumentParser("Convert a Dataset to YOLOv3 format")
ap.add_argument("--dataset_path", required=True, help="Dataset folder")
ap.add_argument("--output_file", required=True, help="Output file")
ap.add_argument("--angle_discretization", default=10.0, help="Angle discretization", type=float)
args = vars(ap.parse_args())

# Select here the filtered list of Scenes
scene_list = [
    'scan_06',
    'scan_07',
    'scan_08',
    #'scan_09',
    'scan_10'
]

# Dataset
dataset = flowbel.Dataset(dataset_path=args['dataset_path'])

# Angle discretization parameter
angle_discretization = args['angle_discretization']

# List of txt scene files
files = sorted(glob.glob(os.path.join(args['dataset_path'], "*.txt")))

new_rows = []  # list with new converted rows picked from each single scene txt file
for f in files:
    basename = os.path.splitext(os.path.basename(f))[0]
    if basename in scene_list:
        print("Loading: {}".format(basename))

        fo = open(f, 'r')
        rows = fo.readlines()

        for counter, r in enumerate(rows):
            #print(basename, counter, r)
            image_path, instances = flowbel.Instance.parseRowString(r)  # transform the file i-th row in a list of Instance(s)

            new_row = os.path.join(os.path.dirname(f),str(image_path))  # create new row with the same image path

            for i in instances:  # iterate each Instance  parsed from row
                if i is not None:  # check if Instance is null (N.B. EACH SPACE CHARACTER ADDED AT THE END OF ROW IS A NULL INSTANCE)
                    converted = flowbel.DARKENTUtils.convertInstanceForAngleClassification2String(i, angle_discretization)  # Apply conversion from Instance -> Yolov3 box
                    # print(i.toString(), " -> ", converted)
                    new_row += " " + converted
            new_rows.append(new_row)


# Write the New Rows to the output file
f = open(args['output_file'], 'w')
for i, nr in enumerate(new_rows):
    f.write(nr)
    if i < len(new_rows) - 1:
        f.write("\n")


classlist = flowbel.DARKENTUtils.convertSimpleClassList(dataset.getDatasetManifest().getPurgedList(), angle_discretization)
manifest_file = args['output_file'] + "_classes.txt"
f = open(manifest_file, 'w')
for i, c in enumerate(classlist):
    f.write(c)
    if i < len(classlist) - 1:
        f.write('\n')