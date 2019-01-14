import flowbel
import cv2
import os
import math
import argparse
import sys
import numpy as np
import json

ap = argparse.ArgumentParser("Convert a Dataset to DARKNET format")
ap.add_argument("--dataset_path", required=True, help="Dataset folder")
ap.add_argument("--output_file", required=True, help="Output file")
ap.add_argument("--angle_discretization", default=10.0, help="Angle discretization", type=float)
args = vars(ap.parse_args())

scene_list = [
    # 'scan_01',
    # 'scan_02',
    # 'scan_03',
    # 'scan_04',
    # 'scan_05',
    # 'scan_06',
    # 'scan_07',
    # 'scan_08',
    # 'scan_09',
    # 'scan_10',
    # 'scan_11',
    # 'scan_12',
    # 'scan_13',
    # 'scan_14',
    'scan_15'
]

dataset = flowbel.Dataset(dataset_path=args['dataset_path'])

whole_annotations = []
rows = []
for name, scene in dataset.scenes.items():
    if name in scene_list:
        for annotation in scene.getImageAnnotations():
            whole_annotations.append(annotation)
            row = str(annotation.getImagePath())
            for i in annotation.getInstances():
                row += ' ' + i.toString()
            rows.append(row)

f = open(args['output_file'], 'w')
for i, r in enumerate(rows):
    f.write(r)
    if i < len(rows) - 1:
        f.write('\n')
f.close()


classes = dataset.getDatasetManifest().getPurgedList()
f = open(args['output_file']+".classes", 'w')
for i, c in enumerate(classes):
    f.write(c)
    if i < len(rows) - 1:
        f.write('\n')
f.close()


# flowbel.DARKENTUtils.exportAnnotationToDarknetDatasetForAngleClassification(
#     args['output_file'], whole_annotations, args['angle_discretization']
# )
# print("Output annotations: ", len(whole_annotations))

# classlist = flowbel.DARKENTUtils.convertSimpleClassList(dataset.getDatasetManifest().getPurgedList(), args['angle_discretization'])
# manifest_file = args['output_file'] + ".classes.txt"
# f = open(manifest_file, 'w')
# for i, c in enumerate(classlist):
#     f.write(c)
#     if i < len(classlist) - 1:
#         f.write('\n')
