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
    'scan_03',
    'scan_06',
    'scan_07',
    'scan_08',
    'scan_09',
    'scan_10',
]

dataset = flowbel.Dataset(dataset_path=args['dataset_path'])


angle_discretization = args['angle_discretization']
if angle_discretization < 0.0:
    angle_discretization = flowbel.AngleDiscretizationExp8()

whole_annotations = []
for name, scene in dataset.scenes.items():
    if name in scene_list:
        for annotation in scene.getImageAnnotations():
            whole_annotations.append(annotation)

flowbel.DARKENTUtils.exportAnnotationToDarknetDatasetForAngleClassification(
    args['output_file'], whole_annotations, angle_discretization
)
print("Output annotations: ", len(whole_annotations))

classlist = flowbel.DARKENTUtils.convertSimpleClassList(dataset.getDatasetManifest().getPurgedList(), angle_discretization)
manifest_file = args['output_file'] + "_classes.txt"
f = open(manifest_file, 'w')
for i, c in enumerate(classlist):
    f.write(c)
    if i < len(classlist) - 1:
        f.write('\n')
