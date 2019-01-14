import flowbel
import cv2
import os
import math
import argparse
import sys
import numpy as np
import json

ap = argparse.ArgumentParser("Convert a Dataset to VOC format")
ap.add_argument("--dataset_path", required=True, help="Dataset folder")
ap.add_argument("--output_folder", required=True, help="Output folder")
args = vars(ap.parse_args())

scene_list = [
    'scan_00',
    'scan_01',
    'scan_02',
    'scan_03',
    'scan_04',
    #'scan_05',
    'scan_06',
    'scan_07',
    'scan_08',
    'scan_09',
    'scan_10',
    'scan_11',
    'scan_12',
    'scan_13',
    'scan_14',
    #'scan_15'
]

dataset = flowbel.Dataset(dataset_path=args['dataset_path'])

whole_annotations = []
for name, scene in dataset.scenes.items():
    if name in scene_list:
        for annotation in scene.getImageAnnotations():
            whole_annotations.append(annotation)

flowbel.VOCUtils.exportAnnotationToVocDatasetForAngleClassification(args['output_folder'], whole_annotations, 10.0)
print("Output annotations: ", len(whole_annotations))


manifest = {
    'output_scenes': scene_list
}
json.dump(manifest, open(
    os.path.join(args['output_folder'], "manifest.json"), 'w'
))
