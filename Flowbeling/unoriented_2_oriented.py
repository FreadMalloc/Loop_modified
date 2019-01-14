import numpy as np
import cv2
from matplotlib import pyplot as plt
from flowbel import Instance
import flowbel
import argparse
import os
import glob
import random

ap = argparse.ArgumentParser("Detector Evaluator")
ap.add_argument("--dataset_path", required=True, help="Dataset folder")
ap.add_argument("--result_file", required=True, help="Result file")
ap.add_argument("--output_file", required=True, help="Output file")
ap.add_argument("--angle_discretization", default=10.0, help="Angle discretization", type=float)
args = vars(ap.parse_args())

dataset = flowbel.Dataset(dataset_path=args['dataset_path'])

debug = False

f = open(args['result_file'], 'r')
fout = open(args['output_file'], 'w')
rows = f.readlines()

for r in rows:
    # print(r)
    chunks = r.split(' ')

    instances_map = {}
    for i in range(1, len(chunks)):
        #instance = dataset.generateInstanceFromClassification(chunks[i], 10.0, estimate_rotated_box=False)
        instance_rot = dataset.generateInstanceFromClassification(chunks[i], args['angle_discretization'], estimate_rotated_box=True)
        if instance_rot.label not in instances_map:
            instances_map[instance_rot.label] = []
        instances_map[instance_rot.label].append(instance_rot)

    output = str(chunks[0])
    for l, instances in instances_map.items():
        instances = sorted(instances, key=lambda i: i.score, reverse=True)
        # print(instances)
        for inst in instances:
            if debug:
                image = cv2.imread(chunks[0])
                inst.draw(image)
            output += ' '+inst.toString(with_score=True)
            if debug:
                cv2.imshow("img", image)
                cv2.waitKey(0)

    fout.write(output)
    fout.write('\n')

    if False:
        # test couples to check IOU
        for a, b in zip(instances[:-1], instances[1:]):

            a.draw(image)
            b.draw(image)
            iou = a.computeIOU(b)
            oiou = a.computeOIOU(b)
            print("Iou", iou, "Oiou", oiou)

            cv2.imshow("img", image)
            cv2.waitKey(0)

f.close()
fout.close()
