import numpy as np
import cv2
from matplotlib import pyplot as plt
from flowbel import Instance
import flowbel
import argparse
import os
import glob
import random

ap = argparse.ArgumentParser("Visualize Darknet Dataset")
ap.add_argument("--dataset_path", required=True, help="Dataset folder")
ap.add_argument("--result_file", required=True, help="Result file")
ap.add_argument("--angle_discretization", required=True, help="Angle Discretization", type=float)
args = vars(ap.parse_args())


dataset = flowbel.Dataset(dataset_path=args['dataset_path'])
dataset_manifest = dataset.getDatasetManifest()

angle_discretization = args['angle_discretization']
if angle_discretization < 0.0:
    angle_discretization = flowbel.AngleDiscretizationExp8()

f = open(args['result_file'], 'r')
rows = sorted(f.readlines())

for r in rows:

    image_path, instances = Instance.parseRowString(r)
    print(r, instances)
    image = cv2.imread(image_path)

    for i in instances:

        if isinstance(angle_discretization, flowbel.AngleDiscretization):
            classes = angle_discretization.classes_size
            object_label = int(i.label / classes)
            inner_label = i.label % classes
            print(i.label, classes, object_label, inner_label)
            angle, arc = angle_discretization.getAngleAndArc(inner_label)
        else:
            classes = int(360.0 / angle_discretization) + 1
            object_label = int(i.label / classes)
            angle_label = i.label % classes
            angle = float(angle_label * angle_discretization)
            arc = angle_discretization

        length = np.linalg.norm(i.x_axis) * 0.5
        angle1 = (angle + arc * 0.5) * np.pi / 180.0
        angle2 = (angle - arc * 0.5) * np.pi / 180.0
        p1 = i.center() + length * np.array([np.cos(angle1), np.sin(angle1)])
        p2 = i.center() + length * np.array([np.cos(angle2), np.sin(angle2)])

        i.draw(image, custom_text="{}_{}".format(
            dataset_manifest.getName(object_label),
            int(angle * 180.0 / np.pi)
        ))
        cv2.line(image, tuple(i.center().astype(int)), tuple(p1.astype(int)), (255, 255, 255), 1)
        cv2.line(image, tuple(i.center().astype(int)), tuple(p2.astype(int)), (255, 255, 255), 1)
        cv2.line(image, tuple(p1.astype(int)), tuple(p2.astype(int)), (255, 255, 255), 1)

    # i = random.choice(instances)
    # # for i in instances:

    # i.draw(image)
    # newinst = Instance.generateRotatedInstance(i.buildBoundingBox(), i.angle(), i.ratio())
    # newinst.draw(image)

    print("OPENING ", image_path, instances)
    cv2.imshow("output", image)
    cv2.waitKey(0)
