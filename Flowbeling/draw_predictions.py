import numpy as np
import cv2
from matplotlib import pyplot as plt
from flowbel import Instance
import flowbel
import argparse
import os
import glob
import time
import random

ap = argparse.ArgumentParser("Detector Evaluator")
ap.add_argument("--dataset_path", required=True, help="Dataset folder")
ap.add_argument("--result_file", required=True, help="Result file")
ap.add_argument("--min_th", default=0.5, help="Min TH", type=float)
ap.add_argument("--save", default=False, action="store_true")
args = vars(ap.parse_args())

dataset = flowbel.Dataset(dataset_path=args['dataset_path'])


f = open(args['result_file'], 'r')
rows = f.readlines()

timestamp = str(time.time())

for counter, r in enumerate(rows):
    # print(r)
    # chunks = r.split(' ')
    # image = cv2.imread(chunks[0])
    # for i in range(1, len(chunks)):
    #     instance = dataset.generateInstanceFromClassification(chunks[i], 10.0, estimate_rotated_box=False)

    #     instance_rot = dataset.generateInstanceFromClassification(chunks[i], 10.0, estimate_rotated_box=True)
    #     if instance.score < 0.2:
    #         continue
    #     print(instance_rot.score)
    #     # instance.draw(image)
    #     instance_rot.draw(image)
    # cv2.imshow("img", image)
    # cv2.waitKey(0)

    # print(chunks[i])

    image_path, instances = Instance.parseRowString(r, args['result_file'])
    if image_path is None or len(image_path) == 0:
        continue
    print(r, instances)
    image = cv2.imread(image_path)

    for i in instances:
        i.dataset_manifest = dataset.dataset_manifest

        if i.score < args['min_th']:
            continue

        print(i)
        if i.unoriented_instance:

            i.draw2(image, fixed_color=(59, 235, 255), custom_text='')
        else:
            i.draw2(image, custom_text=dataset.dataset_manifest.getName(i.label))

    target_color = (255, 255, 255)
    cv2.line(image, (320, 0), (320, 640), target_color, 1)
    cv2.line(image, (0, 240), (640, 240), target_color, 1)
    cv2.circle(image, (320, 240), 100, target_color, 1)
    # i = random.choice(instances)
    # # for i in instances:

    # i.draw(image)
    # newinst = Instance.generateRotatedInstance(i.buildBoundingBox(), i.angle(), i.ratio())
    # newinst.draw(image)

    print("OPENING ", image_path, instances)
    cv2.imshow("output", image)
    cv2.waitKey(0)
    if args['save']:
        cv2.imwrite('/tmp/{}_{}.png'.format(timestamp, counter), image)
