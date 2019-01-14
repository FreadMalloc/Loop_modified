import flowbel
import cv2
import os
import math
import argparse
import sys
import numpy as np


ap = argparse.ArgumentParser("Extract Samples From Dataset")
ap.add_argument("--dataset_path", required=True, help="Dataset folder")
ap.add_argument("--output_path", required=True, help="Output folder")
ap.add_argument("--target_angle", default=0.0, help="Reference sample angle")
ap.add_argument("--scene_name", default='', help="Inner Scene Name")
# ap.add_argument("--class_name", required=True, type=str)
# ap.add_argument("--object_index", required=True, type=int)
# ap.add_argument('--blind', dest='blind', action='store_true')
args = vars(ap.parse_args())


dataset = flowbel.Dataset(dataset_path=args['dataset_path'])

scenes = list(dataset.scenes.values())

if len(args['scene_name']) == 0:
    scene = scenes[0]
else:
    scene = dataset.getSceneByName(args['scene_name'])

if scene is None:
    print("Scene is invalid!")
    sys.exit(0)

output_path = args['output_path']
if not os.path.exists(output_path):
    try:
        os.mkdir(args['output_path'])
    except:
        pass


for k, name in dataset.dataset_manifest.classmap.items():
    target_name = name

    sub, instance, annotation = scene.extractSample(target_name, target_angle=args['target_angle'])
    if sub is None:
        print("No sample for: {}".format(name))
        continue
    image = annotation.loadImage()
    instance.draw(image)
    cv2.imshow("image", image)
    cv2.imshow("sub", sub)
    cv2.waitKey(0)

    output_image = os.path.join(output_path, name+".png")
    cv2.imwrite(output_image, sub)
