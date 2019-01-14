import flowbel
import cv2
import os
import math
import argparse
import sys
import numpy as np
import json
import glob
import imutils

DEFAULT_IMAGE_SIZE = (640, 480)


def cropCenter(img, size):
    h, w = img.shape[:2]
    padding_w = int((w - size[0]) / 2)
    padding_h = int((h - size[1]) / 2)
    return img.copy()[padding_h: padding_h + size[1], padding_w: padding_w + size[0]]


class Model(object):

    def __init__(self, image_path, padding=5, ref_size=DEFAULT_IMAGE_SIZE):
        self.image_path = image_path
        model_full = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        h, w = model_full.shape[:2]
        self.model_full = model_full[5:h - 5, 5: w - 5, :]
        self.template = self.model_full[:, :, :3]
        self.template_mask = self.model_full[:, :, 3]
        self.ref_size = ref_size

    def projectTempate(self, template, instance):
        template = cv2.resize(template, tuple(instance.size().astype(int)))

        th, tw = template.shape[:2]
        pts1 = np.array([
            [0, 0],
            [tw, 0],
            [tw, th]
        ]).astype(np.float32)
        pts2 = i.points[:3, :].astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)

        template_image = template  # cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)
        template_warp = cv2.warpAffine(template_image, M, self.ref_size)
        return template_warp

    def project(self, instance):
        template_warp = self.projectTempate(self.template, instance)
        template_mask_warp = self.projectTempate(self.template_mask, instance)

        template_mask_warp = cv2.blur(template_mask_warp, (3, 3))

        template_warp = template_warp.astype(float) / 255.
        template_mask_warp = template_mask_warp.astype(float) / 255.

        template_mask_warp = np.stack((template_mask_warp, template_mask_warp, template_mask_warp), axis=2)

        template_warp = cv2.multiply(template_warp, template_mask_warp)
        return (template_warp * 255.).astype(np.uint8), (template_mask_warp * 255.).astype(np.uint8)


ap = argparse.ArgumentParser("Generate Synthetic Scenes")
ap.add_argument("--dataset_path", required=True, help="Dataset folder")
ap.add_argument("--models_folder", required=True, help="Models Folder")
ap.add_argument("--backgrounds_folder", required=True, help="Background Folder")
ap.add_argument("--output_folder", required=True, help="Output Folder")

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

scene_tags = {
    'scan_01': 'simple',
    'scan_02': 'simple',
    'scan_03': 'simple',
    'scan_04': 'wood',
    'scan_05': 'wood',
    'scan_06': 'wood',
    'scan_07': 'black',
    'scan_08': 'black',
    'scan_09': 'black',
    'scan_10': 'pollock',
    'scan_11': 'pollock',
    'scan_12': 'pollock',
    'scan_13': 'pollock2',
    'scan_14': 'pollock2',
    'scan_15': 'pollock2'
}


models_files = glob.glob(os.path.join(args['models_folder'], '*'))
models = {}
for f in models_files:
    name = os.path.splitext(os.path.basename(f))[0]
    models[name] = Model(f)

backgrounds_files = glob.glob(os.path.join(args['backgrounds_folder'], '*'))
backgrounds = {}
for f in backgrounds_files:
    name = os.path.splitext(os.path.basename(f))[0]
    backgrounds[name] = cv2.imread(f)


dataset = flowbel.Dataset(dataset_path=args['dataset_path'])


for name, scene in dataset.scenes.items():
    if name in scene_list:
        print(name)
        output_scene_folder = os.path.join(args['output_folder'], name, 'images')
        if not os.path.exists(output_scene_folder):
            try:
                os.makedirs(output_scene_folder)
            except Exception as e:
                print("ERROR ", e)
        for index, first_annotation in enumerate(scene.getImageAnnotations()):

            tag = scene_tags[name]
            current_background = backgrounds[tag].copy()
            current_background = imutils.rotate(current_background, index)
            current_background = cropCenter(current_background, DEFAULT_IMAGE_SIZE)

            #first_annotation = scene.getImageAnnotations()[0]
            print(first_annotation.getImagePath())

            whole_instances_image = np.zeros((DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0], 3), np.uint8)
            whole_mask_image = np.zeros((DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0], 3), np.uint8)

            for i in first_annotation.getInstances():
                classname = dataset.getDatasetManifest().getName(i.label)
                model = models[classname]

                template_warp, template_mask = model.project(i)
                whole_instances_image += template_warp
                whole_mask_image += template_mask

            whole_mask_image_inv = 255 - whole_mask_image

            im1 = whole_instances_image.astype(float) / 255.
            mask1 = whole_mask_image.astype(float) / 255.
            im2 = current_background.astype(float) / 255.
            mask2 = whole_mask_image_inv.astype(float) / 255.

            # cv2.imshow("warp", whole_instances_image)
            # cv2.waitKey(0)
            # cv2.imshow("warp", whole_mask_image)
            # cv2.waitKey(0)
            # cv2.imshow("warp", whole_mask_image_inv)
            # cv2.waitKey(0)

            result = ((im1 * mask1 + im2 * mask2) * 255.).astype(np.uint8)

            if False:
                cv2.imshow("warp", result)
                cv2.waitKey(0)

            output_file = os.path.join(output_scene_folder, first_annotation.getBasename())
            cv2.imwrite(output_file, result)
