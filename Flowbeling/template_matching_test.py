import cv2
import os
import glob
import numpy as np

import imutils

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


def findObject(model, scene, angle_discretization=10.0, method=cv2.TM_CCOEFF_NORMED):
    max_score = -10000000000000000000.0
    max_angle = 0.0
    debug = None
    bbox = [0.0, 0.0]
    angle_discretization = int(angle_discretization)
    for angle in range(0, 360 + angle_discretization, angle_discretization):
        rotated = imutils.rotate_bound(model, angle)
        w, h = rotated.shape[::-1]

        #print(img.shape, img.dtype)

        res = cv2.matchTemplate(img, rotated, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            max_val = -max_val
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        if max_val > max_score:
            max_score = max_val
            max_angle = angle
            debug = rotated
            bbox[0] = tuple(top_left)
            bbox[1] = tuple(bottom_right)
            #print("NEW MAX FOUND", top_left, bottom_right, max_val)

        # output = img .copy()
        # cv2.rectangle(output, top_left, bottom_right, 255, 2)
        # cv2.imshow("debug", output)
        # cv2.waitKey(0)

    return bbox, max_angle, max_val, debug


img = cv2.imread('/Users/daniele/Desktop/to_delete/Compass2018_FLOWBEL/scan_01/images/0000404.jpg', 0)
scene_path = '/Users/daniele/Desktop/to_delete/Compass2018_FLOWBEL/scan_01/images/'
scenes = glob.glob(os.path.join(scene_path, "*.jpg"))
img2 = img.copy()
template = cv2.imread('/Users/daniele/Desktop/to_delete/Compass2018_models/simple/screwdriver.png', 0)
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
for scene in scenes:
    img = cv2.imread(scene, 0)
    bbox, angle, score, debug = findObject(template, img, 10.0)
    print(bbox)
    output = img .copy()
    cv2.rectangle(output, bbox[0], bbox[1], 255, 2)
    cv2.putText(output, "{}".format(angle), bbox[0], cv2.FONT_HERSHEY_PLAIN, 1, (255))

    cv2.imshow("scene", output)
    cv2.imshow("db", debug)
    cv2.waitKey(0)

    for meth in methods:
        img = img2.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img, top_left, bottom_right, 255, 2)
