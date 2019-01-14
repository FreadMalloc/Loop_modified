import numpy as np
import cv2
from matplotlib import pyplot as plt
from flowbel import Instance
import flowbel
import argparse
import os
import glob


class SiftObjectDetector(object):

    MIN_MATCH_COUNT = 2

    def __init__(self):
        # Initiate SIFT detector
        try:
            self.detector = cv2.xfeatures2d.SIFT_create()
        except:
            self.detector = cv2.FeatureDetector_create("SIFT")

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def detect(self, object_image, scene_image, th_ratio=0.75, rigid=True, debug=False):
        try:
            # find the keypoints and descriptors with SIFT
            kp1, des1 = self.detector.detectAndCompute(object_image, None)
            kp2, des2 = self.detector.detectAndCompute(scene_image, None)

            if debug:
                output_image = scene_image.copy()

            matches = self.matcher.knnMatch(des1, des2, k=2)

            points = []

            # store all the good matches as per Lowe's ratio test.
            good = []
            for m, n in matches:
                if m.distance < th_ratio * n.distance:
                    good.append(m)

            # create BFMatcher object
            # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            # print("OK")
            # # Match descriptors.
            # matches = bf.match(des1, des2)

            # # Sort them in the order of their distance.
            # good = sorted(matches, key=lambda x: x.distance)

            if len(good) >= SiftObjectDetector.MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                matchesMask = None
                if rigid:
                    M = cv2.estimateRigidTransform(src_pts, dst_pts, False)
                    M = np.vstack((M, np.array([0, 0, 1])))

                else:
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                    matchesMask = mask.ravel().tolist()

                h, w = object_image.shape[:2]
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                dst = dst.reshape((4, 2))

                p0 = tuple(dst[0, :].astype(int))
                p1 = tuple(dst[3, :].astype(int))
                p2 = tuple(dst[2, :].astype(int))
                p3 = tuple(dst[1, :].astype(int))
                points = np.array([
                    dst[0, :],
                    dst[3, :],
                    dst[2, :],
                    dst[1, :]
                ])
                #print("POINTS: ", points)

                if debug:
                    cv2.line(output_image, p0, p1, (0, 0, 255), 2)
                    cv2.line(output_image, p0, p3, (0, 255, 0), 2)
                    cv2.line(output_image, p1, p2, (255, 0, 0), 2)
                    cv2.line(output_image, p2, p3, (255, 0, 0), 2)

            else:
                if debug:
                    print("Not enough matches are found - %d/%d" % (len(good), SiftObjectDetector.MIN_MATCH_COUNT))
                matchesMask = None
                return None

            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)
            if debug:
                output_image = cv2.drawMatches(object_image, kp1, output_image, kp2, good, None, **draw_params)
                cv2.imshow("debug", output_image)
                cv2.waitKey(0)

            return points
        except Exception as e:
            if debug:
                print("Error: ", e)
            return None


ap = argparse.ArgumentParser("Detector Evaluator")
ap.add_argument("--dataset_path", required=True, help="Dataset folder")
ap.add_argument("--images_list", required=True, help="Images list file")
ap.add_argument("--models_path", required=True, help="Models folder")
ap.add_argument("--results_file", required=True, help="Output Results file")
args = vars(ap.parse_args())


dataset = flowbel.Dataset(dataset_path=args['dataset_path'])
dataset_manifest = dataset.getDatasetManifest()


models_folders = glob.glob(os.path.join(args['models_path'], "*"))
models_folders = sorted([f for f in models_folders if os.path.isdir(f)])
object_names = dataset_manifest.getPurgedList()

# detector
detector = SiftObjectDetector()


# scenes
images_list = []
fo = open(args['images_list'], "r")
rows = fo.readlines()
images = map(lambda row: row.split(' ')[0], rows)
#images = sorted(glob.glob(os.path.join(images_folder, "*.jpg")))

rows = []
for im in images:
    scene = cv2.imread(im)
    instances = []
    for object_name in object_names:
        # model
        for models_folder in models_folders:
            model_path = os.path.join(models_folder, object_name + ".png")
            model = cv2.imread(model_path)

            points = detector.detect(model, scene, debug=False)
            if points is not None:
                label = dataset_manifest.getLabel(object_name)
                pointlist = [label] + np.round(points).astype(int).ravel().tolist()
                #print(label, points, pointlist, ','.join(map(str, pointlist)))
                instance = Instance(label=label, points=points)
                instances.append(instance)

    print("FOund: ", len(instances))
    rows.append(Instance.convertInstancesToRowString(im, instances))
    #print(Instance.convertInstancesToRowString(im, instances))

f = open(args['results_file'], 'w')
for i, r in enumerate(rows):
    f.write(r)
    if i < len(rows) - 1:
        f.write('\n')
