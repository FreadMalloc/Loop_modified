import cv2
import os
import glob
import random
import numpy as np
import math
from scipy.optimize import differential_evolution

IMAGE_SIZE = [640.0, 480.0]


def createVoidImage():
    return np.zeros((int(IMAGE_SIZE[1]), int(IMAGE_SIZE[0])), np.uint8)


class Box(object):

    def __init__(self, center, size, angle):
        self.center = center
        self.size = size
        self.angle = angle
        self.points = []
        self.update()

    def distanceFromCenter(self):
        return np.linalg.norm(np.array([IMAGE_SIZE[0]*0.5, IMAGE_SIZE[1]*0.5]) - self.center)

    def distanceFrom(self, box):
        return np.linalg.norm(self.center - box.center)

    def nominalArea(self):
        return self.size[0] * self.size[1]

    def fromVector(self, z):
        x = (z[0] + 1.0) * IMAGE_SIZE[0] * 0.5
        y = (z[1] + 1.0) * IMAGE_SIZE[1] * 0.5
        self.center = np.array([x, y])
        self.angle = z[2] * np.pi
        self.update()

    def update(self):
        w = self.size[0] * 0.5
        h = self.size[1] * 0.5
        points = np.array([
            [-w, -h],
            [w, -h],
            [w, h],
            [-w, h]
        ])

        rot = np.array([[math.cos(self.angle), -math.sin(self.angle)],
                        [math.sin(self.angle), math.cos(self.angle)]])

        points_rot = np.dot(rot, points.T).T
        self.points = points_rot + self.center

    def draw(self, image, color=(255, 255, 255)):
        self.update()
        if len(image.shape) == 2:
            color = color[0]
        cv2.fillPoly(image, [np.int32(self.points)], color)


def fitness(x, boxes):
    # print(x)

    # print("D: ", center_distances, cross_distances)

    outsides = 0.0
    images = []
    for i, b in enumerate(boxes):
        z = np.array([x[i * 3], x[i * 3 + 1], x[i * 3 + 2]])
        b.fromVector(z)
        image = np.zeros((480, 640), np.uint8)
        b.draw(image)
        area = np.count_nonzero(image.ravel())
        outside = b.nominalArea() - area
        outsides += outside
        b.draw(image)
        images.append(image)

    center_distances = 0.0
    cross_distances = 0.0
    for i in boxes:
        center_distances += i.distanceFromCenter()
        for j in boxes:
            if i != j:
                cross_distances -= i.distanceFrom(j)

    overlaps = 0.0
    for a in images:
        for b in images:
            if a is not b:
                o = cv2.bitwise_and(a, b)
                overlaps += np.count_nonzero(o.ravel())

    output = createVoidImage()
    for b in boxes:
        b.draw(output)
    # print("F: ", overlaps, outsides)
    # cv2.imshow("output", output)
    # cv2.waitKey(1)
    # print("F: ", 0.1 * overlaps, 0.1 * outsides, center_distances, cross_distances)
    # print(cross_distances)
    return 10 * center_distances + overlaps + outsides


boxes = []
bounds = []
for i in range(10):
    box = Box(np.array([0, 0]), np.random.uniform(20, 250, (2,)), np.random.uniform(-np.pi, np.pi))
    boxes.append(box)
    bounds.append((-1.0, 1.0))
    bounds.append((-1.0, 1.0))
    bounds.append((-1.0, 1.0))


def callback(x, convergence):
    print(x, convergence)
    output = createVoidImage()
    for i, b in enumerate(boxes):
        z = np.array([x[i * 3], x[i * 3 + 1], x[i * 3 + 2]])
        b.fromVector(z)
        b.draw(output)
    cv2.imshow("output", output)
    cv2.waitKey(1)


result = differential_evolution(fitness, bounds, args=(boxes,), popsize=20, callback=callback, strategy='best2bin', disp=True)

# while True:
#     z = np.random.uniform(-1.0, 1.0, (3,))
#     box.fromVector(z)

#     box.draw(image)
#     print(z)
#     cv2.imshow("image", image)
#     cv2.waitKey(0)
