import numpy as np
import cv2
from interactivewindow import InteractiveWindow
import sys
import glob
import os
import argparse

current_image = None
current_polygon = np.array([[358, 314],
                            [208, 271],
                            [242, 146],
                            [394, 190]])

orb = cv2.ORB_create(4000)
# orb = cv2.SURF_create()


ap = argparse.ArgumentParser("Tracker Viewer")
ap.add_argument("--folder", required=True, help="Imges folder")
args = vars(ap.parse_args())

# Input
folder = args['folder']
images = sorted(glob.glob(os.path.join(folder, "*.jpg")))

window = InteractiveWindow("frame")


# First Frame Grab
# window.registerCallback(test, event=InteractiveWindow.EVENT_MOUSEDOWN)

current_index = 0
while(current_index < len(images)):
    # Capture frame-by-frame
    current_image = images[current_index % len(images)]
    current_frame = cv2.imread(current_image)

    labelfile = os.path.splitext(current_image)[0] + ".npy"
    data = np.load(labelfile).item()

    output = current_frame.copy()

    for cl, objects in data.items():
        for i, poly in objects.items():
            print(current_image, labelfile, cl, poly)
            if poly.shape[0] == 4:
                p1 = tuple(poly[0, :].astype(int))
                p2 = tuple(poly[1, :].astype(int))
                p3 = tuple(poly[2, :].astype(int))
                p4 = tuple(poly[3, :].astype(int))
                cv2.line(output, p1, p2, (0, 0, 255), 3)
                cv2.line(output, p1, p4, (0, 255, 0), 3)
                cv2.line(output, p2, p3, (255, 0, 0), 3)
                cv2.line(output, p3, p4, (255, 0, 0), 3)
            else:
                output = cv2.polylines(
                    output, [np.int32(poly)], True, 255, 3, cv2.LINE_AA)
            cv2.putText(
                output,
                "{},{}".format(cl, i),
                tuple(poly[0, :].astype(int)),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2
            )

    current_index += 1

    cv2.imshow('frame', output)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cv2.destroyAllWindows()
