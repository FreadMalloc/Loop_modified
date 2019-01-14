import numpy as np
import cv2
from interactivewindow import InteractiveWindow
import sys
import glob
import os
import argparse
import subprocess

class_map = {
    0: "artifact_black",
    1: "artifact_metal",
    2: "artifact_orange",
    3: "artifact_white",
    4: "battery_black",
    5: "battery_green",
    6: "box_brown",
    7: "box_yellow",
    8: "clip",
    9: "glue",
    10: "pendrive",
    11: "screwdriver"
}

ap = argparse.ArgumentParser("Tracker Labeler Batch")
ap.add_argument("--folder", required=True, help="Imges folder")
args = vars(ap.parse_args())

command_template = "python tracker_labeler.py --folder {} --object_index {} --blind --class_name {}"

for i, class_name in class_map.items():
    command = command_template.format(
        args['folder'],
        0,
        class_name
    )
    print(i, command)
    subprocess.call(command.split())
