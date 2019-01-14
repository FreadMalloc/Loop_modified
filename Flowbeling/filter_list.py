import glob
import os
import numpy as np


folder = '/home/daniele/Desktop/temp/LoopMedia/scene3/usb_cam_1_image_raw_compressed'
files = sorted(glob.glob(os.path.join(folder, "*")))
print(files)


out = open('/tmp/files.txt', 'w')
step = 10
for i, f in enumerate(files):
    if i % step == 0:
        print(f)
        out.write(f)
        out.write("\n")
out.close()
