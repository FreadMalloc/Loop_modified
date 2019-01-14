import cv2
import glob
import os
import sys
import numpy as np

videofile = sys.argv[1]
outputfolder = sys.argv[2]
jumps = int(sys.argv[3])

try:
    os.mkdir(outputfolder)
except:
    pass


cap = cv2.VideoCapture(videofile)

counter = 0
current_jumps = 0
ret = True
while(ret):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if current_jumps > 0:
        current_jumps -= 1
        continue
    try:
        outputfile = "frame_{}.jpg".format(str(counter).zfill(5))
        outputfile = os.path.join(outputfolder, outputfile)

        cv2.imwrite(outputfile, frame)
        counter += 1
        print(outputfile, counter, current_jumps)
        current_jumps = jumps
    except:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
