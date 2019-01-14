import os
import glob
import numpy as np
import pprint
import flowbel
import argparse
import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def smooth(x, window_len=5, window='hanning'):
    if x.ndim != 1:
        print(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        print(ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        print(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y


classes_ordered = [
    'artifact_black',
    'artifact_metal',
    'artifact_orange',
    'artifact_white',
    'clip',
    'screwdriver',
    'battery_black',
    'battery_green',
    'box_brown',
    'box_yellow',
    'glue',
    'pendrive',
]


def precisionRecallData(results_files, target_name, target_class):
    files = sorted([f for f in results_files if target_name in f])
    results = []
    for f in files:
        results.append(np.load(f).item())

    data = []
    for result in results:
        data.append((result['th'], result['data'][target_class].precision(), result['data'][target_class].recall()))
    return np.array(data)


ap = argparse.ArgumentParser("Plot Curve Result")
ap.add_argument("--dataset_path", default='/Users/daniele/Desktop/to_delete/Compass2018_FLOWBEL', help="Dataset folder")
ap.add_argument("--results_folder", default='/Users/daniele/Desktop/to_delete/temp/results', help="Results folder")
args = vars(ap.parse_args())

dataset = flowbel.Dataset(dataset_path=args['dataset_path'])
dataset_manifest = dataset.getDatasetManifest()

labels = dataset_manifest.getLabels()


target_name = 'x_30deg#overall'
target_class = 'global'
tag = 'overall'

results_files = glob.glob(os.path.join(args['results_folder'], "*.npy"))

datamap = {}

for l in labels:
    if l >= 0:
        datamap['$Loop_{10}$ ' + '{}'.format(dataset_manifest.getName(l))] = precisionRecallData(results_files, 'loop_10deg#{}'.format(tag), l)


figure(num=None, figsize=(6, 3), dpi=100, facecolor='w', edgecolor='k')


for k, data in datamap.items():
    print(k, data)
    plt.plot(smooth(data[:, 1]), smooth(data[:, 2]))
plt.legend(datamap.keys(), ncol=2)


plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.1, right=0.98)
plt.grid()
plt.xlabel("Precision")
plt.ylabel("Recall")


output_file = os.path.join('/Users/daniele/Desktop/PhD/Loop/19-ICRA/images/performances', "single_" + tag + ".pdf")
plt.savefig(output_file)

plt.show()
