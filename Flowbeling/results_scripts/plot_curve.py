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

plot_type = 'pr'

target_name = 'x_30deg#overall'
target_class = 'global'
tag = 'hard'

results_files = glob.glob(os.path.join(args['results_folder'], "*.npy"))

datamap = {
    '$Loop_{5}$': precisionRecallData(results_files, 'loop_5deg#{}'.format(tag), target_class),
    '$Loop_{10}$': precisionRecallData(results_files, 'loop_10deg#{}'.format(tag), target_class),
    '$Loop_{10}^S$': precisionRecallData(results_files, 'loopSYNTH_10deg#{}'.format(tag), target_class),
    '$Loop_{20}$': precisionRecallData(results_files, 'loop_20deg#{}'.format(tag), target_class),
    '$Loop_{30}$': precisionRecallData(results_files, 'loop_30deg#{}'.format(tag), target_class),
    '$Loop_{45}$': precisionRecallData(results_files, 'loop_45deg#{}'.format(tag), target_class),
}


figure(num=None, figsize=(6, 3), dpi=100, facecolor='w', edgecolor='k')

for k, data in datamap.items():
    fscore = 2 * (data[:, 1]*data[:, 2]) / (data[:, 1] + data[:, 2])
    #print(k, data)
    #print(k, fscore)
    last_recall = 0.0
    average_precision = 0.0
    for i in reversed(range(data.shape[0])):
        delta_recall = data[i, 2] - last_recall
        average_precision += data[i, 1] * delta_recall
        last_recall = data[i, 2]

    print("{} -> mAP: {}".format(k, average_precision))
    if plot_type == 'pr':
        if '^S' in k:
            plt.plot(smooth(data[:, 1]), smooth(data[:, 2]), '--', color='black', lw=2)
        else:
            plt.plot(smooth(data[:, 1]), smooth(data[:, 2]))
    elif plot_type == 'fscore':
        max_th = data[np.argmax(fscore), 0]
        print("MAX TH: ", max_th, np.max(fscore))
        plt.plot(data[:, 0], fscore)

# fig, ax = plt.subplots()
# ax.plot(t, s)
# ax.grid(True)
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.1, right=0.98)
plt.grid()
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend(datamap.keys())

output_file = os.path.join('/Users/daniele/Desktop/PhD/Loop/19-ICRA/images/performances', target_class + "_" + tag + ".pdf")
plt.savefig(output_file)
plt.show()
