import os
import glob
import numpy as np
import pprint
import flowbel
import argparse
import pandas as pd
import sys


def buildResultsMap(results_folder):
    results_files = glob.glob(os.path.join(results_folder, "*.npy"))
    result_map = {}
    for f in results_files:
        result = np.load(f).item()
        basename = os.path.splitext(os.path.basename(f))[0]
        algoname = basename.split('#')[0]
        tag = basename.split('#')[1]
        th = '{:.2f}'.format(float(basename.split('#')[2]))

        if algoname not in result_map:
            result_map[algoname] = {}

        if tag not in result_map[algoname]:
            result_map[algoname][tag] = {}

        result_map[algoname][tag][th] = result
    return result_map


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
    'global'
]

ap = argparse.ArgumentParser("Test Result")
ap.add_argument("--dataset_path", default='/Users/daniele/Desktop/to_delete/Compass2018_FLOWBEL', help="Dataset folder")
ap.add_argument("--results_folder", default='/Users/daniele/Desktop/to_delete/temp/results', help="Results folder")
args = vars(ap.parse_args())

dataset = flowbel.Dataset(dataset_path=args['dataset_path'])
dataset_manifest = dataset.getDatasetManifest()

results_files = glob.glob(os.path.join(args['results_folder'], "*.npy"))

result_map = buildResultsMap(args['results_folder'])


current_tag = 'overall'

ALGO_NAME_MAP = {
    'siftmulti': 'SIFT',
    'bold': 'BOLD',
    'loop_10deg': 'LOOP',
    'loopSYNTH_10deg': 'LOOP_synth'
}
ALGO_TH_MAP = {
    'siftmulti': '1.00',
    'bold': '1.00',
    'loop_10deg': '0.50',
    'loopSYNTH_10deg': '0.35'
}
ALGOS = ['siftmulti', 'bold', 'loop_10deg', 'loopSYNTH_10deg']
VALUES = ['Precision', 'Recall', 'FScore', 'IOU', 'OIOU']

# CReates nested touples of multi columns header
l = []
for v in VALUES:
    for a in ALGOS:
        l.append((v, ALGO_NAME_MAP[a]))

# generate numeric data
mat = []
for field in VALUES:
    for algo in ALGOS:
        th = ALGO_TH_MAP[algo]
        result = result_map[algo][current_tag][th]
        data = result['data']

        for classname in classes_ordered:
            label = dataset_manifest.getLabel(classname)
            if label < 0:
                label = classname
            item = data[label]
            mat.append(item.getValueByName(field))

mat = np.array(mat).reshape((len(classes_ordered), -1), order='F')
print(mat, mat.shape)

# generate PANDAS data
rows = classes_ordered
df = pd.DataFrame(mat, columns=l, index=rows)
df.columns = pd.MultiIndex.from_tuples(df.columns, names=['Method', ''])
df = df.round(2)
latex_text = df.to_latex()

print(latex_text)
print(df)
