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
classes_ordered_reparsed = [
    'artifact_black',
    'artifact_metal',
    'artifact_orange',
    'artifact_white',
    'clip',
    'screwdriver',
    'SUM#Untextured#6',
    'battery_black',
    'battery_green',
    'box_brown',
    'box_yellow',
    'glue',
    'pendrive',
    'SUM#Textured#6',
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


current_tag = 'hard'

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
            if str(classname).startswith('SUM'):
                continue
            label = dataset_manifest.getLabel(classname)
            if label < 0:
                label = classname
            item = data[label]
            mat.append(item.getValueByName(field))

mat = np.array(mat).reshape((len(classes_ordered), -1), order='F')
print(mat, mat.shape)


# Reparse rows with SUM Recaps if any
rows = classes_ordered
reparsed_rows = []
row_pointer = 0
for i, r in enumerate(classes_ordered_reparsed):
    if r.startswith('SUM'):
        key = r.split('#')[1]
        size = int(r.split('#')[2])
        sum_row = np.sum(mat[i - size:i, :], axis=0)
        sum_row = sum_row / float(size)
        reparsed_rows.append(sum_row)
    else:
        reparsed_rows.append(mat[row_pointer, :])
        row_pointer += 1
reparsed_rows = np.array(reparsed_rows)
print("Reparsed", reparsed_rows.shape, "\n", reparsed_rows)


mat = reparsed_rows
chunk_size = 4
for i, r in enumerate(classes_ordered_reparsed):
    latex_row = ''
    separation_line = False
    if 'SUM#' in r:
        r = r.split('#')[1]
        separation_line = True

    if separation_line:
        latex_row += ' \hline '
    latex_row += r.replace('_', '\_')
    for c in range(0, mat[i, :].shape[0], chunk_size):
        chunk = mat[i, c:c + chunk_size]
        max_index = np.argmax(chunk)
        for e, v in enumerate(chunk):
            val = '{:.2f}'.format(v)
            latex_row += ' &'
            if max_index == e:
                latex_row += ' \\textbf{' + val + '}'
            else:
                latex_row += ' ' + val
    latex_row += ' \\\\'
    if separation_line:
        latex_row += ' \hline '
    print(latex_row)


# generate PANDAS data
df = pd.DataFrame(reparsed_rows, columns=l, index=classes_ordered_reparsed)
df.columns = pd.MultiIndex.from_tuples(df.columns, names=['Method', ''])
df = df.round(2)

latex_text = df.to_latex()

print(latex_text)
print(df)
