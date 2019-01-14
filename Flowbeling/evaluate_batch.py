
import glob
import os
import subprocess
import numpy as np

command_template = "python evaluate_results.py --dataset_path {} --test_file {} --result_file {} --confidence_th {} --scene_tags {} --output_name {} --output_folder {}"

tags_conversion_map = {
    'scan_05': 'simple',
    'scan_15': 'hard',
    '': 'overall'
}

dataset_path = '/Users/daniele/Desktop/to_delete/Compass2018_FLOWBEL'
output_folder = '/Users/daniele/Desktop/to_delete/temp/results'
test_file = '/Users/daniele/Desktop/to_delete/test_images.txt'
results_folder = '/Users/daniele/Desktop/to_delete/ICRA2019_results/1sett/'
tags = ['', 'scan_05', 'scan_15']
target_algos = ['loop_20deg', 'loop_30deg', 'loop_45deg']


input_files = sorted(glob.glob(os.path.join(results_folder, "*")))
input_files = sorted([f for f in input_files if 'UNORIENTED' not in f])

algo_map = {}
for f in input_files:
    basename = os.path.splitext(os.path.basename(f))[0]
    basename = basename.replace('results_', '')
    algo_map[basename] = f


th_step = 0.1
ths = [1.0] + [1.0 - 0.05 * i for i in range(1, 21)]
ths = list(map(lambda x: float('{:.2f}'.format(x)), ths))
#ths = [1.0]

for tag in tags:
    tag_name = tags_conversion_map[tag]
    for algo, f in algo_map.items():

        if not any(x in algo for x in target_algos):
            continue

        for th in ths:
            print(algo, th)

            command = command_template.format(
                dataset_path,
                test_file,
                f,
                th,
                tag,
                algo + '#' + tag_name,
                output_folder
            )
            print(command)
            subprocess.call(command.split(' '))
