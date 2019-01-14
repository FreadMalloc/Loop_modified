import numpy as np
import cv2
from matplotlib import pyplot as plt
from flowbel import Instance, InstanceSet, Score
import flowbel
import argparse
import os
import glob
import random
import sys

ap = argparse.ArgumentParser("Detector Evaluator")
ap.add_argument("--dataset_path", required=True, help="Dataset folder")
ap.add_argument("--result_file", required=True, help="Result file")
ap.add_argument("--test_file", required=True, help="Test file")
ap.add_argument("--output_folder", required=True, help="Output folder")
ap.add_argument("--output_name", required=True, help="Output name")
ap.add_argument("--confidence_th", default=0.5, help="Confidence Threshold", type=float)
ap.add_argument("--scene_tags", help="Scene tags", nargs='+', default=[])
ap.add_argument("--debug", default=False, help="Debug mode", action="store_true")
args = vars(ap.parse_args())


dataset = flowbel.Dataset(dataset_path=args['dataset_path'])


test_entries = flowbel.Dataset.loadDataFromFile(args['test_file'])
result_entries = flowbel.Dataset.loadDataFromFile(args['result_file'], args['confidence_th'])

print("ENTRIES", len(test_entries))


def createScore():
    return {
        'TP': 0,
        'FP': 0,
        'P': 0,
        'N': 0,
        'IOU': 0.0,
        'OIOU': 0.0,
        'ALIGN': 0.0
    }


GLOBAL_SCORES = Score()
PERLABEL_SCORES = {}

for l, _ in dataset.getDatasetManifest().getClassMap().items():
    PERLABEL_SCORES[l] = Score()

scene_tags = args['scene_tags']
counter = 0
max_counter = len(test_entries.keys())
for k, entry in test_entries.items():

    flowbel.printProgressBar(counter, max_counter)
    counter += 1

    if len(scene_tags) > 0:
        if not any(tag in k for tag in scene_tags):
            continue

    if k in result_entries:
        result_set = InstanceSet(result_entries[k]['instances'])
        test_set = InstanceSet(entry['instances'])

        if args['debug']:
            image = cv2.imread(k)

        for test_instance in test_set.instances:
            label = test_instance.label

            GLOBAL_SCORES.N += 1
            PERLABEL_SCORES[label].N += 1

            if args['debug']:
                output = image.copy()
                test_instance.draw(output)

            result_instance = test_instance.findMostSimilarInstance(result_set.instances, th=0.5)

            for ri in result_set.instances:
                if ri.label == test_instance.label:

                    GLOBAL_SCORES.P += 1
                    PERLABEL_SCORES[label].P += 1

                    if ri == result_instance:
                        GLOBAL_SCORES.TP += 1
                        PERLABEL_SCORES[label].TP += 1

                        iou = test_instance.computeIOU(result_instance)
                        oiou = test_instance.computeOIOU(result_instance)
                        align = test_instance.computeALIGN(result_instance)

                        GLOBAL_SCORES.IOU += iou
                        PERLABEL_SCORES[label].IOU += iou
                        GLOBAL_SCORES.OIOU += oiou
                        PERLABEL_SCORES[label].OIOU += oiou
                        GLOBAL_SCORES.ALIGN += align
                        PERLABEL_SCORES[label].ALIGN += align

                    else:
                        GLOBAL_SCORES.FP += 1
                        PERLABEL_SCORES[label].FP += 1

                    if args['debug']:
                        print("Hypo", ri)
                        ri.draw(output, (0, 0, 255))

            if args['debug']:
                cv2.imshow("image", output)
                cv2.waitKey(0)
    if args['debug']:
        print(GLOBAL_SCORES)

print(scene_tags)
print(GLOBAL_SCORES)
PERLABEL_SCORES['global'] = GLOBAL_SCORES
for k, score in PERLABEL_SCORES.items():
    name = dataset.getDatasetManifest().getName(k)
    if name is None:
        name = k
    print('{:15}\t{}'.format(name, score))


algo_name = os.path.splitext(os.path.basename(args['result_file']))[0]
algo_th = args['confidence_th']
result = {
    'algo_name': algo_name,
    'th': algo_th,
    'data': PERLABEL_SCORES,
    'tags': scene_tags
}

import pprint
pprint.pprint(result)


output_filename = os.path.join(args['output_folder'], args['output_name'] + "#" + str(algo_th))
np.save(output_filename, result)


# for inst in entry['instances']:
#     print(inst, type(inst))
#     inst.draw(image)

# cv2.imshow("image", image)
# cv2.waitKey(0)
