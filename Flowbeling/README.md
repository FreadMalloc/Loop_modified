# Flowbeling

## Start Tracker

To start the demo tracker launch:

```
python tracker_labeler.py --folder sample_sequence/ --class_name scissor --object_index 0
```

## Start Viewer

To start the viewer on the same folder Tracked with the previous tool:

```
python tracker_viewer.py --folder sample_sequence/
```


## Export Dataset to RAW Manifest

```
python convert_to_raw.py --dataset_path <FLOWBEL_DATASET> --output_file /tmp/test.txt
```

This script exports an original dataset to a Manifest File with this format:

```
<IMAGE_FILE_PATH> <BOX0> <BOX1> <BOX2> ...... <BOXN>
```

with each Box as:

```
label,x0,y0,x1,y1,x2,y2,x3,y3
```

Modify inside the "convert_to_raw.py" script the list of "scene names" to export.


## Evaluate Detector 

```
python detector_evaluator.py --dataset_path <FLOWBEL_DATASET> --images_list <RAW_DATASET_MANIFEST> --models_path <MODELS_FOLDER> --results_file <OUTPUT_FILE>
```
