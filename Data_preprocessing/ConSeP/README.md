## Dataset download
Download the ConSeP dataset from [CoNSeP](https://www.sciencedirect.com/science/article/pii/S1361841519301045)

## Parse the .mat files
After changing the root directory as required:
```
python form_labels.py
```
This forms a folder Annotations in the train and test directories.

## Convert to yolo format
After changing the root directory and save directory as required:
```
python make_yolo_consep.py
```
