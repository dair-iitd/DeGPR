## Train
First modify train.yaml and then:
```
python train.py --img 640 --batch 8 --epochs 300 --data train.yaml --weights [OPTIONAL][pre-trained weights file] --workers 24 --name yolo_muced --mAPl 0.3 --mAPr 0.75 --postreg --embedding_weights [OPTIONAL] --embedding_pca_weights [OPTIONAL]
```
Note : To just calculate baseline numbers, remove --postreg

## Validate
To calculate the Precision, Recall and mAP numbers:
```
python val.py --img 640 --data test.yaml --weights [best trained model weights]
```

## Detect
Please take a look at detect.py for more idea about the flags
```
python detect.py --img 640 --source [img directory] --weights [best model weights]
```
