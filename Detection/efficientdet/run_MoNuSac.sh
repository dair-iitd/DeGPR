python train.py /home/aayush/Aayush/Projects/Celiac_Disease/Detection/Baselines/Dataset/Monusac/train_val/ --model efficientdet_d0 --dataset voc2007 -b 8 --lr 0.008 --sync-bn --opt momentum --warmup-epochs 3 --model-ema --model-ema-decay 0.9966 --epochs 1000 --num-classes 4 --pretrained --postreg --gmm_features 3 --embedding_weights /home/aayush/Aayush/Projects/Celiac_Disease/Detection/Code/Postreg/Contrastive_learning/save/SupCon/MonuSac_Balance_Inc_dec_models_0/SupCon_MonuSac_resnet18_lr_0.0001_decay_0.0001_bsz_8_temp_0.7_trial_0_kfold_0/ckpt_resnet18_epoch_250_temp_0.7.pth  --embedding_pca_weights /home/aayush/Aayush/Projects/Celiac_Disease/Detection/Code/Postreg/Contrastive_learning/save/SupCon/MonuSac_Balance_Inc_dec_models_0/SupCon_MonuSac_resnet18_lr_0.0001_decay_0.0001_bsz_8_temp_0.7_trial_0_kfold_0/pca_MoNuSac_temp07_Inc_Dec_Balance_ep250_90.pkl --resume /home/aayush/chirag/efficientdet/efficientdet-pytorch/output/train/20220829-204038-efficientdet_d0/model_best.pth.tar