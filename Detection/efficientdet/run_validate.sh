# test gmm

# python validate.py /home/aayush/Aayush/Projects/Celiac_Disease/Detection/Baselines/Dataset/voc_kfold/0/test/ --num-gpu 1 --model efficientdet_d0 --dataset voc2007 --num-classes 2 --checkpoint /home/aayush/chirag/efficientdet/efficientdet-pytorch/output/HPC/train/20220830-022821-efficientdet_d0_inc_Dec_balance_celiac_disease/model_best.pth.tar 
# python validate.py /home/aayush/Aayush/Projects/Celiac_Disease/Detection/Baselines/Dataset/voc_kfold/1/test/ --num-gpu 1 --model efficientdet_d0 --dataset voc2007 --num-classes 2 --checkpoint /home/aayush/chirag/efficientdet/efficientdet-pytorch/output/HPC/train/20220831-053422-efficientdet_d1_inc_Dec_balance_celiac_disease/model_best.pth.tar 
# python validate.py /home/aayush/Aayush/Projects/Celiac_Disease/Detection/Baselines/Dataset/voc_kfold/2/test/ --num-gpu 1 --model efficientdet_d0 --dataset voc2007 --num-classes 2 --checkpoint /home/aayush/chirag/efficientdet/efficientdet-pytorch/output/train/efficientDet_d0_lr001_ep1000_inc_dec_balance_celiac_kfold2_gmm/model_best.pth.tar
# python validate.py /home/aayush/Aayush/Projects/Celiac_Disease/Detection/Baselines/Dataset/voc_kfold/3/test/ --num-gpu 1 --model efficientdet_d0 --dataset voc2007 --num-classes 2 --checkpoint /home/aayush/chirag/efficientdet/efficientdet-pytorch/output/train/efficientDet_d0_lr001_ep1000_inc_dec_balance_celiac_kfold3_gmm/model_best.pth.tar 
# python validate.py /home/aayush/Aayush/Projects/Celiac_Disease/Detection/Baselines/Dataset/voc_kfold/4/test/ --num-gpu 1 --model efficientdet_d0 --dataset voc2007 --num-classes 2 --checkpoint /home/aayush/chirag/efficientdet/efficientdet-pytorch/output/HPC/train/20220830-015805-efficientdet_d4_in_Dec_balance_celiac_disease/model_best.pth.tar 

# test baseline

# python validate.py /home/aayush/Aayush/Projects/Celiac_Disease/Detection/Baselines/Dataset/voc_kfold/0/test/ --num-gpu 1 --model efficientdet_d0 --dataset voc2007 --num-classes 2 --checkpoint /home/aayush/chirag/efficientdet/efficientdet-pytorch/output/train/EfficientDet_Baseline_weights/output/train/20220605-153543-efficientdet_d0_kfold0_lr001_ep1000/model_best.pth.tar 
# python validate.py /home/aayush/Aayush/Projects/Celiac_Disease/Detection/Baselines/Dataset/voc_kfold/1/test/ --num-gpu 1 --model efficientdet_d0 --dataset voc2007 --num-classes 2 --checkpoint /home/aayush/chirag/efficientdet/efficientdet-pytorch/output/train/EfficientDet_Baseline_weights/output/train/20220606-012051-efficientdet_d0_kfold1_lr001_ep5000/model_best.pth.tar 
# python validate.py /home/aayush/Aayush/Projects/Celiac_Disease/Detection/Baselines/Dataset/voc_kfold/2/test/ --num-gpu 1 --model efficientdet_d0 --dataset voc2007 --num-classes 2 --checkpoint /home/aayush/chirag/efficientdet/efficientdet-pytorch/output/train/EfficientDet_Baseline_weights/output/train/20220608-002711-efficientdet_d0_kfold2_lr001_ep2500/model_best.pth.tar 
# python validate.py /home/aayush/Aayush/Projects/Celiac_Disease/Detection/Baselines/Dataset/voc_kfold/3/test/ --num-gpu 1 --model efficientdet_d0 --dataset voc2007 --num-classes 2 --checkpoint /home/aayush/chirag/efficientdet/efficientdet-pytorch/output/train/EfficientDet_Baseline_weights/output/train/20220607-004135-efficientdet_d0_kfold3_lr001_ep5000/model_best.pth.tar 
# python validate.py /home/aayush/Aayush/Projects/Celiac_Disease/Detection/Baselines/Dataset/voc_kfold/4/test/ --num-gpu 1 --model efficientdet_d0 --dataset voc2007 --num-classes 2 --checkpoint /home/aayush/chirag/efficientdet/efficientdet-pytorch/output/train/EfficientDet_Baseline_weights/output/train/20220607-150827-efficientdet_d0_kfold4_lr001_ep5000/model_best.pth.tar 

# test consep

# python validate_consep.py /home/aayush/Aayush/Projects/Celiac_Disease/Detection/Baselines/Dataset/ConSep/test/ --num-gpu 1 --model efficientdet_d0 --dataset voc2007 --num-classes 3 --checkpoint /home/aayush/chirag/efficientdet/efficientdet-pytorch/output/HPC/train/20220729-013829-efficientdet_d0_ConSep/model_best.pth.tar 
# python validate_consep.py /home/aayush/Aayush/Projects/Celiac_Disease/Detection/Baselines/Dataset/ConSep/test/ --num-gpu 1 --model efficientdet_d0 --dataset voc2007 --num-classes 3 --checkpoint /home/aayush/chirag/efficientdet/efficientdet-pytorch/output/HPC/train/20220827-041421-efficientdet_d0_Consep_inc_dec_ep250/model_best.pth.tar 


# Monusac

# python validate_monusac.py /home/aayush/Aayush/Projects/Celiac_Disease/Detection/Baselines/Dataset/Monusac/test/ --num-gpu 1 --model efficientdet_d0 --dataset voc2007 --num-classes 4 --checkpoint /home/aayush/chirag/efficientdet/efficientdet-pytorch/output/train/EfficientDet_Baseline_weights/output/train/20220613-171501-efficientdet_d0_monusac_ep1000/model_best.pth.tar 
# python validate_monusac.py /home/aayush/Aayush/Projects/Celiac_Disease/Detection/Baselines/Dataset/Monusac/test/ --num-gpu 1 --model efficientdet_d0 --dataset voc2007 --num-classes 4 --checkpoint /home/aayush/chirag/efficientdet/efficientdet-pytorch/output/HPC/train/20220926-224453-efficientdet_d0-efficientdet_d0_MoNuSac_0d00001_inc_dec_ep2500/model_best.pth.tar 
