## Make the datasets for contrastive pretraining
```
python data_creation/create_contrastive_patches.py
```

## Train the Supervised Contrastive Model
Change directory paths as required and then:
```
python train_inc_dec_size_weight_balance.py
```

## Make the PCA models
Look at make_pca_model.ipynb