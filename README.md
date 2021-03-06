## Results
### Computer vision

**Test**
* ROC AUC 0.697
* MAP 0.183

![cv_test.png](./readme_images/cv_test.png)

### Graph method

**Test**
* ROC AUC 0.702
* MAP 0.199


## Training models
### Computer vision

 0. Generate graph images ```python generate_images.py``` 
 1. Prepare data by ```python prepare_data.py```
 
 2. Adjust config in `config/config_classification.yml`
 
 3. train models run ``python train.py``
 
 4. Watch tensorboad logs `tensorboard --logdir ./lightning_logs/`
 
 5. Collect up-to-date requirements.txt call `pipreqs --force`


### Graph method

1. Run ```python fit_predict_graph.py```


## Data

We will predict the activity (against COVID?) of different molecules. 

Dataset sample:
```
smiles,activity
OC=1C=CC=CC1CNC2=NC=3C=CC=CC3N2,1
CC(=O)NCCC1=CNC=2C=CC(F)=CC12,1
O=C([C@@H]1[C@H](C2=CSC=C2)CCC1)N,1
```

![sample_graph.png](./readme_images/sample_graph.png)