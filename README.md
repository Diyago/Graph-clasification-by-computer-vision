## Results
### Computer vision

**Test**
* ROC AUC 0.697
* MAP 0.183

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