**Train**
ROC AUC 0.989
Precision 0.964
Recall 0.5
F1_score 0.659
MAP 0.901

**Test**
ROC AUC 0.697
MAP 0.183


**3. To train model**

 a. Prepare data by ```prepare_data.py```
 
 b. Adjust config in `config/config_classification.yml`
 
 c. train models run ``train.py``
 
 d. Watch tensorboad logs `tensorboard --logdir ./lightning_logs/`
 
 e. All inference and deploy param specified in `deploy_model` folder
 
 d. Collect up-to-date requirements.txt call `pipreqs --force`
 