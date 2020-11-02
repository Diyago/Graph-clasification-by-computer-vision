import codecs
import os

import pandas as pd
from poyo import parse_string


def prepate_train_csv(path, output_file='all_train.csv'):
    dfs = []
    for folder in os.listdir(path):
        if "train.dir" in folder:
            print(folder, len(os.listdir(path + folder)))
            for file in os.listdir(path + folder):
                if file.endswith(".csv"):
                    path_csv = os.path.join(path, folder, file)
                    df = pd.read_csv(path_csv)
                    df["image_filename"] = (
                            os.path.join(path, folder) + "/" + df["image_filename"]
                    )
                    dfs.append(df)
    dfs = pd.concat(dfs)
    dfs.to_csv(output_file, index=False)
    print('Unique locations', dfs['realative_coordinates'].nunique())
    print('Unique locations with label = correct', dfs[dfs['label'] == 'correct']['realative_coordinates'].nunique())




if __name__ == '__main__':
    with codecs.open("config/config_classification.yml", encoding="utf-8") as ymlfile:
        config_yaml = ymlfile.read()
        config = parse_string(config_yaml)
    prepate_train_csv('./data/smoke_train/')
    prepate_train_csv('./data/smoke_val/', output_file=config['test_inference']['train_csv'])


    # with codecs.open("config/config_classification_fire.yml", encoding="utf-8") as ymlfile:
    #     config_yaml = ymlfile.read()
    #     config = parse_string(config_yaml)
    # prepate_train_csv('./data/fire_train/')
    # prepate_train_csv('./data/fire_val/', output_file=config['test_inference']['train_csv'])
