import torch
import codecs
import os
from common_blocks.utils import convert_model
from models.seresnext import CustomSEResNeXt
from poyo import parse_string

with codecs.open("config/config_classification.yml", encoding="utf-8") as ymlfile:
    config_yaml = ymlfile.read()
    config = parse_string(config_yaml)


if __name__ == '__main__':
    initial_folder = './lightning_logs/se_resnext50_32x4d/trained 3pochs'

    for model in os.listdir(initial_folder):
        # todo read model from config
        convert_model(CustomSEResNeXt(config['model_params']),
                      os.path.join(initial_folder, model),
                      'trained_models'
                      )
