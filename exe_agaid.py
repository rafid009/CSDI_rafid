from main_model import CSDI_Agaid
from dataset_agaid import get_dataloader
from utils import train, evaluate
import numpy as np
import torch
import sys
import os
np.set_printoptions(threshold=sys.maxsize)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 10
config_dict = {
    'train': {
        'epochs': 800,
        'batch_size': 16 ,
        'lr': 1.0e-3
    },      
    'diffusion': {
        'layers': 4, 
        'channels': 64,
        'nheads': 8,
        'diffusion_embedding_dim': 128,
        'beta_start': 0.0001,
        'beta_end': 0.5,
        'num_steps': 50,
        'schedule': "quad"
    },
    'model': {
        'is_unconditional': 0,
        'timeemb': 128,
        'featureemb': 16,
        'target_strategy': "random"
    }
}

file_name = 'ColdHardiness_Grape_Merlot_2.csv'

train_loader, valid_loader = get_dataloader(
    seed=seed,
    filename=file_name,
    batch_size=config_dict["train"]["batch_size"],
    missing_ratio=0.2,
)

model = CSDI_Agaid(config_dict, device).to(device)
model_folder = "./saved_model"
if not os.path.isdir(model_folder):
    os.makedirs(model_folder)

train(
    model,
    config_dict["train"],
    train_loader,
    valid_loader=valid_loader,
    foldername=model_folder,
)
nsample = 50
model.load_state_dict(torch.load(f"{model_folder}/model.pth"))
evaluate(model, valid_loader, nsample=nsample, scaler=1, foldername=model_folder)