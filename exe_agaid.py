from main_model import CSDI_Agaid
from dataset_agaid import get_dataloader
from utils import *
import numpy as np
import torch
import sys
import os
from pypots.imputation import SAITS
from process_data import *
import pickle

np.set_printoptions(threshold=sys.maxsize)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 10
config_dict = {
    'train': {
        'epochs': 1500,
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
        'num_steps': 100,
        'schedule': "quad"
    },
    'model': {
        'is_unconditional': 0,
        'timeemb': 128,
        'featureemb': 16,
        'target_strategy': "random",
        'type': 'CSDI',
        'n_layers': 3, 
        'd_time': 252,
        'n_feature': len(features),
        'd_model': 256,
        'd_inner': 128,
        'n_head': 4,
        'd_k': 64,
        'd_v': 64,
        'dropout': 0.1,
        'diagonal_attention_mask': True
    }
}

file_name = 'ColdHardiness_Grape_Merlot_2.csv'

train_loader, valid_loader = get_dataloader(
    seed=seed,
    filename=file_name,
    batch_size=config_dict["train"]["batch_size"],
    missing_ratio=0.2,
)

model_csdi = CSDI_Agaid(config_dict, device).to(device)
model_folder = "./saved_model"
if not os.path.isdir(model_folder):
    os.makedirs(model_folder)
filename = 'model_csdi.pth'
# train(
#     model_csdi,
#     config_dict["train"],
#     train_loader,
#     valid_loader=valid_loader,
#     foldername=model_folder,
#     filename=filename
# )
nsample = 50
model_csdi.load_state_dict(torch.load(f"{model_folder}/{filename}"))
# evaluate(model_csdi, valid_loader, nsample=nsample, scaler=1, foldername=model_folder)

config_dict = {
    'train': {
        'epochs': 1500,
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
        'num_steps': 100,
        'schedule': "quad"
    },
    'model': {
        'is_unconditional': 0,
        'timeemb': 128,
        'featureemb': 16,
        'target_strategy': "random",
        'type': 'SAITS',
        'n_layers': 3, 
        'd_time': 252,
        'n_feature': len(features),
        'd_model': 256,
        'd_inner': 128,
        'n_head': 4,
        'd_k': 64,
        'd_v': 64,
        'dropout': 0.1,
        'diagonal_attention_mask': True
    }
}

# model_diff_saits_simple = CSDI_Agaid(config_dict, device, is_simple=True).to(device)
model_diff_saits = CSDI_Agaid(config_dict, device, is_simple=False).to(device)
# filename_simple = 'model_diff_saits_simple.pth'
filename = 'model_diff_saits_explode.pth'
# train(
#     model_diff_saits_simple,
#     config_dict["train"],
#     train_loader,
#     valid_loader=valid_loader,
#     foldername=model_folder,
#     filename=filename_simple
# )

train(
    model_diff_saits,
    config_dict["train"],
    train_loader,
    valid_loader=valid_loader,
    foldername=model_folder,
    filename=filename
)
nsample = 100
model_diff_saits.load_state_dict(torch.load(f"{model_folder}/{filename}"))
# model_diff_saits_simple.load_state_dict(torch.load(f"{model_folder}/{filename_simple}"))
# evaluate(model_diff_saits, valid_loader, nsample=nsample, scaler=1, foldername=model_folder)


# filename = "ColdHardiness_Grape_Merlot_2.csv"
# df = pd.read_csv(filename)
# modified_df, dormant_seasons = preprocess_missing_values(df, features, is_dormant=True)
# season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, features, is_dormant=True)
# train_season_df = season_df.drop(season_array[-1], axis=0)
# train_season_df = train_season_df.drop(season_array[-2], axis=0)
# mean, std = get_mean_std(train_season_df, features)
# X, Y = split_XY(season_df, max_length, season_array, features)

# # observed_mask = ~np.isnan(X)

# X = X[:-2]
# X = (X - mean) / std
saits_model_file = f"{model_folder}/model_saits.pth"
# saits = SAITS(n_steps=252, n_features=len(features), n_layers=3, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=3000, patience=200, device=device)

# saits.fit(X)  # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.
# pickle.dump(saits, open(saits_model_file, 'wb'))

saits = pickle.load(open(saits_model_file, 'rb'))

models = {
    'CSDI': model_csdi,
    'SAITS': saits,
    'DiffSAITS': model_diff_saits#,
    # 'DiffSAITSsimple': model_diff_saits_simple
}
mse_folder = "results_samples_simple"

lengths = [100]
print("For All")
for l in lengths:
    print(f"For length: {l}")
    evaluate_imputation(models, mse_folder, length=l, trials=1)
    # evaluate_imputation(models, mse_folder, length=l, trials=20)
    # evaluate_imputation_data(models, length=l)

# feature_combinations = {
#     "temp": ["MEAN_AT", "MIN_AT", "AVG_AT", "MAX_AT"],
#     "hum": ["AVG_REL_HUMIDITY", "MIN_REL_HUMIDITY", "MAX_REL_HUMIDITY"],
#     "dew": ["AVG_DEWPT", "MIN_DEWPT", "MAX_DEWPT"],
#     "pinch": ["P_INCHES"],
#     "wind": ["WS_MPH", "MAX_WS_MPH"],
#     "sr": ["SR_WM2"],
#     "leaf": ["LW_UNITY"],
#     "et": ["ETO", "ETR"],
#     "st": ["ST8", "MIN_ST8", "MAX_ST8"]
# }
# print(f"The exclusions")
# for key in feature_combinations.keys():
#     for l in lengths:
#         print(f"For length: {l}")
#         evaluate_imputation(models, mse_folder, exclude_key=key, exclude_features=feature_combinations[key], length=l, trials=1)
#         evaluate_imputation(models, mse_folder, exclude_key=key, exclude_features=feature_combinations[key], length=l, trials=20)
        # evaluate_imputation_data(models, exclude_key=key, exclude_features=feature_combinations[key], length=l)
# forward_evaluation(models, filename, features)