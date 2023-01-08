from main_model import CSDI_Synth
from dataset_synth import get_dataloader, get_testloader
from utils import evaluate, train
import numpy as np
import torch
import sys
import os
from pypots.imputation import SAITS
import matplotlib.pyplot as plt
import matplotlib
import pickle
from synthetic_data import create_synthetic_data
import json
from json import JSONEncoder
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

np.set_printoptions(threshold=sys.maxsize)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

given_features = ['sin', 'cos2', 'harmonic', 'weight', 'inv']

def evaluate_imputation(models, mse_folder, exclude_key='', exclude_features=None, trials=30, length=100):
    given_features = given_features = ['sin', 'cos2', 'harmonic', 'weight', 'inv'] 
    nsample = 50
    # trials = 30
    season_avg_mse = {}
    num_seasons = 2
    # exclude_features = ['MEAN_AT', 'MIN_AT', 'AVG_AT', 'MAX_AT']
    results = {}
    models['CSDI'].eval()
    models['DiffSAITS'].eval()
    for season in range(num_seasons):
        print(f"For season: {season}")
        mse_csdi_total = {}
        mse_saits_total = {}
        mse_diff_saits_total = {}
        for i in range(trials):
            test_loader = get_testloader(50, len(given_features), 1, exclude_features=exclude_features, length=length, seed=50+10*i)
            for i, test_batch in enumerate(test_loader, start=1):
                output = models['CSDI'].evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points, observed_time, obs_intact, gt_intact = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)
                samples_median = samples.median(dim=1)
                gt_intact = gt_intact.squeeze(axis=0)
                saits_X = gt_intact #test_batch['obs_data_intact']
                saits_output = models['SAITS'].impute(saits_X)

                output_diff_saits = models['DiffSAITS'].evaluate(test_batch, nsample)
                samples_diff_saits, _, _, _, _, _, _ = output_diff_saits
                samples_diff_saits = samples_diff_saits.permute(0, 1, 3, 2)
                samples_diff_saits_median = samples_diff_saits.median(dim=1)

                if trials == 1:
                    results[season] = {
                        'target mask': eval_points[0, :, :].cpu().numpy(),
                        'target': c_target[0, :, :].cpu().numpy(),
                        'csdi_median': samples_median.values[0, :, :].cpu().numpy(),
                        'csdi_samples': samples[0].cpu().numpy(),
                        'saits': saits_output[0, :, :],
                        'diff_saits_median': samples_diff_saits_median.values[0, :, :].cpu().numpy(),
                        'diff_saits_samples': samples_diff_saits[0].cpu().numpy(),
                        # 'diff_saits_median_simple': samples_diff_saits_median_simple.values[0, :, :].cpu().numpy(),
                        # 'diff_saits_samples_simple': samples_diff_saits_simple[0].cpu().numpy()
                    }
                else:

                    for feature in given_features:
                        if exclude_features is not None and feature in exclude_features:
                            continue
                        # print(f"For feature: {feature}")
                        feature_idx = given_features.index(feature)
                        if eval_points[0, :, feature_idx].sum().item() == 0:
                            continue
                        mse_csdi = ((samples_median.values[0, :, feature_idx] - c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx]) ** 2
                        mse_csdi = mse_csdi.sum().item() / eval_points[0, :, feature_idx].sum().item()
                        if feature not in mse_csdi_total.keys():
                            mse_csdi_total[feature] = {"median": mse_csdi}
                        else:
                            mse_csdi_total[feature]["median"] += mse_csdi

                        for k in range(samples.shape[1]):
                            mse_csdi = ((samples[0, k, :, feature_idx] - c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx]) ** 2
                            mse_csdi = mse_csdi.sum().item() / eval_points[0, :, feature_idx].sum().item()
                            if feature not in mse_csdi_total.keys():
                                mse_csdi_total[feature] = {str(k): mse_csdi}
                            else:
                                if str(k) not in mse_csdi_total[feature].keys():
                                    mse_csdi_total[feature][str(k)] = mse_csdi
                                else:
                                    mse_csdi_total[feature][str(k)] += mse_csdi

                        mse_diff_saits = ((samples_diff_saits_median.values[0, :, feature_idx] - c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx]) ** 2
                        mse_diff_saits = mse_diff_saits.sum().item() / eval_points[0, :, feature_idx].sum().item()
                        if feature not in mse_diff_saits_total.keys():
                            mse_diff_saits_total[feature] = {"median": mse_diff_saits}
                        else:
                            mse_diff_saits_total[feature]["median"] += mse_diff_saits

                        for k in range(samples.shape[1]):
                            mse_diff_saits = ((samples_diff_saits[0, k, :, feature_idx] - c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx]) ** 2
                            mse_diff_saits = mse_diff_saits.sum().item() / eval_points[0, :, feature_idx].sum().item()
                            if feature not in mse_diff_saits_total.keys():
                                mse_diff_saits_total[feature] = {str(k): mse_diff_saits}
                            else:
                                if str(k) not in mse_diff_saits_total[feature].keys():
                                    mse_diff_saits_total[feature][str(k)] = mse_diff_saits
                                else:
                                    mse_diff_saits_total[feature][str(k)] += mse_diff_saits

                        mse_saits = ((torch.tensor(saits_output[0, :, feature_idx], device=device)- c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx]) ** 2
                        mse_saits = mse_saits.sum().item() / eval_points[0, :, feature_idx].sum().item()
                        if feature not in mse_saits_total.keys():
                            mse_saits_total[feature] = mse_saits
                        else:
                            mse_saits_total[feature] += mse_saits
        print(f"For season = {season}:")
        if trials > 1:
            print(f"For season = {season}:")
            for feature in given_features:
                if exclude_features is not None and feature in exclude_features:
                    continue
                for i in mse_csdi_total[feature].keys():
                    mse_csdi_total[feature][i] /= trials
                for i in mse_diff_saits_total[feature].keys():
                    mse_diff_saits_total[feature][i] /= trials

                mse_saits_total[feature] /= trials
                print(f"\n\tFor feature = {feature}\n\tCSDI mse: {mse_csdi_total[feature]['median']} \
                \n\tSAITS mse: {mse_saits_total[feature]}\n\tDiffSAITS mse: {mse_diff_saits_total[feature]['median']}")# \

            season_avg_mse[season] = {
                'CSDI': mse_csdi_total,
                'SAITS': mse_saits_total,
                'DiffSAITS': mse_diff_saits_total
            }

    if not os.path.isdir(mse_folder):
        os.makedirs(mse_folder)
    if trials == 1:
        fp = open(f"{mse_folder}/all-sample-results-{exclude_key if len(exclude_key) != 0 else 'all'}-{length}.json", "w")
        json.dump(results, fp=fp, indent=4, cls=NumpyArrayEncoder)
        fp.close()
    else:
        out_file = open(f"{mse_folder}/model_{len(models.keys())}_mse_seasons_{exclude_key if len(exclude_key) != 0 else 'all'}_{length}.json", "w")
        json.dump(season_avg_mse, out_file, indent = 4)
        out_file.close()


def draw_data_plot(results, f, season, folder='subplots', num_missing=100):
    
    plt.figure(figsize=(40,28))
    plt.title(f"For feature = {f} in Season {season}", fontsize=30)

    ax = plt.subplot(411)
    ax.set_title(f'Feature = {f} Season = {season} original data', fontsize=27)
    plt.plot(np.arange(results['real'].shape[0]), results['real'], 'tab:blue')
    ax.set_xlabel('Days', fontsize=25)
    ax.set_ylabel('Values', fontsize=25)
    ax = plt.subplot(412)
    ax.set_title(f'Feature = {f} Season = {season} missing data data', fontsize=27)
    plt.plot(np.arange(results['missing'].shape[0]), results['missing'], 'tab:blue')
    ax.set_xlabel('Days', fontsize=25)
    ax.set_ylabel('Values', fontsize=25)
    ax = plt.subplot(413)
    ax.set_title(f'Feature = {f} Season = {season} CSDI data', fontsize=27)
    plt.plot(np.arange(results['csdi'].shape[0]), results['csdi'], 'tab:orange')
    ax.set_xlabel('Days', fontsize=25)
    ax.set_ylabel('Values', fontsize=25)
    ax = plt.subplot(414)
    ax.set_title(f'Feature = {f} Season = {season} SAITS data', fontsize=27)
    plt.plot(np.arange(results['saits'].shape[0]), results['saits'], 'tab:green')
    ax.set_xlabel('Days', fontsize=25)
    ax.set_ylabel('Values', fontsize=25)
    
    plt.tight_layout(pad=5)
    folder = f"{folder}/{season}"
    if not os.path.isdir(folder):
        os.makedirs(folder)
    plt.savefig(f"{folder}/{f}-imputations-season-{season}-{num_missing}.png", dpi=300)
    plt.close()


def evaluate_imputation_data(models, exclude_key='', exclude_features=None, length=50):
    n_samples = 2
     
    
    nsample = 30
    i = 0
    data_folder = "data_synth"
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    for season in range(n_samples):
        print(f"For season: {season}")
        i += 1
        test_loader = get_testloader(50, len(given_features), 1, length=length, exclude_features=exclude_features, seed=50+10*i)
        for i, test_batch in enumerate(test_loader, start=1):
            output = models['CSDI'].evaluate(test_batch, nsample)
            samples, c_target, eval_points, observed_points, observed_time, obs_data_intact, gt_intact = output
            samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
            c_target = c_target.permute(0, 2, 1)  # (B,L,K)
            eval_points = eval_points.permute(0, 2, 1)
            observed_points = observed_points.permute(0, 2, 1)
            samples_median = samples.median(dim=1)
            gt_intact = gt_intact.squeeze(axis=0)
            # print(f"gt_intact: {gt_intact.shape}")
            saits_output = models['SAITS'].impute(gt_intact)

            for feature in given_features:
                if exclude_features is not None and feature in exclude_features:
                    continue
                # print(f"For feature: {feature}")
                feature_idx = given_features.index(feature)
                # cond_mask = observed_points - eval_points
                missing = gt_intact
                results = {
                    'real': obs_data_intact[0, :, feature_idx].cpu().numpy(),
                    'missing': missing[0, :, feature_idx].cpu().numpy(),
                    'csdi': samples_median.values[0, :, feature_idx].cpu().numpy(),
                    'saits': saits_output[0, :, feature_idx]
                }
                draw_data_plot(results, feature, season, folder=f"{data_folder}/subplots-{exclude_key if len(exclude_key) != 0 else 'all'}", num_missing=length)
                

seed = 10
config_dict_csdi = {
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
        'n_feature': len(given_features),
        'd_model': 256,
        'd_inner': 128,
        'n_head': 4,
        'd_k': 64,
        'd_v': 64,
        'dropout': 0.1,
        'diagonal_attention_mask': True
    }
}

nsample = 30

n_steps = 50
n_features = 5
num_seasons = 32
train_loader, valid_loader = get_dataloader(n_steps, n_features, num_seasons, batch_size=16, missing_ratio=0.2, seed=10, is_test=False)

model = CSDI_Synth(config_dict_csdi, device).to(device)
model_folder = "./saved_model_synth"
if not os.path.isdir(model_folder):
    os.makedirs(model_folder)

train(
    model,
    config_dict_csdi["train"],
    train_loader,
    valid_loader=valid_loader,
    foldername=model_folder,
)


saits_model_file = f"{model_folder}/saits_model_synth.pkl"
saits = SAITS(n_steps=n_steps, n_features=n_features, n_layers=3, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=3000, patience=200, device=device)
X, mean, std = create_synthetic_data(n_steps, num_seasons, seed=10)
saits.fit(X)
pickle.dump(saits, open(saits_model_file, 'wb'))

config_dict_diffsaits = {
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
        'n_feature': len(given_features),
        'd_model': 256,
        'd_inner': 128,
        'n_head': 4,
        'd_k': 64,
        'd_v': 64,
        'dropout': 0.1,
        'diagonal_attention_mask': True
    }
}

model_diff_saits = CSDI_Synth(config_dict_diffsaits, device).to(device)

train(
    model,
    config_dict_diffsaits["train"],
    train_loader,
    valid_loader=valid_loader,
    foldername=model_folder,
)

models = {
    'CSDI': model,
    'SAITS': saits,
    'DiffSAITS': model_diff_saits
}
mse_folder = "results_mse_synth"

lengths = [30]#[10, 25, 40, 45]
print("For All")
for l in lengths:
    print(f"For length: {l}")
    evaluate_imputation(models, mse_folder, length=l, trials=1)
    evaluate_imputation(models, mse_folder, length=l, trials=10)
    # evaluate_imputation_data(models, length=l)

feature_combinations = {
    'sin': ['sin'],
    'cos': ['cos2'],
    'sin-cos': ['sin', 'cos2']
}
print(f"The exclusions")
for key in feature_combinations.keys():
    for l in lengths:
        print(f"For length: {l}")
        evaluate_imputation(models, mse_folder, exclude_key=key, exclude_features=feature_combinations[key], length=l, trials=1)
        evaluate_imputation(models, mse_folder, exclude_key=key, exclude_features=feature_combinations[key], length=l, trials=10)
        # evaluate_imputation_data(models, exclude_key=key, exclude_features=feature_combinations[key], length=l)