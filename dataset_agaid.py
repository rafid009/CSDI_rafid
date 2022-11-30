from process_data import *
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

def parse_data(sample, rate, is_test=False, length=100, include_features=None):
    """Get mask of random points (missing at random) across channels based on k,
    where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""
    if isinstance(sample, torch.Tensor):
        sample = sample.numpy()

    obs_mask = ~np.isnan(sample)
    
    if not is_test:
        shp = sample.shape
        evals = sample.reshape(-1).copy()
        indices = np.where(~np.isnan(evals))[0].tolist()
        indices = np.random.choice(indices, int(len(indices) * rate))
        values = evals.copy()
        values[indices] = np.nan
        mask = ~np.isnan(values)
        mask = mask.reshape(shp)
        obs_intact = values.reshape(shp).copy()
        obs_data = np.nan_to_num(evals, copy=True)
        obs_data = obs_data.reshape(shp)
        obs_data_intact = evals.reshape(shp)
    else:
        a = np.arange(sample.shape[0] - length)
        # print(f"a: {a}\nsample: {sample.shape}")
        start_idx = np.random.choice(a)
        print(f"random choice: {start_idx}")
        end_idx = start_idx + length
        obs_data_intact = sample.copy()
        if include_features is None or len(include_features) == 0:
            obs_data_intact[start_idx:end_idx, :] = np.nan
        else:
            obs_data_intact[start_idx:end_idx, include_features] = np.nan
        mask = ~np.isnan(obs_data_intact)
        obs_intact = obs_data_intact.copy()
        obs_data = np.nan_to_num(obs_intact, copy=True)
        # obs_intact = np.nan_to_num(obs_intact, copy=True)
    return obs_data, obs_mask, mask, sample, obs_intact




def get_mask_mnr(sample, rate):
    """Get mask of random segments (non-missing at random) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""
    flatten_sample = np.reshape(sample, -1)
    k = int(len(flatten_sample) * rate)
    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    for channel in range(mask.shape[1]):
        s_nan = np.random.choice(list_of_segments_index)
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask


def get_mask_bm(sample, rate):
    """Get mask of same segments (black-out missing) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to be preserved
    as per ts imputers"""
    flatten_sample = np.reshape(sample, -1)
    k = int(len(flatten_sample) * rate)
    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    s_nan = np.random.choice(list_of_segments_index)
    for channel in range(mask.shape[1]):
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask


class Agaid_Dataset(Dataset):
    def __init__(self, X, mean, std, eval_length=252, rate=0.2, is_test=False, length=100, exclude_features=None) -> None:
        super().__init__()
        self.eval_length = eval_length
        self.observed_values = []
        self.obs_data_intact = []
        self.observed_masks = []
        self.gt_masks = []
        self.gt_intact = []
        
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        
        include_features = []
        if exclude_features is not None:
            for feature in features:
                if feature not in exclude_features:
                    include_features.append(features.index(feature))
        for i in range(len(X)):
            obs_data, obs_mask, gt_mask, obs_data_intact, gt_intact_data = parse_data(X[i], rate=rate, is_test=is_test, length=length, include_features=include_features)
            self.obs_data_intact.append(obs_data_intact)
            self.gt_masks.append(gt_mask)
            self.observed_values.append(obs_data)
            self.observed_masks.append(obs_mask)
            self.gt_intact.append(gt_intact_data)
        self.gt_masks = torch.tensor(self.gt_masks, dtype=torch.float32)
        self.observed_values = torch.tensor(self.observed_values, dtype=torch.float32)
        self.obs_data_intact = np.array(self.obs_data_intact)
        self.gt_intact = np.array(self.gt_intact)
        self.observed_masks = torch.tensor(self.observed_masks, dtype=torch.float32)
        self.observed_values = ((self.observed_values - self.mean) / self.std) * self.gt_masks
        self.obs_data_intact = ((self.obs_data_intact - self.mean.numpy()) / self.std.numpy()) * self.observed_masks.numpy()
        self.gt_intact = ((self.gt_intact - self.mean.numpy()) / self.std.numpy()) * self.gt_masks.numpy()
        
    def __getitem__(self, index):
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            # "gt_mask": self.gt_masks[index],
            "obs_data_intact": self.obs_data_intact[index],
            "timepoints": np.arange(self.eval_length),
            "gt_intact": self.gt_intact
        }
        if len(self.gt_masks) == 0:
            s["gt_mask"] = None
        else:
            s["gt_mask"] = self.gt_masks[index]
        return s
    
    def __len__(self):
        return len(self.observed_values)

        
def get_dataloader(filename='ColdHardiness_Grape_Merlot_2.csv', batch_size=16, missing_ratio=0.2, seed=10, is_test=False):
    np.random.seed(seed=seed)
    df = pd.read_csv(filename)
    modified_df, dormant_seasons = preprocess_missing_values(df, features, is_dormant=True)
    season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, features, is_dormant=True)
    train_season_df = season_df.drop(season_array[-1], axis=0)
    train_season_df = train_season_df.drop(season_array[-2], axis=0)
    mean, std = get_mean_std(train_season_df, features)
    X, Y = split_XY(season_df, max_length, season_array, features)
    train_dataset = Agaid_Dataset(X[:-2], mean, std, rate=missing_ratio)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = Agaid_Dataset(X[-2:], mean, std, rate=missing_ratio, is_test=is_test)
    if is_test:
        test_loader = DataLoader(test_dataset, batch_size=1)
    else:
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    return train_loader, test_loader

def get_testloader(filename='ColdHardiness_Grape_Merlot_2.csv', missing_ratio=0.2, seed=10, season_idx=-1, exclude_features=None, length=100):
    # np.random.seed(seed=seed)
    df = pd.read_csv(filename)
    modified_df, dormant_seasons = preprocess_missing_values(df, features, is_dormant=True)
    season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, features, is_dormant=True)
    train_season_df = season_df.drop(season_array[-1], axis=0)
    train_season_df = train_season_df.drop(season_array[-2], axis=0)
    mean, std = get_mean_std(train_season_df, features)
    X, Y = split_XY(season_df, max_length, season_array, features)
    # print(f"X: {X.shape}\nidx: {season_idx}")
    X = np.expand_dims(X[season_idx], 0)
    # print(f"X expand: {X.shape}")
    test_dataset = Agaid_Dataset(X, mean, std, rate=missing_ratio, is_test=True, length=length, exclude_features=exclude_features)
    test_loader = DataLoader(test_dataset, batch_size=1)

    return test_loader