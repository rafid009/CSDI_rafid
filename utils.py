import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import json
from json import JSONEncoder
import os
from dataset_agaid import get_testloader, get_testloader_agaid
from dataset_synth import get_testloader_synth
import matplotlib.pyplot as plt
import matplotlib
from main_model import CSDI_Agaid
from pypots.imputation import SAITS
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cross_validate(input_file, config_csdi, config_diffsaits, seed=10):
    seasons_list = [
        '1988-1989', 
        '1989-1990', 
        '1990-1991', 
        '1991-1992', 
        '1992-1993', 
        '1993-1994', 
        '1994-1995', 
        '1995-1996', 
        '1996-1997', 
        '1997-1998',
        '1998-1999',
        '1999-2000',
        '2000-2001',
        '2001-2002',
        '2002-2003',
        '2003-2004',
        '2004-2005',
        '2005-2006',
        '2006-2007',
        '2007-2008',
        '2008-2009',
        '2009-2010',
        '2010-2011',
        '2011-2012',
        '2012-2013',
        '2013-2014',
        '2014-2015',
        '2015-2016',
        '2016-2017',
        '2017-2018',
        '2018-2019',
        '2019-2020',
        '2020-2021',
        '2021-2022'
    ]
    seasons = {
        # '1988-1989': 0,
        # '1989-1990': 1,
        # '1990-1991': 2,
        # '1991-1992': 3,
        # '1992-1993': 4,
        # '1993-1994': 5,
        # '1994-1995': 6,
        # '1995-1996': 7,
        # '1996-1997': 8,
        # '1997-1998': 9,
        # '1998-1999': 10,
        # '1999-2000': 11,
        # '2000-2001': 12,
        # '2001-2002': 13,
        # '2002-2003': 14,
        # '2003-2004': 15,
        # '2004-2005': 16,
        # '2005-2006': 17,
        # '2006-2007': 18,
        # '2007-2008': 19,
        # '2008-2009': 20,
        # '2009-2010': 21,
        # '2010-2011': 22,
        # '2011-2012': 23,
        # '2012-2013': 24,
        # '2013-2014': 25,
        # '2014-2015': 26,
        # '2015-2016': 27,
        # '2016-2017': 28,
        # '2017-2018': 29,
        # '2018-2019': 30,
        # '2019-2020': 31,
        # '2020-2021': 32,
        '2021-2022': 33
    }
    model_folder = "./cv_saved_model"
    for i in seasons.keys():
        season_idx = seasons_list.index(i)
        model_csdi = CSDI_Agaid(config_csdi, device).to(device) 
        if not os.path.isdir(model_folder):
            os.makedirs(model_folder)
        
        filename = f'model_csdi_season_{i}.pth'
        print(f"model_name: {filename}")
        if not os.path.exists(f"{model_folder}/{filename}"):
            cv_train(model_csdi, f"{model_folder}/{filename}", input_file=input_file, season_idx=season_idx, config=config_csdi)
        else:
            model_csdi.load_state_dict(torch.load(f"{model_folder}/{filename}"))

        # saits_model_file = f"{model_csdi}/model_saits_season_{i}.pth"
        # saits = SAITS(n_steps=252, n_features=len(features), n_layers=3, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=3000, patience=200, device=device)
        # saits.fit()
        # pickle.dump(saits, open(saits_model_file, 'wb'))

        model_diff_saits = CSDI_Agaid(config_diffsaits, device, is_simple=False).to(device)
        # if not os.path.isdir(model_folder):
        #     os.makedirs(model_folder)
        filename = f'model_diff_saits_season_{i}_2500.pth'
        print(f"model_name: {filename}")
        if not os.path.exists(f"{model_folder}/{filename}"):
            cv_train(model_diff_saits, f"{model_folder}/{filename}", input_file=input_file, season_idx=season_idx, config=config_diffsaits)
        else:
            model_diff_saits.load_state_dict(torch.load(f"{model_folder}/{filename}"))

        models = {
            'CSDI': model_csdi,
            # 'SAITS': saits,
            'DiffSAITS': model_diff_saits
        }
        mse_folder = "results_cv_2500"
        lengths = [100]#[10, 20, 50, 100, 150]
        print("For All")
        for l in lengths:
            # print(f"For length: {l}")
            trials = 20
            if l == 150:
                trials = 10
            evaluate_imputation(models, mse_folder, length=l, trials=1, season_idx=season_idx)
            evaluate_imputation(models, mse_folder, length=l, trials=trials, season_idx=season_idx)
        # evaluate_imputation(models, mse_folder, trials=1, season_idx=season_idx, random_trial=True)
        # evaluate_imputation(models, mse_folder, trials=20, season_idx=season_idx, random_trial=True)


def cv_train(model, model_file, input_file, config, season_idx, seed=10):
    train_loader, valid_loader = get_dataloader(
        seed=seed,
        filename=input_file,
        batch_size=config["train"]["batch_size"],
        missing_ratio=0.2,
        season_idx=season_idx
    )
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername="",
        filename=model_file
    )


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
    filename="",
    is_saits=False
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + f"/{filename if len(filename) != 0 else 'model_csdi.pth'}"

    # p0 = int(0.6 * config["epochs"])
    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    p3 = int(0.8 * config["epochs"])
    # p4 = int(0.7 * config["epochs"])
    p5 = int(0.6 * config["epochs"])
    # exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    if is_saits:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[p1, p2], gamma=0.1
        )
        # pa
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
        )
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=1000, T_mult=1, eta_min=1.0e-7
    #     )
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20)

    best_valid_loss = 1e10
    model.train()
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        # if epoch_no == 1000:
        #     torch.save(model.state_dict(), output_path)
        #     model.load_state_dict(torch.load(f"{output_path}"))
        # if epoch_no > 1000 and epoch_no % 500 == 0:
        #     torch.save(model.state_dict(), output_path)
        #     model.load_state_dict(torch.load(f"{output_path}"))
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                # print(f"train data: {train_batch}")
                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                # lr_scheduler.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            # exp_scheduler.step()
            # metric = avg_loss / batch_no
            if is_saits:
                lr_scheduler.step()
                # pass
            else:
                lr_scheduler.step()
            
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            model.train()
                # print(
                #     "\n avg loss is now ",
                #     avg_loss_valid / batch_no,
                #     "at",
                #     epoch_no,
                # )

    if filename != "":
        torch.save(model.state_dict(), output_path)
    # if filename != "":
    #     torch.save(model.state_dict(), filename)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time, _, _ = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "mse_total": mse_total / evalpoints_total,
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            with open(
                foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        mse_total / evalpoints_total,
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("MSE:", mse_total / evalpoints_total)
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def evaluate_imputation(models, mse_folder, exclude_key='', exclude_features=None, trials=20, length=-1, season_idx=None, random_trial=False, forecasting=False, data=False, missing_ratio=0.2):
    seasons = {
    '1988-1989': 0,
    '1989-1990': 1,
    '1990-1991': 2,
    '1991-1992': 3,
    '1992-1993': 4,
    '1993-1994': 5,
    '1994-1995': 6,
    '1995-1996': 7,
    '1996-1997': 8,
    '1997-1998': 9,
    '1998-1999': 10,
    '1999-2000': 11,
    '2000-2001': 12,
    '2001-2002': 13,
    '2002-2003': 14,
    '2003-2004': 15,
    '2004-2005': 16,
    '2005-2006': 17,
    '2006-2007': 18,
    '2007-2008': 19,
    '2008-2009': 20,
    '2009-2010': 21,
    '2010-2011': 22,
    '2011-2012': 23,
    '2012-2013': 24,
    '2013-2014': 25,
    '2014-2015': 26,
    '2015-2016': 27,
    '2016-2017': 28,
    '2017-2018': 29,
    '2018-2019': 30,
    '2019-2020': 31,
    '2020-2021': 32,
    '2021-2022': 33,
    }

    seasons_list = [
        '1988-1989', 
        '1989-1990', 
        '1990-1991', 
        '1991-1992', 
        '1992-1993', 
        '1993-1994', 
        '1994-1995', 
        '1995-1996', 
        '1996-1997', 
        '1997-1998',
        '1998-1999',
        '1999-2000',
        '2000-2001',
        '2001-2002',
        '2002-2003',
        '2003-2004',
        '2004-2005',
        '2005-2006',
        '2006-2007',
        '2007-2008',
        '2008-2009',
        '2009-2010',
        '2010-2011',
        '2011-2012',
        '2012-2013',
        '2013-2014',
        '2014-2015',
        '2015-2016',
        '2016-2017',
        '2017-2018',
        '2018-2019',
        '2019-2020',
        '2020-2021',
        '2021-2022'
    ]

    if season_idx is not None:
        season_names = [seasons_list[season_idx]]
    else:
        season_names = ['2020-2021', '2021-2022']

    given_features = [
        'MEAN_AT', # mean temperature is the calculation of (max_f+min_f)/2 and then converted to Celsius. # they use this one
        'MIN_AT',
        'AVG_AT', # average temp is AgWeather Network
        'MAX_AT',
        'MIN_REL_HUMIDITY',
        'AVG_REL_HUMIDITY',
        'MAX_REL_HUMIDITY',
        'MIN_DEWPT',
        'AVG_DEWPT',
        'MAX_DEWPT',
        'P_INCHES', # precipitation
        'WS_MPH', # wind speed. if no sensor then value will be na
        'MAX_WS_MPH', 
        'LW_UNITY', # leaf wetness sensor
        'SR_WM2', # solar radiation # different from zengxian
        'MIN_ST8', # diff from zengxian
        'ST8', # soil temperature # diff from zengxian
        'MAX_ST8', # diff from zengxian
        #'MSLP_HPA', # barrometric pressure # diff from zengxian
        'ETO', # evaporation of soil water lost to atmosphere
        'ETR',
        'LTE50' # ???
    ]
    nsample = 50
    season_avg_mse = {}
    results = {}
    if 'CSDI' in models.keys():
        models['CSDI'].eval()
    if 'DiffSAITS' in models.keys():
        models['DiffSAITS'].eval()

    for season in season_names:
        print(f"For season: {season}")
        if season not in results.keys():
            results[season] = {}
        if season_idx is None:
            season_idx = seasons[season]
        mse_csdi_total = {}
        mse_saits_total = {}
        mse_diff_saits_total = {}
        # mse_diff_saits_simple_total = {}
        CRPS_csdi = 0
        CRPS_diff_saits = 0
        for i in range(trials):
            test_loader = get_testloader(seed=(10 + i), season_idx=season_idx, exclude_features=exclude_features, length=length, random_trial=random_trial, forecastig=forecasting, missing_ratio=missing_ratio)
            for j, test_batch in enumerate(test_loader, start=1):
                if 'CSDI' in models.keys():
                    output = models['CSDI'].evaluate(test_batch, nsample)
                    samples, c_target, eval_points, observed_points, observed_time, obs_intact, gt_intact = output
                    samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                    c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                    eval_points = eval_points.permute(0, 2, 1)
                    observed_points = observed_points.permute(0, 2, 1)
                    samples_median = samples.median(dim=1)
                
                if 'DiffSAITS' in models.keys():
                    output_diff_saits = models['DiffSAITS'].evaluate(test_batch, nsample)
                    if 'CSDI' not in models.keys():
                        samples_diff_saits, c_target, eval_points, observed_points, observed_time, obs_intact, gt_intact = output_diff_saits
                        c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                        eval_points = eval_points.permute(0, 2, 1)
                        observed_points = observed_points.permute(0, 2, 1)
                    else:
                        samples_diff_saits, _, _, _, _, _, _ = output_diff_saits
                    samples_diff_saits = samples_diff_saits.permute(0, 1, 3, 2)
                    samples_diff_saits_median = samples_diff_saits.median(dim=1)
                    samples_diff_saits_mean = samples_diff_saits.mean(dim=1)

                # gt_intact = gt_intact.squeeze(axis=0)
                saits_X = gt_intact #test_batch['obs_data_intact']
                saits_output = models['SAITS'].impute(saits_X)
                
                if data:
                    if 'CSDI' in models.keys():
                        results[season] = {
                            'target mask': eval_points[0, :, :].cpu().numpy(),
                            'target': c_target[0, :, :].cpu().numpy(),
                            # 'csdi_mean': samples_mean[0, :, :].cpu().numpy(),
                            'csdi_median': samples_median.values[0, :, :].cpu().numpy(),
                            'csdi_samples': samples[0].cpu().numpy(),
                            'saits': saits_output[0, :, :],
                            'diff_saits_mean': samples_diff_saits_mean[0, :, :].cpu().numpy(),
                            'diff_saits_median': samples_diff_saits_median.values[0, :, :].cpu().numpy(),
                            'diff_saits_samples': samples_diff_saits[0].cpu().numpy(),
                            # 'diff_saits_median_simple': samples_diff_saits_median_simple.values[0, :, :].cpu().numpy(),
                            # 'diff_saits_samples_simple': samples_diff_saits_simple[0].cpu().numpy()
                            }
                    else:
                         results[season] = {
                            'target mask': eval_points[0, :, :].cpu().numpy(),
                            'target': c_target[0, :, :].cpu().numpy(),
                            'saits': saits_output[0, :, :],
                            'diff_saits_mean': samples_diff_saits_mean[0, :, :].cpu().numpy(),
                            # 'diff_saits_median': samples_diff_saits_median.values[0, :, :].cpu().numpy(),
                            'diff_saits_samples': samples_diff_saits[0].cpu().numpy(),
                        }
                else:
                    for feature in given_features:
                        if exclude_features is not None and feature in exclude_features:
                            continue
                        # print(f"For feature: {feature}, for length: {length}, trial: {random_trial}")
                        feature_idx = given_features.index(feature)
                        if eval_points[0, :, feature_idx].sum().item() == 0:
                            continue
                        if 'CSDI' in models.keys():
                            mse_csdi = ((samples_median.values[0, :, feature_idx] - c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx]) ** 2
                            mse_csdi = math.sqrt(mse_csdi.sum().item() / eval_points[0, :, feature_idx].sum().item())

                            mae_csdi = torch.abs((samples_median.values[0, :, feature_idx] - c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx])
                            mae_csdi = mae_csdi.sum().item() / eval_points[0, :, feature_idx].sum().item()
                            
                            if feature not in mse_csdi_total.keys():
                                mse_csdi_total[feature] = {'rmse': 0, 'mae': 0}
                            
                            mse_csdi_total[feature]["rmse"] += mse_csdi
                            mse_csdi_total[feature]['mae'] += mae_csdi

                        if feature not in mse_diff_saits_total.keys():
                            mse_diff_saits_total[feature] = {'rmse': 0, 'mae': 0, 'diff_rmse_med': 0}

                        mse_diff_saits = ((samples_diff_saits_mean[0, :, feature_idx] - c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx]) ** 2
                        mse_diff_saits = math.sqrt(mse_diff_saits.sum().item() / eval_points[0, :, feature_idx].sum().item())

                        mse_diff_saits_median = ((samples_diff_saits_median.values[0, :, feature_idx] - c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx]) ** 2
                        mse_diff_saits_median = math.sqrt(mse_diff_saits_median.sum().item() / eval_points[0, :, feature_idx].sum().item())

                        mae_diff_saits = torch.abs((samples_diff_saits_mean[0, :, feature_idx] - c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx])
                        mae_diff_saits = mae_diff_saits.sum().item() / eval_points[0, :, feature_idx].sum().item()
                        
                        mse_diff_saits_total[feature]["rmse"] += mse_diff_saits
                        mse_diff_saits_total[feature]["mae"] += mae_diff_saits
                        mse_diff_saits_total[feature]["diff_rmse_med"] += mse_diff_saits_median


                        mse_saits = ((torch.tensor(saits_output[0, :, feature_idx], device=device)- c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx]) ** 2
                        mae_saits = torch.abs((torch.tensor(saits_output[0, :, feature_idx], device=device)- c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx])
                        mse_saits = math.sqrt(mse_saits.sum().item() / eval_points[0, :, feature_idx].sum().item())
                        mae_saits = mae_saits.sum().item() / eval_points[0, :, feature_idx].sum().item()

                        if feature not in mse_saits_total.keys():
                            mse_saits_total[feature] = {'rmse': 0, 'mae': 0}

                        mse_saits_total[feature]['rmse'] += mse_saits
                        mse_saits_total[feature]['mae'] += mae_saits
                CRPS_csdi += calc_quantile_CRPS(c_target, samples, eval_points, 0, 1)
                CRPS_diff_saits += calc_quantile_CRPS(c_target, samples_diff_saits, eval_points, 0, 1)
        print(f"CSDI CRPS: {CRPS_csdi/trials}")
        print(f"DiffSAITS CRPS: {CRPS_diff_saits/trials}")
        if not data:
            print(f"For season = {season}:")
            for feature in features:
                if exclude_features is not None and feature in exclude_features:
                    continue
                # if feature not in mse_csdi_total.keys() or feature not in mse_diff_saits_total.keys():
                #     continue
                if 'CSDI' in models.keys():
                    for i in mse_csdi_total[feature].keys():
                        mse_csdi_total[feature][i] /= trials
                for i in mse_diff_saits_total[feature].keys():
                    mse_diff_saits_total[feature][i] /= trials
                for i in mse_saits_total[feature].keys():
                    mse_saits_total[feature][i] /= trials
                if 'CSDI' in models.keys():
                    print(f"\n\tFor feature = {feature}\n\tCSDI mae: {mse_csdi_total[feature]['mae']}\n\tDiffSAITS mae: {mse_diff_saits_total[feature]['mae']}")
                    print(f"\n\tFor feature = {feature}\n\tCSDI rmse: {mse_csdi_total[feature]['rmse']}\n\tDiffSAITS rmse: {mse_diff_saits_total[feature]['rmse']}\n\tDiffSAITS median: {mse_diff_saits_total[feature]['diff_rmse_med']}\n\tSAITS mse: {mse_saits_total[feature]['rmse']}")# \
                else:
                    print(f"\n\tFor feature = {feature}\n\tDiffSAITS mae: {mse_diff_saits_total[feature]['mae']}")
                    print(f"\n\tFor feature = {feature}\n\tDiffSAITS mse: {mse_diff_saits_total[feature]['rmse']}")# \
                
                # DiffSAITSsimple mse: {mse_diff_saits_simple_total[feature]}")
                # except:
                #     continue
            if 'CSDI' in models.keys():
                season_avg_mse[season] = {
                    'CSDI': mse_csdi_total,
                    'SAITS': mse_saits_total,
                    'DiffSAITS': mse_diff_saits_total#,
                    # 'DiffSAITSsimple': mse_diff_saits_simple_total
                }
            else:
                season_avg_mse[season] = {
                    'SAITS': mse_saits_total,
                    'DiffSAITS': mse_diff_saits_total#,
                }


    
    if not os.path.isdir(mse_folder):
        os.makedirs(mse_folder)
    if data:
        fp = open(f"{mse_folder}/samples-{exclude_key if len(exclude_key) != 0 else 'all'}-l_{length}_{season_names[0] if len(season_names) == 1 else season_names}_random_{random_trial}_forecast_{forecasting}_miss_{missing_ratio}.json", "w")
        json.dump(results, fp=fp, indent=4, cls=NumpyArrayEncoder)
        fp.close()
    else:
        out_file = open(f"{mse_folder}/mse_mae_{exclude_key if len(exclude_key) != 0 else 'all'}_l_{length}_{season_names[0] if len(season_names) == 1 else season_names}_random_{random_trial}_forecast_{forecasting}_miss_{missing_ratio}.json", "w")
        json.dump(season_avg_mse, out_file, indent = 4)
        out_file.close()


def evaluate_imputation_all(models, mse_folder, dataset_name='agaid', batch_size=16, trials=10, length=-1, random_trial=False, forecasting=False, missing_ratio=-1):  
    nsample = 50
    if 'CSDI' in models.keys():
        models['CSDI'].eval()
    if 'DiffSAITS' in models.keys():
        models['DiffSAITS'].eval()

    results = {'csdi': {}, 'diffsaits': {}, 'saits': {}}
    results_rmse = {'csdi': 0, 'diffsaits': 0, 'saits': 0}
    results_crps = {
        'csdi_trials':{}, 'csdi': 0, 
        'diffsaits_trials': {}, 'diffsaits': 0, 
        'saits_trials': {}, 'saits': 0
        }
    for trial in range(trials):
        if dataset_name == 'synth':
            test_loader = get_testloader_synth(n_steps=100, n_features=7, num_seasons=16, seed=(10 + trial), length=length, missing_ratio=missing_ratio, random_trial=random_trial, forecasting=forecasting)
        elif dataset_name == 'physio':
            pass
        else:
            test_loader = get_testloader_agaid(seed=(10 + trial), length=length, missing_ratio=missing_ratio, random_trial=random_trial, forecastig=forecasting)
        
        csdi_rmse_avg = 0
        diffsaits_rmse_avg = 0
        saits_rmse_avg = 0

        csdi_crps_avg = 0
        diffsaits_crps_avg = 0

        for j, test_batch in enumerate(test_loader, start=1):
            if 'CSDI' in models.keys():
                output = models['CSDI'].evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points, observed_time, obs_intact, gt_intact = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)
                samples_median = samples.median(dim=1)
            
            if 'DiffSAITS' in models.keys():
                output_diff_saits = models['DiffSAITS'].evaluate(test_batch, nsample)
                if 'CSDI' not in models.keys():
                    samples_diff_saits, c_target, eval_points, observed_points, observed_time, obs_intact, gt_intact = output_diff_saits
                    c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                    eval_points = eval_points.permute(0, 2, 1)
                    observed_points = observed_points.permute(0, 2, 1)
                else:
                    samples_diff_saits, _, _, _, _, _, _ = output_diff_saits
                samples_diff_saits = samples_diff_saits.permute(0, 1, 3, 2)
                # samples_diff_saits_median = samples_diff_saits.median(dim=1)
                samples_diff_saits_mean = samples_diff_saits.mean(dim=1)

            gt_intact = gt_intact.squeeze(axis=0)
            saits_X = gt_intact #test_batch['obs_data_intact']
            saits_output = models['SAITS'].impute(saits_X)

            ###### RMSE ######

            rmse_csdi = ((samples_median.values - c_target) * eval_points) ** 2
            rmse_csdi = math.sqrt(rmse_csdi.sum().item() / eval_points.sum().item())
            csdi_rmse_avg += rmse_csdi

            rmse_diff_saits = ((samples_diff_saits_mean - c_target) * eval_points) ** 2
            rmse_diff_saits = math.sqrt(rmse_diff_saits.sum().item() / eval_points.sum().item())
            diffsaits_rmse_avg += rmse_diff_saits

            rmse_saits = ((torch.tensor(saits_output, device=device)- c_target) * eval_points) ** 2
            rmse_saits = math.sqrt(rmse_saits.sum().item() / eval_points.sum().item())
            saits_rmse_avg += rmse_saits

            ###### CRPS ######

            csdi_crps = calc_quantile_CRPS(c_target, samples, eval_points, 0, 1)
            csdi_crps_avg += csdi_crps

            diff_saits_crps = calc_quantile_CRPS(c_target, samples_diff_saits, eval_points, 0, 1)
            diffsaits_crps_avg += diff_saits_crps

        results['csdi'][trial] = csdi_rmse_avg / batch_size
        results_rmse['csdi'] += csdi_rmse_avg / batch_size

        results['diffsaits'][trial] = diffsaits_rmse_avg / batch_size
        results_rmse['diffsaits'] += diffsaits_rmse_avg / batch_size

        results['saits'][trial] = saits_rmse_avg / batch_size
        results_rmse['saits'] += saits_rmse_avg / batch_size

        results_crps['csdi_trials'][trial] = csdi_crps
        results_crps['csdi'] += csdi_crps

        results_crps['diffsaits_trials'][trial] = diffsaits_crps_avg / batch_size
        results_crps['diffsaits'] += diffsaits_crps_avg / batch_size

    results_rmse['csdi'] /= trials
    results_rmse['diffsaits'] /= trials
    results_rmse['saits'] /= trials

    results_crps['csdi'] /= trials
    results_crps['diffsaits'] /= trials

    if not os.path.isdir(mse_folder):
        os.makedirs(mse_folder)

    fp = open(f"{mse_folder}/rmse-trials-random-{random_trial}-forecasting-{forecasting}-blackout-{not (random_trial or forecasting)}_l_{length}_miss_{missing_ratio}.json", "w")
    json.dump(results, fp=fp, indent=4)
    fp.close()

    fp = open(f"{mse_folder}/rmse-random-{random_trial}-forecasting-{forecasting}-blackout-{not (random_trial or forecasting)}_l_{length}_miss_{missing_ratio}.json", "w")
    json.dump(results_rmse, fp=fp, indent=4)
    fp.close()
    
    fp = open(f"{mse_folder}/crps-random-{random_trial}-forecasting-{forecasting}-blackout-{not (random_trial or forecasting)}_l_{length}_miss_{missing_ratio}.json", "w")
    json.dump(results, fp=fp, indent=4)
    fp.close()


def draw_data_plot(results, f, season, folder='subplots', num_missing=100):
    
    plt.figure(figsize=(45,32))
    plt.title(f"For feature = {f} in Season {season}", fontsize=30)

    ax = plt.subplot(511)
    ax.set_title('Feature = '+f+' Season = '+season+' original data', fontsize=27)
    plt.plot(np.arange(results['real'].shape[0]), results['real'], 'tab:blue')
    ax.set_xlabel('Days', fontsize=25)
    ax.set_ylabel('Values', fontsize=25)
    ax = plt.subplot(512)
    ax.set_title('Feature = '+f+' Season = '+season+' missing data data', fontsize=27)
    plt.plot(np.arange(results['missing'].shape[0]), results['missing'], 'tab:blue')
    ax.set_xlabel('Days', fontsize=25)
    ax.set_ylabel('Values', fontsize=25)
    ax = plt.subplot(513)
    ax.set_title('Feature = '+f+' Season = '+season+' CSDI data', fontsize=27)
    plt.plot(np.arange(results['csdi'].shape[0]), results['csdi'], 'tab:orange')
    ax.set_xlabel('Days', fontsize=25)
    ax.set_ylabel('Values', fontsize=25)
    ax = plt.subplot(514)
    ax.set_title('Feature = '+f+' Season = '+season+' SAITS data', fontsize=27)
    plt.plot(np.arange(results['saits'].shape[0]), results['saits'], 'tab:green')
    ax.set_xlabel('Days', fontsize=25)
    ax.set_ylabel('Values', fontsize=25)
    ax = plt.subplot(515)
    ax.set_title('Feature = '+f+' Season = '+season+' DiffSAITS data', fontsize=27)
    plt.plot(np.arange(results['diffsaits'].shape[0]), results['diffsaits'], 'tab:olive')
    ax.set_xlabel('Days', fontsize=25)
    ax.set_ylabel('Values', fontsize=25)
    
    plt.tight_layout(pad=5)
    folder = f"{folder}/{season}"
    if not os.path.isdir(folder):
        os.makedirs(folder)
    plt.savefig(f"{folder}/{f}-imputations-season-{season}-{num_missing}.png", dpi=300)
    plt.close()


def evaluate_imputation_data(models, exclude_key='', exclude_features=None, length=50, season_idx=None):
    seasons = {
        '1988-1989': 0,
        '1989-1990': 1,
        '1990-1991': 2,
        '1991-1992': 3,
        '1992-1993': 4,
        '1993-1994': 5,
        '1994-1995': 6,
        '1995-1996': 7,
        '1996-1997': 8,
        '1997-1998': 9,
        '1998-1999': 10,
        '1999-2000': 11,
        '2000-2001': 12,
        '2001-2002': 13,
        '2002-2003': 14,
        '2003-2004': 15,
        '2004-2005': 16,
        '2005-2006': 17,
        '2006-2007': 18,
        '2007-2008': 19,
        '2008-2009': 20,
        '2009-2010': 21,
        '2010-2011': 22,
        '2011-2012': 23,
        '2012-2013': 24,
        '2013-2014': 25,
        '2014-2015': 26,
        '2015-2016': 27,
        '2016-2017': 28,
        '2017-2018': 29,
        '2018-2019': 30,
        '2019-2020': 31,
        '2020-2021': 32,
        '2021-2022': 33,
    }

    seasons_list = [
        '1988-1989', 
        '1989-1990', 
        '1990-1991', 
        '1991-1992', 
        '1992-1993', 
        '1993-1994', 
        '1994-1995', 
        '1995-1996', 
        '1996-1997', 
        '1997-1998',
        '1998-1999',
        '1999-2000',
        '2000-2001',
        '2001-2002',
        '2002-2003',
        '2003-2004',
        '2004-2005',
        '2005-2006',
        '2006-2007',
        '2007-2008',
        '2008-2009',
        '2009-2010',
        '2010-2011',
        '2011-2012',
        '2012-2013',
        '2013-2014',
        '2014-2015',
        '2015-2016',
        '2016-2017',
        '2017-2018',
        '2018-2019',
        '2019-2020',
        '2020-2021',
        '2021-2022'
    ]

    if season_idx is not None:
        season_names = [seasons_list[season_idx]]
    else:
        season_names = ['2020-2021', '2021-2022']

    given_features = [
        'MEAN_AT', # mean temperature is the calculation of (max_f+min_f)/2 and then converted to Celsius. # they use this one
        'MIN_AT',
        'AVG_AT', # average temp is AgWeather Network
        'MAX_AT',
        'MIN_REL_HUMIDITY',
        'AVG_REL_HUMIDITY',
        'MAX_REL_HUMIDITY',
        'MIN_DEWPT',
        'AVG_DEWPT',
        'MAX_DEWPT',
        'P_INCHES', # precipitation
        'WS_MPH', # wind speed. if no sensor then value will be na
        'MAX_WS_MPH', 
        'LW_UNITY', # leaf wetness sensor
        'SR_WM2', # solar radiation # different from zengxian
        'MIN_ST8', # diff from zengxian
        'ST8', # soil temperature # diff from zengxian
        'MAX_ST8', # diff from zengxian
        #'MSLP_HPA', # barrometric pressure # diff from zengxian
        'ETO', # evaporation of soil water lost to atmosphere
        'ETR',
        'LTE50' # ???
    ]
    nsample = 50
    i = 0
    for season in season_names:
        print(f"For season: {season}")
        i += 1
        season_idx = seasons[season]
        test_loader = get_testloader(seed=10 + 10 * i, season_idx=season_idx, length=length, exclude_features=exclude_features)
        for i, test_batch in enumerate(test_loader, start=1):
            output = models['CSDI'].evaluate(test_batch, nsample)
            samples, c_target, eval_points, observed_points, observed_time, obs_data_intact, gt_intact = output
            samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
            c_target = c_target.permute(0, 2, 1)  # (B,L,K)
            eval_points = eval_points.permute(0, 2, 1)
            observed_points = observed_points.permute(0, 2, 1)
            samples_median = samples.median(dim=1)
            gt_intact = gt_intact.squeeze(axis=0)
            
            saits_output = models['SAITS'].impute(gt_intact)

            output_diff_saits = models['DiffSAITS'].evaluate(test_batch, nsample)
            samples_diff_saits, _, _, _, _, _, _ = output_diff_saits
            samples_diff_saits = samples_diff_saits.permute(0, 1, 3, 2)
            samples_diff_saits_median = samples_diff_saits.median(dim=1)

            for feature in given_features:
                if exclude_features is not None and feature in exclude_features:
                    continue
                print(f"For feature: {feature}")
                feature_idx = given_features.index(feature)
                # cond_mask = observed_points - eval_points
                missing = gt_intact
                results = {
                    'real': obs_data_intact[0, :, feature_idx].cpu().numpy(),
                    'missing': missing[0, :, feature_idx].cpu().numpy(),
                    'csdi': samples_median.values[0, :, feature_idx].cpu().numpy(),
                    'saits': saits_output[0, :, feature_idx],
                    'diffsaits': samples_diff_saits_median.values[0, :, feature_idx].cpu().numpy()
                }
                draw_data_plot(results, feature, season, folder=f"subplots_result/subplots-{exclude_key if len(exclude_key) != 0 else 'all'}", num_missing=length)

def graph_bar_diff_multi(diff_folder, GT_values, result_dict, title, x, xlabel, ylabel, season, feature, missing=None, existing=-1):
    plot_dict = {}
    plt.figure(figsize=(32,20))
    for key, value in result_dict.items():
        # print(f"key: {key}")
        if missing is None:
            plot_dict[key] = np.abs(GT_values) - np.abs(value)
        else:
            plot_dict[key] = np.abs(GT_values) - np.abs(value[missing])
    # ind = np.arange(prediction.shape[0])
    # x = np.array(x)
    width = 0.7
    pos = 0
    remove_keys = ['real', 'missing']

    colors = ['tab:orange', 'tab:blue', 'tab:cyan', 'tab:purple']
    # if drop_linear:
    #   remove_keys.append('LinearInterp')
    i = 0
    for key, value in plot_dict.items():
        if key not in remove_keys:
            # print(f"x = {len(x)}, value = {len(value)}")
            print(f"x={x}\npos={pos}\nvalue={value}\nwidth={width}")
            plt.bar(x + pos, value, width, label = key, color=colors[i])
            i += 1
            pos += width

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=25)
    plt.xticks([r + width for r in range(len(x))], [str(i) for i in x],fontsize=20)
    plt.yticks(fontsize=20)
    # plt.axis([0, 80, -2, 3])

    plt.legend(loc='best', fontsize=25)
    plt.tight_layout(pad=5)
    folder = f"{diff_folder}/{season}"
    if not os.path.isdir(folder):
        os.makedirs(folder)
    plt.savefig(f'{folder}/diff-{feature}-{season}-{existing}.png', dpi=300)
    plt.close()

def forward_evaluation(models, filename, features):
    seasons = {
        # '1988-1989': 0,
        # '1989-1990': 1,
        # '1990-1991': 2,
        # '1991-1992': 3,
        # '1992-1993': 4,
        # '1993-1994': 5,
        # '1994-1995': 6,
        # '1995-1996': 7,
        # '1996-1997': 8,
        # '1997-1998': 9,
        # '1998-1999': 10,
        # '1999-2000': 11,
        # '2000-2001': 12,
        # '2001-2002': 13,
        # '2002-2003': 14,
        # '2003-2004': 15,
        # '2004-2005': 16,
        # '2005-2006': 17,
        # '2006-2007': 18,
        # '2007-2008': 19,
        # '2008-2009': 20,
        # '2009-2010': 21,
        # '2010-2011': 22,
        # '2011-2012': 23,
        # '2012-2013': 24,
        # '2013-2014': 25,
        # '2014-2015': 26,
        # '2015-2016': 27,
        # '2016-2017': 28,
        # '2017-2018': 29,
        # '2018-2019': 30,
        # '2019-2020': 31,
        '2020-2021': 32,
        # '2021-2022': 33,
    }
    nsample = 50
    for season in seasons.keys():
        season_idx = seasons[season]
        df = pd.read_csv(filename)
        modified_df, dormant_seasons = preprocess_missing_values(df, features, is_dormant=True)
        season_df, season_array, max_length = get_seasons_data(modified_df, dormant_seasons, features, is_dormant=True)
        train_season_df = season_df.drop(season_array[-1], axis=0)
        train_season_df = train_season_df.drop(season_array[-2], axis=0)
        mean, std = get_mean_std(train_season_df, features)
        X, Y = split_XY(season_df, max_length, season_array, features)
        X = np.expand_dims(X[season_idx], 0)
        lte_idx = features.index('LTE50')
        indices = np.where(~np.isnan(X[0, :, lte_idx]))[0].tolist()
        for i in range(2, len(indices) - 1):
            test_loader = get_forward_testloader(X, mean, std, forward_trial=i, lte_idx=lte_idx)

            for j, test_batch in enumerate(test_loader, start=1):
                output = models['CSDI'].evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points, observed_time, obs_data_intact, gt_intact = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)
                samples_median = samples.median(dim=1)
                gt_intact = gt_intact.squeeze(axis=0)
                saits_output = models['SAITS'].impute(gt_intact)
                start = indices[i]
                mse_saits_same = ((torch.tensor(saits_output[0, start, lte_idx], device=device)- c_target[0, start, lte_idx])) ** 2
                mse_saits_next = ((torch.tensor(saits_output[0, indices[i+1], lte_idx], device=device)- c_target[0, indices[i+1], lte_idx])) ** 2
                
                mse_csdi_same = ((samples_median.values[0, start, lte_idx] - c_target[0, start, lte_idx])) ** 2
                mse_csdi_next = ((samples_median.values[0, indices[i+1], lte_idx] - c_target[0, indices[i+1], lte_idx])) ** 2

                print(f"Season: {season} LTE50 Existing: {i-1} :\n\tSAITS:\n\t\tSame day: {mse_saits_same}\n\t\tNext LTE: {mse_saits_next}\n\tCSDI:\n\t\tSame day: {mse_csdi_same}\n\t\tNext day: {mse_csdi_next}")
                results_for_diff = {
                    'SAITS': saits_output[0, :, lte_idx] * eval_points[0, :, lte_idx].cpu().numpy(),
                    'CSDI': samples_median.values[0, :, lte_idx].cpu().numpy() * eval_points[0, :, lte_idx].cpu().numpy()
                }
                graph_bar_diff_multi(f"diff_LTE", c_target[0, :, lte_idx].cpu().numpy() * eval_points[0, :, lte_idx].cpu().numpy(), results_for_diff, f'Season: {season}, LTE50 prediction existing GT: {i-1}', np.arange(len(c_target[0, :, lte_idx])), 'Days', 'Difference from GT (LTE50)', season, 'LTE50', existing=(i-1))