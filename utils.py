import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import json
from dataset_agaid import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model_csdi.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
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

                print(
                    "\n avg loss is now ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )

    if foldername != "":
        torch.save(model.state_dict(), output_path)


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

                samples, c_target, eval_points, observed_points, observed_time = output
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

def evaluate_imputation(models, mse_folder, trials=30):
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
    '2021-2022': 33,
    }


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
    # trials = 30
    season_avg_mse = {}
    for season in seasons.keys():
        print(f"For season: {season}")
        season_idx = seasons[season]
        mse_csdi_total = {}
        mse_saits_total = {}
        for i in range(trials):
            test_loader = get_testloader(seed=(10 + i), season_idx=season_idx)
            for i, test_batch in enumerate(test_loader, start=1):
                output = models['CSDI'].evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)
                samples_median = samples.median(dim=1)

                saits_X = test_batch['obs_data_intact']
                saits_output = models['SAITS'].impute(saits_X)

                for feature in given_features:
                    feature_idx = given_features.index(feature)
                    # print(f"samp median: {samples_median.values.shape}\ntarget: {c_target.shape}\neval: {eval_points.shape}")
                    
                    
                    mse_csdi = ((samples_median.values[0, :, feature_idx] - c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx]) ** 2
                    mse_csdi = mse_csdi.sum().item() / eval_points[0, :, feature_idx].sum().item()
                    if feature not in mse_csdi_total.keys():
                        mse_csdi_total[feature] = {"median": mse_csdi}
                    else:
                        mse_csdi_total[feature]["median"] += mse_csdi

                    for i in range(samples.shape[1]):
                        mse_csdi = ((samples[0, i, :, feature_idx] - c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx]) ** 2
                        mse_csdi = mse_csdi.sum().item() / eval_points[0, :, feature_idx].sum().item()
                        if feature not in mse_csdi_total.keys():
                            mse_csdi_total[feature] = {str(i): mse_csdi}
                        else:
                            mse_csdi_total[feature][str(i)] += mse_csdi

                        
                    mse_saits = ((torch.tensor(saits_output[0, :, feature_idx], device=device)- c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx]) ** 2
                    mse_saits = mse_saits.sum().item() / eval_points[0, :, feature_idx].sum().item()
                    if feature not in mse_saits_total.keys():
                        mse_saits_total[feature] = mse_saits
                    else:
                        mse_saits_total[feature] += mse_saits
        print(f"For season = {season}:")
        for feature in features:
            for i in mse_csdi_total[feature].keys():
                mse_csdi_total[feature][i] /= trials
            mse_saits_total[feature] /= trials
            print(f"\n\tFor feature = {feature}\n\tCSDI mse: {mse_csdi_total[feature]['median']}\n\tSAITS mse: {mse_saits_total[feature]}")
        season_avg_mse[season] = {
            'CSDI': mse_csdi_total,
            'SAITS': mse_saits_total
        }

    if not os.path.isdir(mse_folder):
        os.makedirs(mse_folder)

    out_file = open(f"{mse_folder}/test_avg_mse_seasons.json", "w")
  
    json.dump(season_avg_mse, out_file, indent = 4)
    
    out_file.close()

