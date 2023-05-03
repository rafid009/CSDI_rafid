import numpy as np
feats = ['sin', 'cos2', 'harmonic', 'weight', 'lin_comb', 'non_lin_comb', 'mixed_history']
feats_v2 = ['sin', 'cos2', 'tan']
feats_v3 = ['sin', 'cos', 'tan']


def add_rn_missing(X, length=-1, rate=-1):
    if length != -1:
        a = np.arange(X.shape[0] - length + 1)
        start = np.random.choice(a)
        end = start + length + 1
        X[start:end] = np.nan
    else:
        shp = X.shape
        sample = X.reshape(-1).copy()
        indices = np.where(~np.isnan(sample))[0].tolist()
        indices = np.random.choice(indices, int(len(indices) * rate))
        sample[indices] = np.nan
        sample = sample.reshape(shp)
        X = sample
    return X
    


def create_synthetic_data(n_steps, num_seasons, seed=10, rate=0.05, length_rate=0.02):
    # num_seasons = 32
    np.random.seed(seed)
    
    synth_features = {
        'sin': (0.00001, 2 * np.pi/3, np.pi/2), # f1
        'cos2': (0, 2*np.pi, np.pi/4), # f2
        'harmonic': np.pi/2, # f3
        'weight': (0.3, 0.6), # f4
        'lin_comb': (0.2, 0.03), # f5
        'non_lin_comb': (0), # f6
        'mixed_history': (0) # f7
    }
    num_steps = n_steps # config['n_steps']
    num_features = len(feats) # config['n_features']
    data = np.zeros((num_seasons, num_steps, num_features))
    # value_range = [(0.1, 0.4, 0.7, 0.99), (11.0, 17.5, 40.5, 61.2), (100.1, 160.2, 500, 1000)]
    
    for i in range(data.shape[0]):
        for feature in synth_features.keys():
            lr = int(np.floor(n_steps * (np.random.rand() * length_rate)))
            args = synth_features[feature]
            if feature == 'sin':
                low = np.random.uniform(args[0], args[0]) + np.random.uniform(0, args[2])
                high = np.random.uniform(args[1], args[1]) + np.random.uniform(0, args[2])
                data[i, :, feats.index(feature)] = np.sin(np.linspace(low, high, data.shape[1]))
            elif feature == 'cos2':
                low = np.random.uniform(args[0], args[0]) + np.random.uniform(0, args[2])
                high = np.random.uniform(args[1], args[1]) + np.random.uniform(0, args[2])
                data[i, :, feats.index(feature)] = (np.cos(np.linspace(low, high, data.shape[1])) ** 2) + (np.random.rand(data.shape[1]) ** (1/8))
            elif feature == 'harmonic':
                f1 = data[i, :, feats.index('sin')]
                f2 = data[i, :, feats.index('cos2')]
                data[i, :, feats.index(feature)] = (synth_features['harmonic'] * f1 + synth_features['harmonic'] * f2) / (1/(f1 + 1e-2) + 1/(f2 + 1e-2))
            elif feature == 'weight':
                f1 = data[i, :, feats.index('sin')]
                f2 = data[i, :, feats.index('cos2')]
                data[i, :, feats.index(feature)] = (synth_features['weight'][0] * f1 + synth_features['weight'][1] * f2) / (1/synth_features['weight'][0] + 1/synth_features['weight'][1])
            elif feature == 'lin_comb':
                data[i, 0, feats.index(feature)] = np.random.uniform(0.001, 1)
                f3 = data[i, 0:-1, feats.index('weight')]
                f2 = data[i, 0:-1, feats.index('cos2')]
                f1 = data[i, 0:-1, feats.index('sin')]
                data[i, 1:, feats.index(feature)] = f1 * 0.5 + f3 * 0.1 + f2 * 0.4
            elif feature == 'non_lin_comb':
                data[i, 0, feats.index(feature)] = np.random.uniform(0.02, 0.06)
                f5 = data[i, 0:-1, feats.index(feature)]
                f2 = data[i, 0:-1, feats.index('cos2')]
                f1 = data[i, 0:-1, feats.index('sin')]
                f3 = data[i, 0:-1, feats.index('weight')]
                data[i, 1:, feats.index(feature)] = f1 * f5 + f2 * f3
            elif feature == 'mixed_history':
                data[i, 0, feats.index(feature)] = np.random.uniform(0, 1)
                data[i, 1, feats.index(feature)] = np.random.uniform(0, 1)
                f1 = data[i, 0:-2, feats.index('sin')]
                f2 = data[i, 0:-2, feats.index('cos2')]
                data[i, 2:, feats.index(feature)] = f1 + f2
            if feature == 'sin' or feature == 'cos2':
                data[i, :, feats.index(feature)] = add_rn_missing(data[i, :, feats.index(feature)], length=lr)
        data[i] = add_rn_missing(data[i], rate=rate)
            # elif feature == 'inv':
            #     data[i, :, feats.index(feature)] = (synth_features['inv'][0] * data[i, :, feats.index('sin')] - synth_features['inv'][1] / data[i, :, feats.index('sin')])
    data_rows = data.reshape((-1, num_features))
    mean = np.nanmean(data_rows, axis=0)
    std = np.nanstd(data_rows, axis=0)
    data = data.reshape((num_seasons, num_steps, num_features))
    return data, mean, std

def create_synthetic_data_v2(n_steps, num_seasons, seed=10, rate=0.05, length_rate=0.02, noise=False):
    # num_seasons = 32
    np.random.seed(seed)

    synth_features = {
        'sin': (0.00001, 2 * np.pi/3, np.pi/2), # f1
        'cos2': (0, 2*np.pi, np.pi/4), # f2
        'tan': (0, np.pi/3, np.pi/16) # f3
    }
    num_steps = n_steps # config['n_steps']
    num_features = len(feats_v2) # config['n_features']
    data = np.zeros((num_seasons, num_steps, num_features))
    # value_range = [(0.1, 0.4, 0.7, 0.99), (11.0, 17.5, 40.5, 61.2), (100.1, 160.2, 500, 1000)]
    
    for i in range(data.shape[0]):
        for feature in synth_features.keys():
            lr = int(np.floor(n_steps * (np.random.rand() * length_rate)))
            args = synth_features[feature]
            if feature == 'sin':
                low = np.random.uniform(args[0], args[0]) + np.random.uniform(0, args[2])
                high = np.random.uniform(args[1], args[1]) + np.random.uniform(0, args[2])
                data[i, :, feats_v2.index(feature)] = np.sin(np.linspace(low, high, data.shape[1]))
                
            elif feature == 'cos':
                low = np.random.uniform(args[0], args[0]) + np.random.uniform(0, args[2])
                high = np.random.uniform(args[1], args[1]) + np.random.uniform(0, args[2])
                data[i, :, feats_v2.index(feature)] = (np.cos(np.linspace(low, high, data.shape[1])))
            elif feature == 'tan':
                low = np.random.uniform(args[0], args[0]) + np.random.uniform(0, args[2])
                high = np.random.uniform(args[1], args[1]) + np.random.uniform(0, args[2])
                f1_sin = np.sin(np.linspace(low, high, data.shape[1]))

                low = np.random.uniform(args[0], args[0]) + np.random.uniform(0, args[2])
                high = np.random.uniform(args[1], args[1]) + np.random.uniform(0, args[2])
                f2_cos = (np.cos(np.linspace(low, high, data.shape[1])) ** 2)

                data[i, :, feats_v2.index(feature)] = f1_sin / (f2_cos + 1e-5) #+ np.random.rand(data.shape[1])
            if noise:
                noise_level = np.random.rand(data.shape[1]) * 0.5
                data[i, :, feats_v2.index(feature)] += noise_level


            if feature == 'sin' or feature == 'cos2' or feature == 'tan':
                data[i, :, feats_v2.index(feature)] = add_rn_missing(data[i, :, feats_v2.index(feature)], length=lr)
            
        data[i] = add_rn_missing(data[i], rate=rate)
            # elif feature == 'inv':
            #     data[i, :, feats.index(feature)] = (synth_features['inv'][0] * data[i, :, feats.index('sin')] - synth_features['inv'][1] / data[i, :, feats.index('sin')])
    data_rows = data.reshape((-1, num_features))
    mean = np.nanmean(data_rows, axis=0)
    std = np.nanstd(data_rows, axis=0)
    data = data.reshape((num_seasons, num_steps, num_features))
    return data, mean, std


def create_synthetic_data_v3(n_steps, num_seasons, seed=10, rate=0.05, length_rate=0.02, noise=False):
    # num_seasons = 32
    np.random.seed(seed)
    
    synth_features = {
        'sin': (0.0001, np.pi/3, np.pi/10), # f1
        'cos': (0, np.pi/3, np.pi/10), # f2
        'tan': (0, np.pi/3, np.pi/10) # f3
    }
    num_steps = n_steps # config['n_steps']
    num_features = len(feats_v3) # config['n_features']
    data = np.zeros((num_seasons, num_steps, num_features))
    # value_range = [(0.1, 0.4, 0.7, 0.99), (11.0, 17.5, 40.5, 61.2), (100.1, 160.2, 500, 1000)]
    
    for i in range(data.shape[0]):
        for feature in synth_features.keys():
            lr = int(np.floor(n_steps * (np.random.rand() * length_rate)))
            args = synth_features[feature]
            if feature == 'sin':
                low = np.random.uniform(args[0], args[0]) + np.random.uniform(0, args[2])
                high = np.random.uniform(args[1], args[1]) + np.random.uniform(0, args[2])
                data[i, :, feats_v3.index(feature)] = np.sin(np.linspace(low, high, data.shape[1]))
                
            elif feature == 'cos':
                low = np.random.uniform(args[0], args[0]) + np.random.uniform(0, args[2])
                high = np.random.uniform(args[1], args[1]) + np.random.uniform(0, args[2])
                data[i, :, feats_v3.index(feature)] = (np.cos(np.linspace(low, high, data.shape[1])) ** 2)
            elif feature == 'tan':
                # low = np.random.uniform(args[0], args[0]) + np.random.uniform(0, args[2])
                # high = np.random.uniform(args[1], args[1]) + np.random.uniform(0, args[2])
                # f1_sin = np.sin(np.linspace(low, high, data.shape[1]))

                # low = np.random.uniform(args[0], args[0]) + np.random.uniform(0, args[2])
                # high = np.random.uniform(args[1], args[1]) + np.random.uniform(0, args[2])
                # f2_cos = (np.cos(np.linspace(low, high, data.shape[1])) ** 2)
                f1_sin = data[i, :, feats_v3.index('sin')]
                f2_cos = data[i, :, feats_v3.index('cos')]
                data[i, :, feats_v3.index(feature)] = f1_sin / (f2_cos + 1e-4) #+ np.random.rand(data.shape[1])
            if noise and feature != 'tan':
                noise_level = np.random.rand(data.shape[1]) * 0.5
                data[i, :, feats_v3.index(feature)] += noise_level


            if feature == 'sin' or feature == 'cos' or feature == 'tan':
                data[i, :, feats_v3.index(feature)] = add_rn_missing(data[i, :, feats_v3.index(feature)], length=lr)
            
        data[i] = add_rn_missing(data[i], rate=rate)
            # elif feature == 'inv':
            #     data[i, :, feats.index(feature)] = (synth_features['inv'][0] * data[i, :, feats.index('sin')] - synth_features['inv'][1] / data[i, :, feats.index('sin')])
    data_rows = data.reshape((-1, num_features))
    mean = np.nanmean(data_rows, axis=0)
    std = np.nanstd(data_rows, axis=0)
    data = data.reshape((num_seasons, num_steps, num_features))
    return data, mean, std