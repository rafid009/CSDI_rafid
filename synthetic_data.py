import numpy as np

def create_synthetic_data(n_steps, num_seasons):
    # num_seasons = 32
    feats = ['sin', 'cos2', 'harmonic', 'weight', 'inv']
    synth_features = {
        'sin': (0.00001, np.pi/2, np.pi/3),
        'cos2': (np.pi/4, 2 * np.pi, np.pi/4),
        'harmonic': np.pi/2,
        'weight': (0.3, 0.6),
        'inv': (0.2, 0.003)
    }
    num_steps = n_steps # config['n_steps']
    num_features = len(feats) # config['n_features']
    data = np.zeros((num_seasons, num_steps, num_features))
    # value_range = [(0.1, 0.4, 0.7, 0.99), (11.0, 17.5, 40.5, 61.2), (100.1, 160.2, 500, 1000)]
    

    for i in range(data.shape[0]):
        for feature in synth_features.keys():
            args = synth_features[feature]
            if feature == 'sin':
                low = np.random.uniform(args[0], args[0] + args[2])
                high = np.random.uniform(args[1], args[1] + args[2])
                data[i, :, feats.index(feature)] = np.sin(np.linspace(low, high, data.shape[1]))
            elif feature == 'cos2':
                low = np.random.uniform(args[0], args[0] + args[2])
                high = np.random.uniform(args[1], args[1] + args[2])
                data[i, :, feats.index(feature)] = (np.cos(np.linspace(low, high, data.shape[1])) ** 2)
            elif feature == 'harmonic':
                data[i, :, feats.index(feature)] = (synth_features['harmonic'] * data[i, :, feats.index('sin')] + synth_features['harmonic'] * data[i, :, feats.index('cos2')]) / (1/data[i, :, feats.index('sin')] + 1/data[i, :, feats.index('cos2')])
            elif feature == 'weight':
                data[i, :, feats.index(feature)] = (synth_features['weight'][0] * data[i, :, feats.index('sin')] + synth_features['weight'][1] * data[i, :, feats.index('cos2')]) / (1/synth_features['weight'][0] + 1/synth_features['weight'][1])
            elif feature == 'inv':
                data[i, :, feats.index(feature)] = (synth_features['inv'][0] * data[i, :, feats.index('sin')] - synth_features['inv'][1] / data[i, :, feats.index('sin')])
    data_rows = data.reshape((-1, num_features))
    mean = np.mean(data_rows, axis=0)
    std = np.std(data_rows, axis=0)
    data = data.reshape((num_seasons, num_steps, num_features))
    return data, mean, std