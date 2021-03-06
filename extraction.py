from __future__ import print_function

import csv
import numpy as np
from six.moves import cPickle as pickle

np.random.seed(1337)  # for reproducibility


def get_features_for_row(row):
    track_feature_array = [row[0], row[1], row[2], row[3:36], row[37], row[38:71], row[72], row[73], row[74],
                           row[75], row[76], row[77:100], row[101], row[102:125], row[126], row[127:146],
                           row[167:190]]
    width = len(max(track_feature_array, key=len))
    height = 17
    zero_array = np.zeros((height, width))

    for index, arr in enumerate(track_feature_array):
        if isinstance(arr, list):
            newarr = np.asarray(map(float, arr))
            newarr.resize(width)
            zero_array[index, :] = newarr
        else:
            newarr = np.asarray(float(arr))
            np.pad(newarr, [0, width - newarr.size], mode='constant')
            zero_array[index, :] = newarr

    np.concatenate(track_feature_array, axis=0)
    zero_array[:track_feature_array.shape[0], :track_feature_array.shape[1]] = track_feature_array
    return track_feature_array


def load_features(filename):
    with open(filename, 'rb') as csvfile:
        next(csvfile)  # skip header
        data_features = []
        data_labels = []
        reader = csv.reader(csvfile)
        for row in reader:
            track_features = get_features_for_row(row)
            data_features.append(track_features)  # features
            data_labels.append(row[-1])  # genre

        return data_features, data_labels


def persist_extracted_features(train_features, test_features, train_labels, test_labels):
    try:
        f = open('data/input.pickle', 'wb')
        save = {
            'train_data': train_features,
            'test_data': test_features,
            'test_labels': test_labels,
            'train_labels': train_labels
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', 'data/input.pickle', ':', e)
        raise


def evens(dataset):
    return dataset[::2]


def odds(dataset):
    return dataset[1::2]


def map_classes(labels, label_map):
    mappings_rev = {v: k for k, v in label_map.items()}
    return map(lambda x: mappings_rev[x], labels)


train_data_file = 'data/genresTrain.csv'
test_data_file = 'data/genresTest.csv'

train_features, train_labels = load_features(train_data_file)
mappings = dict(enumerate(set(train_labels)))
train_label_mapping = map_classes(train_labels, mappings)

test_features = evens(train_features)
train_features = odds(train_features)
train_labels = odds(train_label_mapping)
test_labels = evens(train_label_mapping)

persist_extracted_features(train_features, test_features, train_labels, test_labels)
