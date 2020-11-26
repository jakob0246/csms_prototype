import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold


def drop_id_features(dataset):
    for column in range(dataset.shape[1]):
        if len(dataset[..., column].unique()) == dataset.shape[0]:
            dataset = np.delete(dataset, column, 1)

    return dataset


# remove features with same values (variance = 0):
def drop_constant_features(dataset):
    selector = VarianceThreshold(threshold=0.0)
    dataset_array = selector.fit_transform(dataset)

    indicies = selector.get_support(indices=True)
    dataset_filtered = pd.DataFrame(dataset_array, columns=np.array(dataset.columns)[indicies])

    return dataset_filtered


def simple_automatic_feature_selection(dataset):
    # dataset = pd.DataFrame([[1, 2, 3], [1, 1, 1], [1, 1, 1]], columns=["a", "b", "c"])

    dataset_filtered = dataset.copy()

    # remove features with same values (variance = 0): TODO: categoricals!
    # dataset_filtered = drop_constant_features(dataset_filtered)

    # TODO ... e.g. equidistant spaced, unique features?

    print("[Simple Automatic Feature Selector] Shape before:", dataset.shape, "Shape after:", dataset_filtered.shape)

    return dataset_filtered
