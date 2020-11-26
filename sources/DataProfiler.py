import pandas as pd
import numpy as np

from sklearn.neighbors import LocalOutlierFactor

from scipy.stats import shapiro, normaltest, anderson

import time


def determine_class_std_deviation(dataframe, class_column):
    values_per_class = dataframe[class_column].value_counts()
    std_deviation = values_per_class.std()

    return std_deviation


def determine_normal_distributions(dataframe):
    # drop categoricals: (because ordinal values arent scope of thesis? and nominal values cant be modelled after a distribution)
    columns_to_drop = dataframe.select_dtypes(include=['category']).columns
    dataframe = dataframe.drop(columns=columns_to_drop)

    # TODO: can be removed in the future
    dataframe = dataframe.dropna()

    alpha = 0.05

    results_array = np.zeros((dataframe.shape[1], 1))
    for i, column_name in enumerate(dataframe.columns):
        column = np.array(dataframe[column_name])

        statistics, p = normaltest(column)
        results_array[i] = p > alpha

    return results_array


def determine_outliers(dataframe):
    # use sampling (of size sqrt(number of rows)) if dataframe is too big (> 10000 rows):
    sampling = False
    if dataframe.shape[0] > 10000:
        dataframe = dataframe.sample(n=int(dataframe.shape[0] ** (1 / 2)))
        sampling = True

    classifier = LocalOutlierFactor(algorithm="kd_tree")
    y_pred = classifier.fit_predict(dataframe)
    n_outliers = np.unique(y_pred, return_counts=True)[1][0]

    # normalize number of outliers to dataframe size, to have an equal comparison
    if sampling:
        n_outliers ** 2

    return n_outliers


def profile_data(dataframe_intitial, dataframe, class_column, supervised):
    # TODO: check if Unix or Windows for CPU time, because clock() does not return CPU time for Windows, it returns the normal time

    print("[Data Profiler] Profiling the data ...")

    profiling_time_start = time.time()
    profiling_cputime_start = time.process_time()

    dataframe_missing_class = dataframe.copy()
    dataframe_initial_missing_class = dataframe_intitial.copy()
    if supervised:
        dataframe_missing_class = dataframe_missing_class.drop(columns=[class_column])
        dataframe_initial_missing_class = dataframe_initial_missing_class.drop(columns=[class_column])

    # get outliers:
    n_outliers = determine_outliers(dataframe_missing_class)

    # get singlevariate normal distributions:
    distributions = determine_normal_distributions(dataframe_initial_missing_class)

    # compute high pairwise correlation percentage:
    correlation_threshold = 0.66
    corr_matrix = np.array(dataframe_missing_class.corr())
    n_high_correlations = 0
    for i in range(0, corr_matrix.shape[0] - 1):
        for j in range(i + 1, corr_matrix.shape[0]):
            if corr_matrix[i, j] >= correlation_threshold or corr_matrix[i, j] <= (-1) * correlation_threshold:
                n_high_correlations += 1
    correlation_percentage = n_high_correlations / (0.5 * (corr_matrix.shape[0] ** 2 - corr_matrix.shape[0]))

    if supervised:
        class_std_deviation = determine_class_std_deviation(dataframe, class_column)
    else:
        class_std_deviation = None

    data_profile = {
        "dtypes": dataframe_initial_missing_class.dtypes,
        "n_rows": dataframe_missing_class.shape[0],
        "n_features": dataframe_missing_class.shape[1],
        "n_classes": 0 if class_column == "" or class_column not in dataframe_intitial.columns else len(dataframe[class_column].unique()),
        "nvalues": dataframe_missing_class.count(),
        "nmissing_values": np.sum(pd.Series(((np.ones(dataframe_initial_missing_class.shape[1]) * dataframe_initial_missing_class.shape[0]) - dataframe_initial_missing_class.count().values).astype(int), index=dataframe_initial_missing_class.columns)),
        "correlation": dataframe_missing_class.corr(),
        "covariance": dataframe_missing_class.cov(),
        "outlier_percentage": n_outliers / dataframe_missing_class.shape[0],
        "normal_distribution_percentage": (sum(distributions)[0] / dataframe_missing_class.shape[1]) if len(distributions) != 0 else 0,
        "high_correlation_percentage": correlation_percentage,
        "class_std_deviation": class_std_deviation
    }

    profiling_time_end = time.time()
    profiling_cputime_end = time.process_time()
    number_of_decimals = 6
    print("[Data Profiler] Data profiled! Took " + str(np.around(profiling_time_end - profiling_time_start, decimals=number_of_decimals)) + "s (Real time); " + str(np.around(profiling_cputime_end - profiling_cputime_start, decimals=number_of_decimals)) + "s (CPU time)")

    return data_profile
