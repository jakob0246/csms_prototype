import pandas as pd
import numpy as np

from sklearn.neighbors import LocalOutlierFactor

from scipy.stats import shapiro, normaltest, anderson

import time


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

        # statistics, p = shapiro(column)
        # results_array[0, i] = p > alpha

        statistics, p = normaltest(column)
        results_array[i] = p > alpha

    # results_dataframe = pd.DataFrame(results_array, columns=dataframe.columns, index=["shapiro", "d_agostino"])

    return results_array


def determine_outliers(dataframe):
    classifier = LocalOutlierFactor()
    y_pred = classifier.fit_predict(dataframe)
    n_outliers = np.unique(y_pred, return_counts=True)[1][0]

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

    # describe
    pandas_data_profile_num = dataframe_initial_missing_class.describe()

    pandas_data_profile_cat = pd.DataFrame()
    if not dataframe_initial_missing_class.select_dtypes(include=['category']).empty:
        pandas_data_profile_cat = dataframe_initial_missing_class.describe(include=["category"])

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

    data_profile = {
        "dtypes": dataframe_initial_missing_class.dtypes,
        "n_rows": dataframe_missing_class.shape[0],
        "n_features": dataframe_missing_class.shape[1],
        "nvalues": dataframe_missing_class.count(),
        "nmissing_values": pd.Series(((np.ones(dataframe_initial_missing_class.shape[1]) * dataframe_initial_missing_class.shape[0]) - dataframe_initial_missing_class.count().values).astype(int), index=dataframe_initial_missing_class.columns),
        "correlation": dataframe_missing_class.corr(),
        "covariance": dataframe_missing_class.cov(),
        "min": pandas_data_profile_num.T["min"].copy(),
        "max": pandas_data_profile_num.T["max"].copy(),
        "mean": pandas_data_profile_num.T["mean"].copy(),
        "std_deviation": pandas_data_profile_num.T["std"].copy(),
        "25_percentile": pandas_data_profile_num.T["25%"].copy(),
        "50_percentile": pandas_data_profile_num.T["50%"].copy(),
        "75_percentile": pandas_data_profile_num.T["75%"].copy(),
        "unique": None if pandas_data_profile_cat.empty else pandas_data_profile_cat.T["unique"].copy(),
        "top": None if pandas_data_profile_cat.empty else pandas_data_profile_cat.T["top"].copy(),
        "freq": None if pandas_data_profile_cat.empty else pandas_data_profile_cat.T["freq"].copy(),
        "outlier_percentage": n_outliers / dataframe_missing_class.shape[0],
        "normal_distribution_percentage": sum(distributions[1, ...]) / dataframe_missing_class.shape[1],
        "high_correlation_percentage": correlation_percentage

        # TODO

    }

    profiling_time_end = time.time()
    profiling_cputime_end = time.process_time()
    number_of_decimals = 6
    print("[Data Profiler] Data profiled! Took " + str(np.around(profiling_time_end - profiling_time_start, decimals=number_of_decimals)) + "s (Real time); " + str(np.around(profiling_cputime_end - profiling_cputime_start, decimals=number_of_decimals)) + "s (CPU time)")

    return data_profile
