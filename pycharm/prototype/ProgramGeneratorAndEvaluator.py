from DataClustererSupervised import *
from DataClustererUnsupervised import *
from Helper import print_warning

from sklearn.model_selection import train_test_split
from sklearn import metrics

import pandas as pd


def generate_and_evaluate_program(algorithm, algorithm_parameters, dataset, sample_size, supervised, class_column, sampling=True):
    # choose sample, if wanted:
    dataset_to_evaluate = dataset
    if sampling:
        print("[Generator & Evaluator] Sampling \"" + algorithm + "\" with sample size " + str(sample_size))
        dataset_to_evaluate = dataset.sample(n=sample_size)

    # generate & evaluate program

    if not supervised:
        evaluation_metrics = {  # TODO
            "silhouette_score": None
        }

        if algorithm == "kmeans":
            result_labels = kmeans_clustering(dataset_to_evaluate, algorithm_parameters)
        elif algorithm == "em":
            result_labels = em_clustering(dataset_to_evaluate, algorithm_parameters)
        elif algorithm == "spectral":
            result_labels = spectral_clustering(dataset_to_evaluate, algorithm_parameters)
        elif algorithm == "dbscan":
            result_labels, n_clusters = dbscan_clustering(dataset_to_evaluate, algorithm_parameters)
        elif algorithm == "optics":
            result_labels, n_clusters = optics_clustering(dataset_to_evaluate, algorithm_parameters)
        elif algorithm == "meanshift":
            result_labels, n_clusters = meanshift_clustering(dataset_to_evaluate, algorithm_parameters)
        elif algorithm == "agglomerative":
            result_labels = agglomerative_clustering(dataset_to_evaluate, algorithm_parameters)
        elif algorithm == "affinity":
            result_labels, n_clusters = affinity_clustering(dataset_to_evaluate, algorithm_parameters)
        elif algorithm == "vbgmm":
            result_labels, n_clusters = vbgmm_clustering(dataset_to_evaluate, algorithm_parameters)
        else:
            raise Exception("[Generator & Evaluator] clustering algorithm \"" + algorithm + "\" not supported or spelled incorrectly!")

        evaluation_metrics["silhouette_score"] = metrics.silhouette_score(dataset_to_evaluate, result_labels, metric='euclidean')

        # TODO: accuracy + more
    else:
        evaluation_metrics = {
            "accuracy": None
        }

        X = dataset_to_evaluate.drop(columns=[class_column], axis=1)
        y = dataset_to_evaluate[class_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        if "distance" in algorithm_parameters.keys() and algorithm_parameters["distance"] in ["minkowski_other", "mahalanobis"]:
            try:
                np.linalg.inv(np.cov(X_train))
            except np.linalg.LinAlgError:
                print_warning(f"[Generator & Evaluator] <Warning> Inverse of chosen X_train does not exist (inverse is required for distance {algorithm_parameters['distance']}), setting distance to euclidean ...")
                algorithm_parameters["distance"] == "euclidean"


        if algorithm == "knn":
            evaluation_metrics = knn_clustering(X_train, X_test, y_train, y_test, algorithm_parameters, evaluation_metrics)
        elif algorithm == "svc":
            evaluation_metrics = svc_clustering(X_train, X_test, y_train, y_test, algorithm_parameters, evaluation_metrics)
        elif algorithm == "svc_sgd":
            evaluation_metrics = svc_sdg_clustering(X_train, X_test, y_train, y_test, algorithm_parameters, evaluation_metrics)
        elif algorithm == "nearest_centroid":
            evaluation_metrics = nearest_centroid_clustering(X_train, X_test, y_train, y_test, algorithm_parameters, evaluation_metrics)
        elif algorithm == "radius_neighbors":
            evaluation_metrics = radius_neighbors_clustering(X_train, X_test, y_train, y_test, algorithm_parameters, evaluation_metrics)
        elif algorithm == "nca":
            evaluation_metrics = nca_clustering(X_train, X_test, y_train, y_test, algorithm_parameters, evaluation_metrics)
        else:
            raise Exception("[Generator & Evaluator] clustering algorithm \"" + algorithm + "\" not supported or spelled incorrectly!")

    return evaluation_metrics
