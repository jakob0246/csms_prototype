import json
from functools import reduce


def tune_k():
    # TODO

    return 2


def read_in_knowledge_db_json():
    path = "./KnowledgeDatabases/DecisionRules/KDBDistanceMetrics.json"

    file = open(path)
    json_data = json.load(file)

    return json_data


def select_distance_metric_from_kdb(metadata, hardware, configuration_parameters, distance_metrics):
    threshold_stepsize = 0.5  # TODO: user-param. or integration into KDB ?

    knowledge_db = read_in_knowledge_db_json()

    weights = knowledge_db["weights"]
    thresholds = dict(list(map(lambda x: [x["attribute"], x["borders"]], knowledge_db["threshold_borders"])))
    decision_rules = knowledge_db["decision_rules"]  # list(map(lambda x: {x["algorithm"]: x["attribute"]}, knowledge_db["decision_rules"]))

    # get selection regarding metadata and hardware
    algorithm_scores = dict(list(map(lambda x: [x, 0], distance_metrics)))
    for rule in decision_rules:
        datasets_metadata_value = metadata[rule["attribute"]]

        factor = 0
        for i, threshold in enumerate(thresholds[rule["attribute"]]):
            if datasets_metadata_value >= threshold:
                if i == 0:
                    factor = 1

                factor += threshold_stepsize
            else:
                break

        algorithm_scores[rule["metric"]] += weights[rule["attribute"]] * factor

    # TODO: take configuration_parameters into consideration

    algorithm_scores_list = list(zip(algorithm_scores.keys(), algorithm_scores.values()))

    scores_sum = reduce(lambda a, b: a + b, list(algorithm_scores.values()))
    if scores_sum != 0:
        best_selection = max(algorithm_scores_list, key=lambda x: x[1])[0]
    else:
        best_selection = None

    return best_selection


def select_distance_metric(metadata, hardware, configuration_parameters, algorithm):
    supported_distance_metrics = {
        "kmeans": ["euclidean"],
        "dbscan": ["euclidean", "manhattan", "minkowski_fractional", "minkowski_other", "cosine", "mahalanobis", "canberra", "jensen_shannon"],
        "optics": ["euclidean", "manhattan", "minkowski_fractional", "minkowski_other", "cosine", "mahalanobis", "canberra"],
        "agglomerative": ["euclidean", "manhattan", "cosine"],
        "knn": ["euclidean", "manhattan", "minkowski_fractional", "minkowski_other", "mahalanobis", "canberra"],
        "nearest_centroid": ["euclidean", "manhattan", "minkowski_fractional", "minkowski_other", "cosine", "mahalanobis", "canberra", "jensen_shannon"],
        "radius_neighbors": ["euclidean", "manhattan", "minkowski_fractional", "minkowski_other", "mahalanobis", "canberra"],
        "nca": ["euclidean", "manhattan", "minkowski_fractional", "minkowski_other", "mahalanobis", "canberra"]
    }

    select_distance_metric_from_kdb(metadata, hardware, configuration_parameters, supported_distance_metrics[algorithm])

    # TODO: "Compact or isolated clusters" as system configuration parameter;
    #       "Ignore Magnitude and Rotation", "Measure Distribution Differences", "Grid based distance" too ?


def kmeans_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters):
    parameters = {
        "k": tune_k()
    }

    return parameters


def em_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters):
    parameters = {
        "k": tune_k()
    }

    return parameters


def spectral_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters):
    parameters = {
        "k": tune_k(),
    }

    return parameters


def meanshift_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters):
    parameters = {
        "k": tune_k(),
    }

    return parameters


def optics_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters):
    parameters = {
        "k": tune_k(),
        "distance_metric": select_distance_metric(metadata, hardware, configuration_parameters, "optics")
    }

    return parameters


def dbscan_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters):
    parameters = {
        "min_samples": 10,  # TODO
        "distance": "euclidean",  # TODO
        "epsilon": 10000.0  # TODO
    }

    return parameters


def agglomerative_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters):
    parameters = {
        "k": tune_k(),
    }

    return parameters


def affinity_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters):
    parameters = {
    }

    return parameters


def vbgmm_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters):
    parameters = {
        "max_k": 100,  # TODO
    }

    return parameters


def knn_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters):
    parameters = {
        "k": 10,  # TODO
        "distance": "minkowski"  # TODO
    }

    return parameters


def svc_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters):
    parameters = {
        "degree": 3,  # TODO
        "gamma": "auto"  # TODO
    }

    return parameters


def svc_sgd_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters):
    parameters = {
    }

    return parameters


def nearest_centroid_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters):
    parameters = {
        "distance": "euclidean"  # TODO
    }

    return parameters


def radius_neighbors_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters):
    parameters = {
        "radius": 10000000.0,  # TODO
        "distance": "minkowski"  # TODO
    }

    return parameters


def nca_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters):
    parameters = {
        "k": 10,  # TODO
        "distance": "minkowski"  # TODO
    }

    return parameters


def tune_parameters(algorithm, metadata, hardware, configuration_parameters, n_clusters=2):
    # TODO

    # algorithm = "optics"

    # TODO: more algorithms
    parameters = {}
    if algorithm == "kmeans":
        parameters = kmeans_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters)
    elif algorithm == "em":
        parameters = em_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters)
    elif algorithm == "spectral":
        parameters = spectral_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters)
    elif algorithm == "dbscan":
        parameters = dbscan_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters)
    elif algorithm == "optics":
        parameters = optics_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters)
    elif algorithm == "meanshift":
        parameters = meanshift_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters)
    elif algorithm == "agglomerative":
        parameters = agglomerative_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters)
    elif algorithm == "affinity":
        parameters = affinity_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters)
    elif algorithm == "vbgmm":
        parameters = vbgmm_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters)
    elif algorithm == "knn":
        parameters = knn_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters)
    elif algorithm == "svc":
        parameters = svc_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters)
    elif algorithm == "svc_sgd":
        parameters = svc_sgd_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters)
    elif algorithm == "nearest_centroid":
        parameters = nearest_centroid_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters)
    elif algorithm == "radius_neighbors":
        parameters = radius_neighbors_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters)
    elif algorithm == "nca":
        parameters = nca_parameter_tuning(metadata, hardware, configuration_parameters, n_clusters)

    clustering_parameters = {"kmeans": {"n_clusters": n_clusters},
                             "em": {"n_clusters": n_clusters},
                             "spectral": {"n_clusters": n_clusters},
                             "dbscan": {"min_samples": 1, "epsilon": 100000000.0},  # TODO
                             "optics": {"min_samples": 10},  # TODO
                             "meanshift": {},
                             "agglomerative": {"n_clusters": n_clusters},
                             "affinity": {},
                             "vbgmm": {"max_n_components": 100}}  # n_components != n_clusters, because VBGMM determines n. clusters < n_comp. by itself

    return parameters
