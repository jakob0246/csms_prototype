import json
import itertools
from functools import reduce

import numpy as np

from colorama import Fore
from colorama import Style

from ProgramGeneratorAndEvaluator import generate_and_evaluate_program


def read_in_knowledge_db_json():
    # TODO: check structure

    # information about deriving decision rules from the thesis:
    # - added decision rule for "n_features" if "scalability regarding number of dimensions" was "high" or "very high"
    # - left out much variance, since it can't be computed in a comparative way

    path = "KnowledgeDatabases/DecisionRules/DecisionRulesDistanceMetrics/KDBDistanceMetricsMetadata.json"

    file = open(path)
    json_data = json.load(file)

    return json_data


def read_in_hyper_parameter_config():
    # TODO: check structure

    path = "Configs/HyperParameterConfig.json"

    file = open(path)
    json_data = json.load(file)

    return json_data


def normalize_metric_scores(metric_scores, knowledge_db):
    number_of_decision_rules = len(knowledge_db["decision_rules"])

    for (key, value) in metric_scores.items():
        number_of_algorithm_rules = np.sum(np.array(list(map(lambda x: x["metric"], knowledge_db["decision_rules"]))) == key)
        number_of_other_algorithm_rules = number_of_decision_rules - number_of_algorithm_rules

        metric_scores[key] = value * ((number_of_other_algorithm_rules) / number_of_decision_rules)

    return metric_scores


def select_distance_metric_from_kdb(metadata, hardware, configuration_parameters, distance_metrics):
    threshold_stepsize = 0.5  # TODO: user-param. or integration into KDB ?

    knowledge_db = read_in_knowledge_db_json()

    weights = knowledge_db["weights"]
    thresholds = dict(list(map(lambda x: [x["attribute"], x["borders"]], knowledge_db["threshold_borders"])))
    decision_rules = knowledge_db["decision_rules"]

    # get selection regarding metadata and hardware
    metric_scores = dict(list(map(lambda x: [x, 0], distance_metrics)))
    for rule in decision_rules:
        if rule["metric"] in metric_scores.keys():
            datasets_metadata_value = metadata[rule["attribute"]]

            factor = 0
            for i, threshold in enumerate(thresholds[rule["attribute"]]):
                if datasets_metadata_value >= threshold:
                    if i == 0:
                        factor = 1

                    factor += threshold_stepsize
                else:
                    break

            metric_scores[rule["metric"]] += weights[rule["attribute"]] * factor

    # TODO: take configuration_parameters into consideration

    metric_scores = normalize_metric_scores(metric_scores, knowledge_db)

    metric_scores_list = list(zip(metric_scores.keys(), metric_scores.values()))

    scores_sum = reduce(lambda a, b: a + b, list(metric_scores.values()))
    if scores_sum != 0:
        best_selection = max(metric_scores_list, key=lambda x: x[1])[0]
    else:
        print(f"{Fore.YELLOW}[Parameter Tuner] <Warning> Selected arbitrary distance metric, because the evaluation of all decision rules and algorithms was 0.{Style.RESET_ALL}")
        best_selection = distance_metrics[0]

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

    return select_distance_metric_from_kdb(metadata, hardware, configuration_parameters, supported_distance_metrics[algorithm])

    # TODO: "Compact or isolated clusters" as system configuration parameter;
    #       "Ignore Magnitude and Rotation", "Measure Distribution Differences", "Grid based distance" too ?


def derive_possible_parameters_equally(grid_search_meta_parameters):
    possible_parameters = []

    current_parameter_value = grid_search_meta_parameters["min_boundary"]
    while current_parameter_value <= grid_search_meta_parameters["max_boundary"]:
        possible_parameters.append((current_parameter_value))

        current_parameter_value += grid_search_meta_parameters["step_value"]

    return possible_parameters


# TODO
def derive_possible_parameters_exponentially(grid_search_meta_parameters):
    possible_parameters = []

    # TODO

    return possible_parameters


def derive_possible_parameters_exponentially_naive(grid_search_meta_parameters):
    possible_parameters = []
    if grid_search_meta_parameters["step_value"] == 1:
        # TODO: put constraints somewhere else?
        assert grid_search_meta_parameters["max_boundary"] == grid_search_meta_parameters["min_boundary"], "If step_value = 1, then max_boundary should be the same as min_boundary!"

        return possible_parameters.append(grid_search_meta_parameters["max_boundary"])
    else:
        # TODO: put constraints somewhere else?
        assert grid_search_meta_parameters["max_boundary"] != grid_search_meta_parameters["min_boundary"], "If step_value != 1, then max_boundary should be different to min_boundary!"

    if grid_search_meta_parameters["step_value"] == 2:
        return possible_parameters.extend([grid_search_meta_parameters["min_boundary"], grid_search_meta_parameters["max_boundary"]])

    for i in range(int(np.log10(grid_search_meta_parameters["min_boundary"])), grid_search_meta_parameters["step_value"]):
        possible_parameters.append(10 ** i)

    return possible_parameters


def grid_search_further_parameters(algorithm, initial_parameters, grid_search_meta_parameters, grid_search_parameters_to_tune, sample_size, class_column, dataset, supervised=False):
    possible_parameters = {}
    for algorithm_parameter in grid_search_parameters_to_tune:
        if grid_search_meta_parameters[algorithm_parameter]["step_distance_strategy"] == "equal":
            possible_parameters[algorithm_parameter] = derive_possible_parameters_equally(grid_search_meta_parameters[algorithm_parameter])
        else:
            possible_parameters[algorithm_parameter] = derive_possible_parameters_exponentially_naive(grid_search_meta_parameters[algorithm_parameter])

    all_possible_parameter_combinations = list(itertools.product(*list(possible_parameters.values())))

    best_parameter_combination = {}
    best_parameter_result = {}
    for parameter_combination in all_possible_parameter_combinations:
        parameters_to_evaluate = dict(list(zip(grid_search_parameters_to_tune, parameter_combination)))

        merged_parameters = {**parameters_to_evaluate, **initial_parameters}
        result = generate_and_evaluate_program(algorithm, merged_parameters, dataset, sample_size, supervised, class_column, sampling=True)

        print("[Parameter Tuner] Tested parameters " + str(parameters_to_evaluate) + " for \"" + algorithm + f"\" and got accuracy of {result['accuracy']:.4f}")

        if not supervised:
            # TODO
            pass
        else:
            if best_parameter_combination == {} or result["accuracy"] >= best_parameter_result["accuracy"]:
                best_parameter_result = result.copy()
                best_parameter_combination = parameters_to_evaluate

    return best_parameter_combination


def tune_parameters(algorithm, metadata, hardware, configuration_parameters, learning_type, dataset, class_column, sample_size):
    # TODO

    all_parameters_to_tune = {
        "unsupervised": {
            "kmeans": ["n_clusters"],
            "em": ["n_clusters"],
            "spectral": ["n_clusters"],
            "dbscan": ["min_samples", "epsilon", "distance"],
            "optics": ["min_samples", "distance"],
            "meanshift": [],
            "agglomerative": ["n_clusters", "distance"],
            "affinity": [],
            "vbgmm": ["max_n_components"]
        },
        "supervised": {
            "knn": ["k", "distance"],
            "svc": ["degree"],
            "nearest_centroid": ["distance"],
            "radius_neighbors": ["radius", "distance"],
            "nca": ["k", "distance"],
            "svc_sgd": []
        }
    }

    # TODO: in user configuration file & constraints for it
    grid_search_meta_parameters = read_in_hyper_parameter_config()

    initial_parameters = {}

    if "distance" in all_parameters_to_tune[learning_type][algorithm]:
        initial_parameters["distance"] = select_distance_metric(metadata, hardware, configuration_parameters, algorithm)

    grid_search_parameters_to_tune = set(all_parameters_to_tune[learning_type][algorithm]) - set(["distance"])  # TODO
    best_parameter_combination = grid_search_further_parameters(algorithm, initial_parameters, grid_search_meta_parameters[learning_type][algorithm],
                                                                grid_search_parameters_to_tune, sample_size, class_column, dataset, learning_type == "supervised")

    final_parameters = {**initial_parameters, **best_parameter_combination}

    return final_parameters
