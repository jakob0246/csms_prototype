import pandas as pd
import numpy as np

import json


def read_in_knowledge_db_algorithms(supervised, metadata):
    # TODO: check structure

    # information about deriving decision rules from the thesis:
    # - added decision rule for "high_efficiency" if training and test time was true for supervised algorithms
    # - added decision rule for "medium_to_high_accuracy" if accuracy was "medium" or "high"

    if not supervised:
        if metadata:
            path = "KnowledgeDatabases/DecisionRules/DecisionRulesAlgorithms/KDBAlgorithmsMetadataHardwareUnsupervised.json"
        else:
            path = "KnowledgeDatabases/DecisionRules/DecisionRulesAlgorithms/KDBAlgorithmsHardwareUserParamsUnupervised.json"
    else:
        if metadata:
            path = "KnowledgeDatabases/DecisionRules/DecisionRulesAlgorithms/KDBAlgorithmsMetadataHardwareSupervised.json"
        else:
            path = "KnowledgeDatabases/DecisionRules/DecisionRulesAlgorithms/KDBAlgorithmsUserParamsSupervised.json"

    file = open(path)
    json_data = json.load(file)

    return json_data


def normalize_scores(algorithm_scores, knowledge_db):
    number_of_decision_rules = len(knowledge_db["decision_rules"])

    for (key, value) in algorithm_scores.items():
        number_of_algorithm_rules = np.sum(np.array(list(map(lambda x: x["algorithm"], knowledge_db["decision_rules"]))) == key)
        number_of_other_algorithm_rules = number_of_decision_rules - number_of_algorithm_rules

        algorithm_scores[key] = value * ((number_of_other_algorithm_rules) / number_of_decision_rules)

    return algorithm_scores


def derive_scores_from_metadata_and_hardware(knowledge_db, metadata, hardware, algorithm_set, supervised):
    # can be modified for testing purposes
    threshold_stepsize = 0.5

    hardware_attributes_unsupervised = ["high_parallelization", "low_memory_requirement"]
    hardware_attributes_supervised = ["high_parallelization", "low_memory_requirement_training", "low_memory_requirement_test"]

    hardware_attributes = hardware_attributes_supervised if supervised else hardware_attributes_unsupervised

    if knowledge_db == {}:
        knowledge_db = read_in_knowledge_db_algorithms(supervised, True)

    weights = knowledge_db["weights"]
    thresholds = dict(list(map(lambda x: [x["attribute"], x["borders"]], knowledge_db["threshold_borders"])))
    decision_rules = knowledge_db["decision_rules"]

    algorithm_scores = dict(list(map(lambda x: [x, 0], algorithm_set)))
    # get selection regarding metadata
    for rule in decision_rules:
        if rule["algorithm"] in algorithm_set:
            # reverse way to iterate over thresholds if attribute is memory requirement
            reverse_sign = False

            if rule["attribute"] in hardware_attributes:
                metadata_hardware_value = hardware["cpu_threads"] if rule["attribute"] == "high_parallelization" else hardware["ram"]

                if rule["attribute"] != "high_parallelization":
                    reverse_sign = True
            else:
                metadata_hardware_value = metadata[rule["attribute"]]

            factor = 0
            for i, threshold in enumerate(thresholds[rule["attribute"]]):
                increase_factor = metadata_hardware_value >= threshold if not reverse_sign else metadata_hardware_value <= threshold
                if increase_factor:
                    if i == 0:
                        factor = 1

                    factor += threshold_stepsize
                else:
                    break

            algorithm_scores[rule["algorithm"]] += weights[rule["attribute"]] * factor

    # normalize algorithm_scores
    algorithm_scores = normalize_scores(algorithm_scores, knowledge_db)

    return algorithm_scores


def derive_scores_from_user_parameters(configuration_parameters, algorithm_set, supervised):
    algorithm_scores = dict(list(map(lambda x: [x, 0], algorithm_set)))

    configuration_attributes = ["high_efficiency", "medium_to_high_accuracy", "low_parameter_tuning", "arbitrary_cluster_shape"]

    knowledge_db = read_in_knowledge_db_algorithms(supervised, False)

    weights = knowledge_db["weights"]
    decision_rules = knowledge_db["decision_rules"]

    for rule in decision_rules:
        if rule["algorithm"] in algorithm_set:
            if rule["attribute"] == "arbitrary_cluster_shape":
                if configuration_parameters["system_parameters"]["prefer_finding_arbitrary_cluster_shapes"]:
                    algorithm_scores[rule["algorithm"]] += weights[rule["attribute"]]
            elif rule["attribute"] == "low_parameter_tuning":
                if configuration_parameters["system_parameters"]["avoid_high_effort_of_hyper_parameter_tuning"]:
                    algorithm_scores[rule["algorithm"]] += weights[rule["attribute"]]
            else:
                if configuration_parameters["system_parameters"]["accuracy_efficiency_preference"] == "accuracy":
                    if rule["attribute"] == "medium_to_high_accuracy":
                        algorithm_scores[rule["algorithm"]] += weights[rule["attribute"]]
                else:
                    if rule["attribute"] == "high_efficiency":
                        algorithm_scores[rule["algorithm"]] += weights[rule["attribute"]]

    # normalize algorithm_scores
    algorithm_scores = normalize_scores(algorithm_scores, knowledge_db)

    return algorithm_scores


def select_algorithm(algorithm_set, metadata, hardware, configuration_parameters, knowledge_db_metadata, supervised=False):
    algorithm_scores_metadata_and_hardware = derive_scores_from_metadata_and_hardware(knowledge_db_metadata, metadata, hardware, algorithm_set, supervised)
    algorithm_scores_parameter = derive_scores_from_user_parameters(configuration_parameters, algorithm_set, supervised)

    # sum up algorithm_scores_metadata and algorithm_scores_params_and_hardware
    algorithm_scores_total = algorithm_scores_metadata_and_hardware
    for (algorithm, score) in algorithm_scores_parameter.items():
        algorithm_scores_total[algorithm] += score

    algorithm_scores_list = list(zip(algorithm_scores_total.keys(), algorithm_scores_total.values()))
    best_selection = max(algorithm_scores_list, key=lambda x: x[1])

    return best_selection[0], algorithm_scores_total, knowledge_db_metadata
