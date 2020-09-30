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
            path = "KnowledgeDatabases/DecisionRules/DecisionRulesAlgorithms/KDBAlgorithmsMetadataUnsupervised.json"
        else:
            path = "KnowledgeDatabases/DecisionRules/DecisionRulesAlgorithms/KDBAlgorithmsHardwareUserParamsUnupervised.json"
    else:
        if metadata:
            path = "KnowledgeDatabases/DecisionRules/DecisionRulesAlgorithms/KDBAlgorithmsMetadataSupervised.json"
        else:
            path = "KnowledgeDatabases/DecisionRules/DecisionRulesAlgorithms/KDBAlgorithmsHardwareUserParamsSupervised.json"

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


def derive_scores_from_metadata(knowledge_db, metadata, algorithm_set, supervised):
    # can be modified for testing purposes
    threshold_stepsize = 0.5

    if knowledge_db == {}:
        knowledge_db = read_in_knowledge_db_algorithms(supervised, True)

    weights = knowledge_db["weights"]
    thresholds = dict(list(map(lambda x: [x["attribute"], x["borders"]], knowledge_db["threshold_borders"])))
    decision_rules = knowledge_db["decision_rules"]

    algorithm_scores = dict(list(map(lambda x: [x, 0], algorithm_set)))
    # get selection regarding metadata
    for rule in decision_rules:
        if rule["algorithm"] in algorithm_set:
            datasets_metadata_value = metadata[rule["attribute"]]

            factor = 0
            for i, threshold in enumerate(thresholds[rule["attribute"]]):
                if datasets_metadata_value >= threshold:
                    if i == 0:
                        factor = 1

                    factor += threshold_stepsize
                else:
                    break

            algorithm_scores[rule["algorithm"]] += weights[rule["attribute"]] * factor

    # normalize algorithm_scores
    algorithm_scores = normalize_scores(algorithm_scores, knowledge_db)

    return algorithm_scores


def change_scores_regarding_params_and_hardware_setup(hardware, configuration_parameters, algorithm_set, supervised):
    algorithm_scores = dict(list(map(lambda x: [x, 0], algorithm_set)))

    hardware_attributes = ["high_parallelization", "low_memory_requirement_training", "low_memory_requirement_test"]
    configuration_attributes = ["high_efficiency", "medium_to_high_accuracy", "low_parameter_tuning", "arbitrary_cluster_shape"]

    knowledge_db = read_in_knowledge_db_algorithms(supervised, False)

    weights = knowledge_db["weights"]
    decision_rules = knowledge_db["decision_rules"]

    for rule in decision_rules:
        if rule["algorithm"] in algorithm_set:
            if rule["attribute"] in configuration_attributes:
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
            elif rule["attribute"] in hardware_attributes:
                if rule["attribute"] == "high_parallelization":
                    if hardware["many_cpu_threads"]:
                        algorithm_scores[rule["algorithm"]] += weights[rule["attribute"]]
                elif rule["attribute"] == "low_memory_requirement_training":
                    if hardware["high_ram_amount"]:
                        algorithm_scores[rule["algorithm"]] += weights[rule["attribute"]]
                elif rule["attribute"] == "low_memory_requirement_test":
                    if hardware["high_ram_amount"]:
                        algorithm_scores[rule["algorithm"]] += weights[rule["attribute"]]
            else:
                raise RuntimeError("Unknown decision rule attribute for hardware / user parameter KDB for algorithms!")

    # normalize algorithm_scores
    algorithm_scores = normalize_scores(algorithm_scores, knowledge_db)

    return algorithm_scores


def select_algorithm(algorithm_set, metadata, hardware, configuration_parameters, knowledge_db_metadata, supervised=False):
    algorithm_scores_metadata = derive_scores_from_metadata(knowledge_db_metadata, metadata, algorithm_set, supervised)
    algorithm_scores_params_and_hardware = change_scores_regarding_params_and_hardware_setup(hardware, configuration_parameters, algorithm_set, supervised)

    # sum up algorithm_scores_metadata and algorithm_scores_params_and_hardware
    algorithm_scores_total = algorithm_scores_metadata
    for (algorithm, score) in algorithm_scores_params_and_hardware.items():
        algorithm_scores_total[algorithm] += score

    algorithm_scores_list = list(zip(algorithm_scores_total.keys(), algorithm_scores_total.values()))
    best_selection = max(algorithm_scores_list, key=lambda x: x[1])

    return best_selection[0], algorithm_scores_total, knowledge_db_metadata
