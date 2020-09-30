import pandas as pd
import numpy as np

import json


def read_in_knowledge_db_csv(path="./KnowledgeDatabases/KnowledgeDatabaseUnsupervised.csv"):
    dataframe = pd.read_csv(path, delimiter=";")

    # TODO: check integrity of knowledge db (algorithms in set etc.)

    return dataframe


def read_in_knowledge_db_algorithms(supervised):
    if not supervised:
        path = "KnowledgeDatabases/DecisionRules/KDBAlgorithmsUnsupervised.json"
    else:
        path = "KnowledgeDatabases/DecisionRules/KDBAlgorithmsSupervised.json"

    file = open(path)
    json_data = json.load(file)

    return json_data


def select_algorithm_csv(metadata, hardware, configuration_parameters, supported_algorithms=["kmeans", "em"]):
    knowledge_db = read_in_knowledge_db_csv()

    # get selection regarding metadata and hardware
    algorithm_scores = dict(list(map(lambda x: [x, 0], supported_algorithms)))
    for i, row in knowledge_db.iterrows():
        datasets_metadata_value = metadata[row["attribute"]]

        if row["higher/smaller"] == "higher":
            if datasets_metadata_value >= row["threshold"]:
                algorithm_scores[row["algorithm"]] += row["weight"]
        else:
            if datasets_metadata_value < row["threshold"]:
                algorithm_scores[row["algorithm"]] += row["weight"]

    # TODO: take configuration_parameters into consideration

    best_selection = max(list(algorithm_scores), key=lambda x: x[1])

    return best_selection


def normalize_algorithm_scores(algorithm_scores, knowledge_db):
    number_of_decision_rules = len(knowledge_db["decision_rules"])

    for (key, value) in algorithm_scores.items():
        number_of_algorithm_rules = np.sum(np.array(list(map(lambda x: x["algorithm"], knowledge_db["decision_rules"]))) == key)
        number_of_other_algorithm_rules = number_of_decision_rules - number_of_algorithm_rules

        algorithm_scores[key] = value * ((number_of_other_algorithm_rules) / number_of_decision_rules)

    return algorithm_scores


def select_algorithm(algorithm_set, metadata, hardware, configuration_parameters, knowledge_db, supervised=False):
    if not supervised:
        metadata_attributes = ["outlier_percentage", "n_rows", "n_features", "normal_distribution_percentage"]
        hardware_attributes = []
        configuration_attributes = ["efficient", "accurate"]
    else:
        metadata_attributes = ["outlier_percentage", "n_rows", "n_features", "normal_distribution_percentage"]
        hardware_attributes = []
        configuration_attributes = ["efficient_trainingtime", "accurate_trainingtime", "efficient_testtime", "accurate_testtime"]

    threshold_stepsize = 0.5  # TODO: user-param. or integration into KDB ?

    if knowledge_db == {}:
        knowledge_db = read_in_knowledge_db_algorithms(supervised)

    weights = knowledge_db["weights"]
    thresholds = dict(list(map(lambda x: [x["attribute"], x["borders"]], knowledge_db["threshold_borders"])))
    decision_rules = knowledge_db["decision_rules"]  # list(map(lambda x: {x["algorithm"]: x["attribute"]}, knowledge_db["decision_rules"]))

    # get selection regarding metadata and hardware
    algorithm_scores = dict(list(map(lambda x: [x, 0], algorithm_set)))
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

    # TODO: take configuration_parameters into consideration

    # normalize algorithm_scores
    algorithm_scores = normalize_algorithm_scores(algorithm_scores, knowledge_db)

    algorithm_scores_list = list(zip(algorithm_scores.keys(), algorithm_scores.values()))

    best_selection = max(algorithm_scores_list, key=lambda x: x[1])

    return best_selection[0], algorithm_scores, knowledge_db
