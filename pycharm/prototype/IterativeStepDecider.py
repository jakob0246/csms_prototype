from functools import reduce

import numpy as np


def determine_next_best_algorithm(selected_algorithm, algorithm_scores):
    algorithm_scores_list = list(zip(algorithm_scores.keys(), algorithm_scores.values()))

    algorithm_scores_list.sort(key=lambda x: x[1])
    next_best_algorithm_index = list(map(lambda x: x[0], algorithm_scores_list)).index(selected_algorithm) - 1

    next_best_algorithm = algorithm_scores_list[next_best_algorithm_index][0]

    return next_best_algorithm


def determine_zero_scores_percentage(algorithm_scores):
    just_the_scores = list(algorithm_scores.values())
    zeroes_count = reduce(lambda x, y: x + y, (np.array(just_the_scores) == 0).astype(int))
    percentage = zeroes_count / len(algorithm_scores)

    return percentage


def update_algorithms_knowledge_db(algorithms_knowledge_db, kdb_updating_factor):
    new_algorithms_knowledge_db = algorithms_knowledge_db.copy()

    for borders in new_algorithms_knowledge_db["threshold_borders"]:
        borders["borders"] = list(np.array(borders["borders"]) * kdb_updating_factor)

    return new_algorithms_knowledge_db


def decide_iterative_step(main_iteration, evaluation_results, algorithm_scores, algorithms_knowledge_db, history, kdb_update_count, old_selected_algorithm, max_iterations, supervised=False):
    algorithms_knowledge_db_updated = algorithms_knowledge_db.copy()

    if supervised:
        high_score_threshold = 0.95
        previous_score_threshold = 0.05
    else:
        high_score_threshold = 0.9
        previous_score_threshold = 0.05

    zero_scores_percentage_threshold = 0.2  # TODO: change regarding user preference / determination ?
    kdb_updating_factor = 0.8  # TODO: change regarding user preference / determination ?
    maximum_kdb_updating_boundary = 0.5  # TODO: change regarding user preference / determination ?

    zero_scores_percentage = determine_zero_scores_percentage(algorithm_scores)

    next_iterative_state = None
    decided_algorithm = None
    best_algorithms_parameters = None
    selected_next_best_algorithm = False
    updated_kdb = False

    # traverse decision tree:
    go_to_many_zeroes_in_scores_branch = False
    if (supervised and evaluation_results["accuracy"] > high_score_threshold) or ((not supervised) and evaluation_results["silhouette_score_standardized"] > high_score_threshold):
        next_iterative_state = "stop"
    else:
        if main_iteration == (max_iterations - 1):  # TODO: no iterations left
            next_iterative_state = "stop"

            best_algorithm = None
            best_result = None
            for history_element in history:
                got_new_best_algorithm = False
                if supervised:
                    got_new_best_algorithm = best_result is None or history_element["results"]["accuracy"] >= best_result
                else:
                    got_new_best_algorithm = best_result is None or history_element["results"]["silhouette_score_standardized"] >= best_result

                if got_new_best_algorithm:
                    best_algorithm = history_element["algorithm"]
                    best_algorithms_parameters = history_element["parameters"]
                    best_result = history_element["results"]["accuracy"] if supervised else history_element["results"]["silhouette_score_standardized"]

            decided_algorithm = best_algorithm
        else:
            if main_iteration == 0:
                go_to_many_zeroes_in_scores_branch = True
            else:
                if (supervised and history[main_iteration - 1]["selected_next_best_algorithm"] and evaluation_results["accuracy"] + previous_score_threshold < history[main_iteration - 1]["results"]["accuracy"]) or \
                   ((not supervised) and history[main_iteration - 1]["selected_next_best_algorithm"] and evaluation_results["silhouette_score_standardized"] + previous_score_threshold < history[main_iteration - 1]["results"]["silhouette_score_standardized"]):  # TODO
                    next_iterative_state = "algorithm_selection"
                    decided_algorithm = determine_next_best_algorithm(old_selected_algorithm, algorithm_scores)
                else:
                    go_to_many_zeroes_in_scores_branch = True

    if go_to_many_zeroes_in_scores_branch:
        if zero_scores_percentage > zero_scores_percentage_threshold and (kdb_updating_factor ** kdb_update_count) > maximum_kdb_updating_boundary:
            next_iterative_state = "algorithm_selection"
            algorithms_knowledge_db_updated = update_algorithms_knowledge_db(algorithms_knowledge_db, kdb_updating_factor)
            kdb_update_count += 1
            updated_kdb = True
        else:
            next_iterative_state = "parameter_tuning"
            decided_algorithm = determine_next_best_algorithm(old_selected_algorithm, algorithm_scores)
            selected_next_best_algorithm = True

    new_history_element = {
        "algorithm": None,
        "parameters": None,
        "results": None,
        "next_iterative_step_decision": next_iterative_state,
        "selected_next_best_algorithm": selected_next_best_algorithm,
        "updated_kdb": updated_kdb
    }
    history.append(new_history_element)

    return next_iterative_state, algorithms_knowledge_db_updated, history, decided_algorithm, kdb_update_count, best_algorithms_parameters
