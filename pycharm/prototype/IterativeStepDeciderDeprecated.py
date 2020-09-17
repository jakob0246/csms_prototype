from functools import reduce

import numpy as np


def determine_next_best_algorithm(selected_algorithm, algorithm_scores):
    algorithm_scores_list = list(zip(algorithm_scores.keys(), algorithm_scores.values()))

    algorithm_scores_list.sort(key=lambda x: x[1])
    next_best_algorithm_index = list(map(lambda x: x[0], algorithm_scores_list)).index(selected_algorithm) - 1

    next_best_algorithm = algorithm_scores_list[next_best_algorithm_index][0]

    return next_best_algorithm


def adjust_parameters(algorithm_parameters, parameter_adjustments, negative=False):
    sign = 1
    if negative:
        sign = -1

    for parameter in parameter_adjustments.keys():
        algorithm_parameters[parameter] += (sign * parameter_adjustments[parameter])

    return algorithm_parameters


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


def determine_similar_scores(selected_algorithm, algorithm_history, algorithm_scores):
    same_score_algorithms = set(np.array(algorithm_scores.keys())[(np.array(algorithm_scores.values() == algorithm_scores[selected_algorithm])).astype(int)])
    same_score_algorithms_unevaluated = same_score_algorithms - set(algorithm_history)

    same_score_count = len(same_score_algorithms_unevaluated)

    return same_score_count


def select_same_score_algorithm(selected_algorithm, algorithm_history, algorithm_scores):
    same_score_algorithms = set(np.array(algorithm_scores.keys())[(np.array(algorithm_scores.values() == algorithm_scores[selected_algorithm])).astype(int)])
    same_score_algorithms_unevaluated = same_score_algorithms - set(algorithm_history)

    new_algorithm = same_score_algorithms_unevaluated[0]

    return new_algorithm


def decide_iterative_step(iteration, results, previous_results, selected_algorithm, algorithm_scores, algorithm_parameters,
                          algorithms_knowledge_db, state_history, algorithm_history, adj_parameters_count, other_algorithm_count,
                          kdb_update_count):
    # TODO

    algorithms_knowledge_db_updated = algorithms_knowledge_db.copy()
    algorithm_parameters_updated = algorithm_parameters.copy()

    accuracy_upper_bound = 0.95  # TODO: change regarding user preference
    accuracy_lower_bound = 0.7  # TODO: change regarding user preference
    previous_accuracy_threshold = 0.05  # TODO: change regarding user preference / determination ?
    zero_scores_percentage_threshold = 0.2  # TODO: change regarding user preference / determination ?

    kdb_updating_factor = 0.8  # TODO: change regarding user preference / determination ?
    maximum_kdb_updating_boundary = 0.5  # TODO: change regarding user preference / determination ?

    parameter_adjustments = {  # TODO: change regarding user preference / determination ?
        "k": 1,
    }

    zero_scores_percentage = determine_zero_scores_percentage(algorithm_scores)

    # traverse decision tree:
    low_accuracy_branch = False

    next_state = "algorithm_selection"
    stop_csms = False
    if iteration == 0:
        low_accuracy_branch = True
    else:
        if results["accuracy"] + previous_accuracy_threshold < previous_results["accuracy"]:
            if adj_parameters_count == 0:
                if other_algorithm_count == 0 or other_algorithm_count == 1:
                    selected_algorithm = determine_next_best_algorithm(selected_algorithm, algorithm_scores)
                    other_algorithm_count += 1
                elif other_algorithm_count == 2:
                    selected_algorithm = algorithm_history[len(algorithm_history) - 2]
                    stop_csms = True
                adj_parameters_count += 1
            elif adj_parameters_count == 1:
                next_state = "program_generation_and_evaluation"
                algorithm_parameters_updated = adjust_parameters(algorithm_parameters_updated, parameter_adjustments, negative=True)
                adj_parameters_count += 1
            elif adj_parameters_count == 2:
                selected_algorithm = determine_next_best_algorithm(selected_algorithm, algorithm_scores)
                adj_parameters_count = 0
        else:
            low_accuracy_branch = True

    if low_accuracy_branch:
        if results["accuracy"] < accuracy_lower_bound:
            next_state = "program_generation_and_evaluation"
            algorithm_parameters_updated = adjust_parameters(algorithm_parameters_updated, parameter_adjustments)
        else:
            if results["accuracy"] >= accuracy_upper_bound:
                stop_csms = True
            else:
                if zero_scores_percentage > zero_scores_percentage_threshold and (kdb_updating_factor ** kdb_update_count) > maximum_kdb_updating_boundary:
                    next_state = "algorithm_selection"
                    algorithms_knowledge_db_updated = update_algorithms_knowledge_db(algorithms_knowledge_db, kdb_updating_factor)
                    kdb_update_count += 1
                else:
                    similar_scores = determine_similar_scores(selected_algorithm, algorithm_history, algorithm_scores)
                    if similar_scores > 1:
                        next_state = "program_generation_and_evaluation"
                        selected_algorithm = select_same_score_algorithm(selected_algorithm, algorithm_history, algorithm_scores)
                    else:
                        # just choose different sample:
                        next_state = "program_generation_and_evaluation"

    state_history += [next_state]
    algorithm_history += [selected_algorithm]

    return next_state, stop_csms, algorithms_knowledge_db_updated, algorithm_parameters_updated, state_history, algorithm_history, adj_parameters_count, other_algorithm_count, kdb_update_count