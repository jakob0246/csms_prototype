import itertools

import numpy as np

from Helper import *
from ParameterTunerGridSearch import read_in_hyper_parameter_config, derive_possible_parameters_equally, derive_possible_parameters_exponentially_naive


def determine_num_of_possible_param_combinations(learning_type):
    grid_search_meta_parameters = read_in_hyper_parameter_config()[learning_type]

    max_possible_parameter_combinations = 0
    for algorithm, parameters in grid_search_meta_parameters.items():
        possible_parameters = {}
        for parameter in (set(parameters.keys()) - set(["distance"])):
            if grid_search_meta_parameters[algorithm][parameter]["step_distance_strategy"] == "equal":
                possible_parameters[parameter] = derive_possible_parameters_equally(grid_search_meta_parameters[algorithm][parameter])
            else:
                possible_parameters[parameter] = derive_possible_parameters_exponentially_naive(grid_search_meta_parameters[algorithm][parameter])

        all_possible_parameter_combinations = list(itertools.product(*list(possible_parameters.values())))

        if max_possible_parameter_combinations <= len(all_possible_parameter_combinations):
            max_possible_parameter_combinations = len(all_possible_parameter_combinations)

    return max_possible_parameter_combinations


def determine_max_iterations(sample_size, speedup_multiplier, number_of_algorithms, number_of_rows, learning_type):
    number_of_possible_parameter_combinations = determine_num_of_possible_param_combinations(learning_type)
    number_of_max_possible_parameter_combinations = 1000

    if number_of_possible_parameter_combinations > number_of_max_possible_parameter_combinations:
        print_warning("[Runtime Constrainer] <Warning> Too many possible combinations of (hyper-) parameters inside the configuration file found. Can't ensure possible speedup of the CSMS anymore.")

    max_iterations = int(number_of_rows / (sample_size * number_of_possible_parameter_combinations * speedup_multiplier))

    if max_iterations < 1:
        print_warning("[Runtime Constrainer] <Warning> Speedup multiplier value is set too high, can't ensure this high speedup. Setting max. number of iterations to number of algorithms ...")
        print_warning(" -> Try to decrease the possible combinations of (hyper-) parameters for the maximum number of combinations or the speedup parameter.")
        return number_of_algorithms

    if max_iterations >= number_of_algorithms:
        return number_of_algorithms

    return max_iterations


def determine_sample_size(number_of_rows):
    min_sample_size = 50  # should be 50, but can be changed here
    max_sample_size = int(np.sqrt(number_of_rows))

    if number_of_rows < min_sample_size:
        raise RuntimeError(f"[RuntimeConstrainer] <Error> The number of data points is smaller than {min_sample_size}, therefore the dataset is too small! ({number_of_rows} rows)")

    if max_sample_size <= min_sample_size:
        return min_sample_size

    return max_sample_size  # TODO ?