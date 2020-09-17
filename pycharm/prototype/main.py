from Configurator import get_configuration
from DataIntegrator import read_in_data
from DataPreprocessor import user_feature_selection, initial_preprocessing, prepare_for_unsupervised_learning, \
                             prepare_for_supervised_learning, clean_dataset, scale_and_normalize_features, further_preprocessing
from FeatureSelector import simple_automatic_feature_selection
from DataProfiler import profile_data
from HardwareReader import get_hardware_specs
from AlgorithmSelector import select_algorithm_csv, select_algorithm
from ParameterTunerGridSearch import tune_parameters
from ProgramGeneratorAndEvaluator import generate_and_evaluate_program
from IterativeStepDecider import decide_iterative_step
from KDBSetupManager import check_for_setup_in_knowledge_db, write_setup_to_knowledge_db

from DataClusterer import prepare_for_clustering, automatic_selection, unsupervised_clustering, supervised_learning_using_clustering


supported_algorithms = {
    "unsupervised": {"kmeans", "spectral", "optics", "meanshift", "agglomerative", "affinity", "em", "vbgmm"},  # TODO: "dbscan"
    "supervised": {"knn", "svc", "nearest_centroid", "radius_neighbors", "nca", "svc_sgd"}
}

# get parameters from config.txt
config = get_configuration()

# read in initial dataset and do feature selection based on the users wishes
dataset_initial = read_in_data(config["dataset"]["file_path"], config["dataset"]["csv_delimiter"])
dataset_initial = user_feature_selection(dataset_initial, config["feature_selection"]["features"], config["feature_selection"]["type"])

# initial & further preprocessing of the dataset
dataset_initial = initial_preprocessing(dataset_initial, config["dataset"]["numeric_categoricals"], config["dataset"]["class"], config["general"]["learning_type"])
dataset = further_preprocessing(dataset_initial, config["dataset"]["missing_values"])

# simple data cleaning of erroneous data
# TODO: maybe directly after read in
dataset = clean_dataset(dataset)

# heuristically and simply select features
dataset = simple_automatic_feature_selection(dataset)

# prepare the dataset based on the learning type
if config["general"]["learning_type"] == "unsupervised":
    dataset = prepare_for_unsupervised_learning(dataset, config["dataset"]["numeric_categoricals"], config["dataset"]["class"])
else:
    dataset = prepare_for_supervised_learning(dataset, config["dataset"]["numeric_categoricals"], config["dataset"]["class"])

# scale and normalize features
if config["general"]["feature_scaling_and_normalization"] != "":
    dataset = scale_and_normalize_features(dataset, config["general"]["feature_scaling_and_normalization"],
                                           config["dataset"]["class"], (config["general"]["learning_type"] == "supervised"))  # TODO: how to treat supervised case?

# get all data the clustering selection part and parameter tuning part will exploit
profiled_metadata = profile_data(dataset_initial, dataset, config["dataset"]["class"], (config["general"]["learning_type"] == "supervised"))  # TODO
hardware_specs = get_hardware_specs()
configuration_parameters = config["system_parameters"]

# check if the dataset already ran through the CSMS under this specific user configuration with this similar hardware
setup_result = check_for_setup_in_knowledge_db(config["dataset"]["file_path"], hardware_specs, configuration_parameters,
                                               config["general"]["learning_type"])

remaining_algorithms_set = supported_algorithms[config["general"]["learning_type"]]

if setup_result == {}:
    # TODO: check for constraints
    max_iterations = len(remaining_algorithms_set)  # TODO
    speedup_multiplier = 10
    sample_size = 100  # 50, 1000

    algorithm = None
    algorithm_parameters = {}

    next_iterative_state = "algorithm_selection"
    history = []
    kdb_update_count = 0
    algorithms_knowledge_db = {}
    for iteration in range(max_iterations):
        print(f"\nRunning Iteration [{iteration + 1}] ...")

        # select algorithm:
        if next_iterative_state == "algorithm_selection":
            selected_algorithm, algorithm_scores, algorithms_knowledge_db = select_algorithm(remaining_algorithms_set, profiled_metadata, hardware_specs, configuration_parameters,
                                                                                             algorithms_knowledge_db, supervised=(config["general"]["learning_type"] == "supervised"))

            if iteration == 0 or (iteration != 0 and selected_algorithm != history[iteration - 1]["algorithm"]):
                next_iterative_state = "parameter_tuning"
            else:
                next_iterative_state = "program_generation_and_evaluation"

        # tune parameters:
        if next_iterative_state == "parameter_tuning":
            algorithm_parameters = tune_parameters(selected_algorithm, profiled_metadata, hardware_specs, configuration_parameters,
                                                   config["general"]["learning_type"], dataset, config["dataset"]["class"], sample_size)
            next_iterative_state = "program_generation_and_evaluation"

        # generate and evaluate program:
        if next_iterative_state == "program_generation_and_evaluation":
            results = generate_and_evaluate_program(selected_algorithm, algorithm_parameters, dataset, sample_size, (config["general"]["learning_type"] == "supervised"), config["dataset"]["class"])

            if config["general"]["learning_type"] == "supervised":
                print(" -> Results Iteration [" + str(iteration + 1) + f"]: Got accuracy of {results['accuracy']:.4f} for \"" +
                      selected_algorithm + f"\" (score of {algorithm_scores[selected_algorithm]}) with parameters: " + str(algorithm_parameters))
            else:
                # TODO
                pass

            remaining_algorithms_set -= {selected_algorithm}

        next_iterative_state, algorithms_knowledge_db, history, next_decided_algorithm, kdb_update_count, overall_best_algorithms_parameters = decide_iterative_step(iteration, results, algorithm_scores, algorithms_knowledge_db,
                                                                                                                                                                     history, kdb_update_count, selected_algorithm, max_iterations, (config["general"]["learning_type"] == "supervised"))

        if iteration != 0 and history[iteration]["selected_next_best_algorithm"]:
            algorithm_scores.pop(selected_algorithm)

        history[iteration]["algorithm"] = selected_algorithm
        history[iteration]["parameters"] = algorithm_parameters
        history[iteration]["results"] = results
        history[iteration]["algorithm_scores"] = algorithm_scores

        if next_decided_algorithm != None and iteration != (max_iterations - 1):
            selected_algorithm = next_decided_algorithm

        if overall_best_algorithms_parameters != None:
            algorithm_parameters = overall_best_algorithms_parameters
            selected_algorithm = next_decided_algorithm

        if next_iterative_state == "stop" or len(remaining_algorithms_set) == 0:
            break

    # TODO:
    # write_setup_to_knowledge_db(algorithm, algorithm_parameters, config["dataset"]["file_path"], hardware_specs, configuration_parameters, config["general"]["learning_type"])

else:
    pass

print("\n-*- Running final decision of algorithm \"" + selected_algorithm + "\" with parameters: " + str(algorithm_parameters) + " ...")

end_results = generate_and_evaluate_program(selected_algorithm, algorithm_parameters, dataset, 0,
                                           (config["general"]["learning_type"] == "supervised"), config["dataset"]["class"], sampling=False)

if config["general"]["learning_type"] == "supervised":
    print(f"-*- Got final accuracy of {end_results['accuracy']:.4f}\n")
else:
    print("-*- Got final evaluation results: TODO\n")  # TODO

# TODO: write results / steps / documentation into report
pass
