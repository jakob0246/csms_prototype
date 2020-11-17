import time

from Configurator import get_configuration
from DataIntegrator import read_in_data
from DataPreprocessor import user_feature_selection, initial_preprocessing, prepare_for_unsupervised_learning, \
                             prepare_for_supervised_learning, clean_dataset, scale_and_normalize_features, further_preprocessing
from FeatureSelector import simple_automatic_feature_selection

from DataProfiler import profile_data
from HardwareReader import get_hardware_specs

from AlgorithmSelector import select_algorithm
from ParameterTunerGridSearch import tune_parameters
from ProgramGeneratorAndEvaluator import generate_and_evaluate_program
from IterativeStepDecider import decide_iterative_step
from KDBSetupManager import check_for_setup_in_knowledge_db, write_setup_to_knowledge_db
from ReportGenerator import initialize_report_generation, report_preprocessing_part_to_latex, report_results_part_to_latex, report_selection_part_to_latex, report_setup_part_to_latex, generate_report

from RuntimeConstrainer import *


# if runtime needs to be measured
cputime_start = time.process_time()

# lists all supported algorithms
supported_algorithms = {
    "unsupervised": {"kmeans", "em", "vbgmm", "meanshift", "agglomerative", "optics", "affinity", "spectral"},
    "supervised": {"knn", "svc", "nearest_centroid", "radius_neighbors", "nca", "svc_sgd"}
}

# get parameters from config.txt
config = get_configuration()

report_tex_file = initialize_report_generation(config["dataset"]["file_path"], config["general"]["learning_type"])

# read in initial dataset and do feature selection based on the users wishes
dataset_initial = read_in_data(config["dataset"]["file_path"], config["dataset"]["csv_delimiter"])
dataset_initial = user_feature_selection(dataset_initial, config["feature_selection"]["features"], config["feature_selection"]["type"])

report_statistics_preprocessing = {}
report_statistics_preprocessing["ohe_columns"] = len(list(set(dataset_initial.select_dtypes(include=['category']).columns)))

# initial & further preprocessing of the dataset
dataset_initial = initial_preprocessing(dataset_initial, config["dataset"]["numeric_categoricals"], config["dataset"]["class"], config["general"]["learning_type"])
dataset, report_statistics_preprocessing["rows_with_missings"] = further_preprocessing(dataset_initial)

# simple data cleaning of erroneous data
dataset, report_statistics_preprocessing["non_distinct_rows"] = clean_dataset(dataset)

# heuristically and simply select features
dataset = simple_automatic_feature_selection(dataset)

# prepare the dataset based on the learning type
if config["general"]["learning_type"] == "unsupervised":
    dataset = prepare_for_unsupervised_learning(dataset, config["dataset"]["numeric_categoricals"], config["dataset"]["class"])
else:
    dataset = prepare_for_supervised_learning(dataset, config["dataset"]["numeric_categoricals"], config["dataset"]["class"])

# scale and normalize features for supervised use-case
if config["general"]["learning_type"] == "supervised":
    dataset = scale_and_normalize_features(dataset, config["general"]["feature_scaling_and_normalization"],
                                           config["dataset"]["class"], (config["general"]["learning_type"] == "supervised"))

report_statistics_preprocessing["n_quaniles"] = dataset.shape[0] // 10
report_preprocessing_part_to_latex(report_tex_file, report_statistics_preprocessing)

# get all data the clustering selection part and parameter tuning part will exploit
profiled_metadata = profile_data(dataset_initial, dataset, config["dataset"]["class"], (config["general"]["learning_type"] == "supervised"))
hardware_specs = get_hardware_specs()
configuration_parameters = {
    "system_parameters": config["system_parameters"],
    "system_parameter_preferences_distance": config["system_parameter_preferences_distance"]
}

report_statistics_setup = {
    "metadata": profiled_metadata,
    "hardware": hardware_specs,
    "parameters": configuration_parameters
}
report_setup_part_to_latex(report_tex_file, report_statistics_setup, config["general"]["learning_type"] == "supervised")

# check if the dataset already ran through the CSMS under this specific user configuration with this similar hardware
setup_result = {}
# setup_result = check_for_setup_in_knowledge_db(config["dataset"]["file_path"], hardware_specs, configuration_parameters,
#                                                config["general"]["learning_type"])

remaining_algorithms_set = supported_algorithms[config["general"]["learning_type"]]

if setup_result == {}:
    sample_size = determine_sample_size(dataset.shape[0], dataset, config["dataset"]["class"], (config["general"]["learning_type"] == "supervised"))
    max_iterations = determine_max_iterations(sample_size, config["general"]["speedup_multiplier"], len(remaining_algorithms_set), dataset.shape[0], config["general"]["learning_type"])

    algorithm = None
    algorithm_parameters = {}

    next_iterative_state = "algorithm_selection"
    history = []
    kdb_update_count = 0
    knowledge_db_metadata_hardware = {}
    for iteration in range(max_iterations):
        print(f"\nRunning Iteration [{iteration + 1} / {max_iterations}] ...")

        # select algorithm:
        if next_iterative_state == "algorithm_selection":
            selected_algorithm, algorithm_scores, knowledge_db_metadata_hardware = select_algorithm(remaining_algorithms_set, profiled_metadata, hardware_specs, configuration_parameters,
                                                                                                    knowledge_db_metadata_hardware, supervised=(config["general"]["learning_type"] == "supervised"))

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
            try:
                results = generate_and_evaluate_program(selected_algorithm, algorithm_parameters, dataset, sample_size, (config["general"]["learning_type"] == "supervised"), config["dataset"]["class"])
            except ValueError:
                if config["general"]["learning_type"] == "supervised":
                    results = {"accuracy": 0}
                else:
                    results = {"silhouette_score": -1, "silhouette_score_standardized": 0}

            if config["general"]["learning_type"] == "supervised":
                print(f" -> Results Iteration [{iteration + 1} / {max_iterations}]: Got accuracy of {results['accuracy']:.4f} for \"" +
                      selected_algorithm + f"\" (score of {algorithm_scores[selected_algorithm]:.4f}) with parameters: " + str(algorithm_parameters))
            else:
                print(f" -> Results Iteration [{iteration + 1} / {max_iterations}]: Got silhouette_score_standardized of {results['silhouette_score_standardized']:.4f} for \"" +
                      selected_algorithm + f"\" (selection-score of {algorithm_scores[selected_algorithm]:.4f}) with parameters: " + str(algorithm_parameters))

            remaining_algorithms_set -= {selected_algorithm}

        next_iterative_state, knowledge_db_metadata_hardware, history, next_decided_algorithm, kdb_update_count, overall_best_algorithms_parameters = decide_iterative_step(iteration, results, algorithm_scores, knowledge_db_metadata_hardware,
                                                                                                                                                                            history, kdb_update_count, selected_algorithm, algorithm_parameters, max_iterations, (config["general"]["learning_type"] == "supervised"))

        history[iteration]["selected_algorithm_score"] = algorithm_scores[selected_algorithm]

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
    print(f"-*- Got final silhouette_score_standardized of {end_results['silhouette_score_standardized']:.4f}\n")  # TODO: unsupervised case

cputime_end = time.process_time()
total_cputime = cputime_end - cputime_start
if config["general"]["measure_runtime"]:
    print(f"-*- All in all took {total_cputime:.4f}s (CPU time)")

report_statistics_selection = history
report_selection_part_to_latex(report_tex_file, report_statistics_selection, config["general"]["learning_type"] == "supervised")

report_statistics_result = {
    "algorithm": selected_algorithm,
    "parameters": algorithm_parameters,
    "end_results": end_results,
    "total_cputime": total_cputime
}
report_results_part_to_latex(report_tex_file, config["general"]["learning_type"], dataset, end_results, config["dataset"]["class"],
                             report_statistics_result, config["general"]["learning_type"] == "supervised")

generate_report(report_tex_file, config["general"]["learning_type"], config["dataset"]["file_path"])
