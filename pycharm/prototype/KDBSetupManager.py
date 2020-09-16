import json
import re


def read_in_knowledge_db_setups():
    path = "KnowledgeDatabases/KDBSetups.json"

    # TODO: check structure

    file = open(path)
    json_data = json.load(file)

    return json_data


def get_id_from_path(path):
    path = path.replace(".csv", "")
    path = path.replace(".arff", "")
    path = path.replace(".txt", "")

    split_list = path.split("/")
    id = split_list[len(split_list) - 1]

    return id


def check_for_setup_in_knowledge_db(dataset_path, hardware_specs, configuration_parameters, learning_type):
    setups = read_in_knowledge_db_setups()
    dataset_id = get_id_from_path(dataset_path)

    result = {}
    for setup in setups:
        if dataset_id == setup["setup"]["dataset_id"]:
            if hardware_specs["high_cache_amount"] == setup["setup"]["hardware"]["high_cache_amount"] and hardware_specs["high_ram_amount"] == setup["setup"]["hardware"]["high_ram_amount"] and hardware_specs["many_cpu_threads"] == setup["setup"]["hardware"]["many_cpu_threads"]:
                if configuration_parameters["accuracy_efficiency_preference"] == setup["setup"]["user_configuration"]["accuracy_efficiency_preference"] and configuration_parameters["find_arbitrary_cluster_shapes"] == setup["setup"]["user_configuration"]["find_arbitrary_cluster_shapes"] and configuration_parameters["avoid_high_effort_of_hyper_parameter_tuning"] == setup["setup"]["user_configuration"]["avoid_high_effort_of_hyper_parameter_tuning"]:
                    if learning_type == "unsupervised":
                        if setup["results"]["unsupervised_algorithm"] != {}:
                            result["algorithm"] = setup["results"]["unsupervised_algorithm"]["algorithm"]
                            result["parameters"] = setup["results"]["unsupervised_algorithm"]["parameters"]
                    else:
                        if setup["results"]["supervised_algorithm"] != {}:
                            result["algorithm"] = setup["results"]["supervised_algorithm"]["algorithm"]
                            result["parameters"] = setup["results"]["supervised_algorithm"]["parameters"]
        break

    return result


def write_setup_to_knowledge_db(algorithm, algorithm_parameters, dataset_path, hardware_specs, configuration_parameters, learning_type):
    dataset_id = get_id_from_path(dataset_path)

    new_setup = {
        "setup": {
            "dataset_id": dataset_id,
            "hardware": {
                "high_cache_amount": hardware_specs["high_cache_amount"],
                "high_ram_amount": hardware_specs["high_ram_amount"],
                "many_cpu_threads": hardware_specs["many_cpu_threads"],
            },
            "user_configuration": {
                "accuracy_efficiency_preference": configuration_parameters["accuracy_efficiency_preference"],
                "find_arbitrary_cluster_shapes": configuration_parameters["find_arbitrary_cluster_shapes"],
                "avoid_high_effort_of_hyper_parameter_tuning": configuration_parameters["avoid_high_effort_of_hyper_parameter_tuning"]
            }
        },
        "result": {
            "unsupervised_algorithm": {},
            "supervised_algorithm": {}
        }
    }

    if learning_type == "unsupervised":
        new_setup["result"]["unsupervised_algorithm"]["algorithm"] = algorithm
        new_setup["result"]["unsupervised_algorithm"]["parameters"] = algorithm_parameters
    else:
        new_setup["result"]["supervised_algorithm"]["algorithm"] = algorithm
        new_setup["result"]["supervised_algorithm"]["parameters"] = algorithm_parameters

    setups = read_in_knowledge_db_setups()
    setups.append(new_setup)

    with open("KnowledgeDatabases/KDBSetups_test.json", "w") as file:
        json.dump(setups, file, indent=4)
