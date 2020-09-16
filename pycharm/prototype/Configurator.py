import configparser


def parse_config(path: str = "config.txt") -> dict:
    parser_config = configparser.ConfigParser()
    parser_config.read(path)

    # TODO: exception handling
    assumed_metastructure = {
        "general": ["learning_type", "feature_scaling_and_normalization"],
        "dataset": ["file_path", "numeric_categoricals", "class", "csv_delimiter", "missing_values"],
        "feature_selection": ["type", "features"],
        "system_parameters": ["accuracy_efficiency_preference", "find_arbitrary_cluster_shapes", "find_arbitrary_cluster_shapes",
                              "avoid_high_effort_of_hyper_parameter_tuning"],
        "test_parameters": ["use_categorial_encoding", "show_clusterings"]
    }

    assert list(assumed_metastructure.keys()).sort() == list(parser_config.sections()).sort(), "config: config.txt not correctly formatted!"

    # verify config
    config = {}
    for key_outer in assumed_metastructure.keys():
        assert list(assumed_metastructure[key_outer]).sort() == list(parser_config[key_outer]).sort(), "config: config.txt not correctly formatted!"

        config[key_outer] = {}
        for key_inner in assumed_metastructure[key_outer]:
            config[key_outer][key_inner] = str(parser_config[key_outer][key_inner])

    return config


def get_configuration() -> dict:
    raw_config_dict = parse_config()

    config_dict = raw_config_dict.copy()

    # DONE: preprocess raw config parameters:
    for key_outer in config_dict.keys():
        for key_inner in config_dict[key_outer].keys():
            config_dict[key_outer][key_inner] = config_dict[key_outer][key_inner].lower().strip()

    # TODO: check integrity of user config attributes
    assert config_dict["system_parameters"]["accuracy_efficiency_preference"] in ["efficiency", "accuracy", "none"], \
        "config: \"system_parameters\" -> \"accuracy_efficiency_preference\" setting must be \"efficiency\", \"accuracy\", or \"none\""
    assert config_dict["feature_selection"]["type"] in ["exclude", "include"], \
        "config: \"feature_selection\" -> \"type\" setting must be \"exclude\" or \"include\""
    assert config_dict["general"]["learning_type"] in ["supervised", "unsupervised"], \
        "config: \"general\" -> \"learning_type\" setting must be \"supervised\" or \"unsupervised\""
    assert config_dict["general"]["feature_scaling_and_normalization"] in ["", "standard", "quantile"], \
        "config: \"general\" -> \"feature_scaling_and_normalization\" setting must be \"\", \"standard\" or \"quantile\""
    assert config_dict["dataset"]["missing_values"] in ["cca", "aca", "impute"], \
        "config: \"dataset\" -> \"missing_values\" setting must be \"cca\", \"aca\" or \"impute\""

    # TODO parse raw config: feature-extraction, type-conversions etc.:
    config_dict["feature_selection"]["features"] = list(map(lambda ele: ele.strip(), config_dict["feature_selection"]["features"].split(",")))
    config_dict["dataset"]["numeric_categoricals"] = list(map(lambda ele: ele.strip(), config_dict["dataset"]["numeric_categoricals"].split(",")))

    config_dict["test_parameters"]["show_clusterings"] = config_dict["test_parameters"]["show_clusterings"] == "true"

    config_dict["dataset"]["csv_delimiter"] = config_dict["dataset"]["csv_delimiter"][1]

    # further integrity checks:
    if config_dict["dataset"]["class"] == "" and config_dict["general"]["learning_type"] == "supervised":
        raise RuntimeError("config: \"dataset\" -> \"class\" should be specified if supervised learning is wanted")

    config_dict = raw_config_dict

    return config_dict
