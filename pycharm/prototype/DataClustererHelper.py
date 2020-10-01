def prepare_parameters(parameters):
    modified_parameters = parameters.copy()

    modified_parameters["minkowski_p"] = None

    if modified_parameters["distance"] == "minkowski_other":
        modified_parameters["distance"] = "minkowski"

        # insert p parameter of minkowski distance equation if minkowski distance was chosen
        modified_parameters["minkowski_p"] = 3

    return modified_parameters