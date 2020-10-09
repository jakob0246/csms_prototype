from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier, NeighborhoodComponentsAnalysis, DistanceMetric
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

import numpy as np
from scipy.spatial import distance

from DataClustererHelper import prepare_parameters


def knn_clustering(X_train, X_test, y_train, y_test, parameters, evaluation_metrics):
    # modify parameters to call the clustering algorithm with modified ones, this mainly purposes the distance parameter
    modified_parameters = prepare_parameters(parameters)

    p = modified_parameters["minkowski_p"]
    if modified_parameters["minkowski_p"] is None:
        p = 2

    if modified_parameters["distance"] != "mahalanobis":
        initial_classifier = KNeighborsClassifier(n_jobs=-1, n_neighbors=modified_parameters["k"], metric=modified_parameters["distance"], p=p)
    else:
        initial_classifier = KNeighborsClassifier(n_jobs=-1, algorithm="brute", n_neighbors=modified_parameters["k"], metric="mahalanobis", metric_params={"VI": np.linalg.inv(np.cov(X_train))})

    classifier = initial_classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    evaluation_metrics["accuracy"] = classifier.score(X_test, y_test)

    return evaluation_metrics


def svc_clustering(X_train, X_test, y_train, y_test, parameters, evaluation_metrics):
    initial_classifier = SVC(degree=parameters["degree"])

    classifier = initial_classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    evaluation_metrics["accuracy"] = classifier.score(X_test, y_test)

    return evaluation_metrics


def nearest_centroid_clustering(X_train, X_test, y_train, y_test, parameters, evaluation_metrics):
    # modify parameters to call the clustering algorithm with modified ones, this mainly purposes the distance parameter
    modified_parameters = prepare_parameters(parameters)

    if modified_parameters["distance"] == "minkowski" and modified_parameters["minkowski_p"] is not None:
        initial_classifier = NearestCentroid(metric=lambda x, y: distance.minkowski(x, y, modified_parameters["minkowski_p"]))
    else:
        if modified_parameters["distance"] == "mahalanobis":
            initial_classifier = NearestCentroid(metric="mahalanobis", metric_params={"V": np.cov(X_train)})  # TODO: fix
        else:
            initial_classifier = NearestCentroid(metric=modified_parameters["distance"])

    classifier = initial_classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    evaluation_metrics["accuracy"] = classifier.score(X_test, y_test)

    return evaluation_metrics


def radius_neighbors_clustering(X_train, X_test, y_train, y_test, parameters, evaluation_metrics):
    # modify parameters to call the clustering algorithm with modified ones, this mainly purposes the distance parameter
    modified_parameters = prepare_parameters(parameters)

    if modified_parameters["distance"] != "mahalanobis":
        initial_classifier = RadiusNeighborsClassifier(n_jobs=-1, radius=modified_parameters["radius"], metric=modified_parameters["distance"],
                                                       p=modified_parameters["minkowski_p"])
    else:
        initial_classifier = RadiusNeighborsClassifier(n_jobs=-1, radius=modified_parameters["radius"], metric=modified_parameters["distance"], p=modified_parameters["minkowski_p"],
                                                       algorithm="brute", metric_params={"VI": np.linalg.inv(np.cov(X_train))})

    classifier = initial_classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    evaluation_metrics["accuracy"] = classifier.score(X_test, y_test)

    return evaluation_metrics


def nca_clustering(X_train, X_test, y_train, y_test, parameters, evaluation_metrics):
    # modify parameters to call the clustering algorithm with modified ones, this mainly purposes the distance parameter
    modified_parameters = prepare_parameters(parameters)

    nca = NeighborhoodComponentsAnalysis()

    if modified_parameters["distance"] != "mahalanobis":
        initial_classifier_knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=modified_parameters["k"], metric=modified_parameters["distance"],
                                                      p=modified_parameters["minkowski_p"])
    else:
        initial_classifier_knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=modified_parameters["k"], metric=modified_parameters["distance"],
                                                      p=modified_parameters["minkowski_p"], algorithm="brute", metric_params={"VI": np.linalg.inv(np.cov(X_train))})

    nca.fit(X_train, y_train)
    classifier = initial_classifier_knn.fit(nca.transform(X_train), y_train)

    y_pred = classifier.predict(nca.transform(X_test))

    evaluation_metrics["accuracy"] = classifier.score(nca.transform(X_test), y_test)

    return evaluation_metrics


def svc_sdg_clustering(X_train, X_test, y_train, y_test, parameters, evaluation_metrics):
    initial_classifier = SGDClassifier()

    classifier = initial_classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    evaluation_metrics["accuracy"] = classifier.score(X_test, y_test)

    return evaluation_metrics