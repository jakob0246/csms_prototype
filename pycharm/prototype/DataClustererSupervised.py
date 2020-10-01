from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from scipy.spatial import distance

from DataClustererHelper import prepare_parameters


def knn_clustering(X_train, X_test, y_train, y_test, parameters, evaluation_metrics):
    # modify parameters to call the clustering algorithm with modified ones, this mainly purposes the distance parameter
    modified_parameters = prepare_parameters(parameters)

    p = modified_parameters["minkowski_p"]
    if modified_parameters["minkowski_p"] is None:
        p = 2

    initial_classifier = KNeighborsClassifier(n_neighbors=modified_parameters["k"], metric=modified_parameters["distance"], p=p)

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
        initial_classifier = NearestCentroid(metric=modified_parameters["distance"])

    classifier = initial_classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    evaluation_metrics["accuracy"] = classifier.score(X_test, y_test)

    return evaluation_metrics


def radius_neighbors_clustering(X_train, X_test, y_train, y_test, parameters, evaluation_metrics):
    # modify parameters to call the clustering algorithm with modified ones, this mainly purposes the distance parameter
    modified_parameters = prepare_parameters(parameters)

    initial_classifier = RadiusNeighborsClassifier(radius=modified_parameters["radius"], metric=modified_parameters["distance"],
                                                   p=modified_parameters["minkowski_p"])

    classifier = initial_classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    evaluation_metrics["accuracy"] = classifier.score(X_test, y_test)

    return evaluation_metrics


def nca_clustering(X_train, X_test, y_train, y_test, parameters, evaluation_metrics):
    # modify parameters to call the clustering algorithm with modified ones, this mainly purposes the distance parameter
    modified_parameters = prepare_parameters(parameters)

    nca = NeighborhoodComponentsAnalysis()
    initial_classifier_knn = KNeighborsClassifier(n_neighbors=modified_parameters["k"], metric=modified_parameters["distance"],
                                                  p=modified_parameters["minkowski_p"])

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