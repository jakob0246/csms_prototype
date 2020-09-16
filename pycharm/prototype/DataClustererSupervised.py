from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier


def knn_clustering(X_train, X_test, y_train, y_test, parameters, evaluation_metrics):
    initial_classifier = KNeighborsClassifier(n_neighbors=parameters["k"], metric=parameters["distance"])

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
    initial_classifier = NearestCentroid(metric=parameters["distance"])

    classifier = initial_classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    evaluation_metrics["accuracy"] = classifier.score(X_test, y_test)

    return evaluation_metrics


def radius_neighbors_clustering(X_train, X_test, y_train, y_test, parameters, evaluation_metrics):
    initial_classifier = RadiusNeighborsClassifier(radius=parameters["radius"], metric=parameters["distance"])

    classifier = initial_classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    evaluation_metrics["accuracy"] = classifier.score(X_test, y_test)

    return evaluation_metrics


def nca_clustering(X_train, X_test, y_train, y_test, parameters, evaluation_metrics):
    nca = NeighborhoodComponentsAnalysis()
    initial_classifier_knn = KNeighborsClassifier(n_neighbors=parameters["k"], metric=parameters["distance"])

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