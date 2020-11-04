import numpy as np

from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, OPTICS, MeanShift, AgglomerativeClustering, AffinityPropagation
from sklearn import mixture

import warnings

from DataClustererHelper import prepare_parameters


def determine_n_clusters(labels):
    return len(set(labels)) - (1 if (-1) in labels else 0)


def kmeans_clustering(dataset, parameters):
    k_means_result = KMeans(n_clusters=parameters["n_clusters"]).fit(dataset)

    result_labels = k_means_result.labels_

    return result_labels


def em_clustering(dataset, parameters):
    # TODO: especially fix error!

    em_result = mixture.GaussianMixture(n_components=parameters["n_clusters"], covariance_type='full').fit(dataset)
    result_labels = em_result.predict(dataset)

    return result_labels


def spectral_clustering(dataset, parameters):
    # TODO: especially fix warning!

    warnings.filterwarnings("ignore", category=UserWarning)
    result_labels = SpectralClustering(parameters["n_clusters"]).fit_predict(dataset)
    warnings.filterwarnings("default")

    return result_labels


def dbscan_clustering(dataset, parameters):
    # TODO: especially parameters! + label-result handling!

    # modify parameters to call the clustering algorithm with modified ones, this mainly purposes the distance parameter
    modified_parameters = prepare_parameters(parameters)

    # min_samples = 1 means no noise / outliers possible
    if modified_parameters["distance"] != "mahalanobis":
        dbscan_result = DBSCAN(eps=modified_parameters["epsilon"], min_samples=modified_parameters["min_samples"],
                               metric=modified_parameters["distance"], p=modified_parameters["minkowski_p"]).fit(dataset)
    else:
        dbscan_result = DBSCAN(eps=modified_parameters["epsilon"], min_samples=modified_parameters["min_samples"],
                               metric=modified_parameters["distance"], p=modified_parameters["minkowski_p"], algorithm="brute", metric_params={"VI": np.linalg.inv(np.cov(dataset))}).fit(dataset)

    result_labels = dbscan_result.labels_
    n_clusters = determine_n_clusters(result_labels)

    # if n_clusters <= 1:
    #     raise ValueError("n. of clusters should be > 1 for the DBSCAN-result, check parameters for the clustering algorithm!")

    return result_labels, n_clusters


def optics_clustering(dataset, parameters):
    # TODO: especially parameters! (min_samples cant be 1 ?!) + label-result handling!

    # modify parameters to call the clustering algorithm with modified ones, this mainly purposes the distance parameter
    modified_parameters = prepare_parameters(parameters)

    # min_samples = 1 means no noise / outliers possible
    optics_result = OPTICS(min_samples=modified_parameters["min_samples"], metric=modified_parameters["distance"],
                           p=modified_parameters["minkowski_p"]).fit(dataset)

    result_labels = optics_result.labels_
    n_clusters = determine_n_clusters(result_labels)

    # if n_clusters <= 1:
    #     raise ValueError("n. of clusters should be > 1 for the OPTICS-result, check parameters for the clustering algorithm!")

    return result_labels, n_clusters


def meanshift_clustering(dataset, parameters):
    meanshift_result = MeanShift(n_jobs=-1).fit(dataset)

    result_labels = meanshift_result.labels_
    n_clusters = determine_n_clusters(result_labels)

    return result_labels, n_clusters


def agglomerative_clustering(dataset, parameters):
    # modify parameters to call the clustering algorithm with modified ones, this mainly purposes the distance parameter
    modified_parameters = prepare_parameters(parameters)

    if parameters["distance"] == "euclidean":
        agglomerative_result = AgglomerativeClustering(n_clusters=parameters["n_clusters"], affinity=parameters["distance"]).fit(dataset)
    else:
        agglomerative_result = AgglomerativeClustering(n_clusters=parameters["n_clusters"], affinity=parameters["distance"], linkage="average").fit(dataset)

    result_labels = agglomerative_result.labels_

    return result_labels


def affinity_clustering(dataset, parameters):
    affinity_result = AffinityPropagation(random_state=0).fit(dataset)

    result_labels = affinity_result.labels_
    n_clusters = determine_n_clusters(result_labels)

    return result_labels, n_clusters

def vbgmm_clustering(dataset, parameters):
    if parameters["max_n_components"] > dataset.shape[0] - 1:
        n_components = dataset.shape[0] - 1
    else:
        n_components = parameters["max_n_components"]

    vbgmm_result = mixture.BayesianGaussianMixture(n_components=n_components).fit(dataset)
    result_labels = vbgmm_result.predict(dataset)

    n_clusters = determine_n_clusters(result_labels)

    return result_labels, n_clusters