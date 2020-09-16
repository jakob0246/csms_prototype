from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, OPTICS, MeanShift, AgglomerativeClustering, AffinityPropagation
from sklearn import mixture

import warnings


def determine_n_clusters(labels):
    return len(set(labels)) - (1 if (-1) in labels else 0)


def kmeans_clustering(dataset, parameters):
    k_means_result = KMeans(n_clusters=parameters["k"]).fit(dataset)

    result_labels = k_means_result.labels_

    return result_labels


def em_clustering(dataset, parameters):
    # TODO: especially fix error!

    em_result = mixture.GaussianMixture(n_components=parameters["k"], covariance_type='full').fit(dataset)
    result_labels = em_result.predict(dataset)

    return result_labels


def spectral_clustering(dataset, parameters):
    # TODO: especially fix warning!

    warnings.filterwarnings("ignore", category=UserWarning)

    result_labels = SpectralClustering(parameters["k"]).fit_predict(dataset)

    warnings.filterwarnings("default")

    return result_labels


def dbscan_clustering(dataset, parameters):
    # TODO: especially parameters! + label-result handling!

    dbscan_result = DBSCAN(eps=parameters["epsilon"], min_samples=parameters["min_samples"], metric=parameters["distance"]).fit(dataset)  # min_samples = 1 means no noise / outliers possible

    result_labels = dbscan_result.labels_
    n_clusters = determine_n_clusters(result_labels)

    # if n_clusters <= 1:
    #     raise ValueError("n. of clusters should be > 1 for the DBSCAN-result, check parameters for the clustering algorithm!")

    return result_labels, n_clusters


def optics_clustering(dataset, parameters):
    # TODO: especially parameters! (min_samples cant be 1 ?!) + label-result handling!

    optics_result = OPTICS(min_samples=parameters["min_samples"], metric=parameters["distance"]).fit(dataset)  # min_samples = 1 means no noise / outliers possible

    result_labels = optics_result.labels_
    n_clusters = determine_n_clusters(result_labels)

    # if n_clusters <= 1:
    #     raise ValueError("n. of clusters should be > 1 for the OPTICS-result, check parameters for the clustering algorithm!")

    return result_labels, n_clusters


def meanshift_clustering(dataset, parameters):
    meanshift_result = MeanShift().fit(dataset)

    result_labels = meanshift_result.labels_
    n_clusters = determine_n_clusters(result_labels)

    return result_labels, n_clusters


def agglomerative_clustering(dataset, parameters):
    agglomerative_result = AgglomerativeClustering(n_clusters=parameters["k"], affinity=parameters["distance"]).fit(dataset)

    result_labels = agglomerative_result.labels_

    return result_labels


def affinity_clustering(dataset, parameters):
    affinity_result = AffinityPropagation(random_state=0).fit(dataset)

    result_labels = affinity_result.labels_
    n_clusters = determine_n_clusters(result_labels)

    return result_labels, n_clusters

def vbgmm_clustering(dataset, parameters):
    vbgmm_result = mixture.BayesianGaussianMixture(n_components=parameters["max_n_components"]).fit(dataset)
    result_labels = vbgmm_result.predict(dataset)

    n_clusters = determine_n_clusters(result_labels)

    return result_labels, n_clusters