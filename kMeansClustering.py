import sklearn.datasets
import sklearn.preprocessing
import sklearn.cluster


def kMeansCluster(data,target,num_clusters = 52):
    reshapedData = data.reshape(len(data))

    estimator = sklearn.cluster.KMeans(init="k-means++",n_clusters=numclusters,n_init=10)

    rescaledData = sklearn.preprocessing.scale(reshapedData)

    estimator.fit(reshapedData)

    predY = estimator.predict(rescaledData)
