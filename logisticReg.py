from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot



def logisticReg(data, target, reshape = 0, testProportion = 0.2):
    # takes numpy array-like arguments, reshape int defaulted at 0 for 1D flattening.
    # float for train_test_split defaulted at 20%
    if reshape == 0:
        reshape = data.shape[1] * data.shape[2]

    data = data.reshape(len(data), reshape)

    trainX, testX, trainY, testY = train_test_split(data,target,test_size = testProportion, shuffle = True)

    regClassifier = LogisticRegression()
    regClassifier.fit(trainX,trainY)
    predY = regClassifier.predict(testX)

    sklearn.metrics.plot_confusion_matrix(regClassifier,testX,testY)
    pyplot.show()
