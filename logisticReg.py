from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot
from load_data import getDigits



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

    plot_confusion_matrix(regClassifier,testX,testY)
    pyplot.show()



if __name__ == '__main__':
    data, target = getDigits("C:\\Users\\Tamara\\Desktop\\sklearnpractice\\img",num = 30)
    logisticReg(data,target)
