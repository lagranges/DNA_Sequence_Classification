__author__ = 'vle020518'

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from kmeans import  KMeans
from kernel_kmeans import Kernel_KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from multinomial_NB import multinomial_NB as MyMultinomialNB
from logistic_regression import logistic_regression
import csv
import sys

def ger_raw_data(fileName):
    dic = {"A":0,"C":1,"G":2,"T":3}
    return np.array([ [ dic[e1] for e1 in list(e)] for e in pd.read_csv(fileName,header=None)[0]])


def bagofwords3DataFromCSV(fileName):
    X_raw = pd.read_csv(fileName,header=None)
    # dictionnaire : string -> integer  ex: "AAAA" -> 0, "AAAC" -> 1
    dicts = {}
    chars  = {"A","C","G","T"}
    count = 0
    for i_0 in chars:
        for i_1 in chars:
            for i_2 in chars:
                        dicts[i_0+i_1+i_2] = count
                        count += 1
    def transfer(x):
        rel = np.zeros((np.power(4,3)))
        for i in range(0,98):
            rel[dicts[x[i]+x[i+1]+x[i+2]]] += 1
        return rel

    return np.array(list(map(lambda x: transfer(x),X_raw[0])))
def bagofwords4DataFromCSV(fileName):
    X_raw = pd.read_csv(fileName,header=None)
    # dictionnaire : string -> integer  ex: "AAAA" -> 0, "AAAC" -> 1
    dicts = {}
    chars  = {"A","C","G","T"}
    count = 0
    for i_0 in chars:
        for i_1 in chars:
            for i_2 in chars:
                for i_3 in chars:
                        dicts[i_0+i_1+i_2+i_3] = count
                        count += 1

    def transfer(x):
        rel = np.zeros((np.power(4,4)))
        for i in range(0,98):
            rel[dicts[x[i]+x[i+1]+x[i+2]+x[i+3]]] += 1
        return rel

    return np.array(list(map(lambda x: transfer(x),X_raw[0])))

def bagofwords5DataFromCSV(fileName):
    X_raw = pd.read_csv(fileName,header=None)
    # dictionnaire : string -> integer  ex: "AAAA" -> 0, "AAAC" -> 1
    dicts = {}
    chars  = {"A","C","G","T"}
    count = 0
    for i_0 in chars:
        for i_1 in chars:
            for i_2 in chars:
                for i_3 in chars:
                    for i_4 in chars:
                        dicts[i_0+i_1+i_2+i_3+i_4] = count
                        count += 1
    def transfer(x):
        rel = np.zeros((np.power(4,6)))
        for i in range(0,96):
            rel[dicts[x[i]+x[i+1]+x[i+2]+x[i+3]+x[i+4]]] += 1
        return rel

    return np.array(list(map(lambda x: transfer(x),X_raw[0])))

def bagofwords6DataFromCSV(fileName):
    X_raw = pd.read_csv(fileName,header=None)
    # dictionnaire : string -> integer  ex: "AAAA" -> 0, "AAAC" -> 1
    dicts = {}
    chars  = {"A","C","G","T"}
    count = 0
    for i_0 in chars:
        for i_1 in chars:
            for i_2 in chars:
                for i_3 in chars:
                    for i_4 in chars:
                        for i_5 in chars:
                            dicts[i_0+i_1+i_2+i_3+i_4+i_5] = count
                            count += 1

    def transfer(x):
        rel = np.zeros((np.power(4,6)))
        for i in range(0,96):
            rel[dicts[x[i]+x[i+1]+x[i+2]+x[i+3]+x[i+4]+x[i+5]]] += 1
        return rel

    return np.array(list(map(lambda x: transfer(x),X_raw[0])))

def bagofwords7DataFromCSV(fileName):
    X_raw = pd.read_csv(fileName,header=None)
    # dictionnaire : string -> integer  ex: "AAAA" -> 0, "AAAC" -> 1
    dicts = {}
    chars  = {"A","C","G","T"}
    count = 0
    for i_0 in chars:
        for i_1 in chars:
            for i_2 in chars:
                for i_3 in chars:
                    for i_4 in chars:
                        for i_5 in chars:
                            for i_6 in chars:
                                dicts[i_0+i_1+i_2+i_3+i_4+i_5+i_6] = count
                                count += 1

    def transfer(x):
        rel = np.zeros((np.power(4,7)))
        for i in range(0,95):
            rel[dicts[x[i]+x[i+1]+x[i+2]+x[i+3]+x[i+4]+x[i+5]+x[i+6]]] += 1
        return rel

    return np.array(list(map(lambda x: transfer(x),X_raw[0])))

def bagofwords8DataFromCSV(fileName):
    X_raw = pd.read_csv(fileName,header=None)
    # dictionnaire : string -> integer  ex: "AAAA" -> 0, "AAAC" -> 1
    dicts = {}
    chars  = {"A","C","G","T"}
    count = 0
    for i_0 in chars:
        for i_1 in chars:
            for i_2 in chars:
                for i_3 in chars:
                    for i_4 in chars:
                        for i_5 in chars:
                            for i_6 in chars:
                                for i_7 in chars:
                                    dicts[i_0+i_1+i_2+i_3+i_4+i_5+i_6+i_7] = count
                                    count += 1

    def transfer(x):
        rel = np.zeros((np.power(4,8)))
        for i in range(0,94):
            rel[dicts[x[i]+x[i+1]+x[i+2]+x[i+3]+x[i+4]+x[i+5]+x[i+6]+x[i+7]]] += 1
        return rel

    return np.array(list(map(lambda x: transfer(x),X_raw[0])))


def exportLabelDataFromCSV(fileName):
    y_raw = pd.read_csv(fileName)

    return np.array(y_raw["Bound"])

def compare_classification_kmeans(X,y):
    print("===========Kmeans Comparison===========")
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)
    clr = Kernel_KMeans()
    clr.fit(X_train,y_train)
    print("Kernel_KMeans: %f"%(clr.score(X_test,y_test)))
    clr = KMeans()
    clr.fit(X_train,y_train)
    print("My Nearest Centroid: %f"%(clr.score(X_test,y_test)))
    clr = NearestCentroid()
    clr.fit(X_train,y_train)
    print("Sklearn Nearest Centroid: %f"%(clr.score(X_test,y_test)))

def compare_classification_logistic_regression(X,y):
    print("===========LG Comparison===========")
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)
    clr = logistic_regression()
    clr.fit(X_train,y_train)
    print("My LogisticRegression: %f"%(clr.score(X_test,y_test)))
    clr = LogisticRegression()
    clr.fit(X_train,y_train)
    print("Sklearn LogisticRegression: %f"%(clr.score(X_test,y_test)))

def compare_classification_MultinomialNB(X,y):
    print("===========MutinomialNB Comparison===========")
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)
    clr = MultinomialNB()
    clr.fit(X_train,y_train)
    print("My Multinomial: %f"%(clr.score(X_test,y_test)))
    clr = MyMultinomialNB()
    clr.fit(X_train,y_train)
    print("Sklearn Multinomial: %f"%(clr.score(X_test,y_test)))
    print("")

def proceed_classification(X,y,text="Classification Experiment"):

    print("==========="+text+"===========")
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)
    clr = LogisticRegression()
    clr.fit(X_train,y_train)
    print("Logistic Regession: %f"%(clr.score(X_test,y_test)))
    clr = RidgeClassifier()
    clr.fit(X_train,y_train)
    print("Ridge: %f"%(clr.score(X_test,y_test)))
    clr = MultinomialNB()
    clr.fit(X_train,y_train)
    print("Multinomial: %f"%(clr.score(X_test,y_test)))
    clr = GaussianNB()
    clr.fit(X_train,y_train)
    print("GaussianNB: %f"%(clr.score(X_test,y_test)))
    clr = SGDClassifier()
    clr.fit(X_train,y_train)
    print("SGDClassifier: %f"%(clr.score(X_test,y_test)))
    clr = Perceptron()
    clr.fit(X_train,y_train)
    print("Perceptron: %f"%(clr.score(X_test,y_test)))
    clr = BernoulliNB()
    clr.fit(X_train,y_train)
    print("BernoulliNB: %f"%(clr.score(X_test,y_test)))
    clr = KNeighborsClassifier()
    clr.fit(X_train,y_train)
    print("KNeighbors: %f"%(clr.score(X_test,y_test)))
    clr = NearestCentroid()
    clr.fit(X_train,y_train)
    print("NearestCentroid: %f"%(clr.score(X_test,y_test)))
    clr = RandomForestClassifier()
    clr.fit(X_train,y_train)
    print("RandomForestClassifier: %f"%(clr.score(X_test,y_test)))
    clr = MLPClassifier()
    clr.fit(X_train,y_train)
    print("Neutral network: %f"%(clr.score(X_test,y_test)))
    clr = SVC(kernel="rbf")
    clr.fit(X_train,y_train)
    print("Kernel SVM network: %f"%(clr.score(X_test,y_test)))
    print("\n")

# produire la sortie pour kaggle
def exportResult(classification):

    clr = classification()
    clr.fit(X_0,y_0)
    predict_0 = clr.predict(X_test_0)
    clr = classification()
    clr.fit(X_1,y_1)
    predict_1 = clr.predict(X_test_1)
    clr = classification()
    clr.fit(X_2,y_2)
    predict_2 = clr.predict(X_test_2)

    with open('submission.csv', 'w', newline='') as csvfile:
        fieldnames = ['Id', 'Bound']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for i, y_i in enumerate(predict_0):
            writer.writerow({'Id': i, 'Bound': (int)((y_i+1)/2)})

        for i, y_i in enumerate(predict_1):
            writer.writerow({'Id': 1000+i, 'Bound': (int)((y_i+1)/2)})

        for i, y_i in enumerate(predict_2):
            writer.writerow({'Id': 2000+i, 'Bound': (int)((y_i+1)/2)})
    print("FINISH: the predicted labels have been wrote into submisson.csv")

def exportFinalResult():

    clr = KMeans()
    clr.fit(X_0,y_0)
    print("Dataset 0: %f"%(clr.score(X_0,y_0)))
    predict_0 = clr.predict(X_test_0)
    clr = MyMultinomialNB()
    clr.fit(X_1,y_1)
    print("Dataset 1: %f"%(clr.score(X_1,y_1)))
    predict_1 = clr.predict(X_test_1)
    clr = MyMultinomialNB()
    clr.fit(X_2,y_2)
    print("Dataset 2: %f"%(clr.score(X_2,y_2)))
    predict_2 = clr.predict(X_test_2)
    print(predict_0.shape)

    with open('submission12.csv', 'w', newline='') as csvfile:
        fieldnames = ['Id', 'Bound']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for i, y_i in enumerate(predict_0):
            writer.writerow({'Id': i, 'Bound': (int)((y_i+1)/2)})

        for i, y_i in enumerate(predict_1):
            writer.writerow({'Id': 1000+i, 'Bound': (int)((y_i+1)/2)})

        for i, y_i in enumerate(predict_2):
            writer.writerow({'Id': 2000+i, 'Bound': (int)((y_i+1)/2)})

# change the size of word bag
bagofwordsDataFromCSV = bagofwords6DataFromCSV
X_raw_0 = ger_raw_data("Xtr0.csv")
X_raw_1 = ger_raw_data("Xtr1.csv")
X_raw_2 = ger_raw_data("Xtr2.csv")
X_0 = bagofwordsDataFromCSV("Xtr0.csv")
X_1 = bagofwordsDataFromCSV("Xtr1.csv")
X_2 = bagofwordsDataFromCSV("Xtr2.csv")
y_0 = exportLabelDataFromCSV("Ytr0.csv")
y_1 = exportLabelDataFromCSV("Ytr1.csv")
y_2 = exportLabelDataFromCSV("Ytr2.csv")
X_test_0 = bagofwordsDataFromCSV("Xte0.csv")
X_test_1 = bagofwordsDataFromCSV("Xte1.csv")
X_test_2 = bagofwordsDataFromCSV("Xte2.csv")

def classification_experiment():
    print("===============================================")
    print("===========Classification Experiment===========")
    print("===============================================")
    print("\n")
    print("Raw Data - Dataset 0")
    proceed_classification(X_raw_0,y_0)
    print("Raw Data - Dataset 1")
    proceed_classification(X_raw_1,y_1)
    print("Raw Data - Dataset 2")
    proceed_classification(X_raw_2,y_2)

    print("6-gram Data - Dataset 0")
    proceed_classification(X_0,y_0)
    print("6-gram Data - Dataset 1")
    proceed_classification(X_1,y_1)
    print("6-gram Data - Dataset 2")
    proceed_classification(X_2,y_2)
def exportResult3(classification):

    clr = Kernel_KMeans()
    clr.fit(X_0,y_0)
    print("Dataset 0: %f"%(clr.score(X_0,y_0)))
    predict_0 = clr.predict(X_test_0)
    clr = classification()
    clr.fit(X_1,y_1)
    print("Dataset 1: %f"%(clr.score(X_1,y_1)))
    predict_1 = clr.predict(X_test_1)
    clr = Kernel_KMeans()
    clr.fit(X_2,y_2)
    print("Dataset 2: %f"%(clr.score(X_2,y_2)))
    predict_2 = clr.predict(X_test_2)
    print(predict_0.shape)

    with open('submission9.csv', 'w', newline='') as csvfile:
        fieldnames = ['Id', 'Bound']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for i, y_i in enumerate(predict_0):
            writer.writerow({'Id': i, 'Bound': (int)((y_i+1)/2)})

        for i, y_i in enumerate(predict_1):
            writer.writerow({'Id': 1000+i, 'Bound': (int)((y_i+1)/2)})

        for i, y_i in enumerate(predict_2):
            writer.writerow({'Id': 2000+i, 'Bound': (int)((y_i+1)/2)})

if __name__ == '__main__':
    classification_experiment()
    compare_classification_logistic_regression(X_0,y_0)
    compare_classification_kmeans(X_0,y_0)
    compare_classification_MultinomialNB(X_0,y_0)
