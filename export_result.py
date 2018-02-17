__author__ = 'vle020518'

import pandas as pd
import numpy as np
import csv

# our libraries
from kmeans import  KMeans
from multinomial_NB import multinomial_NB as MyMultinomialNB


def ger_raw_data(fileName):
    dic = {"A":0,"C":1,"G":2,"T":3}
    return np.array([ [ dic[e1] for e1 in list(e)] for e in pd.read_csv(fileName,header=None)[0]])

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

def exportFinalResult():

    clr = KMeans()
    clr.fit(X_0,y_0)
    predict_0 = clr.predict(X_test_0)
    clr = MyMultinomialNB()
    clr.fit(X_1,y_1)
    predict_1 = clr.predict(X_test_1)
    clr = MyMultinomialNB()
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

def exportLabelDataFromCSV(fileName):
    y_raw = pd.read_csv(fileName)

    return np.array(y_raw["Bound"])

# change the size of word bag
bagofwordsDataFromCSV = bagofwords6DataFromCSV
X_0 = bagofwordsDataFromCSV("Xtr0.csv")
X_1 = bagofwordsDataFromCSV("Xtr1.csv")
X_2 = bagofwordsDataFromCSV("Xtr2.csv")
y_0 = exportLabelDataFromCSV("Ytr0.csv")
y_1 = exportLabelDataFromCSV("Ytr1.csv")
y_2 = exportLabelDataFromCSV("Ytr2.csv")
X_test_0 = bagofwordsDataFromCSV("Xte0.csv")
X_test_1 = bagofwordsDataFromCSV("Xte1.csv")
X_test_2 = bagofwordsDataFromCSV("Xte2.csv")

if __name__ == '__main__':
    exportFinalResult()