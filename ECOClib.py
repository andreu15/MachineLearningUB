import sklearn
import numpy as np
import matplotlib.pyplot as plt
import copy


## Implement L1 decoder
class L1_Decoder():


    def distance(self, code1, code2):

        distance = 0
        for i in range(len(code1)):

            distance += abs(code1[i]-code2[i])

        return 0.5*distance


    def predict(self, EC, code):

        distances = []

        for code_class in EC:

            distances.append(self.distance(code_class, code))

        distances = np.array(distances)


        return np.argmin(distances)

## Implementation of L2 decoder
class L2_Decoder():


    def distance(self, code1, code2):

        distance = 0
        for i in range(len(code1)):

            distance += (code1[i]-code2[i]) ** 2

        return np.sqrt(distance)


    def predict(self, EC, code):

        distances = []

        for code_class in EC:

            distances.append(self.distance(code_class, code))

        distances = np.array(distances)


        return np.argmin(distances)

# Implementation of a ECOC classifiers
# It automatically handles the number of classes and features from the data and matrix
# When initializing one must specify the ECOC matrix,  sklearn classifier
# and the decoder to use
class Ecoc():

    def __init__(self, EC, classifier,  use_decoding='L2' ):

        self.EC = EC
        self.classifier = classifier

        if use_decoding == 'L2':  self.decoder  = L2_Decoder()
        if use_decoding == 'L1': self.decoder = L1_Decoder()
        self.classifiers = []

    def create_problem(self, X, y, column): # creates column binary problems

        Xcol = []
        ycol = []

        for data, label in zip(X, y):

            if column[label] == 1:

                Xcol.append(data)
                ycol.append(1)

            elif column[label] == -1:

                Xcol.append(data)
                ycol.append(-1)

            else: pass #for this column this class is not considered

        return Xcol, ycol

    def fit(self, X, y):

        if len(self.EC) != len(np.unique(y)):

            print("ECOC matrix bad defined, avorting process")

        else:

            for column in range(len(self.EC[0])):

                Xcolumn, ycolumn = self.create_problem(X, y, self.EC[:, column])

                classifier_col = copy.copy(self.classifier)

                classifier_col.fit(Xcolumn, ycolumn)

                self.classifiers.append(classifier_col)

    def predict(self, X, show_code=False):

        decoded_codes = []
        for elem in X:
            code = []

            for h in self.classifiers:

                code.append(h.predict(elem.reshape(1,-1))[0])

            if show_code: print("Code for this sample is {}".format(code))
            decoded_code = self.decode(code)
            decoded_codes.append(decoded_code)

        return np.array(decoded_codes)


    def decode(self, code):

        pred = self.decoder.predict(self.EC, code)

        return pred

    def evaluate_accuracy(self, X, y):

        yhat = self.predict(X)
        errs = 0

        for i in range(len(y)):
            if y[i] != yhat[i]:
                errs +=1

        accuracy = 1 - errs/len(y)

        print("Estimated accuracy is {}%".format(accuracy*100))

        return accuracy
