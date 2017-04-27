from main import*
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from decimal import *
import numpy as np
class Naive_implementation:
    def calc_class_prob(self):
        a = Y_train.value_counts()  # count no. of 0's(not spam) and 1's(spam) in y_train
        b = Y_train.count()
        self.prob_spam = a[1] / b
        self.prob_not_spam = a[0] / b

    def initialize_nb_dict(self):
        self.liklehood_notspam = {} #  probablity of words present given mail is not spam
        self.a={}
        self.liklehood_spam = {}# probablity of words present given mail is  spam
        self.non_liklehood_notspam = {}# probablity of words is not present given mail is not spam
        self.non_liklehood_spam = {}# probablity of words is not present given mail is  spam



    def train(self, X,Y):

            self.calc_class_prob()
            self.initialize_nb_dict()
            #frequency_table = pd.DataFrame(X.toarray(), columns=count_vector.get_feature_names())

            row_indices_notspam = np.where(Y == 0)# splitting the X into one's with spam and one's with not spam.here we are getting the indices
            row_indices_spam=np.where(Y == 1)
            data_notspam=X[row_indices_notspam]#data_notspam is data with non-spam mails
            data_spam=X[row_indices_spam ]#data_spam is data with spam mails
            rows,cols=np.shape(data_notspam)

            n1=pd.DataFrame(data_notspam.toarray(), columns=count_vector.get_feature_names())#converting to not spam data to dataframe
            n2 = pd.DataFrame(data_spam.toarray(), columns=count_vector.get_feature_names())
            self.keeys=n1.keys()

            #calculating the liklihood for feature given mail is not a spam
            for key in n1.keys():
                freq_word_absent=np.count_nonzero(n1[key] ==0)
                freq_word_present=np.count_nonzero(n1[key] !=0)
                sum=freq_word_absent+freq_word_present
                self.liklehood_notspam[key]=freq_word_present/float(sum)
                self.non_liklehood_notspam[key] = freq_word_absent / float(sum)
            # calculating the liklihood for feature given  mail is a spam
            for key in n2.keys():
                freq_word_absent = np.count_nonzero(n2[key] == 0)
                freq_word_present = np.count_nonzero(n2[key] != 0)
                sum = freq_word_absent + freq_word_present
                self.liklehood_spam[key] = freq_word_present / sum
                self.non_liklehood_spam[key] = freq_word_absent / sum

#for calculating the probabilty of class with respect to the features present
    def classify_single_elem(self, X_elem):
        prob_spam_feature=1.0
        prob_notspam_feature=1.0
        for key in self.keeys:
            self.a[key]=0
        for key in X_elem.keys() :
          if X_elem[key].item()==1:
            if not (np.isclose(self.liklehood_notspam[key], 0)):# excluding all probabilty which are 0 to remove ovefitting
                prob_notspam_feature *= self.liklehood_notspam[key]
            if not (np.isclose(self.liklehood_spam[key], 0)):
                prob_spam_feature *= self.liklehood_spam[key]

          if  X_elem[key].item()==1:
              if not (np.isclose(self.non_liklehood_notspam[key], 0)):
                  prob_notspam_feature *= self.non_liklehood_notspam[key]
              if not (np.isclose(self.non_liklehood_spam[key], 0)):
                  prob_spam_feature *= self.non_liklehood_spam[key]

        prob_spam_feature *=self.prob_spam
        prob_notspam_feature *=self.prob_not_spam
        sum=prob_notspam_feature+prob_spam_feature
        if prob_notspam_feature/float(sum) >prob_spam_feature/float(sum):
            return 1
        else:
            return 0
    def classify(self, X):
         self.predicted_Y_values = []
         no_rows, no_cols = np.shape(X)
         for ii in range(0, no_rows):
          X_elem = pd.DataFrame(X[ii, :].toarray(), columns=count_vector.get_feature_names())
          prediction = self.classify_single_elem(X_elem)
          self.predicted_Y_values.append(prediction)

         return self.predicted_Y_values

aa=Naive_implementation()
aa.train(training_data,Y_train)
predictions=aa.classify(testing_data)
print(" predictions",predictions)

print('Accuracy score: ', format(accuracy_score(Y_test, predictions)))
print('Precision score: ', format(precision_score(Y_test, predictions)))
print('Recall score: ', format(recall_score(Y_test, predictions)))
print('F1 score: ', format(f1_score(Y_test, predictions)))
