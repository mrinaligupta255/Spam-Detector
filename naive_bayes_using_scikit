from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB

from main import *

# creating a naive bayes classifier using scikit-learn library
naive_bayes = MultinomialNB()
# fitting the training data to the classifier
naive_bayes.fit(training_data, Y_train)
# predicting using testing_data imported from main.py
predictions = naive_bayes.predict(testing_data)
print(predictions)
# calculating accuracy,prediction,recall,F1
print('Accuracy score: ', format(accuracy_score(Y_test, predictions)))
print('Precision score: ', format(precision_score(Y_test, predictions)))
print('Recall score: ', format(recall_score(Y_test, predictions)))
print('F1 score: ', format(f1_score(Y_test, predictions)))
