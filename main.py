import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation  import train_test_split
import matplotlib.pyplot as plt
#1. We have to preapare a Bags Of Words.
#2.count the frequency of each word in each input(each word here is a feature)
# here how to do this------count_vector.fit(df['sms_message'])----words=count_vector.transform(wo).toarray()-
# --frequency_matrix = pd.DataFrame(words,  columns = count_vector.get_feature_names())
#creating a count vector that calculate the frequency of each word present in df['sms_message']
#stop_words='english' => do not count normal english words like is,am,are etc
count_vector = CountVectorizer(analyzer='word',stop_words='english')

df = pd.read_table('SMSSpamDetector',
                   sep='\t',
                   header=None,
                   names=['label', 'sms_message'])
#mapping label ham to 0 and spam to 1
df['label'] = df.label.map({'ham':0, 'spam':1})


#splitting the data into training set and testing set
X_train, X_test, Y_train, Y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1)
print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))

# fitting the X_train and creating a frequency matrix. column represent words and rows represent the message
# fit and tranform are both used for training data but for testing_data only transorm is used (fit is not used)
#-------------count_vector.fit(X_train)
#-------------words=count_vector.transform(X_train)
#-------------frequency_table=pd.DataFrame(training_data.toarray(),columns=count_vector.get_feature_names())
training_data=count_vector.fit_transform(X_train)
#tranforming the testing data but do not fit
testing_data=count_vector.transform([ 'FreeMsg Hey there darling it\'s been 3 week\'s now and no word back!','FROM 88066 LOST £12 HELP','I don\'t think he goes to usf, he lives around here though'])
Y_test=[1,1,0]

# to check for another cases  and accuracy of a dataset,usetesting data=count_vector.transform(X_test) and check with correspomding Y_test
#1=spam,0=not spam




