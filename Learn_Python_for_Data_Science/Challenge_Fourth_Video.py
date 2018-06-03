#It contains code for challenge proposed in fourth video
# Load in necessary libraries for data input and normalization

import numpy as np
import matplotlib.pyplot as plt

def Tweet_Sentiment():
    import tweepy
    from textblob import TextBlob
    import csv

    # Standard twitter developer account So only latest 7 days data only will be available.

    # Step 1 - Authenticate
    consumer_key = 'loRzO0qpBuSVPZEbj0MLr4MnG'
    consumer_secret = 'Q8EhARchwkPelPLBOJCLTi4T3wOveHtNoQMiWpfijCElvEVhAv'

    access_token = '926742176452640768-0Oy0PDrdi3vl0SZYIt3sez3cz0tbolC'
    access_token_secret = 'Pf277RMWl5Ee8DyzFEO2B7kmjdXP6F8jYUYch7YsLpxER'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    # Step 3 - Retrieve Tweets
    # Change text you want to search here.
    text = '#Apple'
    # Corresponding csv file will be generated.

    # To get complete tweet text
    public_tweets = api.search(text, tweet_mode='extended')
    sentiment_Sum = 0
    for tweet in public_tweets:
        analysis = TextBlob(tweet.full_text)
        sentiment_Sum += analysis.polarity

    return sentiment_Sum

def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    lastIndex = window_size
    i = 0
    count = len(series)

    while (lastIndex < count):
        X.append(series[i:lastIndex])
        y.append(series[lastIndex])
        i += 1
        lastIndex += 1

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])

    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

def Build_Model():
    # import keras network libraries
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    import keras

    # given - fix random seed - so we can all reproduce the same results on our default time series
    np.random.seed(0)

    model = Sequential()
    model.add(LSTM(5,input_shape=(window_size,1)))
    model.add(Dense(1))

    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.summary()

    return model

### load in and normalize the dataset
dataset =  np.loadtxt('normalized_apple_prices.csv')
print(dataset)

# Time series plot
plt.plot(dataset)
plt.xlabel('time period')
plt.ylabel('normalized series value')

plt.show()

window_size = 7
X,y = window_transform_series(series = dataset,window_size = window_size)



# split our dataset into training / testing sets
train_test_split = int(np.ceil(2*len(y)/float(3)))   # set the split point

# partition the training set
X_train = X[:train_test_split,:]
y_train = y[:train_test_split]

# keep the last chunk for testing
X_test = X[train_test_split:,:]
y_test = y[train_test_split:]

# NOTE: to use keras's RNN LSTM module our input must be reshaped to [samples, window size, stepsize]
X_train = np.asarray(np.reshape(X_train, (X_train.shape[0], window_size, 1)))
X_test = np.asarray(np.reshape(X_test, (X_test.shape[0], window_size, 1)))

model = Build_Model()

# run your model!
model.fit(X_train, y_train, epochs=1000, batch_size=50, verbose=1)

# generate predictions for training
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# print out training and testing errors
training_error = model.evaluate(X_train, y_train, verbose=0)
print('training error = ' + str(training_error))

testing_error = model.evaluate(X_test, y_test, verbose=0)
print('testing error = ' + str(testing_error))

# plot original series
plt.plot(dataset,color = 'k')

# plot training set prediction
split_pt = train_test_split + window_size
plt.plot(np.arange(window_size,split_pt,1),train_predict,color = 'b')

# plot testing set prediction
plt.plot(np.arange(split_pt,split_pt + len(test_predict),1),test_predict,color = 'r')

# pretty up graph
plt.xlabel('day')
plt.ylabel('(normalized) price of Apple stock')
plt.legend(['original series','training fit','testing fit'],loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

trend = 'Neutral'
sentiment_number = Tweet_Sentiment()
if sentiment_number > 0.0:
    trend = 'Positive'
elif sentiment_number < 0.0:
    trend = 'Negative'

print("the twitter shows " + trend + " trend with polarity " + str(sentiment_number))
