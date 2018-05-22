import tweepy
from textblob import TextBlob
import csv

#Standard twitter developer account So only latest 7 days data only will be available.

# Step 1 - Authenticate
consumer_key = 'loRzO0qpBuSVPZEbj0MLr4MnG'
consumer_secret = 'Q8EhARchwkPelPLBOJCLTi4T3wOveHtNoQMiWpfijCElvEVhAv'

access_token = '926742176452640768-0Oy0PDrdi3vl0SZYIt3sez3cz0tbolC'
access_token_secret = 'Pf277RMWl5Ee8DyzFEO2B7kmjdXP6F8jYUYch7YsLpxER'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# Step 3 - Retrieve Tweets
#Change text you want to search here.
text = 'Lucifer'
#Corresponding csv file will be generated.
filename = text + '.csv'
public_tweets = api.search(text)

for tweet in public_tweets:
    print(tweet.text)

    # Step 4 Perform Sentiment Analysis on Tweets
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)
    print("")

dictData = {}
#TODO:Instead of printing out each tweet, save each Tweet to a CSV file and label each one as either 'positive' or 'negative', depending on the sentiment
#  You can decide the sentiment polarity threshold yourself
for tweet in public_tweets:
    analysis =TextBlob(tweet.text)
    dictData[tweet.text] = ('positive' if analysis.polarity > 0.0 else 'negative')

#Unicode character mappng frequired as tweet may contain unicode characters

with open(filename,'w',encoding='utf-8') as csv_file:
    w = csv.writer(csv_file)
    for key,value in dictData.items():
        s = key + ',' + value
        w.writerow(s)