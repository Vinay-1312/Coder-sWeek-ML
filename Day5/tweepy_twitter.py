# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 13:34:19 2020

@author: dell
"""
import tweepy 
import pandas as pd
import os
consumer_key = "jjVhJeLKnG97eNjEKTzUNKQR7" 
consumer_secret = "XWXTq5gXBusFe7OLtQy2UJ78K1x3tgufUlf3bze9njX4vncFG8"
access_token = "797036028368396288-TwtfQIUHBzWxISUigkK1pfL5H29aWkf"
access_token_secret = "lIJtCe7ZhJXXWGBcRpWnUWkcWuqw5mfzww02pqaBfBEqy"
userID = "realDonaldTrump"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
tweets = api.user_timeline(screen_name=userID, 
                           # 200 is the maximum allowed count
                           count=200,
                           include_rts = False,
                         
                           tweet_mode = 'extended'
                           )
api = tweepy.API(auth)
all_tweets = []
all_tweets.extend(tweets)
oldest_id = tweets[-1].id
while True:
    tweets = api.user_timeline(screen_name=userID, 
                           # 200 is the maximum allowed count
                           count=200,
                           include_rts = False,
                           max_id = oldest_id - 1,
                           # Necessary to keep full_text 
                           # otherwise only the first 140 words are extracted
                           tweet_mode = 'extended'
                           )
    if len(tweets) == 0:
        break
    oldest_id = tweets[-1].id
    all_tweets.extend(tweets)
    print('N of tweets downloaded till now {}'.format(len(all_tweets)))

from pandas import DataFrame
outtweets = [[tweet.id_str, 
              tweet.created_at, 
              tweet.favorite_count, 
              tweet.retweet_count, 
              tweet.full_text.encode("utf-8").decode("utf-8")] 
             for idx,tweet in enumerate(all_tweets)]
df = DataFrame(outtweets,columns=["id","created_at","favorite_count","retweet_count", "text"])
df.to_csv('%s_tweets.csv' % userID,index=False)
df.head(3)
df.to_csv(r'C:\Users\dell\Desktop\IEEE\Day5\tweet.csv', index = False)