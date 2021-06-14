# Twitter scraping based on:
# https://github.com/MartinBeckUT/TwitterScraper
# Some myPolitics results were rescaled on the third axis
# because of the recent changes in the method: [0-1] instead of [-1,1] and reversed conservative-progressive signs

import snscrape.modules.twitter as sntwitter
import pandas as pd
import csv


def get_tweets_to_csv(username, amount=100):
    if len(username) > 0 and amount > 0:
        tweets_list = []
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper("from:"+username).get_items()):
            if i > amount:
                break
            tweets_list.append([tweet.content])

        # tweets_df = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
        tweets_df = pd.DataFrame(tweets_list, columns=['text'])
        tweets_df.to_csv("tweets\\" + username + ".csv", sep=',', index=False)
        print(f"Loaded {tweets_df['text'].size - 1} tweets of {username}")


def get_tweets_to_str(username, amount=100):
    if len(username) > 0 and amount > 0:
        tweets = ""
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper("from:"+username).get_items()):
            if i > amount:
                break
            tweets += tweet.content + "\n"
        return tweets
    else:
        return ""


def load_mypolitics_data(path="mypolitics_data.csv"):
    csv_usernames = []
    csv_horizontals = []
    csv_verticals = []
    csv_thirds = []
    with open(path) as mypolitics_data:
        csv_reader = csv.reader(mypolitics_data, delimiter=",")
        lines = 0
        for row in csv_reader:
            if lines > 0:
                csv_usernames.append(row[0])
                csv_horizontals.append(float(row[1]))
                csv_verticals.append(float(row[2]))
                csv_thirds.append(float(row[3]))
            lines += 1
        print(f'Loaded {lines} lines from csv.')
    return csv_usernames, csv_horizontals, csv_verticals, csv_thirds


if __name__ == "__main__":
    usernames, horizontals, verticals, thirds = load_mypolitics_data()
    for un in usernames:
        get_tweets_to_csv(un, 1000)

