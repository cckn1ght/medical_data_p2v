from pymongo import MongoClient

client = MongoClient()
db = client.dustin_db
collection = db.melatonin


tweet_id_set = set()
for tweet in collection.find():
    tweet_id_set.add(tweet['tweet_id'])


def id_from_awstweets():
    db = client.ravishData
    collection = db.AWSTweets
    for tweet in collection.find():
        yield tweet['tweet_id']
