"""Data Mining Module."""
# pylint: disable=C0330
# pylint: disable=R0914
# pylint: disable=R0912
# pylint: disable=R0915
# pylint: disable=R0902
# pylint: disable=R0903

#import tensorflow
import argparse
import csv
import json
import logging
import pickle
import random
import string
import sys
from typing import Dict, Iterator, List

from nltk import NaiveBayesClassifier, classify, ngrams  # type: ignore
from nltk.corpus import twitter_samples  # type: ignore
from nltk.stem.wordnet import WordNetLemmatizer  # type: ignore
from nltk.tag import pos_tag  # type: ignore

MAX_TWEETS = -1
DIVISION = 25
SUBJECTIVITY_THRESHOLD = 0.30


def mine_tweets(infile: str, tweetout: str, gramout: str) -> None:
    """Classify, prune, and atomize Tweets."""
    logger = logging.getLogger("miner")

    logger.info("Gathering and tokenizing positive tweets")
    positive_tweet_tokens = twitter_samples.tokenized("positive_tweets.json")

    logger.info("Gathering and tokenizing negative tweets")
    negative_tweet_tokens = twitter_samples.tokenized("negative_tweets.json")

    logger.info("Cleaning model tokens")
    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    # Clean tokens
    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(normalize(tokens))

    # Clean tokens
    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(normalize(tokens))

    logger.info("Building Tweet corpus")
    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)  # type: ignore
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)  # type: ignore

    # Mark positive Tweets as such
    positive_dataset = [(tweet_dict, "Positive") for tweet_dict in positive_tokens_for_model]

    # Mark negative Tweets as such
    negative_dataset = [(tweet_dict, "Negative") for tweet_dict in negative_tokens_for_model]

    # Create unified dataset and shuffle it
    dataset = positive_dataset + negative_dataset
    random.shuffle(dataset)

    # Train the data using the first 70% as
    # training data, and the last 30% as
    # testing data.
    logger.info("70% training, 30% testing")
    train_data = dataset[:7000]
    test_data = dataset[7000:]

    logger.info("Training...")
    classifier = NaiveBayesClassifier.train(train_data)

    logger.info("Accuracy is: %s", classify.accuracy(classifier, test_data))

    logger.info("Classifying Tweets")
    tweets = []

    with open(infile, "r") as csv_file:
        logger.info("Opened %s", infile)

        csv_reader = csv.reader(csv_file, delimiter=",")
        logger.info("Attached CSV reader to %s successfully", infile)

        # Counts processed Tweets and rejected ones
        counter: int = 0
        subject_reject: int = 0

        # Iterate
        for tweet in csv_reader:

            # Printing
            if not counter % DIVISION:
                logger.info("Read in %s Tweets so far...", counter)

            # For debugging
            if counter == MAX_TWEETS:
                break

            # Classify Tweet
            new_tweet = Tweet(tweet)
            dist = classifier.prob_classify(
                dict([token, True] for token in new_tweet.cleaned_tokens)  # type: ignore
            )
            new_tweet.positivity = dist.prob("Positive")
            new_tweet.negativity = dist.prob("Negative")
            new_tweet.difference = abs(new_tweet.positivity - new_tweet.negativity)

            # Assess the subjectivity of the Tweet
            if new_tweet.difference > SUBJECTIVITY_THRESHOLD:
                tweets.append(new_tweet)
            else:
                subject_reject += 1

            # Count
            counter += 1

    logger.info("Processed %s Tweets", len(tweets))
    logger.info("%s Tweets were rejected for not being subjective enough", subject_reject)

    # Pickle Tweets
    pickle.dump(tweets, open(tweetout, "wb"))
    logger.info("Pickled %s Tweets", len(tweets))

    # Storing our n-gram occurrences
    gram_scores: List[Dict[str, int]] = [{}, {}, {}, {}, {}]

    # Counting n-grams
    for i in range(1, 5):
        logger.info("Creating %s-grams", i)

        # Iterate
        for tweet in tweets:  # type: ignore

            # Create n-grams
            grams = ngrams(tweet.cleaned_tokens, i)  # type: ignore

            # Count every gram
            for gram in grams:

                # Create record for new n-gram
                if gram not in gram_scores[i]:
                    gram_scores[i][gram] = 1

                # Update existing record
                else:
                    gram_scores[i][gram] += 1

    # Serialize n-grams to file
    with open(gramout, "wb") as gramout_fp:
        pickle.dump(gram_scores, gramout_fp)


class Tweet:
    """Tweet object."""

    def __init__(self, tweet_row: List[str]) -> None:
        """Initialize Tweet object."""
        # Existing members
        self.full_text = tweet_row[0]
        self.created_at = tweet_row[1]
        self.source = tweet_row[2]
        self.tweet_id = tweet_row[3]
        self.retweet_count = tweet_row[4]
        self.favorite_count = tweet_row[5]
        self.user_name = tweet_row[6]
        self.user_id_str = tweet_row[7]
        self.user_handle = tweet_row[8]
        self.user_location = tweet_row[9]
        self.user_desc = tweet_row[10]
        self.user_protected = tweet_row[11]
        self.user_followers = tweet_row[12]
        self.user_created = tweet_row[13]
        self.user_verified = tweet_row[14]
        self.user_tweet_count = tweet_row[15]
        self.cleaned_text = tweet_row[16]
        self.cleaned_tokens = json.loads(tweet_row[17])

        self.positivity = -1
        self.negativity = -1
        self.difference = -1


def normalize(tweet_tokens: List[str]) -> List[str]:
    """Lemmatize a Twitter post.."""
    cleaned_tokens = []

    #  Part of Speech tagging
    for token, tag in pos_tag(tweet_tokens):

        if tag.startswith("NN"):
            pos = "n"
        elif tag.startswith("VB"):
            pos = "v"
        else:
            pos = "a"

        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation:
            cleaned_tokens.append(token.lower())

    return cleaned_tokens


def get_all_words(cleaned_tokens_list: List[List[str]]) -> Iterator[str]:
    """Yield generator for all words."""
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


def get_tweets_for_model(cleaned_tokens_list):  # type: ignore
    """Yield dicts for Tweets."""
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)  # type: ignore


def main() -> int:
    """Execute standalone."""
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument("infile", help="input .CSV file")
    arg_p.add_argument("tweetout", help="output Tweets .CSV file")
    arg_p.add_argument("gramout", help="output n-grams .PICKLE file")

    args = arg_p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s | %(name)s] %(message)s",
    )

    mine_tweets(args.infile, args.tweetout, args.gramout)

    return 0


if __name__ == "__main__":
    sys.exit(main())
