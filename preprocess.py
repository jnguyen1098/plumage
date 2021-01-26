"""Preprocessing module."""
# pylint: disable=C0330
# pylint: disable=R0902

#import tensorflow
import argparse
import csv
import json
import logging
import re
import string
import sys
from typing import List

import preprocessor  # type: ignore
import nltk  # type: ignore
from nltk.stem.wordnet import WordNetLemmatizer  # type: ignore
from nltk.tag import pos_tag  # type: ignore
from nltk.tokenize import word_tokenize  # type: ignore

MAX_TWEETS = -1
DIVISION = 25


def preprocess_tweets(infile: str, outfile: str) -> None:
    """Remove redundant and non-objective posts."""
    logger = logging.getLogger("preprocessor")

    # Number of Tweets read
    counter: int = 0

    # List of all Tweets
    tweets: List[Tweet] = []

    # Begin reading
    with open(infile, "r") as csv_file:

        # CSV reader
        csv_reader = csv.reader(csv_file, delimiter=",")
        logger.info("Attached CSV reader")

        # Number of Tweets deleted due to URL
        url_blocked = 0

        # Iterate
        for tweet in csv_reader:

            # Messaging checkpoints
            if not counter % DIVISION:
                logger.info("Processed %s Tweets", counter)

            # Break at limit
            if counter == MAX_TWEETS:
                break

            # Only add Tweet if it doesn't contain a URL.
            # As per Ejieh's master's thesis, the vast majority
            # of posts with URLs lack any subjectivity.
            ptn = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
            if not bool(re.search(ptn, tweet[0])):
                tweets.append(Tweet(tweet))
            else:
                url_blocked += 1
            counter += 1

    logger.info("Read %s Tweets in total", counter)

    # Finishing message
    logger.info("Only %s Tweets were kept", len(tweets))
    with open(outfile, "w", encoding="utf-8") as output_file:
        tweet_writer = csv.writer(output_file)
        i = 1

        for tweet in tweets:  # type: ignore
            tweet_writer.writerow(
                [
                    tweet.full_text,  # type: ignore
                    tweet.created_at,  # type: ignore
                    tweet.source,  # type: ignore
                    tweet.tweet_id,  # type: ignore
                    tweet.retweet_count,  # type: ignore
                    tweet.favorite_count,  # type: ignore
                    tweet.user_name,  # type: ignore
                    tweet.user_id_str,  # type: ignore
                    tweet.user_handle,  # type: ignore
                    tweet.user_location,  # type: ignore
                    tweet.user_desc,  # type: ignore
                    tweet.user_protected,  # type: ignore
                    tweet.user_followers,  # type: ignore
                    tweet.user_created,  # type: ignore
                    tweet.user_verified,  # type: ignore
                    tweet.user_tweet_count,  # type: ignore
                    tweet.cleaned_text,  # type: ignore
                    json.dumps(tweet.cleaned_tokens),  # type: ignore
                ]
            )

            if not i % DIVISION:
                logger.info("Wrote Tweet #%s", i)
            i += 1
    logger.info("Wrote %s Tweets in total", len(tweets))


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

        # New members
        self.cleaned_text = Tweet.clean_tweet(self.full_text)
        self.cleaned_tokens = Tweet.normalize(word_tokenize(self.cleaned_text))

    @staticmethod
    def clean_tweet(full_text: str) -> str:
        """Remove meaningless data, in-place, from Tweets."""
        # Said Ozcan's preprocessor
        cleaned = str(preprocessor.clean(full_text))

        # Remove any remnant mentions
        cleaned = str(re.sub(r"@[A-Za-z0-9_]+", "", cleaned))

        # Remove non-alpha
        cleaned = str(re.sub(r"[^A-Za-z ]+", "", cleaned))

        return cleaned

    @staticmethod
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


def main() -> int:
    """Execute standalone."""
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument("infile", help="input .CSV file")
    arg_p.add_argument("outfile", help="output .CSV file")

    args = arg_p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s | %(name)s] %(message)s",
    )

    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("wordnet")
    nltk.download("twitter_samples")
    nltk.download("stopwords")

    preprocess_tweets(args.infile, args.outfile)

    return 0


if __name__ == "__main__":
    sys.exit(main())
