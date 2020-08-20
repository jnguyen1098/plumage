"""Analysis module."""
# pylint: disable=R0914
# pylint: disable=R0912
# pylint: disable=R0915
# pylint: disable=R0903

import argparse
import csv
import logging
import pickle
import sys
from typing import Dict, List, Tuple

from nltk import ngrams  # type: ignore
from nltk.corpus import stopwords  # type: ignore

MAX_TWEETS = -1
DIVISION = 25
REPORT_LIMIT = 25


def analyze_tweets(tweetin: str, gramin: str, output: str = "") -> None:
    """Analyze Tweets using prior knowledge."""
    logger = logging.getLogger("analyzer")

    logger.info("Unpickling Tweets")
    tweets = pickle.load(open(tweetin, "rb"))

    logger.info("Unpickling n-grams")
    gram_scores = pickle.load(open(gramin, "rb"))

    logger.info("Initializing aspect sentiments")
    tweets_written = 1
    aspects: List[Dict[str, Aspect]] = [{}, {}, {}, {}, {}]
    for i in range(1, 5):
        for aspect, count in gram_scores[i].items():
            aspects[i][aspect] = Aspect(aspect, count)

    logger.info("Mapping ngram aspects to sentiments")
    for tweet in tweets:
        if tweet.positivity > tweet.negativity:
            for i in range(1, 5):
                grams = ngrams(tweet.cleaned_tokens, i)
                for gram in grams:
                    aspects[i][gram].positive += 1
        else:
            for i in range(1, 5):
                grams = ngrams(tweet.cleaned_tokens, i)
                for gram in grams:
                    aspects[i][gram].negative += 1

        if not tweets_written % DIVISION:
            logger.info("Analyzed tweet #%s", tweets_written)
        tweets_written += 1

    # Create stop words list for presentation. These will only be
    # used to filter out 1- and 2-grams. They provide more useful
    # context in 3- and 4-grams, though
    stop_words = stopwords.words("english")
    alt_stops = [
        "dont",
        "arent",
        "isnt",
        "didnt",
        "hadnt",
        "hasnt",
        "couldnt",
        "shant",
        "shouldnt",
        "wouldns",
        "wasnt",
        "werent",
        "wont",
        "neednt",
        "mustnt",
        "mightnt",
        "thats",
        "get",
        "go",
        "like",
    ]
    for stop in alt_stops:
        stop_words.append(stop)

    # Output 1-grams but without stop-words
    print()
    attempt = j = 0
    sorted_grams = sorted(aspects[1], key=aspects[1].get, reverse=True)
    logger.info("Top 25 1-grams (stop-words removed):")
    logger.info("|             %s-gram             | Count |  Positivity  |  Negativity  |", i)
    logger.info("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    while j < REPORT_LIMIT and attempt < len(sorted_grams):
        if sorted_grams[attempt][0].lower() not in str(stop_words):
            aspect = sorted_grams[attempt][0]
            count = aspects[1][sorted_grams[attempt]].count
            positive = aspects[1][sorted_grams[attempt]].positive
            negative = aspects[1][sorted_grams[attempt]].negative
            logger.info(
                "| %30s | %5s | %3s (%5.4s%%) | %3s (%5.4s%%) |",
                aspect,
                count,
                positive,
                100 * (positive / count),
                negative,
                100 * (negative / count),
            )
            j += 1
        attempt += 1

    # Output 2,4-grams
    for i in range(2, 5):
        print()
        sorted_grams = sorted(aspects[i], key=aspects[i].get, reverse=True)
        logger.info("|             %s-gram             | Count |  Positivity  |  Negativity  |", i)
        logger.info("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
        for j in range(REPORT_LIMIT):
            aspect = " ".join(sorted_grams[j])
            count = aspects[i][sorted_grams[j]].count
            positive = aspects[i][sorted_grams[j]].positive
            negative = aspects[i][sorted_grams[j]].negative
            logger.info(
                "| %30s | %5s | %3s (%5.4s%%) | %3s (%5.4s%%) |",
                aspect,
                count,
                positive,
                100 * (positive / count),
                negative,
                100 * (negative / count),
            )

    # Export to .CSV file if specified
    if output:
        logger.info("Printing report to %s", output)
        output_fp = open(output, "w", encoding="utf-8")
        tweet_writer = csv.writer(output_fp)

    for i in range(1, 5):
        counter = 0
        tmp = sorted(aspects[i], key=aspects[i].get, reverse=True)
        for aspect_txt in tmp:
            aspect = aspects[i][aspect_txt]
            if output:
                tweet_writer.writerow(
                    [
                        i,
                        " ".join(aspect.aspect),
                        aspect.count,
                        aspect.positive,
                        aspect.negative,
                        100 * (aspect.positive / aspect.count),
                        100 * (aspect.negative / aspect.count),
                    ]
                )
            if counter == REPORT_LIMIT:
                break
            counter += 1

    logger.info("File report created" if output else "Done!")


class Aspect:
    """Record for aspect."""

    def __init__(self, aspect: Tuple[str], count: int) -> None:
        """Create new Aspect."""
        self.aspect = aspect
        self.count = count
        self.positive = 0
        self.negative = 0

    def __lt__(self, other) -> bool:  # type: ignore
        """Overload less-than operator."""
        return self.count < other.count  # type: ignore


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


def main() -> int:
    """Execute standalone."""
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument("tweetin", help="input Tweet pickle")
    arg_p.add_argument("gramin", help="input Gram pickle")
    arg_p.add_argument("--output", help="optional output")

    args = arg_p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s | %(name)s] %(message)s",
    )

    analyze_tweets(args.tweetin, args.gramin, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
