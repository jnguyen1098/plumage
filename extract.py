"""Extraction module."""
# pylint: disable=C0330

#import tensorflow
import argparse
import csv
import logging
import os
import sys
import time

import tweepy  # type: ignore


def extract_tweets(secret: str, query: str, outfile: str, count: int = 0, wait: int = 300) -> None:
    """Extract Tweets using the Tweepy API."""
    logger = logging.getLogger("extracter")
    logger.info("Authenticating with Tweepy")

    logger.info("Reading secrets file %s", secret)
    token_fp = open(secret, "r")
    auth = tweepy.OAuthHandler(token_fp.readline().strip(), token_fp.readline().strip())
    auth.set_access_token(token_fp.readline().strip(), token_fp.readline().strip())
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    token_fp.close()

    logger.info("Attempting to authenticate")
    api.verify_credentials()

    logger.info("Authenticated! Examining outfile.")
    if not os.path.exists(outfile):
        logger.info("%s doesn't exist - it will be created.", outfile)
        file_p = open(outfile, "w", encoding="utf-8")
        tweet_writer = csv.writer(file_p)
        tweet_writer.writerow(
            [
                "full_text",
                "created_at",
                "source",
                "id",
                "retweet_count",
                "favorite_count",
                "user_name",
                "user_id_str",
                "user_handle",
                "user_location",
                "user_desc",
                "user_protected",
                "user_followers",
                "user_created",
                "user_verified",
                "user_tweet_count",
            ]
        )
    else:
        logger.info("%s exists - will append.", outfile)
        file_p = open(outfile, "a", encoding="utf-8")
        tweet_writer = csv.writer(file_p)

    logger.info("Starting Tweet extraction for query '%s'", query)

    if not count:
        logger.info("(executing forever)")
    else:
        logger.info("(executing %s times)", count)

    i = 1
    bookmark = "1"

    while True:
        # Our search query.
        #
        # q - search query. We use the -filter:retweets
        #     specifier in order to prune any retweets.
        #     Otherwise we'd have to prune Tweets that
        #     are prefaced with 'RT'
        #
        # lang - English Tweets only
        #
        # count - 100 is the max as per the Twitter API
        #
        # tweet_mode - we use extended tweet mode in
        #     order to access Tweets that are greater
        #     than 140 char. in length this is to keep
        #     legacy Twitter API applications intact
        #
        # result_type - we use recent so as to create
        #     a chronological record of Tweets
        #
        # since_id - we keep track of the last Tweet
        #     saved and use it as a bookmark in order
        #     to only get the Tweets coming after it
        #
        for tweet in api.search(
            q=f"{query} -filter:retweets",
            lang="en",
            count=100,
            tweet_mode="extended",
            result_type="recent",
            max_id=bookmark,
        ):
            # These are the features we write
            tweet_writer.writerow(
                [
                    tweet.full_text,
                    tweet.created_at,
                    tweet.source,
                    tweet.id_str,
                    tweet.retweet_count,
                    tweet.favorite_count,
                    tweet.user.name,
                    tweet.user.id_str,
                    tweet.user.screen_name,
                    tweet.user.location,
                    tweet.user.description,
                    tweet.user.protected,
                    tweet.user.followers_count,
                    tweet.user.created_at,
                    tweet.user.verified,
                    tweet.user.statuses_count,
                ]
            )

            # Flush the stream every time just in case
            file_p.flush()

            # Set the most recent Tweet as a bookmark
            bookmark = tweet.id_str

        # Transparency/monitoring
        limits = api.rate_limit_status()
        rem = limits["resources"]["application"]["/application/rate_limit_status"]["remaining"]
        logger.info("Tweets written to %s (%s hourly API accesses left)", outfile, rem)

        # Do not loop if demo
        if i == count:
            break
        i += 1

        # Respect API
        time.sleep(wait)


def main() -> int:
    """Execute standalone."""
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument("tokenfile", help="see README for details")
    arg_p.add_argument("query", help="search term")
    arg_p.add_argument("outfile", help="output file")

    args = arg_p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s | %(name)s] %(message)s",
    )

    extract_tweets(args.tokenfile, args.query, args.outfile, count=0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
