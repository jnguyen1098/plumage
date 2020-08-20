"""Demo module."""

import argparse
import logging
import sys

import modules.extract as extract
import modules.preprocess as preprocess
import modules.mine as mine
import modules.analyze as analyze

def main() -> int:
    """Execute main."""
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument("tokenfile", help="see README for details")
    arg_p.add_argument("query", help="search term")
    arg_p.add_argument("count", help="number of times to get 100 Tweets")

    args = arg_p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s | %(name)s] %(message)s",
    )

    print()
    logging.info("Initiating extraction module")
    extract.extract_tweets(args.tokenfile, args.query, "_extract", count=int(args.count), wait=1)

    print()
    logging.info("Initiating preprocessing module")
    preprocess.preprocess_tweets("_extract", "_preprocess")

    print()
    logging.info("Initiating mining module")
    mine.mine_tweets("_preprocess", "_tweets", "_grams")

    print()
    logging.info("Initiating analysis module")
    analyze.analyze_tweets("_tweets", "_grams", "_analysis")

    return 0

if __name__ == "__main__":
    sys.exit(main())
