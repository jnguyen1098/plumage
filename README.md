# Plumage
This app performs a sentiment analysis on Twitter posts.

This was a research project I did over the summer—you can read the journal [here](https://github.com/jnguyen1098/plumage/blob/master/QQuibbles.pdf).

You should probably install the dependencies (`python -m pip install -r requirements.txt`)

You need a file in `dev/` called `tokeninfo` containing four lines:
- Consumer key
- Consumer secret
- Access token
- Access secret

After this, you can run a demo using the `makefile` by executing `make demo`.

Edit your parameters (including search query) accordingly in the `makefile`.

This project consists of four modules:

- `extract.py` — extracts raw Tweets into

- `preprocess.py` — cleans up the Tweets and prunes any non-promising posts (i.e. too "objective" for analysis)

- `mine.py` — classifies the Tweets and atomizes them into n-gram aspects for analysis

- `analyze.py` — takes the data from `mine.py` and creates a report determining the sentiments from aspects

`plumage.py` is just a demo driver that runs these four modules in sequence.
