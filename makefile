QUERY = coronavirus

# be careful changing this -- my API key only supports 180 requests an hour
COUNT = 10

demo:
	python3 plumage.py dev/tokeninfo $(QUERY) $(COUNT)
	rm -rf _extract _preprocess _tweets _grams
