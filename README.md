# StackExchange Tag Prediction

splitDataDup.py: remove duplicated examples and split 'Train.csv' to n parts

cleanData.py: remove HTML tags and separate words with regular expression
parsing is based on rottentomato56's parse_posts.py
https://github.com/rottentomato56/kaggle-fb-stackoverflow/blob/master/Final/parse_posts.py

trainOne.py: one-vs-rest logistic regression with SGD

coOccurrence.py: tag-word co-occurrence matrix,  combining SGD models

mean_f1.py: mean F1 score (from Kaggle)

web/ contains scripts to run prediction from a website with flask
The site is based on the Michael Herman's article
https://realpython.com/blog/python/flask-by-example-part-3-text-processing-with-requests-beautifulsoup-nltk/

web/TagPrediction.ipynb: brief documentation of the analysis

