import pandas as pd
import numpy as np

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


# declaring stuff for sentiment analysis and importing the 'clean' reviews dataset
analyzer = SentimentIntensityAnalyzer()
tokenizer = RegexpTokenizer(r'\w+')
stopwords = set(stopwords.words('english'))
compound_values = []
reviews_processed = []

data = pd.read_csv('datasets\\reviews_dataset.csv', header=0)

# processing each review by tokenizing it, removing stopwords and punctuation signs and checking approximate semantic
# value of each review and saving it to data frame
for review in data['review']:
    review_tokenized = tokenizer.tokenize(review)
    review_tokenized = [x.lower() for x in review_tokenized]
    for token in review_tokenized:
        if token in stopwords:
            review_tokenized.remove(token)
    review_processed = " ".join(review_tokenized)
    semantic = analyzer.polarity_scores(review_processed)
    reviews_processed.append(review_processed)
    compound_values.append(semantic['compound'])

data['semantic'] = compound_values
data['review'] = reviews_processed

# adjusting some datasets to have more bad reviews in final set and adjusting semantic values for some
# misleading reviews to give more realistic relation between the review and the score
data.drop(data[(data.title == 'Horizon1') & (data.score > 5)].index, inplace=True)
data.drop(data[(data.title == 'MGSVTPP') & (data.score > 5)].index, inplace=True)
data['semantic'] = np.where((data['semantic'] > 0) & (data['score'] <= 5), data['semantic'] * -1, data['semantic'])
data['semantic'] = np.where((data['semantic'] < 0) & (data['score'] > 5), data['semantic'] * -1, data['semantic'])

# saving final dataset with analysis data to .csv file
data.to_csv('datasets\\reviews_processed.csv', index=False)
