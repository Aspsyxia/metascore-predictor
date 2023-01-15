import pandas as pd
import numpy as np

from nltk.sentiment import SentimentIntensityAnalyzer

# declaring stuff for sentiment analysis and importing the 'clean' reviews dataset
analyzer = SentimentIntensityAnalyzer()
compound_values = []

data = pd.read_csv('datasets\\reviews_dataset.csv', header=0)

# checking approximate semantic value of each review and saving it to data frame
for review in data['review']:
    semantic = analyzer.polarity_scores(review)
    compound_values.append(semantic['compound'])

data['semantic'] = compound_values

# adjusting some datasets to have more bad reviews in final set and adjusting semantic values for some
# misleading reviews to give more realistic relation between the review and the score
data.drop(data[(data.title == 'Horizon1') & (data.score > 5)].index, inplace=True)
data.drop(data[(data.title == 'MGSVTPP') & (data.score > 5)].index, inplace=True)
data['semantic'] = np.where((data['semantic'] > 0) & (data['score'] <= 5), data['semantic'] * -1, data['semantic'])
data['semantic'] = np.where((data['semantic'] < 0) & (data['score'] > 5), data['semantic'] * -1, data['semantic'])

# saving final dataset with analysis data to .csv file
data.to_csv('datasets\\reviews_analyzed.csv', index=False)