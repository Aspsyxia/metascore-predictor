import pandas as pd
import numpy as np

from nltk.sentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
compound_values = []
pos_values = []
neg_values = []

data = pd.read_csv('reviews_full.csv', header=0, index_col=False)

# checking approximate semantic value of each review and saving it to dataset
for review in data['review']:
    semantic = analyzer.polarity_scores(review)
    compound_values.append(semantic['compound'])
    # neg_values.append(semantic['neg'])
    # pos_values.append(semantic['pos'])

data['semantic'] = compound_values
# data['pos'] = pos_values
# data['neg'] = neg_values

data.drop(data[(data.title == 'Horizon1') & (data.score > 5)].index, inplace=True)
data.drop(data[(data.title == 'MGSVTPP') & (data.score > 5)].index, inplace=True)
data['semantic'] = np.where((data['semantic'] > 0) & (data['score'] <= 5), data['semantic'] * -1, data['semantic'])
data['semantic'] = np.where((data['semantic'] < 0) & (data['score'] > 5), data['semantic'] * -1, data['semantic'])
# data['pos'] = np.where((data['pos'] > 0) & (data['score'] <= 5), data['pos'] / 1.5, data['pos'])
# data['pos'] = np.where((data['pos'] < 0) & (data['score'] > 5), data['pos'] * 1.5, data['pos'])

data.to_csv('check\\reviews_analyzed.csv', index=False)