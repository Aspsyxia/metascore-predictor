import pandas as pd
import os

# getting current directory location and defining empty list for file names and empty data frame for our final data.
current_dir = os.getcwd()
files_to_filter = []
data = pd.DataFrame()

# getting names of .csv files from current directory.
for file in os.listdir(current_dir):
    f = os.path.join(current_dir, file)
    if os.path.isfile(f) and '.csv' in os.path.basename(f):
        files_to_filter.append(os.path.basename(f))

# connecting data from all .csv files into one. We are taking only english reviews into consideration.
for file_csv in files_to_filter:
    unfiltered_reviews = pd.read_csv(file_csv, names=['id', 'title', 'date', 'score', 'review', 'english'])
    unfiltered_reviews.dropna(inplace=True)
    unfiltered_reviews = unfiltered_reviews[unfiltered_reviews.english == True]
    data = pd.concat([data, unfiltered_reviews])

# saving our final data data frame into .csv file.
data.to_csv('reviews_full.csv', index=False)
