import pandas as pd
import os

# getting reviews directory location and defining empty list for file names and empty data frame for our final data.
reviews_dir = f"{os.getcwd()}\\csv"
files_to_filter = []
data = pd.DataFrame()

# getting names of .csv files from current directory.
for file in os.listdir(reviews_dir):
    f = os.path.join(reviews_dir, file)
    if os.path.isfile(f) and '.csv' in os.path.basename(f):
        files_to_filter.append(os.path.basename(f))

# connecting data from all .csv files into one. We are taking only english reviews into consideration.
for file_csv in files_to_filter:
    file_path = f"{reviews_dir}\\{file_csv}"
    unfiltered_reviews = pd.read_csv(file_path, names=['id', 'title', 'date', 'score', 'review', 'english'])
    unfiltered_reviews = unfiltered_reviews[unfiltered_reviews.english == True]
    unfiltered_reviews.dropna(inplace=True)
    data = pd.concat([data, unfiltered_reviews])

# saving our final data data frame into .csv file.
data.to_csv('datasets\\reviews_dataset.csv', index=False)
