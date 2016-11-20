import pandas as pd
import prettytable as pt
import numpy as np
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup as bs
from sklearn.feature_extraction.text import CountVectorizer

def cleaner(data_frame, h, r):
    """
    Function to clean up a string
    :param data_frame: data frame
    :param h: column number
    :param r: row
    :return: cleaned data print
    """

    raw_data = str(data_frame[h][r])

    # Remove HTML Markup
    text_data = bs(raw_data, "lxml").get_text()

    # Add space if missing after periods, semi colons and quotes
    text_data = re.sub("(?<=\.)(?=[a-zA-Z])", " ", text_data)
    text_data = re.sub('(?<=\:)(?=[a-zA-Z])', ' ', text_data)
    text_data = re.sub('(?<=\")(?=[a-zA-Z])', ' ', text_data)
    text_data = re.sub('(?<=\[0-9])(?=[a-zA-Z])', ' ', text_data)

    # Remove punctuation
    cleaned_data = "".join(letter for letter in text_data if (letter.isalnum() or letter == " "))

    # Lower case
    lower_case = cleaned_data.lower()

    # Create list of words
    word_list = lower_case.split()

    # Remove stop words
    stops = set(stopwords.words("english"))
    passage = " ".join([word for word in word_list if word not in stops])

    return passage

if __name__ == "__main__":

    # Change output paramaters
    desired_width = 70
    pd.set_option('display.width', desired_width)

    # Read data. line 0 contains headers
    train_data = pd.read_csv('train.tsv', header=0, delimiter='\t')

    # Get headers
    header = np.array(train_data.columns.values)
    print('Fields: \n\n', header.reshape(23,1))

    # Print Data descriptions
    print(train_data.describe())

    rows = train_data['Product Long Description'].size
    print('\nNumber of rows to clean: ', rows, '\n')
    clean_reviews = []

    for i in range(1, rows):
        if i % 1000 == 0: print("review {} of {} completed".format(i, rows))
        clean_reviews.append(cleaner(train_data, header[12], i))

    print('Cleaned Rows subset preview:\n\n', np.array(clean_reviews[:100]).reshape((100,1)))

    # Initialize 'CountVector' Object
    vectorizer = CountVectorizer(analyzer= "word",
                                 tokenizer= None,
                                 preprocessor= None,
                                 stop_words= None,
                                 max_features= 5000
                                 )

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(clean_reviews).toarray()

    vocab = vectorizer.get_feature_names()

    print("\nShape of Train data: ", train_data_features.shape)
    print("\nWords in vectorizor:\n", vocab)

    # Sum count of each word across all descriptions
    dist = np.sum(train_data_features, axis=0)

    print('Training Random Forest....')
    from sklearn.ensemble import RandomForestClassifier

    # Initialize Random Forest Classifier with 100 trees
    forest = RandomForestClassifier(n_estimators= 100)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as response variables
    forest = forest.fit(train_data_features, train_data['tag'][1:])

    # Get Test data
    test_data = pd.read_csv('test.tsv',header=0, delimiter='\t')

    # Verify shape:
    print('Test data size: ', test_data.shape)

    r = len(test_data['Product Long Description'])
    clean_test_reviews = []

    header = np.array(test_data.columns.values)

    for x in range(1, r):

        cleaned_data = cleaner(test_data, header[12], x)

        clean_test_reviews.append(cleaned_data)

    # Create Features
    test_data_features = vectorizer.transform(clean_test_reviews).toarray()

    # Get result
    result = forest.predict(test_data_features)

    # Get output
    output = pd.DataFrame( data={"id" : test_data['Product Name'][1:], "shelf" : result})

    # Push output
    output.to_csv("Bag of Words Model.csv", index=False)


