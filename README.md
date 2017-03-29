Classification of data is a crucial task in the current data science world. There is a huge amount of data and in order to use that data in a productive way for analysis, prediction, learning etc. we need to classify the data in the classes or categories we decide on.
In this project we used supervised machine learning methodologies to classify Yelp reviews into 2 categories, positive or negative. We chose five supervised classification algorithms: Decision Trees, Linear SVC, Naive Bayes, Perceptron and RNN-GRU, and classified the processed dataset into either of the positive or negative class using these algorithms. The effectiveness of each algorithm was analyzed and compared using the precision and recall scores.

Data Preprocessing:
1. Data Set:  For this project we used Yelp dataset[a] for reviews about various businesses. This review data set has more than 2.6 million reviews in the following format:
{"votes": {"funny": 0, "useful": 0, "cool": 0}, "user_id": <user_id>, "review_id": <review_id>, "stars": 2, "date": <date>, "text": <review>, "type": "review", "business_id": <business_id>}

2. Data Selection: Yelp dataset is very huge containing more than 2.6 million reviews. Processing this huge data on limited compute power is next to impossible. Also in the dataset we can find lot of meaningless data. Hence, the number of reviews to be processed had to be reduced by collecting only the quality data that makes our algorithms to perform well. Following rules were implemented to reduce the data from 2.6 million to 17798:
a. [“cool”]>=2 and [“useful”]>3 and [“funny”]=0 and stars = 4 or 5
b. [“cool”]>=1 and [“useful”]>2 and [“funny”]=0 and stars = 1 or 2

3. Data Annotation: We made following decisions on star ratings:
a. Negative review: Star ratings 1 and 2
b. Positive review: Star ratings 4 and 5
Then we annotated each review with the positive or negative sentiment.

4. Feature Selection: The text reviews we selected had to be processed to be used by our algorithm. Following operations were done on the text reviews:
a. Tokenizing (We tried on unigrams, bigrams and trigrams, but settled with unigrams for efficiency, high accuracy and compatibility with other algorithms.)[c][d]
b. Removing the delimiters like “\“; |, | \* | \n| \. | \? | \) | \(”.
c. Pos Tagging. After Pos Tagging only NNP, NN, NNS, NNPS, JJ, JJR, JJS and VBZ tags were selected.[c]

5. Stemming and Lemmatization: The tokens generated from the last step were stemmed using nltk’s PorterStemmer and lemmatized using nltk’s WordNetLemmatizer. This step was to reduce the lexicons size and have only one word of same root.[1][e][f]

Data Formulation:
The pre-processed data had to be formulated in the manner in which our algorithms can use that for training and prediction. We decided to use 12K processed reviews as the training set and remaining 5798 processed reviews as development or prediction set. Each of the set has almost equal percentage of positive and negative processed reviews. Each record finally looks like this:
{“sentiment”: 1, “tokens”: <list of tokens>} (Positive review)
{“sentiment”: 0, “tokens”: <list of tokens>} (Negative review) or
{“sentiment”: -1, “tokens”: <list of tokens>}(For perceptron model)

1. Feature Set: This is the unique set of tokens that were found from the last step. This set can also be called lexicons. Single token can be present in several reviews. But need a single occurrence of each token. This set is used by Naïve Bayes and Perceptron algorithm as the input for learning and predicting.
2. TF-IDF Matrix: Term frequency and Inverse document frequency matrix. This matrix is used by Decision Trees algorithm and Linear SVC algorithm. These algorithms have nothing to do with the actual words/tokens, hence we need to convert them into numbers. So, each token in the lexicon is given an index number and a sparse matrix is created with count of occurrences of each token in the given review. The number of rows in the matrix is equal to the number of review records and number of columns is equal to the count of lexicons.
3. One Hot Sparse Matrix: This matrix is used by RNN-GRU algorithm. This algorithm works on the sequence of the words that occurred in the document. Hence, one hot sparse matrix maintains the order of word occurrence. Each row contains the index of each word in the document as per the lexicons. The column length is fixed to largest review length.

Each of these formulation data structure is created separately for training and development dataset.

Training:
Using the formulated training data the model is trained using our five algorithms. As described earlier Naïve Bayes and Perceptron will be trained using Feature Set, Linear SVC and Decision Trees will be trained using TF-IDF matrix and RNN-GRU will be trained using One Hot Sparse matrix. The output of this step is the trained model.

Prediction:
The trained model will be received from the training step of the project. Using this model and the formulated development data we will predict the sentiments of the development data. Individual algorithm will work on the prediction of the sentiments and finally a list of prediction labels will be returned.

Evaluation:
Using the list of labels/sentiments we received from the prediction step and the list of correct labels/sentiments the evaluation metrics are calculated. Here we calculate precision and recall score for positive and negative sentiments. The algorithms’ accuracy in predicting the sentiments of the reviews is compared using these scores.[g]
Data preprocessing and data formulation steps are common for all the five algorithms. As described each algorithm will use the required formulated data structure and proceed further.