import json
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import pickle
file_prefix = "processed-data/"


def train(train_vecs, train_tags):
    clf = DecisionTreeClassifier(max_depth=15, criterion="entropy")  # construct a decision tree
    clf.fit(train_vecs, train_tags)
    with open("intermediate/dt_trained_dumps.bin", 'wb') as fs:
        fs.write(pickle.dumps(clf))
    print("Classifier Trained...")
    return clf


def classify(clf, vecs, tags):
    predicted = clf.predict(vecs)
    print("accuracy score: ", metrics.accuracy_score(tags, predicted))
    print("precision score: ", metrics.precision_score(tags, predicted, pos_label=None, average='weighted'))
    print("recall score: ", metrics.recall_score(tags, predicted, pos_label=None, average='weighted'))
    print("classification_report: \n ", metrics.classification_report(tags, predicted))


def main():
    print("Reading...")
    with open(file_prefix + "training-dataset.json", "r") as f:
        map_sentiment_train = json.loads(f.read())

    with open(file_prefix + "tf-idf-matrix.json", "r") as f:
        tf_vector_train = json.loads(f.read())

    with open(file_prefix + "dev-dataset.json", "r") as f:
        map_sentiment_dev = json.loads(f.read())

    with open(file_prefix + "dev-tf-idf-matrix.json", "r") as f:
        tf_vector_dev = json.loads(f.read())

    print("Training")

    train_vecs = np.array(tf_vector_train)
    train_tags = np.array(map_sentiment_train)
    clf = train(train_vecs, train_tags)

    print("Classifying")
    dev_vecs = np.array(tf_vector_dev)
    dev_tags = np.array(map_sentiment_dev)

    # with open("intermediate/dt_trained_dumps.bin", 'rb') as fs:
    #     clf = pickle.loads(fs.read())
    classify(clf, dev_vecs, dev_tags)


if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    main()
    end = timeit.default_timer()
    print("\n Time taken: " + str(end - start))