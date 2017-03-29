import json
import numpy as np
from sklearn import metrics, model_selection, tree
from sklearn.naive_bayes import GaussianNB
import pickle
file_prefix = "processed-data/"


def train(train_vecs, train_tags):
    gnb = GaussianNB()  # construct a gaussian based model
    gnb.fit(train_vecs, train_tags)
    with open("intermediate/nb_trained_dumps.bin", 'wb') as fs:
        fs.write(pickle.dumps(gnb))
    print("Classifier Trained...")
    return gnb


def classify(clf, vecs, tags):
    predicted = clf.predict(vecs)
    print("accuracy score: ", metrics.accuracy_score(tags, predicted))
    print("precision score: ", metrics.precision_score(tags, predicted, pos_label=None, average='weighted'))
    print("recall score: ", metrics.recall_score(tags, predicted, pos_label=None, average='weighted'))
    print("classification_report: \n ", metrics.classification_report(tags, predicted))


def main():
    with open(file_prefix + "training-dataset.json", "r") as f:
        map_sentiment_train = json.loads(f.read())

    print("Training.....")
    print("#" * 70)
    with open(file_prefix + "tf-idf-matrix.json", "r") as f:
        tf_vector_train = json.loads(f.read())

    train_vecs = np.array(tf_vector_train)
    train_tags = np.array(map_sentiment_train)
    clf = train(train_vecs, train_tags)
    print("#" * 70)
    print("Training completed\n\n")

    # with open("intermediate/nb_trained_dumps.bin", 'rb') as fs:
    #     clf = pickle.loads(fs.read())

    with open(file_prefix + "dev-dataset.json", "r") as f:
        map_sentiment_dev = json.loads(f.read())
    print("#" * 70)
    with open(file_prefix + "dev-tf-idf-matrix.json", "r") as f:
        tf_vector_dev = json.loads(f.read())
    dev_vecs = np.array(tf_vector_dev)
    dev_tags = np.array(map_sentiment_dev)
    print("Classifying.....")
    classify(clf, dev_vecs, dev_tags)


if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    main()
    end = timeit.default_timer()
    print("\n Time taken: " + str(end - start))