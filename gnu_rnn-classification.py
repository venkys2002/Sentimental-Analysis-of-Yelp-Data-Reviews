import numpy as np
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import skflow,json
from nltk.tokenize import TweetTokenizer

tknzr = TweetTokenizer(preserve_case=True,strip_handles=True)
def tokenizer(iterator):
    for value in iterator:
        yield tknzr.tokenize(value)


#load from data from json
max_lenth=0
with open("train.json",'r') as fid:
    data = json.load(fid)
X_train=[]
y_train=[]
for val in data:
    if len(val['tokens']) >max_lenth:
        max_lenth=len(val['tokens'])
    X_train.append(" ".join(val['tokens']))
    if val['sentiment'] == 1:
        y_train.append(1)
    else:
        y_train.append(0)
with open("dev.json",'r') as fid:
    data = json.load(fid)
X_test=[]
y_test=[]
for val in data:
    X_test.append(" ".join(val['tokens']))
    if val['sentiment'] == 1:
        y_test.append(1)
    else:
        y_test.append(0)

### Process vocabulary

MAX_DOCUMENT_LENGTH = max_lenth
vocab_processor = skflow.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH, tokenizer_fn=tokenizer)
X_train = np.array(list(vocab_processor.fit_transform(X_train)))
X_test = np.array(list(vocab_processor.transform(X_test)))

n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)

### Model

EMBEDDING_SIZE = 100

def rnn_model(X, y):
    """Recurrent neural network model to predict from sequence of words
    to a class."""
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].
    word_vectors = skflow.ops.categorical_variable(X, n_classes=n_words,
                                                   embedding_size=EMBEDDING_SIZE, name='words')
    # Split into list of embedding per word, while removing doc length dim.
    # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
    word_list = skflow.ops.split_squeeze(1, MAX_DOCUMENT_LENGTH, word_vectors)
    # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
    cell = tf.nn.rnn_cell.GRUCell(EMBEDDING_SIZE)
    # Create an unrolled Recurrent Neural Networks to length of
    # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
    _, encoding = tf.nn.rnn(cell, word_list, dtype=tf.float32)
    # Given encoding of RNN, take encoding of last step (e.g hidden size of the
    # neural network of last step) and pass it as features for logistic
    # regression over output classes.
    return skflow.models.logistic_regression(encoding, y)

n_classes = 2
classifier = skflow.TensorFlowEstimator(model_fn=rnn_model, n_classes=n_classes,batch_size=100,
                                        steps=1025, optimizer='Adam', learning_rate=0.01, continue_training=True)


c=0
if c<10:
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(" Precision: {0}".format(metrics.precision_score(y_test,y_pred )))
    print(" Recall: {0}".format(metrics.recall_score(y_test,y_pred)))
    print(" Accuracy: {0}".format(metrics.accuracy_score(y_test,y_pred,)))
    print(" Classification report: {0}".format(metrics.classification_report(y_test,y_pred)))
    c+=1