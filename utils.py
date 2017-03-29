import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re
import pickle
from nltk.corpus import stopwords


def build_lexicon(tokenized_word):
    lexicon = set()
    for i in range(len(tokenized_word)):
        lexicon.update(tokenized_word[i].keys())
    return lexicon


def create_tf_idf_matrix(tokenized_words, ftype):
    if ftype == "train":
        lexicon = build_lexicon(tokenized_words)
        with open("intermediate/lexicon.bin", 'wb') as fs:
            fs.write(pickle.dumps(lexicon))
    else:
        with open("intermediate/lexicon.bin", 'rb') as fs:
            lexicon = pickle.loads(fs.read())
    tf_vector = []
    for i in range(len(tokenized_words)):
        tf_vector.append([tokenized_words[i][word] if word in tokenized_words[i] else 0 for word in lexicon])
    return lexicon, tf_vector


# Our tokenizer that will remove unwanted words based on POS tags
def tokenize(text):
    required_tags = {'NNP': 1, 'NN': 1, 'NNS': 1, 'NNPS': 1, 'JJ': 1, 'JJR': 1, 'JJS': 1, 'VBZ': 1}
    delimiters = ";|,|\*|\n|\.|\?|\)|\("
    pos_tags = nltk.pos_tag(text.split())
    ret_list = []
    for pos_tag in pos_tags:
        if pos_tag[1] in required_tags:
            split = re.split(delimiters, pos_tag[0])
            if len(split) > 1:
                for s in split:
                    if len(s) > 0:
                        ret_list.append(s)
            else:
                ret_list.append(pos_tag[0])
    return ret_list


def tokenizer(text):
    stop = set(stopwords.words('english'))
    stop_map = {}
    for s in stop:
        stop_map[s] = 1
    delimiters = ";|,|\*|\n|\.|\?|\)|\("
    tokens = text.split()
    ret_list = []
    wl = WordNetLemmatizer()
    ps = PorterStemmer()
    for token in tokens:
        token = token.lower()
        split = re.split(delimiters, token)
        if len(split) > 1:
            for s in split:
                if len(s) > 0 and s not in stop_map:
                    ret_list.append(ps.stem(wl.lemmatize(s)))
        else:
            if split not in stop_map:
                ret_list.append(ps.stem(wl.lemmatize(split)))


# class Model():
#     def __init__(self, bigram, freq):
#         self.bigram = bigram
#         self.frequency = freq
#
#     def __cmp__(self, other):
#         return other.frequency - self.frequency
#
#
# def build_lexicon_dt(text):
#     cleanText = []
#     delimiters = ";|,|\*|\n|\.|\?|\)|\("
#     for t in text:
#         split = re.split(delimiters, t)
#         if len(split) > 1:
#             for s in split:
#                 if len(s) > 0:
#                     cleanText.append(s)
#         else:
#             cleanText.append(t)
#     bigrams = list(nltk.bigrams(cleanText))
#     bigrams_freq = {}
#     for bigram in bigrams:
#         if bigram in bigrams_freq:
#             bigrams_freq[bigram].frequency += 1
#         else:
#             bigrams_freq[bigram] = Model(bigram, 1)
#     lexicons = list(bigrams_freq.values())
#     lexicons.sort(key=lambda x: x.frequency)
#     return [l.bigram for l in lexicons]


def build_count_dict(tokens):
    map_text_count = {}
    for token in tokens:
        if token in map_text_count:
            map_text_count[token] += 1
        else:
            map_text_count[token] = 1
    return map_text_count
