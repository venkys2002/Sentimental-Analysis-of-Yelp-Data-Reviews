class Restaurant:
    def __init__(self, restaurant_review):
        self.review_id = restaurant_review['review_id']
        self.stars = restaurant_review['stars']
        self.text = restaurant_review['text']
        self.sentiment = "Positive" if restaurant_review['stars'] >=3 else "Negative"

    def __repr__(self):
        # return [self.business_id, self.stars, self.text, self.sentiment]
        return self.sentiment


class Model:
    # serialize the data and store it in a file nbmodal.txt. This file will be used as input to nbclassify.py
    def __init__(self, vocab_size, positive_r_count, negative_r_count, positive_w_count, negative_w_count, word_dict=None):
        self.vocab_size = vocab_size
        self.positive_r_count = positive_r_count
        self.negative_r_count = negative_r_count
        self.positive_w_count = positive_w_count
        self.negative_w_count = negative_w_count
        self.word_dict = word_dict
