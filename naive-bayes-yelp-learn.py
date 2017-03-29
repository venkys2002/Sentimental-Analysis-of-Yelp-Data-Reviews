import json
import model
import time


def print_values(vocabulary_size, positive_review_count, negative_review_count, positive_wordcount, negative_wordcount):
    print("Vocabulary Size:" + str(vocabulary_size))
    print("Positive review count :" + str(positive_review_count))
    print("Negative review count :" + str(negative_review_count))
    print("Total Words in Positive:" + str(positive_wordcount))
    print("Total Words in Negative:" + str(negative_wordcount))

    # print(pprint(word_dict))
    # print(spam_dict.keys())


def yelp_review_learn():
    # review_list will contain the the required values for each json review
    # reviews_list = []
    word_dict = dict()

    positive_wordcount = 0     # No of words in spam
    negative_wordcount = 0      # No of words in ham
    positive_review_count = 0    # No of files in spam
    negative_review_count = 0     # No of file in ham

    # # read the data from json and store the required data for each review and append it to reviews_list
    # with open('train_yelp_dataset_review.json') as json_data:
    #     restaurant_review = json_data.readline()
    #     while restaurant_review.strip() is not "":
    #         restaurant_review_obj = model.Restaurant(json.loads(restaurant_review))
    #         reviews_list.append(restaurant_review_obj)
    #         restaurant_review = json_data.readline()

    # print(len(reviews_list))

    # calculate the values necessary for naive baiyes model
    with open("train.json") as data_file:
        reviews_list = json.load(data_file)

    for restaurant_review in reviews_list:
        # print(restaurant_review.business_id + ", " + str(restaurant_review.stars) + ", " + restaurant_review.sentiment
        # + ", " + restaurant_review.text)

        if restaurant_review['sentiment'] == 0:
            negative_review_count += 1
        else:
            positive_review_count += 1

        words = restaurant_review['tokens']
        for w in words:
            if w not in word_dict:
                word_dict[w] = {"positive_count": 0, "negative_count": 0}
                if restaurant_review['sentiment'] == 0:
                    word_dict[w]['negative_count'] += 1
                    negative_wordcount += 1
                else:
                    word_dict[w]['positive_count'] += 1
                    positive_wordcount += 1
            else:
                if restaurant_review['sentiment'] == 0:
                    word_dict[w]['negative_count'] += 1
                    negative_wordcount += 1
                else:
                    word_dict[w]['positive_count'] += 1
                    positive_wordcount += 1

    vocabulary_size = len(word_dict)
    print_values(vocabulary_size,positive_review_count, negative_review_count, positive_wordcount, negative_wordcount)

    # serializing the data
    model_obj = model.Model(vocabulary_size, positive_review_count, negative_review_count, positive_wordcount,
                            negative_wordcount, word_dict)
    model_str = json.dumps(vars(model_obj))
    fs = open("yelp_model_newdata.txt", "w")
    fs.write(model_str)
    fs.close()

start_time = time.time()
if __name__ == "__main__": yelp_review_learn()
print("--- %s seconds ---" % (time.time() - start_time))