import json
import model
import math
import time

start_time = time.time()


def yelp_classify():
    with open("yelp_model_newdata.txt") as data_file:
        data = json.load(data_file)

    # calculate some data to be used in Naive Baye's calculation
    probab_positive = data["positive_w_count"]/data["vocab_size"]
    probab_negative = data["negative_w_count"]/data["vocab_size"]

    # used during accuracy and recall calculation
    positive_review_count = 0
    negative_review_count = 0

    # No of files classified as ham or spam
    negative_classified_true = 0
    positive_classified_true = 0
    negative_classified_false = 0
    positive_classified_false = 0

    # review_list will contain the the required values for each json review
    # reviews_list = []

    fout = open("nboutput.txt", "w")
    # with open('dev_yelp_dataset_review.json') as json_data:
    #     restaurant_review = json_data.readline()
    #     while restaurant_review.strip() is not "":
    #         restaurant_review_obj = model.Restaurant(json.loads(restaurant_review))
    #         reviews_list.append(restaurant_review_obj)
    #         restaurant_review = json_data.readline()

    with open("dev.json") as data_file:
        reviews_list = json.load(data_file)

    for restaurant_review in reviews_list:
        words = restaurant_review['tokens']
        probab_positive_words = 0
        probab_negative_words = 0

        for w in words:
            # Add one smoothing
            if w in data['word_dict']:
                probab_negative_words += math.log((data["word_dict"][w]["negative_count"] + 1)/(data["negative_w_count"] +
                                                                                          data["vocab_size"]))
                probab_positive_words += math.log((data["word_dict"][w]["positive_count"] + 1)/(data["positive_w_count"] +
                                                                                            data["vocab_size"]))

        prob_negative_review = math.log(probab_negative) + probab_negative_words
        prob_positive_review = math.log(probab_positive) + probab_positive_words

        if restaurant_review['sentiment'] == 0:
            negative_review_count += 1
            if prob_negative_review > prob_positive_review:
                negative_classified_true += 1
                fout.writelines("Negative " + "\n")
            else:
                negative_classified_false += 1
                fout.writelines("Positive " + "\n")
        else:
            positive_review_count += 1
            if prob_positive_review > prob_negative_review:
                positive_classified_true += 1
                fout.writelines("Negative " + "\n")
            else:
                positive_classified_false += 1
                fout.writelines("Positive " + "\n")

    accuracy_negative = negative_classified_true/negative_review_count
    accuracy_positive = positive_classified_true/positive_review_count

    # calculating precision for negative and positive
    precision_positive = positive_classified_true/(positive_classified_true + negative_classified_false)
    precision_negative = negative_classified_true/(negative_classified_true + positive_classified_false)

    # calculating recall for ham and spam
    recall_positive = positive_classified_true/positive_review_count
    recall_negative = negative_classified_true/negative_review_count

    # calculating f1 score for ham and spam
    f1_positive = (2 * precision_positive * recall_positive)/(precision_positive + recall_positive)
    f1_negative = (2 * precision_negative * recall_negative)/(precision_negative + recall_negative)

    print("Total negative review:" + str(negative_review_count))
    print("Total positive review:" + str(positive_review_count))
    print("Total negative classified true:" + str(negative_classified_true))
    print("Total negative classified false:" + str(negative_classified_false))
    print("Total positive classified true:" + str(positive_classified_true))
    print("Total positive classified false:" + str(positive_classified_false))
    print()
    print("negative accuracy:" + str(accuracy_negative))
    print("positive accuracy:" + str(accuracy_positive))
    print()
    print("positive precision:" + str(precision_positive))
    print("positive recall:" + str(recall_positive))
    print("positive F1 Score:" + str(f1_positive))
    print()
    print("negative precision:" + str(precision_negative))
    print("negative recall:" + str(recall_negative))
    print("negative F1 Score:" + str(f1_negative))


start_time = time.time()
if __name__ == "__main__":yelp_classify()
print("--- %s seconds ---" % (time.time() - start_time))