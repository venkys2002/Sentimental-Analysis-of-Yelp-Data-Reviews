import os,sys,json
total_ham_files = 0
total_spam_files = 0
ham_classified_true = 0
spam_classified_true = 0
ham_classified_false = 0
spam_classified_false = 0


def calculate_accuracy(feature):
    precision = 1
    recall = 1
    f1Score = 1
    if 'ham' in feature:
        precision = ham_classified_true/(ham_classified_true+ham_classified_false)
        recall = ham_classified_true/(total_ham_files)
    elif 'spam' in feature:
        precision = spam_classified_true/(spam_classified_false+spam_classified_true)
        recall = spam_classified_true/(total_spam_files)
    f1Score = (2*precision*recall)/(precision+recall)
    return precision*100, recall*100, f1Score*100

def classify(input_file,output):

    global total_ham_files,total_spam_files, ham_classified_true, spam_classified_true, ham_classified_false, spam_classified_false

    with open('per_model_final.txt',mode='r',encoding=None) as fptr:
        train_data = json.loads(fptr.read())
        feature_weights = train_data['feature_weights']
        updated_bias = train_data['updated_bias']
    with open(output,mode='w') as fw_ptr:
        with open(input_file,mode="r",encoding=None) as json_fp:
            json_data = json.loads(json_fp.read())
            for each_item in json_data:
                if(each_item["sentiment"] == 1):
                    total_ham_files += 1
                elif(each_item["sentiment"] == 0):
                    total_spam_files += 1

                alpha = 0

                tokens = each_item["text"]
                for each_token in tokens:
                    if each_token in feature_weights:
                        alpha += feature_weights[each_token]
                alpha += updated_bias
                if( alpha > 0):
                    classify_file = "Negative\n"
                    fw_ptr.write(classify_file)

                    if(each_item["sentiment"] == 0):
                        spam_classified_true += 1
                    else:
                        spam_classified_false += 1

                else:

                    classify_file = "Positive\n"
                    fw_ptr.write(classify_file)

                    if(each_item["sentiment"] == 1):
                        ham_classified_true += 1
                    else:
                        ham_classified_false += 1

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_filename = sys.argv[2]
    classify(input_file,output_filename)

    print('Precision Recall F1 score')
    print(calculate_accuracy('ham'))
    print(calculate_accuracy('spam'))