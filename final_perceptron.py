import os,sys,random,json,timeit
from collections import defaultdict

filename_dict = dict()
filelist = []
feature_weights = defaultdict(int)
avg_weights = defaultdict(int)
json_dict = defaultdict(int)


def learn_model():
    global filelist,feature_weights,filename_dict,avg_weights,json_dict
    bias = 0
    beta = 0
    counter = 1
    for i in range(0,30):
        random.shuffle(filelist)
        for file in filelist:
            alpha = 0
            y = filename_dict[file]['label']
            feature_list = filename_dict[file]['tokens']
            #Calculate the activation value for the Current file
            for feature in feature_list:
                alpha += feature_weights[feature]
            alpha += bias

            if((y*alpha) <= 0):
                for feature in feature_list:
                    feature_weights[feature] += y
                bias += y

                for feature in feature_list:
                    avg_weights[feature] += y
                beta += y*counter
            counter += 1

    for each_key in avg_weights.keys():
        avg_weights[each_key] = feature_weights[each_key] - (1/counter)*avg_weights[each_key]
    beta = bias - (1/counter)*beta

    json_dict = {'updated_bias': beta, 'feature_weights': avg_weights}



if __name__ == '__main__':

    start = timeit.default_timer()
    with open('train.json',mode='r',encoding=None) as fp:
        json_data = json.loads(fp.read())

        file_name = 1
        for each_item in json_data:
            print(each_item["sentiment"])
            if(each_item["sentiment"] == 0):
                filename_dict[file_name] = {'label': 1, 'tokens': each_item["tokens"], 'filename':file_name}
                filelist.append(file_name)
                file_name += 1
            else:
                filename_dict[file_name] = {'label': -1, 'tokens': each_item["tokens"], 'filename':file_name}
                filelist.append(file_name)
                file_name += 1
    learn_model()
    with open('per_model_final.txt', 'w') as fp:
        fp.write(json.dumps(json_dict))
    total_time = timeit.default_timer() - start
    print(total_time)