import json
#
# wfs = open("yelp_reviews.json", 'w+')
# total = 0
# with_stars = 0
# with open("yelp_academic_dataset_review.json", "r") as fs:
#     raw_data = fs.readline()
#     while raw_data:
#         data = json.loads(raw_data.strip())
#         total += 1
#         if data["stars"] != 0:
#             with_stars += 1
#             wfs.write(raw_data)
#             wfs.write('\n')
#         raw_data = fs.readline()
# wfs.close()
# print(total)
# print(with_stars)

# import json
#
# record_list = []
# positive_list = []
# negative_list = []
# with open("yelp_academic_dataset_review.json", "r") as fs:
#     raw_data = fs.readline()
#     while raw_data:
#         data = json.loads(raw_data.strip())
#         positive_map = {}
#         negative_map = {}
#         if data["stars"] < 3:
#             negative_map["votes"] = data["votes"]
#             negative_map["text"] = data["text"]
#             negative_map["stars"] = data["stars"]
#             negative_list.append(negative_map)
#         if data["stars"] > 3 and (data["votes"]["cool"] > 1 and data["votes"]["useful"] > 3) and data["votes"]["funny"] == 0:
#                 positive_map["votes"] = data["votes"]
#                 positive_map["text"] = data["text"]
#                 positive_map["stars"] = data["stars"]
#                 positive_list.append(positive_map)
#         # if (data["votes"]["cool"] > 1 and data["votes"]["useful"] > 3) and data["votes"]["funny"] == 0:
#         #     my_map["text"] = data["text"]
#         #     my_map["stars"] = data["stars"]
#         #     record_list.append(my_map)
#         raw_data = fs.readline()

# with open("raw-data/positive.json", 'w') as fs:
#     fs.write(json.dumps(positive_list))
#
# with open("raw-data/negative.json", 'w') as fs:
#     fs.write(json.dumps(negative_list))

# import random
# import math
# random.shuffle(record_list)
#
# l = len(record_list)
# i, j = 0, 0
# train = math.floor(0.8*l)
# dev = math.ceil(0.2*l)
#
# with open("raw-data/train.json", 'w') as fs:
#     fs.write(json.dumps(record_list[:train]))
#
# with open("raw-data/dev.json", 'w') as fs:
#     fs.write(json.dumps(record_list[train:]))


# while j < train:
#     with open("raw-data/train.json", "w") as ftrain:
#         ftrain.write(record_list[i] + "\n")
#         i+=1
#         j+=1
# j = 0
# while j < dev:
#     with open("raw-data/dev.json", "w") as fdev:
#         fdev.write(record_list[i] + "\n")
#         i+=1
#         j+=1




# new_list = []
# with open("raw-data/negative.json", 'r') as fs:
#     data = json.loads(fs.read())
#     for d in data:
#         if (d["votes"]["useful"] > 2 and d["votes"]["cool"] >= 1) and d["votes"]["funny"] == 0:
#             new_list.append(d)
#     print(len(new_list))
#
# with open("raw-data/negative_1.json", 'w') as fs:
#     fs.write(json.dumps(new_list))
#
# new_list = []
# with open("raw-data/positive.json", 'r') as fs:
#     data = json.loads(fs.read())
#     for d in data:
#         if (d["votes"]["cool"] >= 2 and d["votes"]["useful"] > 3) and d["votes"]["funny"] == 0:
#             new_list.append(d)
#     print(len(new_list))
# with open("raw-data/positive_1.json", 'w') as fs:
#     fs.write(json.dumps(new_list))

# p_list = []
# with open("raw-data/positive_1.json", 'r') as fs:
#     p_list = json.loads(fs.read())
#     print(len(p_list))
# n_list = []
# with open("raw-data/negative_1.json", 'r') as fs:
#     n_list = json.loads(fs.read())
#     print(len(n_list))
# p_list.extend(n_list)
# print(len(p_list))
# import random
# random.shuffle(p_list)
# random.shuffle(p_list)
# random.shuffle(p_list)
# random.shuffle(p_list)
# random.shuffle(p_list)
#
# with open("yelp_reviews.json", 'w') as fs:
#     fs.write(json.dumps(p_list[:13000]))

# from utils import tokenizer
#
# import timeit
# start = timeit.default_timer()
# data = []
# reduced_data = []
#
#
# with open("yelp_reviews.json", 'r') as fs:
#     data = json.loads(fs.read())
#     import random
#     random.shuffle(data)
#     i = 0
#     #print(len(data))
#     for d in data:
#         dic = {}
#         dic["sentiment"] = 0 if d["stars"] < 3 else 1
#         dic["text"] = d["text"]
#         dic["tokens"] = tokenizer(d["text"])
#         reduced_data.append(dic)
#         # if i == 5000:
#         #     break
#         # i+=1
# print(timeit.default_timer()-start)
# with open("raw-data/train.json", 'w') as train:
#     train.write(json.dumps(reduced_data[:10000]))
# with open("raw-data/dev.json", 'w') as train:
#     train.write(json.dumps(reduced_data[10000:]))
