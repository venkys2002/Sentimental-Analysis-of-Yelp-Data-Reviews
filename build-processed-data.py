import json
from utils import create_tf_idf_matrix, build_count_dict
file_prefix = "processed-data/"


def main(ftype):
    file = "raw-data/train.json" if ftype == "train" else "raw-data/dev.json"
    with open(file, 'r') as f:
        data = json.loads(f.read())
        map_sentiment = []
        tokens_count = []
        for record in data:
            try:
                sentiment = record["sentiment"]
                tokens = record["tokens"]
                map_sentiment.append(sentiment)
                tokens_count.append(build_count_dict(tokens))
            except:
                print(record)

    file = file_prefix + "training-dataset.json" if ftype == "train" else file_prefix + "dev-dataset.json"
    with open(file, "w") as f:
        f.write(json.dumps(map_sentiment))

    lexicon, tf_vector = create_tf_idf_matrix(tokens_count, ftype)
    file = file_prefix + "tf-idf-matrix.json" if ftype == "train" else file_prefix + "dev-tf-idf-matrix.json"
    with open(file, "w") as f:
        f.write(json.dumps(tf_vector))


if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    main("train")
    main("dev")
    end = timeit.default_timer()
    print("\n Time taken: " + str(end - start))
