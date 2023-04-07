# from transformers import OFATokenizer
# pip install gensim
import argparse
import gensim.downloader as api
import re
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def postprocess(tag_list):
    unnecessary_tag = ['in', 'the', 'thing']
    return [tag for tag in tag_list if tag not in unnecessary_tag]


def getpreferredencoding(do_setlocale=True):
    return "UTF-8"


def cap2hashtag(cap_list, w2v, bert):

    core = []
    relative = []

    # define variables
    num_of_inputs = int(len(cap_list)/3)
    num_of_cores = 5 if num_of_inputs < 3 else int(num_of_inputs * 2)
    num_of_total_tags = int(num_of_inputs * 4)

    print(cap_list)

    for i in range(len(cap_list)):
        cap_list[i] = re.sub('[^a-zA-Z]+', ' ', cap_list[i])

    docs = ". ".join(cap_list)

    n_gram_range = (1, 1)
    stop_words = "english"
    count = CountVectorizer(ngram_range=n_gram_range,
                            stop_words=stop_words).fit([docs])
    candidates = count.get_feature_names_out()  # list

    doc_embedding = bert.encode([docs])
    candidate_embeddings = bert.encode(candidates)

    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0]]  # list
    # keywords = [candidates[index]
    #             for index in distances.argsort()[0][-num_of_cores:]]

    idx = 0

    # while len(core) <= num_of_cores:
    #     kw = keywords[idx]
    #     kw = re.sub('[^a-zA-Z]+', ' ', kw)
    #     idx += 1
#
    #     try:
    #         relatives = w2v.most_similar(kw)
    #         top_n = sorted(relatives, key=lambda x: x[1], reverse=True)[0]
#
    #         relative.extend([x for x, y in top_n if x ==
    #                         re.sub('[^a-zA-Z]+', ' ', x)])
    #         core.append(kw)
#
    #     except:
    #         print("not tokenized", kw)
    #         continue
    #     cnt = 0

    print("키워드:", keywords)
    print("코어:", core)

    # len(core) 가 0 이 아니면 계속 실행
    # 근데 여기서 len(core) 은 시작때 0인데 그럼 While 문 자체가 안도는거아님?

    while len(keywords) != 0:  # not keywords.empty()

        if len(core) == num_of_cores:
            break

        kw = keywords.pop(0)

        print("====================================")
        print("키워드 하나:", kw)
        core.append(kw)
        print("코어:", core)

        try:
            relatives = w2v.most_similar(kw)
            top_n = sorted(relatives, key=lambda x: x[1], reverse=True)[0]

            relative.extend([x for x, y in top_n if x ==
                            re.sub('[^a-zA-Z]+', ' ', x)])

        except:
            print("not tokenized", kw)
            continue

    relative.extend(keywords[idx:num_of_total_tags])

    relative = postprocess(relative)
    core = postprocess(core)

    core = ['#{} '.format(x) for x in core]
    if 'people' in relative or 'ppl' in relative:
        relative.extend(['friends', 'family'])

    relative = ['#{} '.format(x) for x in relative]

    # print(core, relative)
    # print(len(core), len(relative))
    print(core, relative)
    return core, relative

# # if __name__ == "__main__":
# #     cap_list = ['cheetah running in the grass',
# #                 'an elephant walking along a dirt road',
# #                 'a giraffe standing in a grassy field with a mountain in the background']

# #     cap2hashtag(cap_list)
