from transformers import OFATokenizer
# pip install gensim
import argparse
import yake
import re
from gensim.models import KeyedVectors


def cap2hashtag(cap_list):

    core = []
    relative = []
    impression = []

    # define variables
    num_of_inputs = len(cap_list)
    num_of_cores = 5 if num_of_inputs < 3 else num_of_inputs * 2
    num_of_relative_per_image = 2
    
    word2vec = KeyedVectors.load_word2vec_format("eng_w2v") # 모델 로드
    docs = ". ".join(cap_list)

    tokenizer = OFATokenizer.from_pretrained('OFA-base')

    language = "en"
    max_ngram_size = 3
    deduplication_threshold = 0.9
    deduplication_algo = 'seqm'
    num_of_keywords = num_of_cores

    kw_extractor = yake.KeywordExtractor(lan=language,
                                         n=max_ngram_size,
                                         dedupLim=deduplication_threshold,
                                         dedupFunc=deduplication_algo,
                                         top=num_of_keywords,
                                         stopwords=None)

    keywords = kw_extractor.extract_keywords(docs)
    keywords.sort(key=lambda x: x[1], reverse=True)

    if 'background' in keywords:
        num_of_inputs += 1

    for kw, v in keywords[:num_of_inputs]:
        kw = re.sub('[^a-zA-Z]+', ' ', kw)

        if kw == 'background':
            continue

        try:
            relatives = model.wv.most_similar('Ġ' + kw.split(' ')[0])
            top_3 = sorted(relatives, key=lambda x: x[1], reverse=True)[
                :num_of_relative_per_image]
            relative.extend([re.sub('Ġ', '', x) for x, y in top_3])
            core.append(kw)

        except:
            print(kw)

    core = ['#{}'.format(x) for x in core]
    relative = ['#{}'.format(x) for x in relative]
    impression = ['#{}'.format(x) for x in impression]

    # print(core, relative)
    # print(len(core), len(relative))
    print(core, relative, impression)
    return core, relative, impression

# if __name__ == "__main__":
#     cap_list = ['cheetah running in the grass',
#                 'an elephant walking along a dirt road',
#                 'a giraffe standing in a grassy field with a mountain in the background']

#     cap2hashtag(cap_list)
