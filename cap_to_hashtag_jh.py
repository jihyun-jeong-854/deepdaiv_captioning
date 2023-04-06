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

def cap2hashtag(cap_list, w2v, bert):

    core = []
    relative = []
    impression = []

#     # define variables
    num_of_inputs = len(cap_list)/3
    num_of_cores = 5 if num_of_inputs < 3 else num_of_inputs * 2
    num_of_total_tags = num_of_inputs * 4
    
    print(cap_list)
    for i in range(len(cap_list)):
        cap_list[i] = re.sub('[^a-zA-Z]+', ' ', cap_list[i])
        
    docs = ". ".join(cap_list)

    n_gram_range = (1, 1)
    stop_words = "english"
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([docs])
    candidates = count.get_feature_names_out() # list

    doc_embedding = bert.encode([docs])
    candidate_embeddings = bert.encode(candidates)

    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0]] # list

    idx = 0
    while len(core) <= num_of_cores:
        kw = keywords[idx]
        kw = re.sub('[^a-zA-Z]+', ' ', kw)
        idx +=1
    
        try:
            relatives = w2v.most_similar(kw)
            top_n = sorted(relatives, key=lambda x: x[1], reverse=True)[0]
            
            relative.extend([x for x,y in top_n if x == re.sub('[^a-zA-Z]+', ' ', x)])
            core.append(kw)
            
        except:
            print("not tokenized", kw)
            continue
     
      
    relative.extend(keywords[idx:num_of_total_tags])
    
    relative = postprocess(relative)
    core = postprocess(core)
    
    core = ['#{} '.format(x) for x in core]
    if 'people' in relative or 'ppl' in relative:
        relative.extend(['friends','family'])
        
    relative = ['#{} '.format(x) for x in relative]
    impression = ['#{} '.format(x) for x in impression]

    # print(core, relative)
    # print(len(core), len(relative))
    print(core, relative, impression)
    return core, relative, impression

# # if __name__ == "__main__":
# #     cap_list = ['cheetah running in the grass',
# #                 'an elephant walking along a dirt road',
# #                 'a giraffe standing in a grassy field with a mountain in the background']

# #     cap2hashtag(cap_list)
