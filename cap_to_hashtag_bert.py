from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import locale
import re

def getpreferredencoding(do_setlocale=True):
    return "UTF-8"


def cap2hashtag(cap_list):

    locale.getpreferredencoding = getpreferredencoding

    core = []
    relative = []
    impression = []

    cap_list=sum(cap_list, [])

    docs = ". ".join(cap_list)
    num_of_inputs = len(cap_list)
    num_of_cores = 5 if num_of_inputs < 3 else num_of_inputs * 2
    num_of_relative_per_image = 2

    # keword extraction setting
    n_gram_range = (1, 1)
    stop_words = "english"
    count = CountVectorizer(ngram_range=n_gram_range,
                            stop_words=stop_words).fit([docs])
    candidates = count.get_feature_names_out()  # list

    # keyword extraction model
    ke_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    doc_embedding = ke_model.encode([docs])
    candidate_embeddings = ke_model.encode(candidates)

    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index]
                for index in distances.argsort()[0][-num_of_cores:]]  # list

    # w2v model
    #w2v_model = gensim.models.Word2Vec.load('1minwords')

    # postprocessing for core, relatives
    for kw in keywords:
        kw = re.sub('[^a-zA-Z]+', ' ', kw)

        if kw == 'background':
            continue

        try:
            # relatives = w2v_model.wv.most_similar('Ġ' + kw.split(' ')[0])
            # top_3 = sorted(relatives, key=lambda x: x[1], reverse=True)[:num_of_relative_per_image]
            # relative.extend([re.sub('Ġ', '', x) for x, y in top_3])
            core.append(kw)

        except:
            print(kw)

    core = ['#{}'.format(x) for x in core]
    relative = ['#{}'.format(x) for x in relative]
    impression = ['#{}'.format(x) for x in impression]

    return core, relative, impression


# if __name__ == "__main__":
#     cap_list = ['cheetah running in the grass',
#                 'an elephant walking along a dirt road',
#                 'a giraffe standing in a grassy field with a mountain in the background',
#                 'the whole world tastes like daffodil daydream.',
#                 'on the crosstown expressway this morning.',
#                 'You ate breakfast, yes? Breakfast']

#     cap2hashtag(cap_list)
