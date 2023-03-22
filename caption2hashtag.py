from transformers import OFATokenizer, OFAModel
# pip install gensim
import gensim
import argparse
import yake


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_file_dir', default='test-img/dir2/caption.txt', help='caption dir')
    args = parser.parse_args()

    f = open(args.caption_file_dir,'r')
    model = gensim.models.Word2Vec.load('1minwords')
    docs =  f.read()   
    tokenizer = OFATokenizer.from_pretrained('OFA-base')
    
    language = "en"
    max_ngram_size = 3
    deduplication_threshold = 0.9
    deduplication_algo = 'seqm'
    num_of_keywords = 10
    
    kw_extractor = yake.KeywordExtractor(lan = language,
                                         n = max_ngram_size,
                                         dedupLim = deduplication_threshold,
                                         dedupFunc = deduplication_algo,
                                         top = num_of_keywords,
                                         stopwords = None)
    
    keywords = kw_extractor.extract_keywords(docs)
    keywords.sort(key = lambda x : x[1], reverse = True)
    
    print(docs)
    for kw, v in keywords:
        print("Keyphrase: ",kw, ": score", v)
