import pandas as pd

hashtag=pd.read_csv("hashtag_db.csv")
num_categories=len(hashtag)
categories=list(hashtag.iloc[:,0])

hashtag_dict={}
for i in range(len(hashtag)):
    hashtags=hashtag.iloc[:,1][i].replace(" ", "").split("#")[1:]
    hashtags=list(set(hashtags))
    hashtag_dict[hashtag.iloc[:, 0][i]]=hashtags