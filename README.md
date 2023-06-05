# deepdaiv_captioning
<img width="945" alt="Screenshot 2023-04-06 at 8 41 11 PM" src="https://github.com/rachel618/deepdaiv_captioning/assets/67910258/05b2cab3-b680-42c2-a199-ea897c159cd5">

## 1. Image Captioning

먼저, 여러장의 이미지를 받아서 캡션을 생성해주었습니다. 사진 당 캡션을 3개씩 생성하였고 업로드한 모든 사진의 캡션을 합쳐서 하나의 문서를 생성하게 됩니다.

### multi input

여러 장의 사진을 받는 방식에 대해서 상당히 많은 고민을 하였는데요,
단순히 사진을 한장씩 받아서 각 사진에 대해 따로 얻은 캡션을 활용할지, 여러 장의 진을 합쳐서 동일한 크기로 만들어서 모델에 inference는 한번만 할지 등등 다양한 의견이 나왔습니다.

하지만 여러장의 사진을 합쳐서 모델에 input으로 넣을 경우 상당히 general 한 caption이 생성되었습니다.
예를 들어 해변 사진이 여러 장 있는 경우 해변에 누워있는 사람, 크루즈, 스노쿨링 등의 자세한 정보들은 놓치고 
해변에서의 하루 이러한 식으로 캡션이 나왔습니다. 여기서 해시태그를 생성할 수 있는 유효한 정보를 얻기는 어렵다고 생각해서 사진 별로 따로 inference 하는 방식을 채택하였습니다.

여러 장의 사진을 세로로 합치는 방식, 여러 장의 사진을 가로로 합치는 방식 등으 시도해 본 결과 또한 사진 별로 고르게 caption을 하지 않고 특정 부분에 집중해서 captioning을 하는 경향을 보였습니다.
캡션 개수를 늘려보았지만 마지막 사진에 대한 caption은 거의 없었고 한 장의 사진을 각각 inference 했을 때 정확도가 높았습니다.

### Versatile Diffusion(VD)
<img width="821" alt="Screenshot 2023-04-06 at 9 14 42 PM" src="https://github.com/rachel618/deepdaiv_captioning/assets/67910258/ee476a64-b54b-4546-9e36-b70cfbcdfb37">


데모 : https://huggingface.co/spaces/shi-labs/Versatile-Diffusion
Image Captioning 모델에는 BLIP, OFA 등 다양한 모델이 있는데 데모를 고려해서 inference time이 길지 않으면서 정확도가 괜찮은 Versatile-Diffusion으로 선택하였습니다.

VD는 통합된 multi-flow 멀티모달 diffusion framework로,
저희가 사용한 image to text 외에도 text to image , image 변형, editable I2T2I 까지 지원합니다. 
editable I2T2I는 모델이 image captioning 한 후 caption을 수정하여 다시 input으로 넣어주면 수정된 caption에 맞게 기존 이미지를 수정해주는 것입니다.

이렇게 VD는 input으로 다양한 타입을 받을 수 있도록 다른 flow 들을 동시에 결합하여 학습시켰습니다. diffuser network은  input과 output 데이터 타입에 맞게 여러 네트워크를 그룹 별로 나누고, 공유하고, 바꾸는 프로토콜이 핵심입니다.
<img width="411" alt="Screenshot 2023-04-06 at 10 01 30 PM" src="https://github.com/rachel618/deepdaiv_captioning/assets/67910258/c2407c63-fcce-4c30-a2b6-c48bb8a632fa">

위 그림에서 나타난것 처럼 모든 diffuser layer는 global layer, data layer, context layer로 분류됩니다. 

각각의 layer 이름에서 기능을 어느정도 유추할 수 있는데요. global layer는 모든 task에 공유되는 layer이고, context layer와 data layer는 각각 대응되는 타입의 input, output이 들어올 때 사용됩니다.
예를 들어, image to text를 할 때 diffuser는 image context layer 와 text data layer 를 사용합니다.

이렇게 여러 task에 같이 쓰이는 layer들이 있어서 single flow 모델(text to image 만 되는 Latent Diffusion Model, Stable Diffusion 등)보다 의미론적인 정보를 더 잘 이해하는 것 같습니다.

## 2. Extract Keyword from Caption

### Yake

[YAKE! ](https://www.notion.so/YAKE-dabe536b203b4b5893b77b43f3f51ad7?pvs=21)

간단히 말하면 yake 는 총 5개의 지표를 활용하여 문서 내 단어의 중요도를 계산하는 통계모델입니다. 
여러 장점이 있지만 yake는 “문서” 에서의 키워드 추출 알고리즘이기 때문에 글쓰기에서 강조하는 기법들에 의존합니다. 대문자 용어가 소문자 용어에 비해 더 중요하고, 문서의 맨 처음 나타나는 용어가 중간이나 끝에 나타나는 용어보다 중요하다고 가정합니다. 

하지만 여러 장의 사진 중에 첫번째 사진에 대한 캡션이 가장 중요하지 않을 수도 있고, 캡션에서 문장 중간에 강조하기 위한 단어를 대문자용어로 사용하는 경우는 없기 때문에 기승전결이 없는 여러 캡션을 합친 문서에서 키워드를 뽑을 때는 적합하지 않아서 YAKE 대신 DistilBert를 사용하기로 하였습니다.

### Distill Bert

많은 양의 웹 문서로, 아주 큰 언어 모델을 학습시켰다는 개념인 ‘Large Language Model’ 중에서도 BERT는 그 파급력도 높고 유명한 모델입니다. 우리가 외국어를 배운다고 생각했을 때, 주어진 문장에서 빈칸에 들어갈 외국어 단어를 맞추는 방식을 떠올려 보세요. 버트는 바로 그런 방식으로 인간이 쓰는 자연어를 학습한답니다.

 버트는 그 성능이 우수하고, Fine Tuning (질문-답변, 감성분석 등 특정한 용도에 맞게 쓸 수 있도록 부가적인 학습을 시키는 것) 만으로 다양한 Task에 사용할 수 있다는 점에 있어서 좋은 모델입니다. 그렇지만 연구자들은 이 버트를 더욱 경량화 해낼 방법을 고안해 냈습니다. 그 결과가 DistilBERT입니다.

 Distillation은 우리말로 증류입니다. 큰 혼합액에서 정수만 쏙 뽑아 내는 것처럼, 딥러닝에서는 Knowledge Distillation이라는 개념을 통해 큰 모델이 잘 학습한 지식을 작은 모델이 흡수할 수 있도록 하는 개념이 도입되었습니다. Knowledge Distillation에 대한 개념은 잘 정리된 [웹 참고문헌](https://cpm0722.github.io/paper-review/distilbert-a-distilled-version-of-bert-smaller-faster-cheaper-and-lighter)을 보시길 추천드립니다.

 ‘허깅페이스’ 는 ‘트랜스포머’ 모델 가족들을 편리하게 쓸 수 있도록 된 라이브러리입니다. 버트도, 디스틸벌트도 트랜스포머 가족이기 때문에 ‘SentenceTransformer’ 이라는 모듈 안에 디스틸벌트 이름을 넣어주는 것만으로도 쉽게 모델을 불러올 수 있습니다. 아래 코드가 디스틸벌트로 키워드를 뽑아낸 것에 해당합니다. 그 인수를 항목별로 나열하겠습니다.

- docs : 우리가 만든 n개의 캡션을 이어 붙인 긴 문장 (input)
- CountVectorizer : 단어 벡터화 (숫자로 바꿔주는 것)
- candidates : stop word(me, mine처럼 정보가 없는 단어) 를 제외한 문장내 단어들

- **ke_model : distill BERT 모델**
- doc_embedding = ke_model.encode([docs]) : 인풋 문장을 distillBERT의 인코더를 이용하여 통과시킴. 인풋 문장 전체적인 정보가를 띄고 있다고 볼 수 있음. (문장의 의미)
- candidate_embeddings : 인풋 문장의 단어들을 인코딩함으로써 그 정보가를 해석함. distillBERT는 기본적으로 많은 수의 문서로 학습한 Large Model의 속성을 지녔기 때문에, ‘배경’ ‘그러나’ 이런 단어보다는 ‘여행’ ‘사과’ 이런 단어가 더 중요하다는 점을 안다.
- distance : 문장과 단어들 사이의 거리를 계산하여, **문장의 의미를 가장 잘 나타내는 단어들**을 고르도록 한다.

```python
# keword extraction setting
n_gram_range = (1, 1)
stop_words = "english"
count = CountVectorizer(ngram_range=n_gram_range,
                        stop_words=stop_words).fit([docs])
candidates = count.get_feature_names_out()  # list

# keyword extraction model
ke_model = SentenceTransformer(
    'sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1')
doc_embedding = ke_model.encode([docs])
candidate_embeddings = ke_model.encode(candidates)

distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index]
            for index in distances.argsort()[0][-num_of_cores:]]  # list
```

## 3. Recommend relevant Tags

### Word2Vec

captioning 모델이 놓진 의미론적(semantic) 정보를 보강하기 위해 Word2Vec 모델을 사용하였습니다.
처음에는 최대한 일상을 기록한 데이터로 학습시키려고 했지만 생성된 caption에서 keyword로 뽑은 단어가 word2vec의 학습 데이터에 없는 경우가 많았습니다. 그런 경우 단어의 유사도를 계산하지 못하기 때문에 관련 해시태그를 추천해 줄 수 없었습니다.

diffusion 모델에서 사용하고 있는 GPT, Optimus 의 vocab 사이즈가 워낙 크고 이 vocab 들을 대부분 포함하는 데이터셋을 구하기가 어려워서 Twitter (2B tweets, 27B tokens, 1.2M vocab,uncased) 데이터로 pretrain 된 758MB 크기의 모델을 사용하였습니다.

## 4. Impression Tags

저희가 인스타그램 해시태그 제공 서비스를 기획했을 때, 고려했던 부분 중 하나가 수요층이었습니다. 인스타그램에 대한 해시태그를 생성하고자 하는 사람이 적지만 정확한 해시태그 추천을 바라는지, 많은 양의 추천을 바라는지, 혹은 단순히 노출수에 유리한 해시태그 추천을 바라는지 알 수 없었기 때문입니다. 우리는 그 모든 것을 선택 가능한 요소로 두도록 결정하였습니다.

 impression은 인스타그램 바이럴을 목적으로 하는 정보를 제공하는 웹 페이지의 2023 best hashtags를 참조하였습니다. 이곳에는 #fashion, #pet, #Reels 등 다양한 주제로 빈도 높게 많이 같이 쓰이고, 또 따라서 노출에 유리한 해시태그들이 공개되어 있습니다.

 이미지의 정보를 이용하여 그룹화된 추천을 하거나, 더 섬세한 추천을 하는 것도 가능했지만 인퍼런스 시간과 사용편의성을 위하여 사용자가 직접 체크박스그룹을 통하여 실시간 선택하는 것이 좋겠다고 결정했습니다. 사용자는 14가지 그룹에서 원하는 해시태그 카테고리를 복수선택할 수 있습니다.

- 표 보기
    
    
    | like | '#likes', '#like', '#follow', '#likeforlikes', '#love', '#instagood', '#instagram', '#followforfollowback', '#followme', '#photooftheday', '#bhfyp', '#instalike', '#photography', '#l', '#instadaily', '#me', '#picoftheday', '#beautiful', '#myself', '#likeforfollow', '#fashion', '#smile', '#followers', '#likeforlike', '#followback', '#f', '#followforfollow', '#comment', '#likesforlikes', '#bhfyp'] |
    | --- | --- |
    | fashion | '#fashion', '#love', '#style', '#styleinspo', '#ootd', '#outfitoftheday', '#whatiwore', '#shoppingaddict', '#beautydoesnthavetobepain', '#currentlywearing', '#instastyle', '#lookgoodfeelgood', '#fashionblogger', '#fashionista', '#fashionstyle', '#fashionable', '#fashiongram', '#fashionblog', '#fashionaddict', '#fashionphotography' |
    | food | '#food', '#foodporn', '#foodie', '#instafood', '#foodphotography', '#foodstagram', '#yummy', '#instagood', '#love', '#foodblogger', '#foodlover', '#delicious', '#homemade', '#healthyfood', '#photooftheday', '#picoftheday', '#dinner', '#foodgasm', '#foodies', '#tasty', '#cooking', '#instadaily', '#lunch', '#bhfyp', '#restaurant', '#healthy' |
    | pet | '#pet', '#dog', '#pets', '#dogsofinstagram', '#cute', '#petsofinstagram', '#dogs', '#cat', '#love', '#puppy', '#cats', '#petstagram', '#animals', '#animal', '#instadog', '#instagram', '#dogstagram', '#doglover', '#catsofinstagram', '#dogoftheday', '#doglovers', '#petlovers', '#doglife', '#instapet', '#catstagram', '#instagood', '#cachorro', '#catlover' |
    | tech | '#techie', '#latesttech', '#ilovemygadgets', '#gadgetsgalore', '#apple', '#android', '#applevsandroid', '#wearabletech', '#VR', '#mobile', '#makinglifeeasier', '#tech', '#technology', '#technews', '#gadgets', '#instatech', '#software', '#innovation' |
    | wedding | '#wedding', '#bride', '#weddingdress', '#weddingphotography', '#weddingday', '#weddinginspiration', '#makeup', '#weddingplanner', '#prewedding', '#bridal', '#bridetobe', '#weddingphotographer', '#weddings', ' #groom', '#engagement', '#party', '#weddingideas', '#makeupartist', '#indianwedding', '#casamento' |
    | fitness | '#fitness', '#gym', '#workout', '#fitnessmotivation', '#motivation', '#fit', '#bodybuilding', '#training', '#health', '#lifestyle', '#fitfam', '#healthylifestyle', '#sport', '#healthy', '#gymlife', '#life', '#crossfit', '#personaltrainer', '#goals', '#exercise', '#muscle' |
    | travel | #travel', '#nature', '#photography', '#travelphotography', '#photooftheday', '#instagood', '#travelgram', '#picoftheday', '#photo', '#beautiful', '#art', '#naturephotography', '#wanderlust', '#adventure', '#instatrave', '#travelblogger', '#landscape', '#summer', '#trip', '#explore' |
    | holiday | '#stockingstuffers', '#christmasdecor', '#spookyhalloween', '#happyhalloween', '#thanksgivingtable', '#turkeyorham', '#valentineformyvalentine', '#happyfourth', '#newyearseve', '#newyearsresolution', '#holidaycrazy', '#holidayspirit', '#kwanza', '#hanukkahgift', '#underthetree' |
    | photo | '#photography', '#photooftheday', '#photo', '#picoftheday', '#photographer', '#model', '#photoshoot', '#portrait', '#beauty', '#travelphotography', '#canon', '#selfie', '#landscape', '#sunset', '#fotografia', '#portraitphotography', '#photographylovers', '#artist', '#nikon' |
    | music | '#music', '#hiphop', '#rap', '#musician', '#singer', '#musica', '#dj', '#rock', '#dance', '#song', '#guitar', '#viral', '#producer', '#newmusic', '#musicvideo', '#instamusic', '#livemusic', '#pop', '#concert', '#trap', '#beats' |
    | art | '#art', '#artist', '#drawing', '#artwork', '#painting', '#artistsoninstagram', '#illustration', '#digitalart', '#design', '#sketch' |
    | nature | '#nature', '#naturephotography', '#sunset', '#flowers', '#wildlife', '#mountains', '#hiking', '#explore', '#outdoors', '#naturelover', '#beach', '#forest' |
    | Reels | '#reels', '#reelsvideo', '#reelsinstagram', '#reelsindia', '#holareels', '#reelsbrasil', '#reelsteady', '#instagramreels', '#reelsinsta', '#instareels', '#reelsofinstagram', '#k', '#music', '#bhfyp', '#disney', '#tiktokindia', '#videoftheday', '#instareel', '#foryoupage', '#fyp' |

# 3. 데모

 데모는 그라디오로 진행했습니다. 그라디오는 머신러닝 모델을 웹 인터페이스에서 구현할 수 있는 모델로 사용이 간편하고 여러가지 기능을 구현 가능해 간편한 웹 데모 페이지를 만들 때 사용됩니다. 예를 들어 이미지가 주어졌을 때 이미지에 어떤 객체가 존재하는지 분류하는 머신러닝 모델을 구현했다고 하면 입력은 이미지, 출력은 이미지에 해당하는 텍스트(고양이, 강아지,…) 가 될 것입니다. 그라디오는 이처럼 [머신러닝 모델, 입력값의 형식, 출력값의 형식] 에 해당하는 pair와 ui 디자인 코드를 넣었을 때 아래와 같은 페이지를 구현해주는 서비스입니다.

 사전 학습된 모델이 매우 컸기 때문에 램 사용량이 많아 Google Colab의 유료 버전을 이용했습니다. 
 <img width="1608" alt="Untitled" src="https://github.com/rachel618/deepdaiv_captioning/assets/67910258/fb13fdf7-2821-4ac8-83b6-2543fd8c10e3">



 데모 페이지는 위와 같습니다. (1)번의 sliding bar을 이용해 입력 사진의 개수를 조절할 수 있습니다. 최소 1장에서 최대 10장의 사진을 입력으로 받을 수 있으며, (2)번의 체크박스를 이용해 사용자의 기호에 따라 뽑고 싶은 해시태그의 종류를 결정할 수 있습니다. 

 (2)번의 ‘Do you want more affluent recommendation using wordmap?’ 이라는 옵션을 통해 사진 또는 context와 관련된 해시태그를 더 뽑을 수 있고, ‘Do you want more hashtags for impression?’ 이라는 옵션을 체크해 상단에 노출을 목적으로 하는 해시태그(f4f, follow2follow) 등을 추가로 뽑을 수 있습니다. 

 (3)번은 이미지 입력 부분입니다. 로컬 파일 또는 구글 검색 결과를 드래그-앤-드롭 방식으로 입력할 수 있고, 입력 창을 누르면 파일 탐색기로 이동해 직접 파일을 선택할 수도 있습니다. 아래 (4)번의 submit 버튼을 누르게 되면 (5)번, 아래의 textbox에 해시태그가 생성됩니다. 위의 그림은 체크박스를 모두 공란으로 두었기에 ‘The most relative hashtags of your photos’ 부분에만 해시태그가 생성된 것을 확인할 수 있습니다. 체크박스에 체크를 한다면 아래의 ‘More affluent recommendation results’ 부분과 ‘Also these are for impressions’ 부분에도 해시태그가 생성됩니다.

 최종적으로 (6)번, 각 해시태그 옆에 있는 Accept All 버튼을 클릭하면 (7)번, 맨 아래의 텍스트박스에 해시태그가 옮겨지게 되고 원하는 결과를 복사해서 사용할 수 있게 됩니다. 

데모 최종 예시
<img width="945" alt="demo-example" src="https://github.com/rachel618/deepdaiv_captioning/assets/67910258/d9e06189-1763-4c04-97bd-bb1c4e35a29c">

* 해당 주소는 지금 작동하지 않습니다.
[https://860fce14df5772e8f4.gradio.live](https://860fce14df5772e8f4.gradio.live/)

[https://a1bda5134557dbbbac.gradio.live](https://a1bda5134557dbbbac.gradio.live/)



# 사용방법

    git clone https://github.com/rachel618/deepdaiv_captioning.git
    cd deepdaiv_captioning
    git clone --single-branch --branch feature/add_transformers https://github.com/OFA-Sys/OFA.git 
    pip install OFA/transformers/
    git clone https://huggingface.co/OFA-Sys/OFA-base
    python Demo.py





