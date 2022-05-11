# 0511 자연어처리 강의

#  자연어처리 -> text처리
# 음성인식 -> 파동
# 번역 -> 유사도분석 통해 연관성기반 매칭
# 요약 , 분류 ( 스팸메일 분류 --> 광고성 메세지 특징 찾아서 )

# 챗봇 : setiment analysis 텍스트에 녹아있는 감성, 의견을 파악
#       tokenization 토큰화 : 문장을 형태소 단위로 자르는 것
#       named entity recognize : 텍스트로부터 주제 파악하기
#       namalization : 의도된 오타 파악하기 -> 일반화, 정구회
#       dependency parsing : 문장 구성성분의 분석


# ex> siri --> 질문에 대한 답을 하는일
#       음성 데이터로부터 특징을 추출 --> 단어 파악, 단어의 특징 추출
#       언어별로 갖고있는 특성을 반영 --> ex> 영어 : SVO , 한국어 : SOV 등의 차이
#       deep learning : 이미 학습된 데이터로부터 음성 신호 처리
#       HMM ( hidden markov model ) : 앞으로 나올 주제의 예측
#       similarity analysis : 음성 신호가 어떤 기준에 부합하느냐 : 질문? 감탄? 혼잣말? 구분

# 번역기
#       encoding : 유사도 기반 자연어의 특징 추출 , 필요없는 부분 제거
#       time series modeling : 시간에 따른 데이터 처리
#       attention mechanism : 번역에 필요한 부분만 집중해서 유사도 기반 단어 매칭
#       self attention : 문장 사이의 상관관계를 분석
#       transformer : attention 구조를 활용한 번역


"""
텍스트 전처리 과정

1. 토큰화
--> 문장을 자르는 것(단어 기준)

2. 정제 및 추출
--> 중요한 단어만 놔두고 자르기 , 불필요한 단어 제거 (cleaning)

3. 인코딩
--> 남겨진 단어들을 숫자로 바꿈



언어의 형태소

ex> 화분에 예쁜 꽃이 피었다
형태소 단위로 나누기
-자립 형태소 : 명사, 수사, 부사, 감탄사
-의존 형태소: 조사, 어미, 어간


언어 전처리 과정
sentence -> tokenization -> Cleaning, stemming
-> encoding -> sorting -> padding, similarity

토큰화 : 주어진 문장에서 "의미 부여"가 가능한 단위를 찾는다
어려운 예시 : 어제 삼성 라이온즈가 기아 타이거즈를 5:3으로 꺾고 위닝 시리즈를 거두었습니다.

--> 5:3을 하나의 단어로 볼 것인지 5 : 3이라는 3개의 단어로 볼 것인지 예외처리를 통해
    처리해 준 다음에 진행해야 함 단순히 구두점이나 특수문자를 전부 제거하는걸로는 안됨

* 표준화된 토큰화 방법
표준 토큰화( Treebank tokenization )
ex> Model-based reinforcement learning don't need a value function for the policy.
"""
# from nltk.tokenize import TreebankWordTokenizer
# tokenizer = TreebankWordTokenizer()
# text = "Model-based reinforcement learning don't need a value function for the policy."
# print(tokenizer.tokenize(text))
#--> list형태로 return


"""
문장 토큰화 : 문장 단위로 의미를 나누기

한국어 토큰화의 어려움 단어마다 예외처리를 해줘야하기 떄문에

토큰화 --> 패키지 사용 but 패키지 마다의 특성을 파악한 후 진행하기 
"""


"""
* 데이터 정제(cleaning)
--> 데이터 사용 목적에 맞춰 노이즈 제거 
1. 대문자 vs 소문자 US->us 의미가 변화되기 때문에 예외처리 꼼꼼히 진행해야함
2. 출현 횟수가 적은 단어의 제거
3. 길이가 짧은 단어, 지시(대)명사, 관사의 제거


*데이터 추출
어간(stem): 단어의 의미를 담은 핵심
접사(Affix): 단어에 추가 용법을 부여
ex> lectures에서의 s
    playing에서의 ing
    지우기!
    
*어간 추출(stemming)
porter algorithm : 대표적인 stemming방법 
ex> formal(ize) , toler(ance) , electri(cal)

*표제어 추출 
ex> is, are -> be

*표제어 추출 vs 어간 추출
표제어 추출은 단어의 품사정보 포함, 어간 추출은 품사 정보 포함 x 
단어의 뜻이 분명한 단어만 모여있다면 굳이 표제어 추출 하지 않아도 ok 
그러나 bear처럼 품사 정보에 따라 의미가 변하는 단어가 있으므로 표제어 추출 선호 

*불용어 stopword
문장에서 대세로 작용하지 않는, 중요도가 낮은 단어 제거 -> 불용어 목록이 없다면 추가해서 만들어야
지워줄 필요가 있을때는 지워주는게 좋음 ( 메모리, 시간, 해석차이 절감효과)
import nltk 
nltk.download('stopword')
from nltk.corpus import stopwords
print(stopsords.words('english')[:5])

*불용어 제거방법
1. 불용어 목록을 받아오기
2. 정제할 문장의 토큰화
3. 비교해가면서 제거 

for w in word_tokens:
    if w not in stop_words:
        result.append(w)
print(word_tokens)
print(result)



*정수 인코딩 (integer-encoding)
-> 처음보는 단어들에 번호를 붙여서 인코딩 봤던건 이전의 번호를 그대로 사용 
--> 단어 등장 횟수 count후 빈도수 높은 단어들을 앞의 숫자로 인코딩
        : 숫자가 작을수록 사용하는 메모리 감소, search시 앞에서 걸리기 때문에 속도 감소

*정수 인코딩 1. Dictionary
1. 문장의 토큰화 - 불용어 및 대문자 제거 과정을 거친다.
2. 빈 단어 dictionary vocab={}을 만든다
3. 토큰화된 각 단어에 대해서
        단어가 vocab에 없으면 -> vocab[단어]= 0  (단어, 등장횟수)
        단어가 vocab에 있으면 -> vocab[단어] += 1

*정수 인코딩 2. 빈도순 정렬
-> python의 enumerate연산의 역할

mylist = ['english','Math','Science']
for n,name in enumerate(mylist):
    print("Course " {} , Number " {}".format(name, n))
    
빈도수로 정렬된 dict list
vocab = [('apple',8),('July',6),('piano','4),('cup',2),('orange',1)]
key값만 따로 빼온 후 enumerate로 encoding
word2inx = {word[0] : index + 1 for index, word in enumerate(vocab)}



* Padding ( zero-padding )
인코딩된 길이가 전부 다르면 컴퓨터가 처리하기 힘듬 패딩 통해 길이를 맞춰줌
문장의 길이가 가장 긴 숫자를 value값으로 정한 뒤, 만약 문장의 길이가 value값보다 작으면
0으로 채워서 문장길이 맞추기~~

1. 문장들에 대해 정수 인코딩을 거친다
2. 각 문장에 대해서: 해당 문장이 가장 긴 문장의 길이보다 작을 경우 0을 추가


* One - hot encoding
-> 분류 모델에서 cross-entropy를 사용하기 위해서는 0,1로 이뤄진 벡터를 사용하는게 유리

* word2vec encoding : 단어의 유사성을 인코딩에 반영 -> 인코딩 벡터가 비슷하다 = 단어가 유사하다
->유사한 단어들끼리 유사도가 비슷하도록 encoding


*TF- IDF (term frequency - inverse document frequency )
-> 단어들의 중요한 정도를 "가중치"로 매기는 방법
            TF - IDF = tf(d,t) X idf(d,t)
d : 특정 문서 번호
t : 특정 단어 번호 

TF : 특정 문자 d에서 특정 단어 t의 등장횟수
ex) tf(1, sky) --> 횟수 return 
idf(d,t)는 log를 취한 역수값

둘을 곱하면 tf-idf
"""


#2장~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`

"""
언어 모델
1. 통계 기반 언어 모델
개별 단어 = w 
단어 sequence = W = {w1, w2 ...} 나는, 밥을 , 먹는다. 로 매칭 -> 단어 sequence는 개별 단어의 합 
n개의 단어로 이루어진 sequence가 등장할 확률은 개별단어의 곱

* markov chain 
    w1  -> w2 -> w3 한 방향으로만 영향을 미친다는 주장
    한국어 언어 모델에서는 사용하기 어려움 어순이 바뀌기 때문에 
    
ex>  오늘 9시에 퀴즈를 봤어야 했는데 늦잠을 자는 바람에 _____
 9번째의 예상 단어를 맞추기 위해서는 8개까지의 문장, 9개까지 말한 문장도 많아야하지만
 표본이 너무 적음 -> 문장이 나올 확률의 희소성 문제 "sparsity problem"
 
* N - gram Language Model
->통해 문장의 희소성 문제 해결 가능 

ex )Deep learning has become a core technique for data scientists. 
    N = 2의 model 사용시 ~ p(core | become a ) 앞의 2개의 단어만 참조!
    정확성은 많이 떨어지지만 희소성문제는 해결 가능

    
* 유사도 분석

*벡터 유사도 Cosine Metric : 벡터는 크기와 방향을 갖는 성질
 -> 크기는 같지만 방향이 다를떄 cosine metirc 사용 
 -> 벡터의 내접값을 방향으로 나누면 cos0가 나옴 0의 의미는 각도
 -> 180도의 cosine값은 -1 , 90도의 값은 0 , 0도의 값은 1
 -> 같은 크기와 방향을 갖는 두 벡터= 거의 일치함 = cosine 1로 cosine값이 제일 큼
 ->cosine metric값을 사용하면 단어의 유사도를 파악할 수 있지 않을까 하는 생각
 
 
*벡터의 내적과 norm
 -> 두 벡터의 내적은 norm에 cos0
 -> 내적은 성분끼리 곱한 다음에 더하는것
 -> 벡터의 norm값은 각 요소의 제곱의 루트
 
 ex> [1,1,0]
     [1,0,-1]
     
내적 = 1x1 + 1x0 + 0x-1 = 1
norm = 루트에 성분의 제곱 = 루트2

--> 두 벡터 사이의 코사인 유사도는 1 = 루트2 x 루트2 x cos0
                               cos0 = 0.5
                               
in python에서의 norm계산 : np.linalg.norm(a)
               내적 계산 : atb
               
               

* 문장 유사도 분석 : BoW( Bag-of-Words)사용
--> 단어의 등장횟수를 벡터와 시켜서 문장과 문장 사이의 유사도를 구할 수 있음
1. 토큰화 한 후 딘어의 빈도수 체크 
2. 각각의 norm 구한 후 내적값 계산 
3. cos0 구하기 


*벡터 유사도 Euclidean Metric : 거리 개념 
--> 각도 사이의 길이
--> 거리값의 norm값이다 d = 루트(x1-x2)제곱+(y1-y2)제곱

*레벤슈타인 거리 Levenshtein Distance
-> 단어 사이의 거리를 나타내는 대표적인 척도 , 단어a를 단어b로 수정하기 위한 "최소 횟수"
    (단어의 삽입, 삭제, 변경 )
    
*레벤슈타인 거리 : Tabular Method 통해 구할 수 있음
ex> 데이타마닝 -> 데이터마이닝

*자카르드 거리
->단어와 집합사이의 관계

"""


#3장~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`텍스트마이닝

"""
*Word2Vec -> 정수 인코딩/원핫인코딩의 단점 (단어와 단어 사이의 관계 반영x) 보완
          -> 원핫인코딩 한계 :1. 메모리 , 2. 희소 표현 Sparse Representation 벡터 중에 0이 많고 1이 적은것
              -> 뉴럴 네트워크할때 문제 생길 가능성 3. 연관성 문제 
        = Word2Vec이란 압축시킨것
              
*Word Embedding
-> 밀집 표현 ( Dense Representaiton) 희소 표현 문제를 보완, 
    -> 벡터의 차원을 원하는 대로 설정할 수 있음 
    -> 데이터를 이용해서 표현을 학습함 -> NN 사용 
    =단어를 밀집 벡터의 형태로 표현하는 방법이 워드 임베딩
    =밀집 벡터는 워드 임베딩 과정을 통해 나온 결과이기 때문에 임베딩 벡터라고도 함
    
*Word2Vec : NN를 사용해서 밀집표현 

 단어를 모두 레이블링하는건 어렵기 때문에 한 문장안에 붙어있는 단어, 
 문단 안에 자주 등장하는 단어들끼리
 상관관계가 높지 않을까-?
 
*Word2Vec Methods

1. CBOW (Continuous Bag od Words)
    ->주변 단어를 활용해 중간 단어를 예측(prediction보다는 학습을 위한 레이블링을 해줌)
    주변 단어의 갯수도 설정해 줘야함 
    ex> I studied hard for the exam 에서의 중심단어가 for, 주변단어2면 hard, the
    -->중간에 for가 있으면 주변 단어가 hard, the일 확률이 높겠구나~ 하고 레이블링
    
    *CBOW 데이터셋 구성
    문장을 토큰화-정제,추출-인코딩(ex정수인코딩)gndp 1이 중심 주변2,3  2가 중심 주변 1,3,4
    이런 식으로 dataset구성  
    
    *CBOW: Network structure
    주변 단어를 모두 input : input layer의 갯수 #neighbor X2 (좌우) X 원핫인코딩된 값까지
    projection layer 
    output layer -> 예측값이랑 정답이랑 cross entropy 비교해서 loss 구함
    
    *CBOW: Weight Matrix
    input layer - projection layer(hx1 :중심단어의 압축) - output layer(Mx1 원핫인코딩갯수)
    (h<M)
    중에서 projection layer를 중심단어의 Dense Rep라고 함 -> 이 값을 새로운 워드임베딩 벡터로 사용
    
    
    ===주변단어에서 중심단어로 forword후 loss값 구한뒤
    back propagation 두개의 weight metrics를 학습시키는게 목표!
    
    

2. Skip-Gram
->중간 단어를 활용해 주변 단어를 예측 CBOW와 네트워크 구조는 똑같음 but forword, backword가 반대임


3. SGNS : SkipGram with negative sampling 
    ->skipgram과 CBOW의 문제점
    : 단어 수가 많아지면 예측력이 떨어짐 

SGNS를 통해 단점 극복하고자 함
-negative sampling 은 Word2Vec 학습 과정에서 학습 대상의 단어와 관련이 높은 단어에 보다 집중!
-skipgram이 중심-> 주변 예측이라면
-SGNS는 선택된 두 단어가 중심단어와 주변단어 관계인가?를 알고자 함 OX퀴즈

#2 레이블 2개짜리 
ex> I studied hard for the exam

for -> SkipGram -> hard
for,hard -> SGNS -> 0.95    1:Yes, 0:NO binary이기 때문에 연산량 적음



"""

"""
다시 언어모델로!

* NN - gram Language Model
-> N- gram model에 뉴럴 적용

ex> Neural network is essential for machine learning technique.
        input                       output
[essential, for, machine]  -->     'learning'


*NNLM : network structure

 input - hidden - output 
 한계 : 정해진 길이의 과거 정보만을 참조하므로 함축된 정보를 파악할 ㅅ ㅜ없음
        문장의 길이가 달라진 경우 한계점이 명확함
        
* NNLM vs Word2Vec(CBOW)
-> NNLM은 앞에걸로 뒤에 예측, Word2Vec은 주변걸로 중심 예측 




"""
