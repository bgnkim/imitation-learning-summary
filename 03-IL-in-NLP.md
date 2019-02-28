# Imitation Learning in NLP

- 이 문서는 개인적인 정리 결과물이며 신뢰성을 담보할 수 없으니 논문을 직접 읽는 것을 권한다.
- 이 문서는 수식이 많으므로 Typora로 보는 것이 좋다.

[TOC]



## 품사 분석(POS Tagging)

### [Daume et al., 2009][] (SEARN)

알고리즘은 [imitation learning](01-IL.md/#daume-et-al-2009-(searn)) 참고

- Data: CoNLL 2000 (Syntactic Chunking task, English)

  - Training 8936 sentences, Testing 2012 sentences.
- Features

  > The standard **base features** are: the chunk length, the word (original, lower cased, stemmed, and original-stem), the case pattern of the word, the first and last 1, 2, and 3 characters, and the part of speech and its first character. We additionally consider membership features for lists of names, locations, abbreviations, stop words, etc. The **meta features** we use are, for any base feature b, b at position i (for any sub-position of the chunk), b before/after the chunk, the entire b-sequence in the chunk, and any 2- or 3-gram tuple of bs in the chunk.

- What is the problem?

  - Argmax problem (decoding problem; pre-image problem)

    - 최적화 문제를 푸는 것이 복잡한 구조에서는 불가능에 가깝고, 근사치를 구하기 위해 제한조건을 가하면 실제 답을 찾지 못하는 문제가 발생함.

  - Independent classifiers: 각 시점 간의 독립성 가정

    > However, unlike SEARN, the stacked sequential learning framework is effectively limited to sequence labeling problems. This limitation arises from the fact that it implicitly assumes that the set of decisions one must make in the future are always going to be same, regardless of decisions in the past. **In many applications, ... this is not true. The set of possible choices available at time step $i$ is heavily dependent on past choices.**

- Result

  - F~1~ = 96.98 (SEARN).
  - Note: CRF 96.48



## 개체명 인식 (Named Entity Recognition)

### [Daume et al., 2009][] (SEARN)

알고리즘은 [imitation learning](01-IL.md/#daume-et-al-2009-(searn)) 참고

- Data: CoNLL 2002 (NER task; Spanish)
  - Training 8324 sentences, Testing 1517 sentences.
  - Also trained on 300-sentence samples from the training set.
- Features

  > For each label, each label pair and each label triple, a feature counts the number of times this element is observed in the output. Furthermore, the standard set of input features includes the words and simple functions of the words (case markings, prefix and suffix up to three characters) within a window of ±2 around the current position.

- Result
  - F~1~ = 95.91 (small set), 98.11 (large set)
  - Note: CRF 93.94 (small set); SVM classifier 97.31 (large set)


## 공지시어 해소 (Coreference resolution)

### [Clark & Manning, 2015][] (Model Stacking)

- What is the problem?
  - 기존의 모형은 전체의 정합성을 보지 않고 최선만을 선택하여 Coreference로 묶인 것들의 정합성이 낮았다.
- Idea
  - Mention pair 모형을 통해 만들어진 score 값을 사용해서 clustering을 하도록 Imitation learning 사용
  - 탐색 영역을 줄이기 위해서 score가 한계치 이상인 것만 사용.
  - 계산 속도를 높이기 위해서 memoization 사용.
- Data: CoNLL 2012 Shared Task dataset
  - 2802 training documents, 343 development documents, 345 test documents.
- Structure
  - Mention pair scores > DAgger > Action
  - State: 현재의 clustering 상태
  - Action: 무엇을 묶을 것인가
- result
  - MUC F~1~ 72.59, **B^3^ F~1~ 60.44, CEAF~φ4~ F~1~ 56.02, CoNLL F~1~ 63.02**
  - MAX of 6 SOTA Systems: MUC F~1~ 72.84, B^3^ F~1~ 58.71, CEAF~φ4~ F~1~ 55.61, CoNLL F~1~ 61.71


## 구문 구조 분석 (Constituency parsing)

### [Bengio et al., 2015][] (Scheduled sampling)

- What is the problem?

  > The main problem is that **mistakes made early in the sequence generation process are fed as input to the model** and can be quickly amplified because the model might be in a part of the state space it has **never seen** at training time.
  
  - 훈련 시점과 사용 시점에서 동작하는 방식이 다름 (Distributional drift).
  - beam search로도 불충분함.

- Idea
  - Curriculum learning (Scheduled sampling)
    시간에 따라 점진적으로 정답의 사용 비율을 낮추고 시스템 생성 데이터의 사용 비율을 높임.
    즉, $\pi = \beta\pi_E + (1-\beta)\hat\pi$를 사용해 롤아웃하고, $\beta$는 1부터 0으로 낮춰나감. (논문에서는 inverse sigmoid decay를 사용함: $\beta(t) = \frac{k}{(k+e^{t/k})}$.)

- Data: WSJ 22 dataset
  - 40K training instances. Tested on the WSJ 22 development set.
  - Words in dictionary: 90K
- Structure:

  - Embedding (512) > 1 LSTM layer (512 units) > attention > 128 targets

- Result
  - F1 88.08, 88.68 (with Dropout)
  - Baseline LSTM: 86.54


## 의존구문 분석 (Dependency parsing)

### [Goldberg & Nivre, 2012][] (Dynamic Oracle)

- What is the problem?
  - 기존 방법인 Static oracle은, parse tree를 구성하는 방법이 여러가지일 경우에도 한가지 방법만 학습할 수 밖에 없었고, 훈련 집합에 없는 tree를 구성하는 방법을 알지 못했다.

    > One important use for a dynamic oracle is in training a parser that (a) is not restricted to a particular canonical order of transitions and (b) can handle configurations that are not part of any gold sequence, thus mitigating the effect of error propagation.

- Idea
  - Sequence를 지정할 게 아니라, 각 상태마다 어떤 행동이 최적인지를 cost로 표현해보자.
  - 즉, expert의 cost 함수를 프로그래밍해서 각 action이 좋은지 아닌지 학습시키자.

- Data
  - English model
    - Training set: Penn-WSJ Treebank Sections 2-21
    - Test set: WSJ section 22, 23, British National Corpus (1K), Brown Corpus, Football Corpus, QuestionBank, English Web Treebank.
  - Other languages
    - CoNLL 2007 Shared Task.
    - Arabic, Basque, Catalan, Chinese, Czech, English, Greek, Hungarian, Italian, Turkish
- Structure

  - Perceptron-style learning
- Result (English, WSJ 22 only)
  - UAS 91.24, LAS 88.76
  - Baseline (Static Oracle): UAS 90.31, LAS 87.88

### [Ballesteros et al., 2016][]

- What is the problem?
  > Although this setup obtains very good performance, the training and testing conditions are mismatched in the following way: at training time the historical context of an action is always derived from the gold standard, but at test time, it will be a model prediction.
  
  - 훈련 시점과 사용 시점에서 동작하는 방식이 다름 (Distributional drift).
  - LSTM으로 훈련하더라도 마찬가지임.
  
- Idea

  - Dynamic oracle + LSTM

- Data
  - English PTB and Chinese CTB-5. (Following the setting of others)
  - CoNLL 2009 Shared Task

- Structure
  - Using original work (Dyer et al., 2015)
  - Several modifications + imitation learning
    - Using arc-hybrid instead of arc-standard
    - Sample action s.t. $a ~ \alpha \pi$.
      - 신경망의 예측 결과가 양 극단으로 치닫는 경우가 많아, 이를 보정함.

- Result
  - English PTB
    - **UAS 93.56, LAS 91.42**
    - Baseline (Dyer et al.) UAS 93.04, LAS 90.87
  - Chinese CTB-5
    - **UAS 87.65, LAS 86.21**
    - Baseline (Dyer et al.) UAS 86.85, LAS 85.63
  - CoNLL 2009 (English)
    - UAS 92.22, LAS 89.87
    - Max of 3 SOTA: **UAS 93.22, LAS 91.23**

### [Le & Fokkens, 2017][] (Approx. Policy Gradient)

- What is the problem?
  - 정말 Reinforcement Learning (Imitation learning)을 적용하면 성능이 올라갈까?
- Idea
  - Chen & Manning (2014)에 RL만 더하여 성능을 비교함.
- Data
  - Penn Treebank 3 (Chen & Manning, 2014를 그대로 따름.)
  - German section of SPMRL-2014 dataset.
- Structure
  - Arc-standard, Arc-eager, Swap-standard 시스템을 사용함.
  - 각 시스템별로 세가지 방법 제안
    - RL-Oracle: $\pi_E$ 만 따라감
    - RL-Random: $\pi_m$을 따라 $k$개의 자취를 추출하여 사용한 후 버림.
    - RL-Memory: RL-Random과 같이 추출하되, 보상이 높은 순서대로 $k$개를 남겨서 저장하고, 각 자취는 잊어버릴 확률의 값을 $\rho$로 임의 배정받음.
- Result
  - Arc-standard (PTB)
    - RL-Memory (RL-Random): UAS 92.2, LAS 90.6
    - Baseline: UAS 91.3, LAS 89.4
  - Arc-eager (PTB)
    - RL-Memory (RL-Random): UAS 89.8, LAS 87.4
    - Baseline: UAS 88.3, LAS 85.8
  - Swap-standard (SPMRL)
    - RL-Memory (RL-Random): UAS 87.6, LAS 84.6
    - Baseline: UAS 84.3, LAS 81.3

## 자연어 생성 (NLG, Image captioning, etc...)

### [Bengio et al., 2015][] (Scheduled sampling)

- Task: Image captioning
- Idea: Curriculum learning
- Data: MSCOCO dataset
  - 75K training images, 5K developmental set images.
  - Words in dictionary: 8,857
- Structure:
  - Embedding (512 dim) > 1 LSTM layer (512 units) > Embedding (512)
  - Image는 Embedding의 초기값으로 간주함.
- Result
  - BLEU 30.6, METEOR 24.3, CIDER 92.1
  - Baseline: BLEU 28.8, METEOR 24.2, CIDER 89.5


# 참고문헌

## 학회 튜토리얼 자료들

[EACL2017 Tutorial]: https://github.com/sheffieldnlp/ImitationLearningTutorialEACL2017	" EACL2017 Tutorial on Imitation Learning "
[ICML2018 Tutorial]: https://sites.google.com/view/icml2018-imitation-learning/	" Imitation Learning Tutorial (Retrieved at 18.11.30) "

## 논문들

[Ballesteros et al., 2016]: https://arxiv.org/pdf/1603.03793.pdf "Ballesteros et al. (2016). Training with Exploration Improves a Greedy Stack LSTM Parser. EMNLP '16"
[Bengio et al., 2015]: https://arxiv.org/abs/1506.03099	"Bengio et al. (2015). Scheduled Sampling or Sequence Prediction with Recurrent Neural Networks. NIPS '15"
[Clark & Manning, 2015]: https://cs.stanford.edu/people/kevclark/resources/clark-manning-acl15-entity.pdf	"Clark & Manning. (2015). Entity-Centric Coreference Resolution with Model Stacking, ACL '15"
[Daume et al., 2009]: https://link.springer.com/article/10.1007/s10994-009-5106-x "Daume et al. (2009). Search-based structured prediction. Machine Learning 75"
[Goldberg & Nivre, 2012]: http://www.aclweb.org/anthology/C12-1059 "Goldberg & Nivre. (2012). A Dynamic Oracle for Arc-Eager Dependency Parsing. COLING '12"
[Goodman et al., 2016]: http://aclweb.org/anthology/P16-1001 "Goodman et al. (2016). Noise reduction and targeted exploration in imitation learning for Abstract Meaning Representation parsing. ACL '16"
[Le & Fokkens, 2017]: http://www.aclweb.org/anthology/E17-1064 "Le & Fokkens. (2017). Tackling Error Propagation through Reinforcement Learning: A Case of Greedy Dependency Parsing. EACL '17"

## 아직 읽지 않은 논문들

* 체크는 읽은 논문들

- [x] [Chang et al., 2015]: https://arxiv.org/pdf/1502.02206.pdf	"Chang et al. (2015). Learning to search better than your teacher. ICML '15"
- [x] [Ranzato et al., 2016]: https://arxiv.org/abs/1511.06732	"Ranzato et al. (2016). Sequence level training with recurrent neural networks. ICLR '16"
- [ ] [Vlachos & Craven, 2011]: http://www.aclweb.org/anthology/W/W11/W11-0307.pdf	"Vlachos & Craven. (2011). Search-based Structured Prediction applied to Biomedical Event Extraction. CoNLL '11"
- [x] [Vlachos & Clark, 2014]: http://www.aclweb.org/anthology/Q14-1042	"Vlachos & Clark. (2014). A new corpus and imitation learning framework for context-dependent semantic parsing. TOACL 2"
- [x] [Lampouras & Vlachos, 2016]: https://aclweb.org/anthology/C/C16/C16-1105.pdf	"Lampouras & Vlachos. (2016). Imitation learning for language generation from unaligned data. COLING '16"
- [ ] [Goldberg & Nivre, 2013]: https://www.aclweb.org/anthology/Q/Q13/Q13-1033.pdf	"Goldberg & Nivre. (2013). Training Deterministic Parsers with Non-Deterministic Oracles. TOACL 1"
- [x] [Tsuruoka et al., 2011]: http://www.anthology.aclweb.org/W/W11/W11-0328.pdf	"Tsuruoka et al. (2011). Learning to lookahead: Can History-Based Models Rival Globally Optimized Models? CoNLL '11"
- [ ] [Daume III, 2009]: http://www.umiacs.umd.edu/~hal/docs/daume09unsearn.pdf	"Daume III. (2009). Unsupervised Search-based Structured Prediction. ICML '09"
- [ ] [Liang et al., 2009]: http://www.aclweb.org/anthology/P09-1011	"Liang et al. (2009). Learning Semantic Correspondences with Less Supervision. ACL '09"

  

