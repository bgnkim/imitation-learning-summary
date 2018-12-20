# Inverse Reinforcement Learning (WITH Model)

*참조: [ICML2018 Tutorial][]*

시스템이 스스로 하게 하는 것은,

- 시스템이 목표를 좇아 효율적으로 행동하게 하는 것이고
- 다시 말해, 시스템이 보상을 최대화하는 방향으로 행동하게 하는 것이며

따라서 **강화학습**과 연결된다.

다만, 보통의 경우 전문가는 행동의 목표를 알지만 그것을 수식으로 적지는 못하는 상황이다.



<u>[전제조건]</u>

- $S$, $A$, $\pi^*$는 주어진다.
- 환경 모형 $\theta$는 일단은 알고 있다고 전제하자.



<u>[할 일]</u>

* 알지 못하는 보상함수 $R^*$를 찾아 최종적으로는 전문가의 행동을 따라하거나, 더 나은 행동을 해야 한다. 다시 말해
  $$
  \pi^* = \arg\max_{\pi} \mathbb{E}_{s\sim p_\pi} \left[ R^*(s) | \pi \right]
  $$

  가 성립하는 $R^*$를 찾아야 한다. 또는 Value 함수로 표기하면 아래와 같다.
  $$
  \mathbb{E}\left[\left.\sum_t \gamma^t R^*(s) \right| \pi^* \right] \ge \mathbb{E}\left[\left.\sum_t \gamma^t R^*(s) \right| \pi \right] \quad\forall\pi
  $$






### 정의


이렇게 보상함수를 찾아 학습하는 방법을 Inverse Reinforcement Learning 또는 Inverse Optimal Control이라 한다.

[Abbeel & Ng, 2004][]가 따르는 정의

> The problem of deriving a reward function from observed behavior is referred to as inverse reinforcement learning (Ng & Russell, 2000)
>

[Ratliff et al., 2006][]에서 언급한 정의

> The goal in IRL is to observe an agent acting in an MDP (with unknown reward function) and extract from the agent's behavior the reinforcement function it is attempting to optimize.

[Ziebart et al., 2008][]이 따르는 정의

> ... to recovering a reward function that induces the demonstrated behavior with the search algorithm serving to "stitch-together" long, coherent sequences of decisions that optimize that reward function.

[Finn et al., 2016][]이 따르는 정의

> Inverse optimal control (IOC) or inverse reinforcement learning (IRL) provide an avenue for addressing this challenge[defining a cost function] by learning a cost function directly from expert demonstrations, e.g. ...



<u>[일반적인 방법]</u>

1. $\mathcal{T}_E$를 사용해서 $R_w$를 학습 (보통 $R_w = w^\top \phi(s, a)$로 가정)
2. 학습된 $R_w$와 주어진 데이터를 사용해서 $\pi$ 학습 (강화학습 알고리즘 사용)
3. $\pi$와 $\pi_E$를 비교하여 차이가 많이 난다면 1부터 반복.

문제는, 같은 목표를 표현하는 **$R_w​$가 여러개**라는 것. 어떤 기준으로 여러 개 중 하나를 골라야 할까?



## New Concept: Feature Expectation

$R_w = w^\top \phi(s, a)$ 라고 가정하면, 다음이 성립한다.
$$
\mathbb{E}\left[\left. \sum_t \gamma^t R_w(s) \right| \pi \right] = \mathbb{E}\left[\left. \sum_t \gamma^tw^\top\phi(s) \right| \pi\right] = w^\top\cdot\mathbb{E}\left[\left.\sum_t \gamma^t\phi(s)\right| \pi\right]
$$
$ \mathbb{E}[\sum_t \gamma^t\phi(s)|\pi]$ 는 $\pi$에 따라 변화하며, 이외의 변인은 없다.

따라서, 이를 **feature expectation**이라 하고, $\mu(\pi)$ 또는 $\mu$로 표기한다.



Feature expectation을 활용해 식 (2)를 다시 서술하면 다음과 같다.
$$
w^\top \cdot \mu(\pi^*) \ge w^\top \cdot \mu(\pi)\quad \forall\pi
$$


꼭 어디서 많이 본 문제 같다.



## Maximum Margin Methods

Support vector machine의 기초가 되는 Maximum margin classifier는 보통 다음 최대화 문제의 해인 $w$를 찾는다.
$$
\max_{\zeta, w}\quad \zeta\\
\textrm{s.t.}\quad w^\top x^+ \ge w^\top x^- + \zeta\quad \forall x^+ \forall x^-\\
\quad \|w\|_2 \le 1
$$


식 (4)는 식 (5)와 닮았다. $\mu_E:=\mu(\pi_E)$가 유일한 positive example이라고 두면, 다음과 같이 Maximum margin 형태로 만들어 볼 수 있다.



### [Abbeel & Ng, 2004][] (QP version)

$\epsilon$은 허용할 오차 범위를 나타내며, $\mu_E$는 $\mathcal{T}_E$를 표본으로 하여 근사할 수 있다.

1. 임의로 $\pi_0$를 선택한다.
2. $\zeta_m > \epsilon$인 동안 $m=1,2,\cdots$을 증가시키면서 반복한다:
   1. $\mu(\pi_m)$을 계산한다  (실제 가능한 경우를 모두 구하든, Monte Carlo 샘플링을 하든...)
   2. **(보상함수의 추정)** $\zeta_m := \max_{w:\|w\|_2<1} \min_{n: 0\le n<m} w^\top (\mu_E - \mu(\pi_n))$를 이차계획법으로 풀어낸다.
      이 때, $w_m$은 해당 $\zeta_m$을 만드는 $w$값이 된다.
   3. **(최적 정책 찾기)** 강화학습 알고리즘을 사용해 보상함수가 $R_{w_m}$일때의 최적 정책 $\pi_m$을 찾는다.
3. $\{\pi_m | m = 0, 1, \cdots\}$ 를 반환한다.



* 이 절차는 최대 $M = O(\frac{d}{(1-\gamma)^2\epsilon^2}\log\frac{d}{(1-\gamma)\epsilon})$ 만큼의 반복수에 종료된다.
* $1-\delta$ 만큼의 확률로 최대 $M$회의 반복수로 종료되려면, 아래 조건으로 충분하다:

$$
|\mathcal{T}_E| \ge \frac{2d}{(\epsilon(1-\gamma))^2}\log\frac{2d}{\delta}.
$$



### Abbeel & Ng, 2004 (Projection version)

위의 QP version에서 Step 2-2를 다음과 같이 변경한다.

1. $m=1$이라면 $w_1=\mu_E - \mu(\pi_0)$ 이고 $\zeta_1 = \|w_1\|_2$ 이다. 아래는 $m>1$일 때를 나타낸다.
2. $\bar{\mu}_{m-1} := \bar{\mu}_{m-2} + \frac{(\mu_{m-1} - \bar\mu_{m-2})^\top (\mu_E - \bar{\mu}_{m-2})}{(\mu_{m-1} - \bar\mu_{m-2})^\top(\mu_{m-1}-\bar\mu_{m-2})} (\mu_{m-1}-\bar\mu_{m-2})$와 같이 계산하여, $\mu_E$를 선분 $\bar\mu_{m-2}\mu_{m-1}$에 사영시킨다. (단, $\bar\mu_0 = \mu(\pi_0)$)
3. $w_m = \mu_E-\bar\mu_{m-1}$ 로 설정하여 weight를 업데이트한다.
4. $\zeta_m = \|w_m\|_2$로 계산한다.



* QP Version과 동일한 성질을 갖는다.



### [Ratliff et al. 2006][] (MMP)

이 방법은 여러개의 MDP가 있다고 가정한다. $k$를 각 MDP를 구분하는 index로 사용하자.

다음과 같이 multiple MDP 상황으로 확장할 수 있다 ([CMU10703][] 참고):

1. 각 MDP의 해집합 $\Pi_k$를 공집합으로 초기화한다.

2. 다음 최대화 문제를 풀어서 $w$를 구한다.
   $$
   \min_w\quad \|w\|_2^2 + C\sum_k \zeta_k\\
   \textrm{s.t.}\quad w^\top \mu(\pi_{E,k}) \ge w^\top \mu(\pi_k) + l(\pi_{E, k}, \pi_k)-\zeta_k\quad \forall k, \pi_k\in\Pi_k.
   $$

3. 각 $k$ 마다:

   1. 찾은 $w$를 사용하여, 각 $k$마다 가장 크게 위반된 조건^1^을 나타내는 $\pi_k$를 찾는다^2^:

   $$
   \pi_k = \arg\max_{\pi_k} w^\top \mu(\pi_k) + l(\pi_{E,k}, \pi_k)
   $$

   2. $\pi_k$를 $\Pi_k​$에 더한다.

4. 만약 새로 더해지는 $\pi_k$가 없었다면, $\Pi_k$ 들을 반환한다.



주^1^: SVM의 Support Vector와 비슷하다.

주^2^: 환경 모형을 안다면 계산하도록 하고, 모른다면 강화학습을 사용한다.



### [Syed & Schapire, 2008][] (MWAL)

게임이론의 접근법을 사용해서 $w$를 찾는 min player와 $\pi$를 찾는 max player의 평형점을 찾도록 접근한다.

이 과정에서 feature expectation의 차이, 즉 $\mu - \mu_E$를 모든 $\pi$에 대해 모은 matrix를 게임 행렬 $G$로 둔다 (즉, $G(k, j) = \mu(\pi_k)(j) - \mu_E(j)$).

1. $\beta = \left( 1+\sqrt{\frac{2\ln k}{T}} \right)^{-1}$.
2. 모든 $\vec{\mu}\in\mathbb{R}^d$에 대해 $\tilde G: (\vec{\mu}) \mapsto \frac{1-\gamma}{4} (\vec{\mu} - \vec{\hat{\mu}}_E  \oplus 2)$로 정의한다 ($\oplus$는 element-wise addition).
   $\vec{w}_1 = [1, 1, 1, \cdots, 1]\in \mathbb{R}^d$로 설정한다.
3. $m=1, \cdots, M$까지의 iteration 동안:
   1. $\mathbf{w}_m = \frac{1}{\|\vec{w}_m\|_1}\vec{w}_m$.
   2. $R_{\mathbf{w}_m}$를 보상함수로 하여 최적 정책 $\hat\pi_m$ 를 $\epsilon_P$ 오차 이내로 추정한다.
   3. 찾은 최적 정책을 바탕으로 $\mathbf{\mu}(\hat\pi_m)$을 $\epsilon_F$ 오차 이내로 추정하여 $\hat\mu_m$이라 한다.
   4. $\vec{w}_{m+1} = \vec{w}_m\odot \beta e^{\tilde{G}(\hat{\mu}_m)}$으로 설정한다. ($\odot$는 element-wise multiplication, 지수 연산도 element-wise.)
4. $\{\hat\pi_m | m=1, 2, \cdots, M\}$ 을 반환한다. (또는 이들이 균등하게 섞인 정책을 반환한다.)



* 결과로 얻은 정책은 $\pi_E​$보다 좋을 수도 있다([Abbeel & Ng, 2004][]나 [Ratliff et al., 2006][]과 달리 상한이 존재하지 않는다.)
* 결과로 얻은 정책이 $\pi_E$보다 $\epsilon$ 오차 범위 이내에서 좋을 확률이 $1-\delta$이길 원한다면, 다음으로 충분하다 (정리의 상세한 형태는 [Syed & Schapire, 2007][] 참고).

$$
M \ge \frac{9\ln d}{2(\epsilon'(1-\gamma))^2},\\
|\mathcal{T}_E| \ge \frac{2}{(\epsilon'(1-\gamma))^2}\ln\frac{2d}{\delta}
$$



### Maximum Margin의 문제

Maximum Margin은 정규화(regularization)를 통해 $\pi$가 한정되게 하여 여러 가능한 정책 중에 하나를 고르는 문제를 해결하려 했지만. 여전히, 

- 거의 모든 행동과 합치되는 보상함수나 정책은 없다.
- Feature expectation이 동일하도록 만들 수 있는 **정책들의 조합이 너무 많다**. (정책을 확률적으로 조합하면 만들 수 있다.)
  - 만약 $\pi_1$과 $\pi_2$가 $\mu(\pi_1) \simeq \mu_E$, $\mu(\pi_2)\simeq \mu_E$ 였다면,
    임의의 $\alpha$에 대해 $\mu(\alpha\pi_1 + (1-\alpha)\pi_2)\simeq\mu_E$도 성립한다.
  - Sub-optimal Expert의 정책을 사용하거나, 알고리즘이 관련된 상태공간을 모두 고려하지 못할 때 많이 발생한다. ([Ziebart, 2008][])



## Maximum Entropy Methods

다시, $R$을 모르는데 $R$을 극대화하는 $\pi$를 찾아야 하는 문제가 되었다.

전문가를 살펴보면, 전문가는 보상을 극대화하는 자취를 선호한다. 다시 말해서, 자취의 분포는 보상과 연관이 있다.

그러면, 간단하게 softmax 형태를 취해서 이렇게 생각해 볼 수 있을것이다.
$$
R(\tau) = \sum_{s_j} w^\top \phi(s)\\
P(\tau) \propto e^{R(\tau)}
$$

좋은 확률분포는, Entropy가 최대인 확률분포이다.

즉, $w = \arg\max_w \sum_i \log P(\tau)$를 Gradient ascent로 풀어보자. 
$$
L(w) := \sum_{\tau\in\mathcal{T}_E} \log P(\tau|w)\\
\Rightarrow \nabla L(w) = \frac{1}{N}\sum_i \phi(\tau_i) - \sum_\tau P(\tau|w)\phi(\tau) = \frac{1}{N}\sum_i \phi(\tau_i) - \sum_s D_s \phi(s)
$$
이 때, $D_s$는 $s$를 방문하는 빈도 값이다. 이를 활용해서 다음과 같이 $w$를 계산할 수 있다.



### [Ziebart, 2008][] (MaxEntIRL)

**Backward pass** 

$\sum P(\tau) = 1$을 만들기 위해서, 총합을 계산하는 과정으로, 계산하기 쉬운 $\mathcal{S}_F$부터 역으로 연산함.

1. 모든 $s\in\mathcal{S}_F$에 대해 $Z_s = 1$로 설정한다.

2. Time horizon만큼 다음을 반복해서 계산한다.
   $$
   Z_{a_{i,j}} = \sum_k \theta(s_i, a_{i,j}, s_k)e^{R(s_i|w)}Z_{s_k}\\
   Z_{s_i} = \sum_{a_{i,j}} Z_{a_{i,j}} + \mathbb{I}(s_i\in F)
   $$






**Local action probability computation**

각 상태별로 행동을 취할 확률을 계산함. $s_i$를 지나가는 경로 중에서 $a_{i,j}$를 거친 경로의 비율.
$$
P(a_{i,j}|s_i)=\frac{Z_{a_{i,j}}}{Z_{s_i}}
$$

**Forward pass**

이제 각 시점별 상태 방문 빈도 값인 $D_{s,t}$를 계산한다

1. $D_{s_i, t} = P(s_i \in \mathcal{S}_I)$.
2. Time horizon만큼 다음을 반복해서 계산한다

$$
D_{s_k, t+1} = \sum_{s_i}\sum_{a_{i,j}} D_{s_i, t}P(a_{i,j}|s_i)\theta(s_i, a_{i,j}, s_k).
$$

**Summing frequencies**

빈도값을 다 더한다: $D_{s_i} = \sum_t D_{s_i, t}$



이제 $D_s$를 사용해서 gradient ascent를 할 수 있다.



### Maximum Entropy Method의 문제

우리는 사실...

- 환경 모형 $\theta$를 안다고 전제하고 있다. (모르는 경우가 더 많다)
- $R_w=w^\top \phi$로 선형성을 전제하고 있다. (선형이 아닌 경우가 일반적이다)



이 제한조건들도 풀어버릴 수 있을까? → [Inverse Reinforcement Learning (WITHOUT Model)](./InverseReinforcementLearningWOModel.md)





------

**References:**

[ICML2018 Tutorial]: https://sites.google.com/view/icml2018-imitation-learning/	" Imitation Learning Tutorial (Retrieved at 18.11.30) "

[Abbeel & Ng, 2004]: https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf "Abbeel & Ng. (2004). Apprenticeship Learning via Inverse Reinforcement Learning. ICML04"

[Ratliff et al., 2006]: https://dl.acm.org/citation.cfm?id=1143936 "Ratliff et al. (2006). Maximum Margin Planning. ICML06"
[CMU10703]: http://www.andrew.cmu.edu/course/10-703/ "Deep Reinforcement Learning and Control (Retrieved at 18.11.30)"

[Syed & Schapire, 2008]: https://papers.nips.cc/paper/3293-a-game-theoretic-approach-to-apprenticeship-learning.pdf "Syed & Schapire. (2007). A Game-Theoretic Approach to Apprenticeship Learning. NIPS08"
[Ziebart, 2008]: https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf "Ziebart. (2008). Maximum Entropy Inverse Reinforcement Learning. AAAI08"

[Finn et al., 2016]: https://arxiv.org/abs/1603.00448 "Finn et al. (2016). Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization. ICML16"

