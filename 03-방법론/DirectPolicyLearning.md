# Direct Policy Learning

*참조: [ICML2018 Tutorial][]*

알고리즘이 **경험하지 못한 상태**를 어떻게 처리할 것인지를 생각해보면, 가장 간단한 답은:

- 전문가에게 묻는 것
- 다른 알고리즘이 어떻게 하는지 보는 것

이다. 즉, **누가 어떻게 하는지 보는 것**이다.



<u>[전제조건]</u> 

- 학습하는 동안 사람이나 기타 전문가에게 실시간으로 물어볼 수 있다.
- 또는, 학습하기 위해서 임의의 상태에서 학습을 시작하거나, 학습과 시뮬레이션을 자유롭게 오갈 수 있다.
  - 예: 언어처리에서 품사 분석 과정은 문장 중간에서 시작할 수 있다.



[<u>일반적인 방법</u>]

1. 초기 정책 $\pi_0$를 수립한다.
2. $m$을 증가시키면서 $\pi_{m+1}$을 $\pi_m$으로부터 다음과 같이 구성한다:
   1. $\pi_m$의 롤아웃 과정을 통해 $\mathcal{T}$를 구성한다.
   2. $\mathcal{T}$으로부터 상태 $s\in S$의 출현 확률 분포 $p_{\pi_m}$을 구성한다.
   3. 전문가나 학습 데이터를 참조해서 각 상태에서의 정답 행동 $\mathcal{T}_E$를 구성한다.
   4. $\mathcal{T}_E$, $p_{\pi_m}$을 활용해서 $\pi_{m+1}$을 갱신한다.



어떻게 업데이트 해야 할까?

## Naive Approach: $m$만으로 $m+1$ 만들기

$\mathcal{T}_E$과 $p_{\pi_m}$만으로 $\pi_{m+1}$을  만들 수 있지 않을까? 즉, $\lim_{m\to\infty} \pi_m \simeq \pi_E$ 아닐까?

- 일단 알고리즘이 경험하지 못한 상태는 $\mathcal{T}_E$로 보완할 수 있게 되었다.
- 그러면, $\mathcal{T}_E$가 일관되어야, 즉 $\pi_0, \pi_1, \cdots, \pi_m$ 아래에서 상태의 출현 분포가 비슷해야 한다.
- 최소한, $p_{\pi_m}(s) > 0$이라면 $(\forall i>m)\ [p_{\pi_i}(s)>0]$ 이어야 한다.



**근데, 그렇지 않다는게 문제다**.

- 매 반복마다 $\pi_m$ 이 변하고 있으므로, 이전에 방문한 상태 $s$를 다시 방문하지 않을수도 있다.
- 방문하지 않은 상태에 대한 정책은 이전 정책을 유지하지 않을 가능성이 높다.
- 극단적으로, $\pi_m$은 수렴하지 않고 진동할 가능성이 있다: 방문했다가, 방문하지 않았다가, ...



그래서, 이전에 방문했었던 상태의 정책을 기억해야 한다.

## Stochastic Mixing: 정책 누적하기

정책은 확률분포이므로, 확률분포를 섞듯이 섞어볼 수 있다.



### [SEARN][]

1. $\pi_E$로 $\pi_0$를 초기화한다.
2. $\pi_m$ 이 $\pi_E$ 에 비해서 잘 하지 못하는 동안^1^ , 다음을 반복한다:
   1. 훈련 집합 $D = \emptyset$ 으로 초기화한다.
   2. $\mathcal{T}_E$ 에 포함된 전문가의 행동열 $\tau_i=(s_0, a_0, s_1, a_1, \cdots)$에 대해:
      1. $\pi_m$을 $s_0, s_1, \cdots, s_T$ 상태에서 롤아웃하여 $\hat{\tau}_i$를 얻는다^2^.
      2. 각 시간 $t$마다:
         1. $\phi_t := \phi (\mathbf{s}, a_0, a_1, \cdots, a_t)$를 계산한다.
         2. $\pi_m$의 행동을 선택하지 않고 다른 행동 $a$ 를 선택하는 경우를 생각하기 위해,
            행동을 $a$로 교체한 $\pi_{m,a}$를 생각하고 $a$를 선택할 때의 보상 $r_a$ 를 $\min_{a'} \mathbb{E}_{\tau \sim p_{\pi_{m,a'}}} l(\tau, \tau_i) - \mathbb{E}_{\tau\sim p_{\pi_{m,a}}} l(\tau, \tau_i) $ 로 계산한다.
      3. $D$ 에 $(\phi, \mathbf{r})$를 추가한다.
   3. $D$를 학습데이터로 하여 적당한 지도학습 알고리즘으로 $\hat{\pi}_{m+1}$을 훈련한다.
   4. $\pi_{m+1}=\beta\hat{\pi}_{m+1}+(1-\beta)\pi_m$ 으로 업데이트한다. **(정책을 섞는다)**
3. $\tilde{\pi} = \frac{\pi_M - (1-\beta)^M\pi_E}{1-(1-\beta)^M}$으로 지정해서 $\pi_E$의 효과를 제거한다.



주^1^: 상수 $C$와 $\beta$ 에 대해서, $m\le C/\beta$ 인 동안이면 충분하다. (논문에 $C$가 무엇인지는 안 적혀있다.)

주^2^: SEARN은 고정된 입력에 labeling을 하는 task를 수행하므로, $s_0, s_1, \cdots, s_T$는 $\pi$에 무관하게 고정이다.



### [SMILe][]

1. $\pi^*$로 $\pi_0$를 초기화한다.
2. 적당한 횟수 $M$동안^2^ 다음을 반복한다:
   1. $\pi_m$을 롤아웃하여 크기 $B$인 $\mathcal{T}$을 얻는다.
   2. $\mathcal{T}$에 나타나는 상태분포를 기준으로 전문가에게 물어서 $\mathcal{T}_E$를 만든다.
   3. $\mathcal{T}_E$ 를 학습데이터로 하여 적당한 지도학습 알고리즘으로 $\hat{\pi}_{m+1}$을 훈련한다.
   4. $\pi_{m+1} = \beta \hat{\pi}_{m+1} + (1-\beta) \pi_m$로 업데이트한다. **(정책을 섞는다)**
3. $\tilde{\pi} = \frac{\pi_M - (1-\beta)^M\pi_E}{1-(1-\beta)^M}$으로 지정해서 $\pi_E$의 효과를 제거한다.



주^2^: $\alpha = \frac{\sqrt{3}}{T^2\sqrt{\log T}}$, $M=\frac{2\log T}{\alpha}$ 일 때, Regret $T[\mathbb{E}_{s\sim P(s|\tilde{\pi})}l(\pi^*, \tilde{\pi})-\min_{\pi'}\mathbb{E}_{s\sim P(s|\pi')}l(\pi^*, \pi')]$가 $T^2$에 비례하게 된다. (Regret의 상한)



### SEARN, SMILe은 어떤 특징이 있나?

- Imitation learning을 Stochastic한 지도학습의 순서열로 환원(reduction)시켰다.
- 수렴성이 **보장**된다. 그러나 오차가 0이 되는 것은 아니다. ($T^2$에 비례한다)
- **부정확한 정책**을 지속적으로 시도해야 한다. 시도할 때마다 지속적으로 Expert의 정책을 알아야 한다.



**문제는...**

* $T$가 크면 잘 돌지 않게 된다 (오류도 $T^2$에 비례하지만 수행 시간도 만만치 않다)
* 계속해서 혼합되는 $\pi_m$이 다른 것들에 비해 나빠도 제거할 방법이 없다.



## Data Aggregation: 상태 누적하기

그래서, 결과가 아닌 결과의 재료를 모아보기로 한다.



### [DAgger][]

1. $D$ 를 $\emptyset$으로 초기화한다.
2. $\hat{\pi}_0$를 임의로 선택한다. (전문가 정책일 필요 없다)
3. 적당한 횟수 $M$ 동안^3^ 반복한다:
   1. $\pi_m := \beta_m\pi_E +(1-\beta_m)\hat{\pi}_m$으로 롤아웃하여 크기 $B$인 $\mathcal{T}$을 만든다.
   2. $\mathcal{T}$에 나타나는 상태분포를 기준으로 전문가에게 물어서 $\mathcal{T}_E^{(m)} := \{(s, \pi^*(s)) | s\in\mathcal{T}(\pi_m)\}$을 구성한다.
   3. $D := D \cup \mathcal{T}_E^{(m)}$으로 갱신한다. **(상태 분포를 누적한다)**
   4. $D$를 훈련집합으로 하여 적당한 지도학습 알고리즘을 사용해 $\hat{\pi}_{m+1}$을 훈련한다.
4. 별도로 구성된 검증 집합을 기준으로, 가장 좋은 성능을 보이는 $\hat{\pi}_m$을 선택한다.



주^3^: 일반화 오류를 줄여서 최적해를 찾을 확률이 $1-\delta$가 되려면, $BM=O(-T^2\log\delta)$ 로 설정하면 된다.
(Strong convex loss인 경우 $O(T\log(T/\delta))$).



### DAgger는 어떤 특징이 있나?

- (SEARN, SMILe과 같이) 역시 수렴성이 보장되고, 부정확한 정책을 시도하며, 지속적으로 전문가에게 물어야 한다.
- Imitation Learning을 No-regret Online Learning으로 환원하여 오차를 줄였다.
  (즉, 오차 $\mathbb{E}_{s\sim p_\tilde{\pi}}l(\pi^*, \tilde{\pi})-\min_{\pi'}\mathbb{E}_{s\sim p_{\pi'}}l(\pi^*, \pi')$ 가 0에 가까워지는 Online learning이다)



**그러나 여전히 문제는...**

- 부정확한 정책을 실행할 수 있을 때만 사용가능하다.
- 지속적으로 전문가에게 물을 수 있고, 물어볼 방법이 있어야 한다.
- 전문가가 모든것을 정확히 안다고 가정한다. 즉, suboptimal 전문가를 고려하지 않는다.
- 전체 작업의 보상이 각 행동의 보상으로 균등분할되지 않는다.





그래서... 시스템이 **혼자서 할 수 있으면** 좋겠다.

- 시뮬레이션을 통해 상태 변화를 생각해보게 하자.
- 전문가에게는 가끔만 물어보자.
- 전문가보다 나은 정책을 찾아보자.
- 보상을 어떻게 나눌지 모르겠으니 알아서 보상을 얻게 하자.

→ [Inverse Reinforcement Learning (With Model)](./InverseReinforcementLearningWithModel.md)

 

-----

**References:**

[ICML2018 Tutorial]: https://sites.google.com/view/icml2018-imitation-learning/	" Imitation Learning Tutorial (Retrieved at 18.11.30) "
[SEARN]: https://link.springer.com/article/10.1007/s10994-009-5106-x "Daume et al. (2009). Search-based structured prediction. Machine Learning 75"
[SMILe]: http://proceedings.mlr.press/v9/ross10a/ross10a.pdf "Ross and Bagnell (2010). Efficient Reductions for Imitation Learning. AISTATS 2010"
[DAgger]: http://proceedings.mlr.press/v15/ross11a/ross11a.pdf "Ross et al. (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. AISTATS 2011"