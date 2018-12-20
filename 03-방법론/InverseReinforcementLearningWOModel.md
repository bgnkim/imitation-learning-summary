# Inverse Reinforcement Learning (WITHOUT Model)

*참조: [ICML2018 Tutorial][]*

지금까지 시스템이 스스로 하게 하기 위해서 강화학습을 활용했지만 다음과 같은 문제가 남아있다.

- 환경 모형 $\theta$를 안다고 전제하고 있다. (모르는 경우가 더 많다)
- $R_w=w^\top \phi$로 선형성을 전제하고 있다. (선형이 아닌 경우가 일반적이다)



이제 이 전제를 없애보도록 하자.



<u>[전제조건]</u>

- $S$, $A$, $\pi^*$는 주어진다.
- 환경 모형 $\theta$는 **이제 알지 못한다**.
- 찾고자 하는 보상함수 $R$은 **더 이상 선형이거나 아핀(Affine) 함수가 아니다**. (Deep Learning을 쓰면 된다)



## 환경 모형 없애기

MaxEntIRL을 생각해보자. $\tau$의 확률분포와 보상함수의 관계를 설명하면서 다음과 같은 수식을 사용했다.
$$
P(\tau|w) \propto e^{R_w(\tau)}
$$
확률의 총합은 1이다. 따라서, 
$$
P(\tau|w) = \frac{e^{R_w(\tau)}}{\int e^{R_w(\tau')} d\tau'}
$$
분모의 함수를 $Z(w) := \int e^{R_w(\tau')} d\tau'$ 라고 두자. 여기서 문제는 $Z$를 구하기 위해 모든 $\tau$를 더해야 한다는 데 있다.

게다가 Gradient descent를 사용하려면 $Z$를 구하지 않을 수 없으므로, 방법이 필요하다.
$$
L(w) = \frac{1}{N}\sum_{\tau_i\in \mathcal{T}_E} R_w(\tau_i) + \log Z(w)\\
\nabla L(w) = \frac{1}{N}\sum_{\tau_i\in \mathcal{T}_E}\nabla R_w(\tau_i) + \frac{1}{Z(w)}\nabla Z(w)
$$


### Sampling?

실제 $\tau$의 분포 $\theta$를 안다면, $\tau$를 $\theta$에서 표본추출함으로써 근사치를 계산할 수 있다. 즉 표본 $S$를 $\theta$에 따라 추출했다면
$$
Z(w) \simeq \sum_{\tau \in S} e^{R_w(\tau)},\quad \nabla Z(w) \simeq \sum_{\tau\in S} R_w(\tau)e^{R_w(\tau)}
$$
이와 같이 계산할 수 있다. 하지만 $\theta$를 구할 수 없다면 $S$를 뽑을 수 없다.



### Importance Sampling

표본을 추출하기 위한 분포를 모른다면, 적당한 분포를 제안하고 그 분포에 따라 뽑은 다음 효과를 없애면 안될까?

적당한 임의의 확률분포 $q$를 생각하자. 그러면, 다음과 같이 근사할 수 있다.
$$
\int e^{R_w(\tau)}d\tau = \int \frac{e^{R_w(\tau)}}{q(\tau)}q(\tau)d\tau = \mathbb{E}\left[ \frac{e^{R_w(\tau)}}{q(\tau)} \right] \simeq \frac{1}{|S|}\sum_{\tau_i\in S} \frac{e^{R_w(\tau_i)}}{q(\tau_i)}
$$
분포 $q$를 알고 $R_w$는 매 iteration마다 추정값이 있으므로 계산이 가능해졌다. 이제, 다음과 같이 계산할 수 있다. 
$$
L(w) \simeq \frac{1}{N}\sum_{\tau_i\in\mathcal{T}_E} R_w(\tau_i) + \log\frac{1}{|S|}\sum_{\tau_i\in S} \frac{e^{R_w(\tau_i)}}{q(\tau_i)}\\
\nabla L(w) \simeq \frac{1}{N}\sum_{\tau_i\in \mathcal{T}_E}\nabla R_w(\tau_i) + \frac{1}{Z(w)}\sum_{\tau_i\in S} \frac{e^{R_w(\tau_i)}}{q(\tau_i)}\nabla R_w(\tau_i)
$$


### [Finn et al., 2016][] (GCL; Guided Cost Learning)

한발 더 나아가서, 분포 $q$를 상황에 맞게 갱신해나가고, 분포를 $k$개 사용해서 추출한다면, 다음과 같은 형태의 알고리즘을 얻는다.

1. 모든 $k$에 대해 $q_k(\tau)$를 임의 정책이나 $\pi_E$를 사용하여 $\mathcal{T}$를 구성, 초기화한다.
2. 누적해서 추출된 표본집합 $\mathcal{D}_{samp} = \emptyset$ 라 하자.
3. $m=1$부터 $M$까지,
   1. 각 $q_k(\tau)$에서 표본 $\mathcal{D}_{traj}$를 추출한다.
   2. 표본 업데이트: $\mathcal{D}_{samp} = \mathcal{D}_{samp}\cup \mathcal{D}_{traj}$
   3. $\mathcal{D}_{samp}$와 Gradient descent를 활용해서 $R_w$를 갱신한다: $k=1$부터 $K$까지
      1. $\hat{\mathcal{D}}_{demo}$를 $\mathcal{T}_E$에서 추출한다.
      2. $\hat{\mathcal{D}}_{samp}$를 $\mathcal{D}_{samp}$에서 추출하고, $\hat{\mathcal{D}}_{demo}$를 추가한다.
      3. 식 (6)을 추출된 표본들을 사용해서 계산한다.
      4. $w$를 $\nabla L(w)$를 사용해 갱신한다.
   4. $q_k(\tau)$를 $D_{traj}$와 ([Levine & Abbeel, 2014][])를 사용하여 $q_{k+1}(\tau)$로 갱신한다.
4. $w$와 $q(\tau)$를 반환한다.



다시 보면, 생성된 $\mathcal{D}_{samp}$와 실제 분포인 $\mathcal{T}_E$가 혼합되어 $R_w$를 갱신하는데, $R_w$는 $\mathcal{T}_E$를 더 선호하도록 되어있다.

즉 3.1-3.2, 3.4는 Generator, 3.3은 Discriminator를 훈련하고 있다.







------

**References:**

[ICML2018 Tutorial]: https://sites.google.com/view/icml2018-imitation-learning/	" Imitation Learning Tutorial (Retrieved at 18.11.30) "
[CMU10703]: http://www.andrew.cmu.edu/course/10-703/	"Deep Reinforcement Learning and Control (Retrieved at 18.11.30)"
[Finn et al., 2016]: https://arxiv.org/abs/1603.00448	"Finn et al. (2016). Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization. ICML16"
[Ho & Ermon, 2016]: https://arxiv.org/pdf/1606.03476.pdf	"Ho & Ermon. (2016). Generative adversarial imitation learning. NIPS16"
[Syed et al., 2008]: http://rob.schapire.net/papers/SyedBowlingSchapireICML2008.pdf	"Syed et al. (2008). Apprenticeship learning using linear programming. ICML08"

[Schulman et al., 2015]: https://arxiv.org/pdf/1502.05477.pdf "Schulman et al. (2015). Trust Region Policy Optimization. ICML15"
[Tamar et al., 2016]: https://arxiv.org/pdf/1602.02867.pdf "Tamar et al. (2016). Value Iteration Networks. NIPS16"

