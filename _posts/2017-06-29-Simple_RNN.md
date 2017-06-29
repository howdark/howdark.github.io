---
title: "Simple RNN Algorithm"
author: "Seongbong Kim"
date: 2017-06-29 23:35:00 +0900
categories: jekyll update deep-learning
permalink: /blog/:title
comments: true
use_math: true
---

## RNN Algorithm - 1 (Simple RNN)

#### 1. Parameter 초기화

-   Simple RNN은 3개의 Parameter를 보유
    -   Input weight ($U$)
    -   이전 state를 반영하기 위한 weight ($W$)
    -   Output weight ($V$)
-   초기화 방법
    -   Random 초기화
    -   0 초기화
    -   Xavier 초기화
    -   등등

![SimpleRNN1](/assets/DL_algorithm/RNN/SimpleRNN1.png)

<br>
#### 2. Forward Propagation

-   Input으로 RNN 내부 연산을 거친 후 output을 산출하는 과정
-   Time step 1에서부터 t까지 모든 연산 수행 (최초 State는 $s_0$)

![SimpleRNN2](/assets/DL_algorithm/RNN/SimpleRNN2.png) ![SimpleRNN3](/assets/DL_algorithm/RNN/SimpleRNN3.png)

-   1단계 : 현재 State 산출
<br>
    -   $s_t = f(Ux_t + Ws_{(t-1)})$
<br>
        -   $s_t$ : new state
        -   $s_{(t-1)}$ : old state
        -   $x_t$ : t 시점의 input vector
        -   $f$는 $tanh$, $ReLU$ 등 nonlinear activation함수
<br>
-   2단계 : Output 산출
<br>
    -   $o_t = softmax(Vs_t)$
<br>
        -   $o_t$ : t 시점의 output vector

<br>
#### 3. Loss Calculation

-   예측된 output과 실제 값과의 loss를 계산

![SimpleRNN4](/assets/DL_algorithm/RNN/SimpleRNN4.png)

-   Loss functions
    -   Cross Entropy
    -   Mean Squared Error
    -   Sum of Squres
    -   Softmax Loss
    -   Negative log-likelihood
    -   등등

<br>
#### 4. Parameter Optimization

-   Stochastic Gradient Descent (SGD)
    -   Parameter들을 loss가 감소하는 방향으로 밀어가는 알고리즘
    -   Loss가 감소하는 방향 : $\frac{\partial L}{\partial U}$,$\frac{\partial L}{\partial V}$, $\frac{\partial L}{\partial W}$ (Gradient of Loss)
    -   미는 양 : $\alpha$  (Learning Rate)
<br>
-   $\alpha$는 초기 설정된 상수. 그럼 Gradient는 어떻게 계산?
    → **BPTT** (Back Propagation Through Time)
    -   최종 Loss로부터 역으로 Gradient 산출
    -   Chain rule을 적용하여 t시점부터 최초 Node까지 산출
<br>
-   **BPTT** 를 통해 산출된 Gradient로 SGD 1 step 수행
    -   SGD 과정에서 Parameter가 갱신되며, 이 때가 학습되는 시점
    -   (Forward Propagation → Loss → BPTT → SGD) : 1 Step ▶ 반복


![SimpleRNN5](/assets/DL_algorithm/RNN/SimpleRNN5.png)

<br>
#### 5. Problem of BPTT

-   Long Back Propagation
    -   $\frac{\partial L}{\partial W}=(\frac{\partial L}{\partial \hat y})(\frac{\partial \hat y}{\partial s_t})(\frac{\partial s_t}{\partial W})$ : t 시점에서 Loss에 대한 W의 Gradient
    -   여기서 $s_t$는 $s_{t-1}$에 영향을 받기 때문에$(s_t = tanh(Ux_t+Ws_{t-1}))$, $s_{t-1}$의 Gradient를 산출하여 합산해줘야 함
    -   U에 대해서도 L과 유사한 상황임

-   이전 State의 Gradient 산출
    -   $s_{t-1} = (\frac{\partial L}{\partial \hat y})(\frac{\partial \hat y}{\partial s_t})(\frac{\partial s_t}{\partial s_{t-1}})(\frac{\partial s_{t-1}}{\partial W})$
    -   $s_{t-2} = (\frac{\partial L}{\partial \hat y})(\frac{\partial \hat y}{\partial s_t})(\frac{\partial s_t}{\partial s_{t-1}})(\frac{\partial s_{t-1}}{\partial s_{t-2}})(\frac{\partial s_{t-2}}{\partial W})$
    -   $s_{t-3} = (\frac{\partial L}{\partial \hat y})(\frac{\partial \hat y}{\partial s_t})(\frac{\partial s_t}{\partial s_{t-1}})(\frac{\partial s_{t-1}}{\partial s_{t-2}})(\frac{\partial s_{t-2}}{\partial s_{t-3}})(\frac{\partial s_{t-3}}{\partial W})$
    -   $\cdots$
-   먼 과거의 gradient는 chain rule로 계속 곱해진 후에 합산되므로 최종 gradient에 주는 영향도가 미미해짐 (Vanishing Problem*)
<br>
※   Vanishing Problem : tanh나 sigmoid 함수의 경우 미분값이 0에서 최대 1(sigmoid는 ¼)로 [증명](http://www.jmlr.org/proceedings/papers/v28/pascanu13.pdf)이 되었으며, 이를 계속 곱해나가면 gradient가 0으로 수렴

<br>
#### 다음에서 계속 (LSTM 소개)
