---
title: "Machine Learning with R - W5"
author: "Seongbong Kim"
date: 2016-01-19 00:10:00 +0900
categories: jekyll update machine-learning
---

이번 포스트에서는 Week 5에 소개된 Neural Networks와 관련된 Exercise를 R로 구현해 보겠습니다.


## Programming Exercise 4:Neural Networks Learning

**Neural Networks** 을 연습하는 과제입니다.
 이번 과제는 Exercise 3과 동일하게 0~9 사이의 handwritten digits를 neural networks를 이용해 분류하는 과제입니다.

### Dataset
이번 과제에 사용되는 데이터는 20x20 pixel handwritten digits (X)와 실제 label(y) 5000개 관측치가 주어집니다.
(Exercise4 데이터 중 ex4data1.mat 파일이 필요합니다.)
<a href="http://s3.amazonaws.com/spark-public/ml/exercises/on-demand/machine-learning-ex4.zip">다운로드</a>



과제에서 사용할 parameter들을 setting 합니다. 이번 과제에서는 20x20 pixel의 이미지를 1행으로 취급하여 길이가 400인 matrix로 만들 예정이기에 `input_layer_size`는 400. hidden layer는 1개를 두고 hidden unit은 25개로 설정할 예정이어서 `hidden_layer_size`는 25. 예측하려는 숫자의 개수인 `num_labels`는 10.(0~9)

```r
## Setup the parameters you will use for this exercise
# 20x20 Input Images of Digits
input_layer_size  <- 400

# 25 hidden units
hidden_layer_size <- 25

# 10 labels, from 1 to 10   
num_labels <- 10

# (note that we have mapped "0" to label 10)                          
```


### Loading and visualizing the data
지난 과제와 같이 랜덤하게 샘플 이미지를 선택해서 보여주는 섹션입니다. 지난 과제에 작성한 `displayData` 함수를 가져오겠습니다.

```r
# Function that display sample data
displayData <- function(X, example_width=NULL){

  # setting example width & height
  if(is.null(example_width)){
    example_width <- round(sqrt(ncol(X)))
  }

  m <- nrow(X)
  n <- ncol(X)
  example_height <- n / example_width

  display_rows <- floor(sqrt(m))
  display_cols <-ceiling(m / display_rows)

  pad <- 1

  # matrix array for images
  display_array <- matrix(1, nrow=(pad + display_rows*(example_height + pad)),
                          ncol=(pad + display_cols*(example_width + pad)))


  # assign examples into display_array
  curr_ex <- 1

  for(i in 1:display_rows){

    for(j in 1: display_cols){

      if(curr_ex > m){
        break
      }
      else{
        max_val <- max(abs(X[curr_ex,]))
        re <- matrix(X[curr_ex,], nrow=example_height, ncol=example_width) / max_val
        display_array[pad + (j-1) * (example_height + pad) + (1:example_height),
                      pad + (i-1) * (example_width + pad) + (1:example_width)] <- re  
        curr_ex <- curr_ex + 1
      }

    }
  }

  # display matrix array
  image(display_array, col=gray(seq(1,0,length=100)))

}
```

지난 과제와 마찬가지로 `.mat` 파일을 불러옵니다. (R.matlab 패키지 로딩이 필요합니다.) `X`와 `y`를 할당하고 `displayData`를 수행합니다.


```r
library(R.matlab)
## === Part 1: Loading and Visualizing Data ===
exdata <- readMat('ex4data1.mat')
X <- exdata$X
y <- exdata$y
m <- nrow(X)

sel <- runif(100, 1, m)

displayData(X[sel,])
```

![Exercise Data](/assets/coursera/machine-learning/ex4/unnamed-chunk-4-1.png)

### Model representation

과제에서 주어진 `Theta1`, `Theta2` 값을 불러와서 이후에 작성할 cost function과 gradient function을 검증해 봅니다. `Theta1`은 input layer를 hidden layer로 변환하는 matrix인데 변환 시 intercept term을 추가하기 때문에 25 by (400+1) 행렬이 되고, `Theta2`는 hidden layer를 output으로 변환하는 matrix여서 총 output 수에 맞춰 10 by (25+1) 행렬이 됩니다. 추후에 최적화 함수를 돌리기 편하게 `Theta1`과 `Theta2`를 합쳐서 `nn_params`라는 변수에 저장합니다.


```r
## === Part 2: Loading Parameters ===
exweights <- readMat('ex4weights.mat')
Theta1 <- exweights$Theta1        # 25 x 401
Theta2 <- exweights$Theta2        # 10 x 26

nn_params <- rbind(matrix(Theta1, ncol=1), matrix(Theta2, ncol=1))
```


### Feedforward and cost function

`nnCostFunction` 함수를 작성하는 부분입니다. cost를 계산하려면 주어진 input layer 값에 theta1, theta2를 가지고 feedforward propagation을 수행하여 output 값과 실제 y값의 차이를 계산해야 합니다. y는 10x10 행렬로 변환시키고 각 열이 1~10을 나타내도록 하는데, 각 열번호와 동일한 행에 해당하는 원소만 1이고 나머지는 0으로 표현합니다. Neural Networks의 hypothesis 함수는 `sigmoid` 함수를 사용하기 때문에 `sigmoid` 함수 정의도 필요합니다.


```r
# Sigmoid Function
sigmoid <- function(z){

  g <- 1 / (1 + exp(-z))
  return(g)

}

# Neural Networks Cost Function
nnCostFunction <- function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda){

  ## Extract theta1 & theta2 from nn_params
  Theta1 <- matrix(nn_params[1:(hidden_layer_size*(input_layer_size + 1))],
                   nrow=hidden_layer_size, ncol=(input_layer_size + 1))
  Theta2 <- matrix(nn_params[(1 + hidden_layer_size*(input_layer_size + 1)):length(nn_params)],
                   nrow=num_labels, ncol=(hidden_layer_size + 1))

  m <- nrow(X)

  J <- 0
  Theta1_grad <- matrix(0, nrow(Theta1), ncol(Theta1))
  Theta2_grad <- matrix(0, nrow(Theta1), ncol(Theta1))


  ## Feedforward propagation
  a_1 <- cbind(1, X)  # 5000 x 401
  z_2 <- a_1 %*% t(Theta1)    # 5000 x 25
  a_2 <- cbind(1, sigmoid(z_2))   # 5000 x 26
  z_3 <- a_2 %*% t(Theta2)    # 5000 x 10
  h_theta <- sigmoid(z_3)     # 5000 x 10

  Y <- matrix(0, m, num_labels)     # 5000 x 10

  for(i in 1:num_labels){
    Y[, i] <- y == i
  }

  ## Calculate cost without regularization
  cost_i <- (((-1) * Y * log(h_theta)) - ((1-Y) * log(1-h_theta))) / m
  J <- sum(cost_i)  


  ## Add regularization term
  reg <- (sum(sum(Theta1[,2:ncol(Theta1)]^2)) + sum(sum(Theta2[,2:ncol(Theta2)]^2))) * lambda / (2 * m)
  J <- J + reg

  return(J)

}
```

과제에서는 regularization term이 없는 cost함수와 있는 함수에 대해 각각 cost를 산출하도록 되어 있는데 위에 `nnCostFunction`을 정의할 때 regularized cost function을 구현하였으니, lambda에 0을 대입해서 regularization term이 없는 함수를 검증하고 1을 대입해서 regularized cost function을 검증해 봅니다.


```r
## === Part 3: Compute Cost (Feedforward) ===
lambda <- 0

J <- nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
print(paste("The value of J must be around 0.287 : ", J))
```


     [1] "The value of J must be around 0.287 :  0.287629165161319"


```r
## === Part 4: Implement Regularization ===
lambda <- 1

J <- nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
print(paste("The value of J must be around 0.384 : ", J))
```


     [1] "The value of J must be around 0.384 :  0.383769859090924"



### Backpropagation and gradient function

점점더 comment를 달기 어려워 지는 개념입니다. Feedforward로 구해진 output과 y사이의 차이를 delta로 놓고, output에서의 delta 값이 나오기 위해서는 이전 layer에서의 편차 값이 얼마인지 계산해 나가는 방식이 backpropagation입니다. 결과적으로 layer와 layer 사이의 편차 값(Delta)을 산출해서 이 편차도 최소화를 하는 것이 이번 과제의 목적이 되겠네요. 결국 이 편차를 산출하는 함수들을 작성해야 합니다. 과제에서는 `nnCostFunction`에 모두 구현을 하는데 최적화 함수인 `optim`에 gradient 함수를 별도로 적어야 해서 `nnGradFunction`에 backpropagation을 구현하도록 하겠습니다. (강의를 이해하고 넘어오셔야 함수가 이해될 듯 합니다.) 이 과정에서 `sigmoidGradient`라는 sigmoid를 미분한 결과를 산출하는 함수가 필요해서 별도로 정의하도록 하겠고, 기존 Regression들과 다르게 `initial_theta`가 0으로 구성된 벡터일 경우 backpropagation이 변화가 생기지 않기 때문에, (-epsilon, epsilon) 사이의 값으로 `initial_theta`를 구성합니다.

```r
# Sigmoid Gradient Function
sigmoidGradient <- function(z){

  if(class(z) == "numeric"){
    z <- matrix(z, nrow = 1)
  }

  g <- matrix(0, dim(z)[1], dim(z)[2])
  g <- sigmoid(z) * (1-sigmoid(z))

}

# Randomly initialize weights
randInitializeWeights <- function(L_in, L_out){
  W <- matrix(0, L_out, L_in)

  epsilon_init <- 0.12
  W <- matrix(runif(L_out * (L_in + 1), 0, 1) * 2 * epsilon_init - epsilon_init, L_out, L_in + 1)

}


# Neural Networks Gradient Function
nnGradFunction <- function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda){

  Theta1 <- matrix(nn_params[1:(hidden_layer_size*(input_layer_size + 1))],
                   nrow=hidden_layer_size, ncol=(input_layer_size + 1))
  Theta2 <- matrix(nn_params[(1 + hidden_layer_size*(input_layer_size + 1)):length(nn_params)],
                   nrow=num_labels, ncol=(hidden_layer_size + 1))

  m <- nrow(X)

  J <- 0
  Theta1_grad <- matrix(0, nrow(Theta1), ncol(Theta1))
  Theta2_grad <- matrix(0, nrow(Theta1), ncol(Theta1))


  ## Feedforward propagation
  a_1 <- cbind(1, X)  # 5000 x 401
  z_2 <- a_1 %*% t(Theta1)    # 5000 x 25
  a_2 <- cbind(1, sigmoid(z_2))   # 5000 x 26
  z_3 <- a_2 %*% t(Theta2)    # 5000 x 10
  h_theta <- sigmoid(z_3)     # 5000 x 10

  Y <- matrix(0, m, num_labels)     # 5000 x 10

  for(i in 1:num_labels){
    Y[, i] <- y == i
  }

  ## Backpropagation
  Delta_1 <- 0
  Delta_2 <- 0

  for(i in 1:m){

    delta_3 <- as.matrix(h_theta[i,]) - as.matrix(Y[i,])    # 10 x 1
    delta_2 <- (t(Theta2[,2:ncol(Theta2)]) %*% delta_3) * t(sigmoidGradient(z_2[i,]))   # 25 x 1

    Delta_2 <- Delta_2 + delta_3 %*% a_2[i,]    # 10 x 26
    Delta_1 <- Delta_1 + delta_2 %*% a_1[i,]    # 25 x 401

  }

  Theta1_grad <- 1/m * Delta_1
  Theta2_grad <- 1/m * Delta_2


  ## Compute Gradient
  T1g_c <- ncol(Theta1_grad)
  T2g_c <- ncol(Theta2_grad)
  Theta1_grad[,2:T1g_c] <- Theta1_grad[,2:T1g_c] + ((lambda / m) * Theta1[,2:ncol(Theta1)])
  Theta2_grad[,2:T2g_c] <- Theta2_grad[,2:T2g_c] + ((lambda / m) * Theta2[,2:ncol(Theta2)])


  grad <- rbind(matrix(Theta1_grad, ncol=1), matrix(Theta2_grad, ncol=1))

  return(grad)

}
```


### Training Neural Networks

함수들의 code가 상당히 길었는데, 작성한 함수들을 가지고 training을 해서 최적화된 `Theta1`, `Theta2`가 합쳐진 `nn_params`와 `cost` 값을 산출하겠습니다.

```r
## === Part 6: Initializing Pameters ===

initial_Theta1 <- randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 <- randInitializeWeights(hidden_layer_size, num_labels)

initial_nn_params <- rbind(matrix(initial_Theta1, ncol=1), matrix(initial_Theta2, ncol=1))


## === Part 8: Training NN ===
it <- 100
lambda <- 1

system.time(res <- optim(initial_nn_params, fn=nnCostFunction, gr=nnGradFunction, method="CG", control=list(maxit=it, type=1),
             input_layer_size = input_layer_size, hidden_layer_size=hidden_layer_size,
             num_labels=num_labels, X=X, y=y, lambda=lambda))
```


        user  system elapsed
       48.27    1.17   49.50


```r
nn_params <- res$par
cost <- res$value

print(paste("The cost result is : ", cost))
```


     [1] "The cost result is :  0.615709829630765"


### Prediction

`Theta1`에 대해 시각화를 해보면 '이미지에 이런 가중치를 가해서 변환을 시키면 구분이 되는가 보구나..' 하면서 그럴싸한 그림이 나오는데 큰 의미는 없어 보입니다.

```r
## === Part 9: Visualize Weights ===
displayData(Theta1[,2:ncol(Theta1)])
```

![Weight Visualization](/assets/coursera/machine-learning/ex4/unnamed-chunk-10-1.png)

이 모델에 대해 정확도를 구해보면 아래와 같이 높은 정확도가 나옵니다.

```r
# Predict Function
predict <- function(Theta1, Theta2, X){

  m <- nrow(X)
  num_labels <- nrow(Theta2)

  h1 <- sigmoid(cbind(1, X) %*% t(Theta1))
  h2 <- sigmoid(cbind(1, h1) %*% t(Theta2))
  p <- matrix(apply(h2, 1, function(x) which(x==max(x))), m, 1)

  return(p)
}

## === Part 10: Implement Predict ===

pred <- predict(Theta1, Theta2, X)
accuracy <- mean(pred==y)
print(paste("The accuracy of this model is : ", accuracy))
```


     [1] "The accuracy of this model is :  0.9752"
