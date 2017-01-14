---
title: "Machine Learning with R - W1"
author: "Seongbong Kim"
date: 2016-01-15 00:00:00 +0900
categories: jekyll update machine-learning
---

Week 1과 2에는 Introduction과 기본적인 Linear Algebra에 대한 내용을 다루고 있고 단변량,다변량 Linear Regression의 Learning 알고리즘을 다루고 있습니다. 그 동안 Linear Regression은 `lm`과 같은 구현된 함수를 가지고 결과값만 활용을 했었지 어떻게 최적값을 찾아내는지에 대한 고민은 없었는데 이 강의를 통해서 대략적인 원리를 알게 됐습니다.

동영상 강의를 다 요약할 수는 없고, 포스팅을 하는 목적도 R로 과제를 수행하는 것에 있기에 과제 PDF를 따라가며 포스팅을 하겠습니다.

## Programming Exercise 1: Linear Regression


**Linear Regression with Single Variable** 을 연습하는 과제입니다.
단변량 Linear Regression을 이용해 도시의 인구수 기준으로 도시의 수익을 예측해 보도록 하겠습니다.

### Dataset

Dataset은 도시의 인구수(X), 도시의 수익(y) 2개의 변수에 대해 각 97개의 관측치를 사용합니다. (Exercise1 데이터 중 ex1data1.txt 파일이 필요합니다.)

### Plotting the Data

우선 Data가 어떻게 생겼는지 보여주기 위해서 Plotting을 하는 부분이 되겠습니다.

```r
## === Part 2 : Plotting ===
data <- read.table("ex1data1.txt", sep=",")  # Read comma separated data
X <- as.matrix(data[, 1])
y <- as.matrix(data[, 2])
m <- length(y)   # number of training examples

# Plot Data
plot(X, y,
    ylab = 'Profit in $10,000s',
    xlab = 'Population of City in 10,000s',
    type = "p", col = "red", pch = 'x', cex = 1)
```

![Excercise1 Data](/assets/coursera/machine-learning/ex1/unnamed-chunk-2-1.png)

### Gradient Decent

Gradient Decent는 한국어로는 *경사 하강법*으로 번역되며 *함수의 기울기를 구하여 기울기가 낮은 쪽으로 계속 이동시켜서 극값에 이를 때까지 반복시키는 것이다.* 라고 위키에 설명되어 있습니다. 자세한 설명은 비디오 강의를... 어쨌든 우리의 목적은 주어진 x에 대해 hypothesis 함수를 적용했을 때 y값과의 차이를 가장 작게 만드는 것입니다. 이후 hypothesis 함수 계산 및 Cost Function/Gradient Function 계산은 모두 행렬 형식으로 계산이 되기 때문에 For문이 없이도 빠르게 계산이 되는 장점이 있습니다. 행렬 연산 기능이 R에 없으면 어쩌나 했으나 역시 기우였고 잘 구현이 되는군요. 행렬 연산을 위해서 θ~0~와 계산이 될 수 있게 1로 이루어진 column을 추가합니다.

```r
## === Part 3 : Gradient descent ===

# Add a column of ones to X
X <- cbind(matrix(1, nrow=m), data[, 1])

# Initialize fitting parameters
theta <- mat.or.vec(2, 1)   

# Some gradient descent settings
iterations <- 1500
alpha <- 0.01;
```

hypothesis 함수에 X를 적용한 값과 y와의 차이를 이용한 Cost를 계산하는 함수입니다.

```r
# Function that compute cost function J of theta

computeCost <- function(X, y, theta){

  m <- length(y)
  J <- 0

  J <- J + (((t(theta) %*% t(X)) - t(y)) %*%
            t((t(theta) %*% t(X)) - t(y))) / (2*m)

  return(J)

}
```

Initial Theta값을 가지고 계산한 최초 Cost를 표시합니다.

```r
# Compute and display initial cost

J <- computeCost(X, y, theta)
J
```


              [,1]
     [1,] 32.07273


지정한 iteration 수 만큼 Loop를 돌며 cost가 작아지는 방향으로 이동하는 함수입니다. (Gradient Descent)

```r
# Function that gradient descent to learn theta
gradientDescent <- function(X, y, theta, alpha, num_iters){

  m <- length(y)

  J_history <- mat.or.vec(num_iters, 1)

  for(iter in 1:num_iters){

    theta <- theta - t(alpha / m * (((t(theta) %*% t(X)) - t(y)) %*% X))

    J_history[iter] <- computeCost(X, y, theta)

  }

  return(list(theta=theta, J_history=J_history))

}
```

초기에 Initial Theta 벡터를 `[0 0]`으로 놓고 Gradient Descent 함수를 돌리고 나면 1500 iteration 이후의 Theta 값으로 `[-3.63 1.16]`이 반환됩니다. iteration 간 Cost가 감소하는 것도 Plot을 통해 볼 수 있습니다.

```r
# run gradient descent
grad_result <- gradientDescent(X, y, theta, alpha, iterations)
theta <- grad_result$theta
J_history <- grad_result$J_history

# print theta to screen
theta
```


               [,1]
     [1,] -3.630291
     [2,]  1.166362


```r
# Plot cost history over iteration
plot(1:iterations, J_history)
```

![cost history over iteration](/assets/coursera/machine-learning/ex1/unnamed-chunk-7-1.png)

그럼 이렇게 나온 Theta 벡터가 얼마나 X들을 잘 설명하는지 Plot을 그려보면 꽤 잘 설명하는 Line이 그려집니다.

```r
# Plot the linear fit
plot(X[,2:ncol(X)], y, ylab = 'Profit in $10,000s', xlab = 'Population of City in 10,000s',
     type = "p", col = "red", pch = 'x', cex = 1)
abline(theta, lwd = 2, col = "blue")
```

![Linear Fit](/assets/coursera/machine-learning/ex1/unnamed-chunk-8-1.png)

### Prediction

Population size에 따른 Profit을 예측해보면 아래와 같습니다.

```r
# Predict values for population sizes of 35,000 and 70,000
predict1 <- c(1, 3.5) %*% theta
predict2 <- c(1, 7) %*% theta
print(paste("Size 35,000 : ", predict1," / Size 70,000 : ", predict2))
```


     [1] "Size 35,000 :  0.451976786770177  / Size 70,000 :  4.53424501294471"
