---
title: "Machine Learning with R - W2"
author: "Seongbong Kim"
date: 2016-01-15 00:10:00 +0900
categories: jekyll update machine-learning
permalink: /blog/:title
comments: true
---

이번 포스트에서는 Week 2에 소개된 Multi-variable linear regression과 관련된 Exercise를 R로 구현해 보겠습니다.


## Programming Exercise 1: Linear Regression


**Linear regression with multiple variables** 을 연습하는 과제입니다.
 Multi-variable Linear Regression 과제는 집 크기와 방의 숫자 데이터를 바탕으로 집 값을 예측하는 과제가 되겠습니다.

### Dataset
이번 과제에 사용되는 데이터는 집 크기(X1), 방의 수(X2), 집 가격(y)의 3개 변수에 대해 47개 관측치가 주어집니다.
(Exercise1 데이터 중 ex1data2.txt 파일이 필요합니다.)
    <a href="https://s3.amazonaws.com/spark-public/ml/exercises/on-demand/machine-learning-ex1.zip">Coursera Machine Learning Exercise1(다운)</a>

### Feature Normalization



우선 데이터를 불러와서 X와 y에 저장을 해주고


```r
## === Part 1 : Feature Normalization ===
# Load Data
data <- read.table("ex1data2.txt", sep=",")
X <- as.matrix(data[, 1:2])
y <- as.matrix(data[, 3])
m <- length(y)
```

집 크기는 방의 숫자보다 1000배 이상 값이 크기 때문에 Normalize 하여 데이터를 사용하려 합니다.
X를 입력 받으면 각각의 feature에 대해 Normalize를 수행하는 `featureNormalize`라는 함수를 작성합니다.


```r
# Function that normalize features
featureNormalize <- function(X) {
  X_norm <- X
  mu <- mat.or.vec(1, ncol(X))
  sigma <- mat.or.vec(1, ncol(X))

  mu <- matrix(apply(X, 2, mean), nrow = 1)
  sigma <- matrix(apply(X, 2, sd), nrow = 1)
  X_norm <- matrix(
                mapply(
                    function(X, mu, sigma){
                        (t(X) - mu) / sigma
                    },
                    X=t(X), mu=mu, sigma=sigma),
              ncol=2, byrow=T)

  return(list(X=X_norm, mu=mu, sigma=sigma))
}
```

 X에 대해 Normalize를 수행합니다. Normalize에 사용된 `mu`(평균)과 `sigma`(표준편차)는 아래와 같습니다.


```r
# Scale features and set them to zero mean
norm_result <- featureNormalize(X)
X <- norm_result$X
mu <- norm_result$mu
sigma <- norm_result$sigma
print(paste("mean : ", mu, " / std : ", sigma))
```


     [1] "mean :  2000.68085106383  / std :  794.70235353389"
     [2] "mean :  3.17021276595745  / std :  0.7609818867801"


θ~0~과 행렬곱을 수행하기 위해서 1로 이루어진 column을 X에 추가하고,
learning rate, interation 횟수, Initial theta를 설정합니다.


```r
# Add intercept term to X
X <- cbind(matrix(1, nrow = m), X)


## ============ Part 2 : Gradient Descent =============

# Choose some alpha value
alpha <- 0.01
num_iters <- 400

# Init Theta and Run Gradient Descent
theta <- mat.or.vec(3, 1)
```

Multi-variable Linear Regression의 Cost를 산출하는 Function입니다. 행렬 연산을 Coding을 하니 Single-variable Linear Regression의 Cost 산출 함수와 동일합니다. 행렬 연산으로 수식을 구현하니 이런 점이 매력적이더군요.


```r
computeCostMulti <- function(X, y, theta){

  m <- length(y)
  J <- 0

  J <- J + (((t(theta) %*% t(X)) - t(y)) %*%
            t((t(theta) %*% t(X)) - t(y))) / (2*m)

  return(J)

}
```

Gradient Descent 함수도 행렬 연산으로 수식을 구현하여서 Single-variable과 동일한 함수입니다.


```r
gradientDescentMulti <- function(X, y, theta, alpha, num_iters){

  m <- length(y)
  J_history <- mat.or.vec(num_iters, 1)

  for(iter in 1:num_iters){

    theta <- theta - t(alpha / m * (((t(theta) %*% t(X)) - t(y)) %*% X))

    J_history[iter] <- computeCostMulti(X, y, theta)

  }

  return(list(theta=theta, J_history=J_history))

}
```


작성한 함수를 가지고 X, y, Initial theta, Learning Rate, Iteration 횟수를 입력하여 최적의 Theta 값을 산출해 봅니다.

```r
grad_result <- gradientDescentMulti(X, y, theta, alpha, num_iters)
theta <- grad_result$theta
J_history <- grad_result$J_history
theta
```


                [,1]
     [1,] 334302.064
     [2,] 100087.116
     [3,]   3673.548


iteration에 따른 Cost 변화 값은 아래와 같습니다.


```r
# Plot the convergence graph
plot(J_history, type="l", lwd = 2,
    xlab='Number of iterations', ylab='Cost J')
```

![Cost over iterations](/assets/coursera/machine-learning/ex1/unnamed-chunk-9-1.png)

1650 sqft에 3-bedroom을 가진 집의 가격은?!


```r
# Estimate the price of a 1650 sq-ft, 3 br house
pred1 <- (matrix(c(1650, 3), ncol=2) - mu) / sigma
pred1 <- cbind(1, pred1)
predict1 <- pred1 %*% theta
print(paste("price by gradient : ", predict1))
```


     [1] "price by gradient :  289314.620337776"


추가적으로 Gradient Descent 방식이 아닌 Normal Equation 방식으로 최적해를 찾아보는 것이 Excercise에 있습니다.
Normal Equation은 주어진 $Ax = b$ 라는 행렬 식에서 좌변과 우변의 sum of square를 최소화 하는 수식입니다.
($A^TAx = A^Tb$)

Normal Equation을 구현하는 함수를 작성합니다. (`normalEqn`)


```r
## ============ Part 3 : Normal Equations =============

# Function that compute the closed-form solution to linear regression
normalEqn <- function(X, y){
  # require(MASS)
  theta = mat.or.vec(ncol(X), 1)

  theta = solve(t(X) %*% X) %*% t(X) %*% y
  # theta = ginv(t(X) %*% X) %*% t(X) %*% y

}
```

X와 y 데이터를 새로이 다시 불러와서 Intercept Column을 추가하고, `normalEqn` 함수에 입력합니다. `normalEqn`으로 산출된 theta 값은 아래와 같습니다.


```r
# Load Data
data <- read.table("ex1data2.txt", sep=",")
X <- as.matrix(data[, 1:2])
y <- as.matrix(data[, 3])
m <- length(y)

# Add intercept term to X
X <- cbind(matrix(1, nrow = m), X)

theta <- normalEqn(X, y)
theta
```


              [,1]
        89597.9095
     V1   139.2107
     V2 -8738.0191


산출된 Theta 값으로 1650 sqft에 3-bedroom을 가진 집의 가격을 구해보면?!


```r
# Estimate the price of a 1650 sq-ft, 3 br house
pred2 <- c(1, 1650, 3)
predict2 <- pred2 %*% theta
print(paste("price by normalEqn : ", predict2))
```


     [1] "price by normalEqn :  293081.464334895"


차이는 4000 정도 있지만 근사한 값이 산출되는 것을 볼 수 있습니다. Coding도 단순하고 개념적으로도 어렵지는 않지만, Normal Equation의 경우 역행렬을 계산해야 하기 때문에 행렬이 커질수록 Computing Cost가 급격히 증가하는 단점이 있습니다.
