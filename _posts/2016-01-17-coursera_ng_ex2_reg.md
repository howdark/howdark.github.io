---
title: "Machine Learning with R - W3"
author: "Seongbong Kim"
date: 2016-01-17 00:10:00 +0900
categories: jekyll update machine-learning
---

이번 포스트에서는 Week 3에 소개된 Regularized Logistic Regression과 관련된 Exercise를 R로 구현해 보겠습니다.


## Programming Exercise 2: Regularized Logistic Regression

**Regularized Logistic Regression** 을 연습하는 과제입니다.
 Regularized Logistic Regression을 이용해 Microchip 조립공장에서 test 1, 2의 결과를 가지고 quality assurance 결과를 예측하는 과제입니다.

### Dataset
이번 과제에 사용되는 데이터는 Microchip Test1(X1), Microchip Test1(X2), QA 결과(y)의 3개 변수에 대해 118개 관측치가 주어집니다.
(Exercise2 데이터 중 ex2data2.txt 파일이 필요합니다.)
<a href="http://s3.amazonaws.com/spark-public/ml/exercises/on-demand/machine-learning-ex2.zip">다운로드</a>

### Plotting Data


우선 데이터를 불러와서 X와 y에 저장을 해줍니다.


```r
# Data Load
data <- read.table("ex2data2.txt", sep=",")
X <- as.matrix(data[,1:2])
y <- as.matrix(data[,3])
```

다음은 데이터가 어떻게 구성되어 있는지 Plot을 하는 단계인데, 과제에서는 `plotData(X,y)`라는 함수를 별도로 제공을 하고 있어서 `plotData` 함수를 생성합니다. QA 합격한 point는 +와 green으로, QA 불합격한 point는 -와 red로 표시하는 함수 입니다.

```r
# Function that plotting data
plotData <- function(X,y,xlim=NULL,ylim=NULL){

  plot(X[which(y==1), 1], X[which(y==1), 2], pch="+", cex = 2, col = "green",
       xlab='', ylab='', xlim=xlim, ylim=ylim)
  points(X[which(y==0), 1], X[which(y==0), 2], pch="-", cex = 2, col = "red")}
```

그럼 `plotData` 함수를 사용해서 Plotting을 해보겠습니다.


```r
# Plotting data
plotData(X, y, xlim=c(-0.9,1.1), ylim=c(-0.9,1.35))
title(xlab='Microchip Test 1', ylab='Microchip Test 2')
legend(0.8,1.3, legend=c("y = 1","y = 0"), pch = c("+", "-"), col=c("green", "red"))
```

![Exercise Data](/assets/coursera/machine-learning/ex2/unnamed-chunk-4-1.png)

### Add Features
Plotting 결과 1차원의 Decision Boundary로는 합격과 불합격을 구분할 수 없기 때문에 과제에서는 feature를 추가해서 Logistic Regression을 수행하려 합니다. `mapFeature` 함수가 과제에서 Matlab 코드로 제공되는데 이 함수는 X1과 X2에 대해 6차까지 polynomial 조합을 산출하는 함수로서 아래와 같이 구현할 수 있습니다.


```r
# Feature mapping function with degree 6
mapFeature <- function(X1, X2, degree=6){

  m1 <- length(X1)
  m2 <- length(X2)

  if(m1 != m2){
    print('Length of X1 and X2 are different.')
    return(NA)
  }

  result <- sapply(0:degree, function(i){
    sapply(0:i, function(j){
      X1^(i-j) * X2^j
    })
  })

  return(matrix(unlist(result), nrow=m1))

}
```

주어진 X1과 X2에 대해 `mapFeature` 함수를 수행하여 6차 다항변수 조합을 수행하고, X의 feature 수에 대응하는 `initial_theta`를 0 함수로 설정합니다. 이번에는 Regularized Logistic Regression을 수행할 예정이기에 Regularize Coefficient인 `lambda` 값도 설정해 줘야 합니다.

```r
## =============== Part 1 : Regularized Logistic Regression ===============

# Add polynomial Features
X <- mapFeature(X[,1], X[,2])

# Initialize fitting parameters
initial_theta <- mat.or.vec(ncol(X), 1)

# Set regularization parameter lambda to 1
lambda <- 1
```


### Cost Function and Gradient Function
Regularized logistic regression의 cost 함수와 gradient 함수를 작성합니다. (sigmoid 함수 포함)


```r
# Sigmoid Function
sigmoid <- function(z){
  g <- 1 / (1 + exp(-z))
  return(g)
}

# Cost Function
costFunctionReg <- function(theta, X, y, lambda){

  m <- length(y)

  J <- 0
  h_theta <- sigmoid(t(theta) %*% t(X))

  theta_j <- theta
  theta_j[1] <- 0

  J <- (-1 * (t(y) %*% t(log(h_theta))) - (t(1-y) %*% t(log(1-h_theta))))/m +
    (lambda / (2 * m)) * (t(theta_j) %*% theta_j)

  return(J)
}

# Gradient Function
gradFunctionReg <- function(theta, X, y, lambda){

  m <- length(y)

  h_theta <- sigmoid(t(theta) %*% t(X))

  theta_j <- theta
  theta_j[1] <- 0

  grad <- t(((h_theta - t(y)) %*% X) / m) + lambda / m * theta_j

  return(grad)
}
```

zero 행렬인 `initial_theta`의 cost를 산출해 봅니다.


```r
# Compute and display initial cost and gradient for regularized logistic regression
cost_result <- costFunctionReg(initial_theta, X, y, lambda)
```

### Optimization

이제 구현된 cost function과 gradient function을 가지고 최적화를 수행해 보겠습니다.
`initial_theta`와 `lambda`를 설정해주고 `optim` 함수를 활용해 theta 값을 최적화합니다.


```r
## =============== Part 2 : Regularization and Accuracies ===============

# Initialize fitting parameters
initial_theta <- mat.or.vec(ncol(X), 1)

# Set regulaization parameter lambda to 1
lambda <- 1

# optimization
opt_result <- optim(initial_theta, fn = function(t) costFunctionReg(t,X,y,lambda),
                    gr = function(t) gradFunctionReg(t,X,y,lambda), method="Nelder-Mead",
                    control=list(maxit=2000))
theta <- opt_result$par
cost <- opt_result$value
```

`optim`의 결과인 `opt_result`는 **List** 형식으로 par 항목은 최적화된 `theta`에 해당하는 값을, value 항목은 최적화된 `cost` 값을 반환합니다. 이렇게 최적화된 값을 가지고 Decision Boundary를 그려보겠습니다. 과제에서 제공되는 `plotDecisionBoundary` 함수를 R로 구현하여 사용해 보겠습니다.


```r
# Function that plotting decision boundary
plotDecisionBoundary <- function(theta, X, y, xlim=NULL, ylim=NULL){

  plotData(X[,2:3], y, xlim=xlim, ylim=ylim)

  if(ncol(X) <= 3){
    plot_x <- c(min(X[,2])-2, max(X[,2])+2)
    plot_y <- (-1 / theta[3]) * (theta[2]*plot_x + theta[1])  # theta0 + theta1*x1 + theta2*x2 = 0

    lines(plot_x, plot_y)
  }
  else{

    u <- seq(from=-1, to=1.5, length.out=50)
    v <- seq(from=-1, to=1.5, length.out=50)

    z <- mat.or.vec(length(u), length(v))

    for(i in 1:length(u)){
      for(j in 1:length(v)){
        z[i,j] <- mapFeature(u[i],v[j]) %*% theta
      }
    }

    z <- t(z)

    contour(u, v, z, levels = 0, add = T)

  }
}
```

구현한 `plotDecisionBoundary` 함수로 그래프를 그려 보겠습니다.

```r
# Plot Decision Boundary
plotDecisionBoundary(theta, X, y, xlim=c(-0.9,1.1), ylim=c(-0.9,1.35))
legend(0.8,1.3, legend=c("y = 1","y = 0"), pch = c("+", "-"), col=c("green", "red"))
```

![Data with Decision Boundary](/assets/coursera/machine-learning/ex2/unnamed-chunk-11-1.png)

### Prediction
지정한 theta값을 X에 적용해 예측 값을 반환하는 `predict` 함수를 작성한 후, 실제 y값과 비교를 통해 모델 정확도를 살펴 보겠습니다.

```r
# Predict Function
predict <- function(theta, X){

  m <- nrow(X)
  p <- rep(0, m)
  prob <- sigmoid(t(theta) %*% t(X))
  p_one <- prob >= 0.5
  p_zero <- prob < 0.5
  p[p_one] <- 1  
  p[p_zero] <- 0

  return(p)
}

# Predict and calculate accuracy
p <- predict(theta, X)
print(paste('Train Accuracy : ', mean(p==y)*100))
```


     [1] "Train Accuracy :  83.0508474576271"
