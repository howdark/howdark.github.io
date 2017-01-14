---
title: "Machine Learning with R - W3"
author: "Seongbong Kim"
date: 2016-01-17 00:00:00 +0900
categories: jekyll update machine-learning
---

이번 포스트에서는 Week 3에 소개된 Logistic Regression과 관련된 Exercise를 R로 구현해 보겠습니다.


## Programming Exercise 2: Logistic Regression


**Logistic Regression** 을 연습하는 과제입니다.
 Logistic Regression 1차 시험과 2차 시험 성적을 가지고 입학 허가 여부를 예측하는 과제가 되겠습니다.

### Dataset
이번 과제에 사용되는 데이터는 1차 시험성적(X1), 2차 시험성적(X2), 입학 허가 여부(y)의 3개 변수에 대해 100개 관측치가 주어집니다.
(Exercise2 데이터 중 ex2data1.txt 파일이 필요합니다.)
<a href="http://s3.amazonaws.com/spark-public/ml/exercises/on-demand/machine-learning-ex2.zip">다운로드</a>

### Plotting Data

우선 데이터를 불러와서 X와 y에 저장을 해줍니다.

```r
data <- read.table("ex2data1.txt", sep=",")
X <- as.matrix(data[,1:2])
y <- as.matrix(data[,3])
```

다음은 데이터가 어떻게 구성되어 있는지 Plot을 하는 단계인데, 과제에서는 `plotData(X,y)`라는 함수를 별도로 제공을 하고 있어서 `plotData` 함수를 생성합니다. 입학 허가된 point는 +와 green으로, 입학 불허된 point는 -와 red로 표시하는 함수 입니다.


```r
plotData <- function(X,y,xlim=NULL,ylim=NULL){

  plot(X[which(y==1), 1], X[which(y==1), 2], pch="+", cex = 2, col = "green",
       xlab='', ylab='', xlim=xlim, ylim=ylim)
  points(X[which(y==0), 1], X[which(y==0), 2], pch="-", cex = 2, col = "red")}
```

그럼 `plotData` 함수를 사용해서 Plotting을 해보겠습니다.


```r
## =============== Part 1 : Plotting ===============
plotData(X,y)

title(xlab='Exam 1 score', ylab='Exam 2 score')
legend(88,99, legend=c("Admitted","Not admitted"), pch = c("+", "-"), col=c("green", "red"))
```

![Exercise Data](/assets/coursera/machine-learning/ex2/ex2-1-1.png)


### Compute Cost and Gradient
이전 과제와 마찬가지로 Logistic Regression의 Cost함수를 구현하고 Gradient Descent 방식으로 Cost를 최소화 할 예정입니다.

초기 값들을 Setting 합니다. `X`는 Intercept Term과 연산을 위해 1로 구성된 Column을 추가해 주고, `initial_theta`는 0으로 이루어진 `(n + 1) * 1` 행렬로 생성합니다.


```r
## =============== Part 2 : Compute Cost and Gradient ===============

# Setup the data matrix appropriately, and add ones for the intercept term
m <- nrow(X)
n <- ncol(X)

# Add intercept term to x and X_test
X <- cbind(matrix(1, nrow=m), X)

# Initialize fitting parameters
initial_theta = mat.or.vec(n+1,1)
```


#### Cost Function
Cost를 계산하는 `costFunction` 함수를 작성합니다. `costFunction` 내부에서는 logit함수를 구현하는 `sigmoid` 함수를 사용하기 때문에 `sigmoid` 함수를 먼저 작성하고 `costFunction` 함수를 작성합니다.

```r
# sigmoid function
sigmoid <- function(z){

  g <- 1 / (1 + exp(-z))

  return(g)

}

# function of computing cost
costFunction <- function(theta, X, y){

  m <- length(y)
  J <- 0

  h_theta <- sigmoid(t(theta) %*% t(X))
  J <- J + (-1 * (t(y) %*% t(log(h_theta))) - t(1-y) %*% t(log(1-h_theta))) / m

  return(J)

}
```


#### Gradient Function
Gradient를 계산하는 `gradFunction` 함수를 작성합니다.

```r
gradFunction <- function(theta, X, y){

  m <- length(y)

  h_theta <- sigmoid(t(theta) %*% t(X))

  grad <- t(((h_theta - t(y)) %*% X) / m)

  return(grad)

}
```


#### Compute and display initial cost and gradient
`initial_theta`를 가지고 cost와 gradient 값을 구해봅니다.

```r
cost <- costFunction(initial_theta, X, y)
grad <- gradFunction(initial_theta, X, y)
print(paste('Cost at initial theta (zeros): ', cost))
```


     [1] "Cost at initial theta (zeros):  0.693147180559945"


```r
print('Gradient at initial theta (zeros): ')
```


     [1] "Gradient at initial theta (zeros): "


```r
print(grad)
```


             [,1]
         -0.10000
     V1 -12.00922
     V2 -11.26284



### Optimizing
이전 과제에서는 `for-loop`를 사용하여 gradient descent를 수행했으나, 이번 과제에서는 `fminunc`라는 함수를 이용해서 gradient descent를 수행하도록 하고 있습니다.(`fminunc`는 *function minimization with unconstraint* 의 약자?라고 생각됩니다.) Matlab 과제여서 함수도 Matlab 코드로 작성하여 제공되는데, R에서는 `optim`이라는 함수로 동일한 최적화를 수행할 수 있습니다.


```r
## =============== Part 3 : Optimizing using fminunc ===============

opt_result <- optim(initial_theta, fn = function(t) costFunction(t,X,y),
                    gr = function(t) gradFunction(t,X,y))
theta <- opt_result$par
cost <- opt_result$value
```
`optim`함수는 최적화 할 변수의 초기값을 넣어주고(이 과제의 경우 `initial_theta`), 최적화 할 함수를 지정하고(`costFunction`), gradient 함수를 지정하여(`gradFunction`) 수행합니다.
결과 값은 **List** 형식으로 반환되는데 그 중에 `par` 항목이 최적화된 변수이고 `value`는 최적화된 변수를 함수에 대입했을 때의 값입니다.
반환된 값을 가지고 plot을 해보겠습니다.
과제에서는 `plotDecisionBoundary`라는 함수를 별도로 작성하도록 해서 동일하게 작성합니다.

```r
plotDecisionBoundary <- function(theta, X, y, xlim=NULL, ylim=NULL){

  plotData(X[,2:3], y, xlim=xlim, ylim=ylim)

  if(ncol(X) <= 3){
    plot_x <- c(min(X[,2])-2, max(X[,2])+2)
    plot_y <- (-1 / theta[3]) * (theta[2]*plot_x + theta[1])
    # theta0 + theta1*x1 + theta2*x2 = 0

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

Decision Boundary를 그려봅니다.

```r
plotDecisionBoundary(theta, X, y)
```

![Data with Decision Boundary](/assets/coursera/machine-learning/ex2/ex2-1-2.png)

### Prediction

1차 시험 45점, 2차 시험 85점을 맞을 경우 입학 허가가 나올 확률은 얼마인지 예측해 보도록 하겠습니다.


```r
## =============== Part 4 : Predict and Accuracies ===============

prob <- sigmoid(c(1, 45, 85) %*% theta)
print(paste('For a student with scores 45 and 85, we predict an admission probability of ', prob))
```


     [1] "For a student with scores 45 and 85, we predict an admission probability of  0.776354125968299"



Logistic Regression의 경우 예측값이 확률로 나오기 때문에 0.5 이상일 경우 1로, 0.5 미만일 경우 0으로 예측하는 `predict` 함수를 작성합니다.

```r
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
```

전체 점수 조합에 대해 우리의 theta값으로 입학 허가 여부를 예측했을 때 정확도가 얼마인지 확인해 봅니다.

```r
# Compute accuracy on our training set
p <- predict(theta, X)
print(paste('Train Accuracy : ', mean(p==y)*100))
```


     [1] "Train Accuracy :  89"
