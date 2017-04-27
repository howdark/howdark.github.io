---
title: "Machine Learning with R - W4"
author: "Seongbong Kim"
date: 2016-01-18 00:10:00 +0900
categories: jekyll update machine-learning
permalink: /blog/:title
comments: true
---

이번 포스트에서는 Week 4에 소개된 Neural Networks와 Logistic Regression과 관련된 Exercise를 R로 구현해 보겠습니다.


## Programming Exercise 3: Multi-class Classification and Neural Networks

**Multi-class Classification** 을 연습하는 과제입니다.
 이번 과제는 0~9 사이의 handwritten digits를 logistic regression과 neural networks를 이용해 분류하는 과제입니다.

### Dataset
이번 과제에 사용되는 데이터는 20x20 pixel handwritten digits (X)와 실제 label(y) 5000개 관측치가 주어집니다.
(Exercise3 데이터 중 ex3data1.mat 파일이 필요합니다.)
<a href="http://s3.amazonaws.com/spark-public/ml/exercises/on-demand/machine-learning-ex3.zip">다운로드</a>



### Loading and visualizing the data
우선 예제파일이 `.mat` 파일로 주어지기 때문에(Matlab data file), 이 파일을 읽으려면 `R.matlab` package가 필요합니다. `install.packages("R.matlab")`으로 package가 설치 되었다고 가정하겠습니다. `ex3data1.mat` 파일을 `readMat()`함수로 불러오면 **List** 형식으로 변수가 저장이 됩니다. `X`와 `y`를 각각 저장합니다.


```r
## =============== Part 1 : Loading and Visualizing Data ===============
library(R.matlab)

# Load Training Data
exdata <- readMat('ex3data1.mat')
X <- exdata$X
y <- exdata$y
m <- nrow(X)
```

Loading한 데이터가 어떤 데이터인지 살펴보기 위해 5000개 중 100개의 index를 sampling해서 이미지를 불러오는 단계입니다. `displayData`라는 함수를 과제에서 제공하는데 아래와 같이 구현해 봤습니다.


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

100개의 관측치를 샘플링해서 subset을 만들고 `displayData`함수를 실행하면 아래와 같이 그림이 나옵니다. (샘플링 결과에 따라 그림은 다르게 표시될 수 있습니다.)


```r
rand_indices <- sample(5000,100)
sel <- X[rand_indices,]

displayData(sel)
```

![Exercise Data](/assets/coursera/machine-learning/ex3/unnamed-chunk-4-1.png)

### Vectorize Logistic Regression
다음은 과제를 위한 parameter들을 setting 합니다. `input_layer_size`는 20x20을 1-dimension vector로 변환해 400의 크기를 가지고, 예측하려는 label의 숫자는 0~9로 `num_labels`는 10을 지정합니다. Regularization 상수 `lambda`는 임의로 0.1을 할당합니다.


```r
## =============== Part 2 : Vectorize Logistic Regression ===============

## Setup the parameters you will use for this part of the exercise
input_layer_size <- 400   # 20x20 input images of Digits
num_labels <- 10    # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)

lambda <- 0.1
```

과제에서는 `oneVsAll`함수를 작성하라고 되어 있는데 이 안에는 앞선 과제에서 수행한 것과 마찬가지로 **cost function** 과 **gradient function** 을 작성하고 optimizing을 수행하는 많은 내용이 들어 있습니다. 지난 과제에 작성했던 Logistic Regression의 cost function과 gradient function을 그대로 가져오면 됩니다. 아래와 같이 함수를 작성하고 `oneVsAll`함수를 작성하도록 하겠습니다.


```r
# Sigmoid Function
sigmoid <- function(z){
  g <- 1 / (1 + exp(-z))
  return(g)
}


# Logistic Regression cost function
lrcostFunctionReg <- function(theta, X, y, lambda){

  m <- nrow(y)
  J <- 0

  h_theta <- sigmoid(t(theta) %*% t(X))

  theta_j <- theta
  theta_j[1] <- 0

  J <- (-1 * (t(y) %*% t(log(h_theta))) - (t(1-y) %*% t(log(1-h_theta))))/m +
    (lambda / (2 * m)) * (t(theta_j) %*% theta_j)

  return(J)

}

# Logistic Regression gradient function
lrgradFunctionReg <- function(theta, X, y, lambda){

  m <- nrow(y)

  h_theta <- sigmoid(t(theta) %*% t(X))

  theta_j <- theta
  theta_j[1] <- 0

  grad <- t(((h_theta - t(y)) %*% X) / m) + lambda / m * theta_j

  return(grad)

}
```


### One-vs-all Classification

`oneVsAll`함수를 작성해 보도록 하겠습니다. 이 함수가 하는 역할은 400개의 feature를 가진 X가 1부터 10까지(0을 10으로 표현) 각각의 label과 가장 일치하는 최적 hypothesis 함수(theta)를 산출하는 과정을 coding하였습니다. 과제에서는 `fmincg`라는 Matlab 함수를 사용했는데, 이 함수는 **Conjugate gradients** method를 사용한 최적화 함수여서 `oneVsAll` 함수 내에서 `optim` 함수를 사용할 때 `method="CG"`를 지정했습니다.


```r
oneVsAll <- function(X, y, num_labels, lambda, maxit=500){

  m <- nrow(X)
  n <- ncol(X)

  all_theta <- matrix(0, nrow=num_labels, ncol=n + 1)       # 10 x 401

  X <- cbind(matrix(1, nrow=m), X)

  for(i in 1:num_labels){

    initial_theta <- matrix(0, nrow=n+1, ncol=1)
    y_tmp <- y
    oneIdx <- which(y_tmp == i)
    zeroIdx <- which(y_tmp != i)
    y_tmp[oneIdx,] <- 1
    y_tmp[zeroIdx,] <- 0
    res <- optim(initial_theta, fn=lrcostFunctionReg, X=X, y=y_tmp, lambda=lambda,
                 control=list(maxit=maxit),
                 gr=lrgradFunctionReg,
                 method="CG")
    all_theta[i,] <- res$par

  }

  return(all_theta)
}
```

어렵게 작성된 `oneVsAll` 함수에 `X`, `y`, `num_labels`, `lambda`를 입력하고 결과값이 나오는 시간을 산출해 봅니다.


```r
system.time(all_theta <- oneVsAll(X, y, num_labels, lambda, maxit=50))
```


        user  system elapsed
       23.10    5.86   29.02



### Prediction
산출된 `all_theta`는 hypothesis 함수의 상수로서 `X`와 행렬 연산을 통해 `y`를 예측할 수 있습니다. `predictOneVsAll`함수를 만들어서 `all_theta`와 `X`를 입력하면 예측값을 반환하도록 하겠습니다.


```r
# Function that predict y for given alltheta
predictOneVsAll <- function(all_theta, X){

  m <- nrow(X)
  num_labels <- nrow(X)

  p <- matrix(0, nrow(X), 1)

  X <- cbind(matrix(1, nrow=m), X)

  h_theta <- sigmoid(X %*% t(all_theta))

  p <- apply(h_theta, 1, function(x) which(x==max(x)))
  return(p)
}
```

예측 결과의 정확도는 아래와 같습니다.


```r
## =============== Part 3 : Predict for One-Vs-All ===============

pred <- predictOneVsAll(all_theta, X)

accuracy <- mean(pred == y) * 100
print(paste("Accuracy of this model is : ", accuracy))
```


     [1] "Accuracy of this model is :  92.02"

___

## Neural Networks
이 부분은 동일한 Data를 가지고 neural networks를 이용해 label을 예측해 보는 맛보기 실습입니다.

### Loading Parameters
과제에서 이미 neural networks에 사용할 weight들을 산출해서 제공하고 있습니다. 이 weight들을 불러와서 feedforward propagation을 수행할 예정입니다. Data를 새로 불러오고 역시 `readMat`함수로 `ex3weights.mat` 파일을 불러온 뒤 `Theta1`과 `Theta2`에 할당합니다. neural networks의 hidden layer는 1개, unit은 25개로 설정합니다.


```r
## =========== Part 1: Loading Data =============

# Load Training Data
exdata <- readMat('ex3data1.mat')

X <- exdata$X
y <- exdata$y

m <- nrow(X)

## ================ Part 2: Loading Parameters ================
# Setup parameters
input_layer_size <- 400
hidden_layer_size <- 25
num_labels <- 10

# Loading Parameters
thetas <- readMat('ex3weights.mat')
Theta1 <- thetas$Theta1      # 401 x 25
Theta2 <- thetas$Theta2      # 26 x 10
```


### Feedforward Propagation and Prediction
Feedforward propagation을 수행하여 label을 예측하는 `predict` 함수를 작성하고 정확도를 산출하면 아래와 같습니다.

```r
# Function that feedforward propagation
predict <- function(Theta1, Theta2, X){

  m <- nrow(X)
  num_labels <- nrow(Theta2)

  X <- cbind(matrix(1,nrow=m), X)
  a2 <- sigmoid(X %*% t(Theta1))
  a2 <- cbind(matrix(1,nrow=m), a2)
  a3 <- sigmoid(a2 %*% t(Theta2))

  p <- apply(a3, 1, function(x) which(x == max(x)))

  return(p)
}


## ================= Part 3: Implement Prediction =================

pred <- predict(Theta1, Theta2, X)
accuracy <- mean(pred==y)
print(paste("Accuracy of this model is : ", accuracy))
```


     [1] "Accuracy of this model is :  0.9752"
