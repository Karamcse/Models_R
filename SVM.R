## loading libraries
library(pROC)
library(e1071)


## function for svm classification
SVMClassification <- function(X_train,y,X_test=data.frame(),cv=5,scale=T,kernel="radial",cost=1,epsilon=0.1,seed=123,metric="auc")
{
  # defining evaluation metric
  score <- function(a,b,metric)
  {
    switch(metric,
           auc = auc(a,b),
           logloss = -(sum(log(1-b[a==0])) + sum(log(b[a==1])))/length(a),
           precision = length(a[a==b])/length(a))
  }
  
  cat("Preparing Data\n")
  X_train$order <- seq(1, nrow(X_train))
  X_train$result <- as.factor(y)
  
  set.seed(seed)
  X_train$randomCV <- floor(runif(nrow(X_train), 1, (cv+1)))
  
  cat(cv, "-fold Cross Validation\n", sep = "")
  for (i in 1:cv)
  {
    X_build <- subset(X_train, randomCV != i, select = -c(order, randomCV))
    X_val <- subset(X_train, randomCV == i) 
    
    # building model
    model_svm <- svm(result ~., data = X_build, scale=scale, kernel=kernel, cost=cost, epsilon=epsilon, probability=T)
    
    # predicting on validation data
    pred_svm <- predict(model_svm, X_val, probability=T)
    X_val <- cbind(X_val, data.frame("pred_svm"=attr(pred_svm, "probabilities")[,2]))
    
    # predicting on test data
    if (nrow(X_test) > 0)
    {
      pred_svm <- predict(model_svm, X_test, probability=T)
    }
      
    cat("CV Fold-", i, " ", metric, ": ", score(X_val$result, X_val$pred_svm, metric), "\n", sep = "")
    
    # initializing outputs
    if (i == 1)
    {
      output <- X_val
      if (nrow(X_test) > 0)
      {
        X_test <- cbind(X_test, data.frame("pred_svm"=attr(pred_svm, "probabilities")[,2]))
      }      
    }
    
    # appending to outputs
    if (i > 1)
    {
      output <- rbind(output, X_val)
      if (nrow(X_test) > 0)
      {
        X_test$pred_svm <- (X_test$pred_svm * (i-1) + attr(pred_svm, "probabilities")[,2])/i
      }            
    }
    
    gc()
  } 

  # final evaluation score
  output <- output[order(output$order),]
  cat("\nSVM ", cv, "-Fold CV ", metric, ": ", score(output$result, output$pred_svm, metric), "\n", sep = "")

  output <- subset(output, select = c("order", "pred_svm"))
  
  # returning CV predictions and test data with predictions
  return(list(output, X_test))  
}


## function for svm regression
SVMRegression <- function(X_train,y,X_test=data.frame(),cv=5,scale=T,kernel="radial",cost=1,epsilon=0.1,seed=123,metric="mae")
{
  # defining evaluation metric
  score <- function(a,b,metric)
  {
    switch(metric,
           mae = sum(abs(a-b))/length(a),
           rmse = sqrt(sum((a-b)^2)/length(a)),
           rmspe = sqrt(sum(((a-b)/a)^2)/length(a)))
  }
  
  cat("Preparing Data\n")
  X_train$order <- seq(1, nrow(X_train))
  X_train$result <- as.numeric(y)
  
  set.seed(seed)
  X_train$randomCV <- floor(runif(nrow(X_train), 1, (cv+1)))
  
  cat(cv, "-fold Cross Validation\n", sep = "")
  for (i in 1:cv)
  {
    X_build <- subset(X_train, randomCV != i, select = -c(order, randomCV))
    X_val <- subset(X_train, randomCV == i) 
    
    # building model
    model_svm <- svm(result ~., data = X_build, scale=scale, kernel=kernel, cost=cost, epsilon=epsilon)
    
    # predicting on validation data
    pred_svm <- predict(model_svm, X_val)
    X_val <- cbind(X_val, pred_svm)
    
    # predicting on test data
    if (nrow(X_test) > 0)
    {
      pred_svm <- predict(model_svm, X_test)
    }
      
    cat("CV Fold-", i, " ", metric, ": ", score(X_val$result, X_val$pred_rf, metric), "\n", sep = "")
    
    # initializing outputs
    if (i == 1)
    {
      output <- X_val
      if (nrow(X_test) > 0)
      {
        X_test <- cbind(X_test, pred_svm)
      }      
    }
    
    # appending to outputs
    if (i > 1)
    {
      output <- rbind(output, X_val)
      if (nrow(X_test) > 0)
      {
        X_test$pred_svm <- (X_test$pred_svm * (i-1) + pred_svm)/i
      }            
    }
    
    gc()
  } 

  # final evaluation score
  output <- output[order(output$order),]
  cat("\nSVM ", cv, "-Fold CV ", metric, ": ", score(output$result, output$pred_svm, metric), "\n", sep = "")

  output <- subset(output, select = c("order", "pred_svm"))
  
  # returning CV predictions and test data with predictions
  return(list(output, X_test))  
}

