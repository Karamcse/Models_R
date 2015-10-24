## loading libraries
library(pROC)
library(randomForest)


## function for random forest classification
RandomForestClassification <- function(X_train,y,X_test=data.frame(),cv=5,ntree=50,nodesize=5,seed=123,metric="auc",plot=0,importance=0)
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
    model_rf <- randomForest(result ~., data=X_build, ntree=ntree, nodesize=nodesize)
    
    if (plot == 1)
    {
      varImpPlot(model_rf)
    }
    
    if (importance == 1)
    {
      print(model_rf$importance)
    }
    
    # predicting on validation data
    pred_rf <- predict(model_rf, X_val, type = "prob")[,2]
    X_val <- cbind(X_val, pred_rf)
    
    # predicting on test data
    if (nrow(X_test) > 0)
    {
      pred_rf <- predict(model_rf, X_test, type = "prob")[,2]
    }
      
    cat("CV Fold-", i, " ", metric, ": ", score(X_val$result, X_val$pred_rf, metric), "\n", sep = "")
    
    # initializing outputs
    if (i == 1)
    {
      output <- X_val
      if (nrow(X_test) > 0)
      {
        X_test <- cbind(X_test, pred_rf)
      }      
    }
    
    # appending to outputs
    if (i > 1)
    {
      output <- rbind(output, X_val)
      if (nrow(X_test) > 0)
      {
        X_test$pred_rf <- (X_test$pred_rf * (i-1) + pred_rf)/i
      }            
    }
    
    gc()
  } 

  # final evaluation score
  output <- output[order(output$order),]
  cat("\nRandomForest ", cv, "-Fold CV ", metric, ": ", score(output$result, output$pred_rf, metric), "\n", sep = "")

  output <- subset(output, select = c("order", "pred_rf"))
  
  # returning CV predictions and test data with predictions
  return(list(output, X_test))  
}


## function for random forest regression
RandomForestRegression <- function(X_train,y,X_test=data.frame(),cv=5,ntree=50,nodesize=5,seed=123,metric="mae",plot=0,importance=0)
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
    model_rf <- randomForest(result ~., data=X_build, ntree=ntree, nodesize=nodesize)
    
    if (plot == 1)
    {
      varImpPlot(model_rf)
    }
    
    if (importance == 1)
    {
      print(model_rf$importance)
    }
    
    # predicting on validation data
    pred_rf <- predict(model_rf, X_val)
    X_val <- cbind(X_val, pred_rf)
    
    # predicting on test data
    if (nrow(X_test) > 0)
    {
      pred_rf <- predict(model_rf, X_test)
    }
      
    cat("CV Fold-", i, " ", metric, ": ", score(X_val$result, X_val$pred_rf, metric), "\n", sep = "")
    
    # initializing outputs
    if (i == 1)
    {
      output <- X_val
      if (nrow(X_test) > 0)
      {
        X_test <- cbind(X_test, pred_rf)
      }      
    }
    
    # appending to outputs
    if (i > 1)
    {
      output <- rbind(output, X_val)
      if (nrow(X_test) > 0)
      {
        X_test$pred_rf <- (X_test$pred_rf * (i-1) + pred_rf)/i
      }            
    }
    
    gc()
  } 

  # final evaluation score
  output <- output[order(output$order),]
  cat("\nRandomForest ", cv, "-Fold CV ", metric, ": ", score(output$result, output$pred_rf, metric), "\n", sep = "")

  output <- subset(output, select = c("order", "pred_rf"))
  
  # returning CV predictions and test data with predictions
  return(list(output, X_test))  
}

