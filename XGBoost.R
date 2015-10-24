## loading libraries
library(pROC)
library(xgboost)


## function for xgboost
XGBoost <- function(X_train,y,X_test=data.frame(),cv=5,objective="binary:logistic",eta=0.1,max.depth=5,nrounds=50,gamma=0,min_child_weight=1,seed=123,metric="auc")
{
  # defining evaluation metric
  score <- function(a,b,metric)
  {
    switch(metric,
           auc = auc(a,b),
           mae = sum(abs(a-b))/length(a),
           rmse = sqrt(sum((a-b)^2)/length(a)),
           rmspe = sqrt(sum(((a-b)/a)^2)/length(a)),
           logloss = -(sum(log(1-b[a==0])) + sum(log(b[a==1])))/length(a),
           precision = length(a[a==b])/length(a))
  }
  
  cat("Preparing Data\n")
  X_train$order <- seq(1, nrow(X_train))
  X_train$result <- as.numeric(y)
  
  # converting data to numeric
  for (i in 1:ncol(X_train))
  {
    X_train[,i] <- as.numeric(X_train[,i])
  }
  
  if (nrow(X_test)>0)
  {
    for (i in 1:ncol(X_test))
    {
      X_test[,i] <- as.numeric(X_test[,i])
    }    
  }
  
  X_train[is.na(X_train)] <- -1
  X_test[is.na(X_test)] <- -1

  X_test2 <- X_test
  
  set.seed(seed)
  X_train$randomCV <- floor(runif(nrow(X_train), 1, (cv+1)))
  
  # cross-validation
  cat(cv, "-fold Cross Validation\n", sep = "")
  for (i in 1:cv)
  {
    X_build <- subset(X_train, randomCV != i)
    X_val <- subset(X_train, randomCV == i)
    
    build <- as.matrix(subset(X_build, select = -c(order, randomCV, result)))
    val <- as.matrix(subset(X_val, select = -c(order, randomCV, result)))
    test <- as.matrix(X_test2)
    
    build_label <- as.matrix(subset(X_build, select = c('result')))
    
    # building model
    model_xgb <- xgboost(build,build_label,objective=objective,eta=eta,max.depth=max.depth,nrounds=nrounds,gamma=gamma,min_child_weight=min_child_weight,nthread=-1,verbose=0,eval.metric="auc")
    
    # predicting on validation data
    pred_xgb <- predict(model_xgb, val)
    X_val <- cbind(X_val, pred_xgb)
    
    # predicting on test data
    if (nrow(X_test) > 0)
    {
      pred_xgb <- predict(model_xgb, test)
    }
    
    cat("CV Fold-", i, " ", metric, ": ", score(X_val$result, X_val$pred_xgb, metric), "\n", sep = "")
    
    # initializing outputs
    if (i == 1)
    {
      output <- X_val
      if (nrow(X_test) > 0)
      {
        X_test <- cbind(X_test, pred_xgb)
      }      
    }
    
    # appending to outputs
    if (i > 1)
    {
      output <- rbind(output, X_val)
      if (nrow(X_test) > 0)
      {
        X_test$pred_xgb <- (X_test$pred_xgb * (i-1) + pred_xgb)/i
      }            
    }
    
    gc()
  } 
  
  # final evaluation score
  output <- output[order(output$order),]
  cat("\nXGBoost ", cv, "-Fold CV ", metric, ": ", score(output$result, output$pred_xgb, metric), "\n", sep = "")
  
  output <- subset(output, select = c("order", "pred_xgb"))
  
  # returning CV predictions and test data with predictions
  return(list(output, X_test))  
}
