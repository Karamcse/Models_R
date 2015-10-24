# loading library
library(pROC)
library(e1071)


NaiveBayesClassification <- function(X_train,y,X_test=data.frame(),cv=5,seed=123,metric="auc")
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
  
  set.seed(seed)
  X_train$randomCV <- floor(runif(nrow(X_train), 1, (cv+1)))
  
  # cross validation
  cat(cv, "-fold Cross Validation\n", sep = "")
  for (i in 1:cv)
  {
    X_build <- subset(X_train, randomCV != i, select = -c(order, randomCV))
    X_val <- subset(X_train, randomCV == i) 
    
    # building model
    model_nb <- naiveBayes(result ~., data=X_build)
    
    # predicting on validation data
    pred_nb <- predict(model_nb, X_val, type="raw")[,2]
    X_val <- cbind(X_val, pred_nb)
    
    # predicting on test data
    if (nrow(X_test) > 0)
    {
      pred_nb <- predict(model_nb, X_test, type = "raw")[,2]
    }
    
    cat("CV Fold-", i, " ", metric, ": ", score(X_val$result, X_val$pred_nb, metric), "\n", sep = "")
    
    # initializing outputs
    if (i == 1)
    {
      output <- X_val
      if (nrow(X_test) > 0)
      {
        X_test <- cbind(X_test, pred_nb)
      }      
    }
    
    # appending to outputs
    if (i > 1)
    {
      output <- rbind(output, X_val)
      if (nrow(X_test) > 0)
      {
        X_test$pred_nb <- (X_test$pred_nb * (i-1) + pred_nb)/i
      }            
    }
    
    gc()
  } 
  
  # final evaluation score
  output <- output[order(output$order),]
  cat("\nnaiveBayes ", cv, "-Fold CV ", metric, ": ", score(output$result, output$pred_nb, metric), "\n", sep = "")
  
  output <- subset(output, select = c("order", "pred_nb"))
  
  # returning CV predictions and test data with predictions
  return(list(output, X_test))  
}
