# loading library
library(kknn)
library(pROC)


kNNClassification <- function(X_train,y,X_test=data.frame(),cv=5,k=10,distance=2,kernel="optimal",seed=123,metric="auc")
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
  X_train$result <- as.factor(y)
  
  set.seed(seed)
  X_train$randomCV <- floor(runif(nrow(X_train), 1, (cv+1)))
  
  # cross-validation
  cat(cv, "-fold Cross Validation\n", sep = "")
  for (i in 1:cv)
  {
    X_build <- subset(X_train, randomCV != i, select = -c(order, randomCV))
    X_val <- subset(X_train, randomCV == i) 
    
    # building model
    model_knn <- kknn(result ~., X_build, X_val, k=k, d=distance, kernel=kernel)
    
    # predicting on validation data
    pred_knn <- model_knn$prob[,2]
    X_val <- cbind(X_val, pred_knn)
    
    # predicting on test data
    if (nrow(X_test) > 0)
    {
      model_knn <- kknn(result ~., X_build, X_test, k = k, d = distance, kernel = kernel)
      pred_knn <- model_knn$prob[,2]
    }
    
    cat("CV Fold-", i, " ", metric, ": ", score(X_val$result, X_val$pred_knn, metric), "\n", sep = "")
    
    # initializing outputs
    if (i == 1)
    {
      output <- X_val
      if (nrow(X_test) > 0)
      {
        X_test <- cbind(X_test, pred_knn)
      }      
    }
    
    # appending to outputs
    if (i > 1)
    {
      output <- rbind(output, X_val)
      if (nrow(X_test) > 0)
      {
        X_test$pred_knn <- (X_test$pred_knn * (i-1) + pred_knn)/i
      }            
    }
    
    gc()
  } 
  
  # final evaluation score
  output <- output[order(output$order),]
  cat("\nkNN ", cv, "-Fold CV ", metric, ": ", score(output$result, output$pred_knn, metric), "\n", sep = "")
  
  output <- subset(output, select = c("order", "pred_knn"))
  
  # returning CV predictions and test data with predictions
  return(list(output, X_test))  
}
