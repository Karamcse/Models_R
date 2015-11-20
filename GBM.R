# loading library
library(pROC)
library(gbm)


GBMRegression <- function(X_train,y,X_test=data.frame(),cv=5,n.trees=50,n.minobsinnode=5,interaction.depth=2,shrinkage=0.001,seed=123,metric="rmse")
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
  
  # cross-validation
  cat(cv, "-fold Cross Validation\n", sep = "")
  for (i in 1:cv)
  {
    X_build <- subset(X_train, randomCV != i, select = -c(order, randomCV))
    X_val <- subset(X_train, randomCV == i) 
    
    # building model
    model_gbm <- gbm(result ~.,data=X_build,n.trees=n.trees,n.minobsinnode=n.minobsinnode,interaction.depth=interaction.depth,shrinkage=shrinkage)
    
    # predicting on validation data
    pred_gbm <- predict(model_gbm, X_val, n.trees)
    X_val <- cbind(X_val, pred_gbm)
    
    # predicting on test data
    if (nrow(X_test) > 0)
    {
      pred_gbm <- predict(model_gbm, X_test)
    }
    
    cat("CV Fold-", i, " ", metric, ": ", score(X_val$result, X_val$pred_gbm, metric), "\n", sep = "")
    
    # initializing outputs
    if (i == 1)
    {
      output <- X_val
      if (nrow(X_test) > 0)
      {
        X_test <- cbind(X_test, pred_gbm)
      }      
    }
    
    # appending to outputs
    if (i > 1)
    {
      output <- rbind(output, X_val)
      if (nrow(X_test) > 0)
      {
        X_test$pred_gbm <- (X_test$pred_gbm * (i-1) + pred_gbm)/i
      }            
    }
    
    gc()
  } 
  
  # final evaluation score
  output <- output[order(output$order),]
  cat("\nGBM ", cv, "-Fold CV ", metric, ": ", score(output$result, output$pred_gbm, metric), "\n", sep = "")
  
  output <- subset(output, select = c("order", "pred_gbm"))
  
  # returning CV predictions and test data with predictions
  return(list(output, X_test))  
}


