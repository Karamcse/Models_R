## loading libraries


## function for GLM
GLM <- function(X_train,y,X_test=data.frame(),cv=5,seed=123,metric="rmse")
{
  # defining evaluation metric
  score <- function(a,b,metric)
  {
    switch(metric,
           mae = sum(abs(a-b))/length(a),
           precision = length(a[a==b])/length(a),
           rmse = sqrt(sum((a-b)^2)/length(a)),
           rmspe = sqrt(sum(((a-b)/a)^2)/length(a)))           
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
    model_glm <- glm(result ~., data=X_build, family="gaussian")
    
    # predicting on validation data
    pred_glm <- predict(model_glm, X_val)
    X_val <- cbind(X_val, pred_glm)
    
    # predicting on test data
    if (nrow(X_test) > 0)
    {
      pred_glm <- predict(model_glm, X_test)
    }
    
    cat("CV Fold-", i, " ", metric, ": ", score(X_val$result, X_val$pred_glm, metric), "\n", sep = "")
    
    # initializing outputs
    if (i == 1)
    {
      output <- X_val
      if (nrow(X_test) > 0)
      {
        X_test <- cbind(X_test, pred_glm)
      }      
    }
    
    # appending to outputs
    if (i > 1)
    {
      output <- rbind(output, X_val)
      if (nrow(X_test) > 0)
      {
        X_test$pred_glm <- (X_test$pred_glm * (i-1) + pred_glm)/i
      }            
    }
    
    gc()
  } 
  
  # final evaluation score
  output <- output[order(output$order),]
  cat("\nGLM ", cv, "-Fold CV ", metric, ": ", score(output$result, output$pred_glm, metric), "\n", sep = "")
  
  output <- subset(output, select = c("order", "pred_glm"))
  
  # returning CV predictions and test data with predictions
  return(list(output, X_test))  
}
