# loading library
library(pROC)
library(glmnet)


GLMNET <- function(X_train,y,X_test=data.frame(),cv=5,family="gaussian",alpha=1,nlambda=100,seed=123,metric="auc")
{
  # defining evaluation metric
  score <- function(a,b,metric)
  {
    switch(metric,
           accuracy = sum(abs(a-b)<=0.5)/length(a),
           auc = auc(a,b),
           logloss = -(sum(log(1-b[a==0])) + sum(log(b[a==1])))/length(a),
           mae = sum(abs(a-b))/length(a),
           precision = length(a[a==b])/length(a),
           rmse = sqrt(sum((a-b)^2)/length(a)),
           rmspe = sqrt(sum(((a-b)/a)^2)/length(a)))           
  }
  
  cat("Preparing Data\n")
  X_train$order <- seq(1, nrow(X_train))
  X_train$result <- as.numeric(y)
  
  X_test2 <- X_test
  
  set.seed(seed)
  X_train$randomCV <- floor(runif(nrow(X_train), 1, (cv+1)))
  
  cat(cv, "-fold Cross Validation\n", sep = "")
  for (i in 1:cv)
  {
    X_build <- subset(X_train, randomCV != i)
    X_val <- subset(X_train, randomCV == i)
    
    build <- as.matrix(subset(X_build, select = -c(order, randomCV, result)))
    val <- as.matrix(subset(X_val, select = -c(order, randomCV, result)))
    test <- as.matrix(X_test2)
    
    build_label <- as.factor(as.matrix(subset(X_build, select = c('result'))))
    
    model_glm <- glmnet(build,build_label,family=family,alpha=alpha,nlambda=nlambda)
    
    pred_glm <- predict(model_glm,val,type="response",s=0.01)
    X_val$pred_glm <- pred_glm
    
    if (nrow(X_test) > 0)
    {
      pred_glm <- predict(model_glm, test, type = "response", s = 0.01)
    }
    
    cat("CV Fold-", i, " ", metric, ": ", score(X_val$result, X_val$pred_glm, metric), "\n", sep = "")
    
    if (i == 1)
    {
      output <- X_val
      if (nrow(X_test) > 0)
      {
        X_test$pred_glm <- pred_glm
      }      
    }
    
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
  
  output <- output[order(output$order),]
  cat("\nGLMNET ", cv, "-Fold CV ", metric, ": ", score(output$result, output$pred_glm, metric), "\n", sep = "")
  
  colnames(X_test)[ncol(X_test)] <- "pred_glm"
  
  output <- subset(output, select = c("order", "pred_glm"))
  return(list(output, X_test))  
}
