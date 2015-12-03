## initialize h2o
library(h2o)
localh2o <- h2o.init()


## function for random forest
GBM <- function(X_train,y,X_test=data.frame(),cv=5,transform="none",distribution="gaussian",ntrees=100,max_depth=5,min_rows=1,learn_rate=0.1,seed=235,metric="rmse")
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
  
  # loading library for auc
  if (metric == "auc")
  {
    library(pROC)
  }
  
  # cleaning data
  cat("Preparing Data\n")
  X_train$order <- seq(1,nrow(X_train))
  X_train$target <- y
  
  X_train[is.na(X_train)] <- -1
  X_test[is.na(X_test)] <- -1
  
  set.seed(seed)
  X_train$randomCV <- floor(runif(nrow(X_train), 1, (cv+1)))
  
  if (nrow(X_test) > 0)
  {
    X_test_h2o <- as.h2o(X_test, destination_frame="X_test_h2o")    
  }
  
  # cross-validation
  cat(cv, "-fold Cross Validation\n", sep = "")
  for (i in 1:cv)
  {
    X_build <- subset(X_train, randomCV != i, select=-c(order,randomCV))
    X_val <- subset(X_train, randomCV == i)
    
    X_build_h2o <- as.h2o(X_build, destination_frame="X_build_h2o")
    X_val_h2o <- as.h2o(X_val, destination_frame="X_val_h2o")
    
    # building model
    model_gbm <- h2o.gbm(x=names(X_build)[-ncol(X_build)], y="target", training_frame=X_build_h2o, distribution=distribution, ntrees=ntrees, max_depth=max_depth, learn_rate=learn_rate, min_rows=min_rows, seed=seed)
    
    # predicting on validation data
    pred_gbm <- as.data.frame(predict(model_gbm, X_val_h2o))
    names(pred_gbm) <- "pred_gbm"
    X_val <- cbind(X_val, pred_gbm)
    
    # predicting on test data
    if (nrow(X_test) > 0)
    {
      pred_gbm <- as.data.frame(predict(model_gbm, X_test_h2o))
      names(pred_gbm) <- "pred_gbm"
    }
    
    cat("CV Fold-", i, " ", metric, ": ", score(X_val$target, X_val$pred_gbm, metric), "\n", sep = "")
    
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
  cat("\nGBM ", cv, "-Fold CV ", metric, ": ", score(output$target, output$pred_gbm, metric), "\n", sep = "")
  
  output <- subset(output, select = c("order", "pred_gbm"))
  
  # returning CV predictions and test data with predictions
  return(list("train"=output, "test"=X_test))  
}

