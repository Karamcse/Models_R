## initialize h2o
library(h2o)
localh2o <- h2o.init(nthreads=-1)


## function for random forest
RandomForest <- function(X_train,y,X_test=data.frame(),cv=5,transform="none",ntrees=100,max_depth=5,sample_rate=1,min_rows=1,seed=235,metric="rmse",importance=0)
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
    model_rf <- h2o.randomForest(x=names(X_build)[-ncol(X_build)], y="target", training_frame=X_build_h2o, ntrees=ntrees, max_depth=max_depth, sample_rate=sample_rate, min_rows=min_rows, seed=seed)
    
    if (importance == 1)
    {
      print(h2o.varimp(model_rf))
    }
    
    # predicting on validation data
    pred_rf <- as.data.frame(predict(model_rf, X_val_h2o))
    names(pred_rf) <- "pred_rf"
    X_val <- cbind(X_val, pred_rf)
    
    # predicting on test data
    if (nrow(X_test) > 0)
    {
      pred_rf <- as.data.frame(predict(model_rf, X_test_h2o))
      names(pred_rf) <- "pred_rf"
    }
    
    cat("CV Fold-", i, " ", metric, ": ", score(X_val$target, X_val$pred_rf, metric), "\n", sep = "")
    
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
  return(list("train"=output, "test"=X_test))  
}

