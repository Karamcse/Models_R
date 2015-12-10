## initialize h2o
library(h2o)
localh2o <- h2o.init(nthreads=2)


## function for random forest
DeepLearning <- function(X_train,y,X_test=data.frame(),cv=5,transform="none",distribution="AUTO",activation="Rectifier",epochs=50,rate=0.005,hidden=c(100,100),l1=0,l2=0,max_w2=Inf,train_samples_per_iteration=50,input_dropout_ratio=0,hidden_dropout_ratios=c(0.5,0.5),seed=235,metric="rmse")
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
    model_dl <- h2o.deeplearning(x=names(X_build)[-ncol(X_build)], y="target", training_frame=X_build_h2o, distribution=distribution, nfolds=5, activation=activation, epochs=epochs, rate=rate, hidden=hidden, l1=l1, l2=l2, train_samples_per_iteration=train_samples_per_iteration, max_w2=max_w2)
    
    # predicting on validation data
    pred_dl <- as.data.frame(predict(model_dl, X_val_h2o))
    names(pred_dl)[1] <- "pred_dl"
    if (ncol(pred_dl) == 3)
    {
      pred_dl <- pred_dl[,3]
    }
    
    X_val <- cbind(X_val, pred_dl)
    
    # predicting on test data
    if (nrow(X_test) > 0)
    {
      pred_dl <- as.data.frame(predict(model_dl, X_test_h2o))
      names(pred_dl)[1] <- "pred_dl"
      if (ncol(pred_dl) == 3)
      {
        pred_dl <- pred_dl[,3]
      }
    }
    
    cat("CV Fold-", i, " ", metric, ": ", score(X_val$target, X_val$pred_dl, metric), "\n", sep = "")
    
    # initializing outputs
    if (i == 1)
    {
      output <- X_val
      if (nrow(X_test) > 0)
      {
        X_test <- cbind(X_test, pred_dl)
      }      
    }
    
    # appending to outputs
    if (i > 1)
    {
      output <- rbind(output, X_val)
      if (nrow(X_test) > 0)
      {
        X_test$pred_dl <- (X_test$pred_dl * (i-1) + pred_dl)/i
      }            
    }
    
    gc()
  } 
  
  # final evaluation score
  output <- output[order(output$order),]
  cat("\nDeepLearning ", cv, "-Fold CV ", metric, ": ", score(output$target, output$pred_dl, metric), "\n", sep = "")
  
  # returning CV predictions and test data with predictions
  return(list("train"=output, "test"=X_test))  
}

