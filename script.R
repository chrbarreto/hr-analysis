
########## 1. Reading and viewing dataset ########## 

library(readr)
HR = read_csv('HR.csv', col_names= T)

# shuffling dataset
HR <- HR[sample(nrow(HR)),]

########## 2. Applying PCA for merging components ##########

# discarding non-numeric columns 
HR_without_chars <- HR[,c(1:8)]

# running PCA after removing class and scaling dataset for data normalization
HR_princomp <- princomp(scale(HR_without_chars[,-7]))
print(HR_princomp$loadings)

# turning variables into PCA components
HR_pca <- predict(HR_princomp, newdata = HR_without_chars[,-7])

# removing last component, keeping a 85.7% variance
HR_pca <- HR_pca[,-7]

# getting class column from original dataset
left <- HR[,c(7)]

# merging dataset with class after PCA components calculation
dataset_after_pca <- cbind(HR_pca, left)

##### 3. Applying RELIEF for selecting features #####

library(FSelector)

# running RELIEF algorithm for estimating variables' weights
weights <- relief(left ~ ., HR)

# selecting top 5 more influent variables 
top_features <- cutoff.k(weights, 5)

# creating a new dataset including only variables selected by RELIEF 
# p.s.: RELIEF selected numerical variables at all times,
# which means that probably it does not work well with categorical variables 
dataset_after_relief <- HR[, c("satisfaction_level", "last_evaluation", 
                               "number_project", "average_montly_hours", "time_spend_company", "left")]


########## 4. SVM ##########

library(e1071)
# function for iterating over folds and estimating accuracy measures
iterate_folds_svm <- function(dataset, k, cost_input, kernel_type){
  precisionSum <- 0
  recallSum <- 0
  errorSum <- 0
  # partitioning input dataset into k folds
  folds <- cut(seq(1, nrow(dataset)), breaks=k,labels=FALSE)
  print("Printing partial confusion matrixes...")
  for(i in 1:k){
    testIndexes <- which(folds==i,arr.ind=TRUE)
    # selecting 1 fold for testing and k-1 for training
    testData <- dataset[testIndexes, ]
    trainData <- dataset[-testIndexes, ]
    
    # applying SVM
    fit <- svm(left~., data = trainData, kernel=kernel_type, scale = TRUE,
               cost=cost_input, type = "C")
    # applying model to training subset
    prediction <- predict(fit, testData, type="class")
    # --------------------------------------------
    
    # confusion matrix
    tab <- table(prediction, testData$left) 
    print(tab)
    TP <- ifelse(dim(tab)[1] == 1, 0, tab[c(2), c(2)])  # true positive
    FP <- ifelse(dim(tab)[1] == 1, 0, tab[c(2), c(1)]) # false positive
    FN <- tab[c(1), c(2)]  # false negative
    TN <- tab[c(1), c(1)]  # true negative
    P <- TP + FN
    # calculating error, recall and precision for each fold
    error <- (FN + FP) / (TP + FP + FN + TN)
    precision <- TP/(TP + FP)
    recall <- TP/(P)
    
    # adding calculated values to final results
    precisionSum <- precisionSum + precision
    recallSum <- recallSum + recall
    errorSum <- errorSum + error
  }
  # returning an average of the values obtained in all k iterations
  values <- list(errorSum/k, recallSum/k, precisionSum/k)
  print(paste("Cost of error: ", cost_input))
  print(paste("Kernel: ", kernel_type))
  print(paste("... the average error is ", values[1]))
  print(paste("... the average recall is ", values[2]))
  print(paste("... the average precision is ", values[3]))
  
}

########## 5. Naive Bayes ##########

library(e1071)
# function for iterating over folds and estimating accuracy measures
iterate_folds_bayes <- function(dataset, k){
  precisionSum <- 0
  recallSum <- 0
  errorSum <- 0
  print("Printing partial confusion matrixes...")
  # partitioning input dataset into k folds
  folds <- cut(seq(1,nrow(dataset)), breaks=k,labels=FALSE)
  for(i in 1:k){
    testIndexes <- which(folds==i,arr.ind=TRUE)
    # selecting 1 fold for testing and k-1 for training
    testData <- dataset[testIndexes, ]
    trainData <- dataset[-testIndexes, ]
    
    # applying model to training subset
    fit <- naiveBayes(left ~., data = dataset)
    prediction <- predict(fit, testData)
    
    # confusion matrix
    tab <- table(prediction, testData$left)
    print(tab)
    TP <- ifelse(dim(tab)[1] == 1, 0, tab[c(2), c(2)])  # true positive
    FP <- ifelse(dim(tab)[1] == 1, 0, tab[c(2), c(1)]) # false positive
    FN <- tab[c(1), c(2)]  # false negative
    TN <- tab[c(1), c(1)]  # true negative
    P <- TP + FN
    # calculating error, recall and precision for each fold
    error <- (FN + FP) / (TP + FP + FN + TN)
    precision <- TP/(TP + FP)
    recall <- TP/(P)
    
    # adding calculated values to final results
    precisionSum <- precisionSum + precision
    recallSum <- recallSum + recall
    errorSum <- errorSum + error
  }
  # returning an average of the values obtained in all k iterations
  values <- list(errorSum/k, recallSum/k, precisionSum/k)
  print(paste("... the average error is ", values[1]))
  print(paste("... the average recall is ", values[2]))
  print(paste("... the average precision is ", values[3]))
  
}


########## 6. Neural Networks ##########

# function for iterating over folds and estimating accuracy measures
iterate_folds_nn <- function(dataset, k, num_neurons_input, learn_input, index_class){
  precisionSum <- 0
  recallSum <- 0
  errorSum <- 0
  
  # generating a formula which contains all of the incoming variables to compare them against the "left" class
  n <- colnames(dataset)
  form <- as.formula(paste("left ~", paste(n[!n %in% "left"], collapse = " + ")))
  print("Printing partial confusion matrixes...")
  # partitioning input dataset into k folds
  folds <- cut(seq(1,nrow(dataset)), breaks=k,labels=FALSE)
  for(i in 1:k){
    # selecting 1 fold for testing and k-1 for training
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- dataset[testIndexes, ]
    trainData <- dataset[-testIndexes, ]
    
    # applying neural networks
    fit <- neuralnet(form, data = trainData, hidden=num_neurons_input, 
                     learningrate = learn_input, err.fct ="ce",
                     linear.output = FALSE, stepmax=1e6)
    # applying model to training subset
    output <- compute(fit, testData[, -(index_class)])
    p <- output$net.result
    prediction <- ifelse(p>0.5, 1, 0)
    
    # confusion matrix
    tab <- table(prediction, testData[, index_class])
    print(tab)
    TP <- ifelse(dim(tab)[1] == 1, 0, tab[c(2), c(2)])  # true positive
    FP <- ifelse(dim(tab)[1] == 1, 0, tab[c(2), c(1)]) # false positive
    FN <- tab[c(1), c(2)]  # false negative
    TN <- tab[c(1), c(1)]  # true negative
    P <- TP + FN
    
    # calculating error, recall and precision for each fold
    error <- (FN + FP) / (TP + FP + FN + TN)
    precision <- TP/(TP + FP)
    recall <- TP/(P)
    
    # adding calculated values to final results
    precisionSum <- precisionSum + precision
    recallSum <- recallSum + recall
    errorSum <- errorSum + error
  }
  # returning an average of the values obtained in all k iterations
  values <- list(errorSum/k, recallSum/k, precisionSum/k)
  print(paste("Number of neurons: ", num_neurons_input))
  print(paste("Learning rate: ", learn_input))
  print(paste("... the average error is ", values[1]))
  print(paste("... the average recall is ", values[2]))
  print(paste("... the average precision is ", values[3]))
}



########## 7. Running model functions ##########

# We chose to use 5 folds for all models
k <- 5

### SVM ###
print("======= SVM =======")

dataset_list <- list(HR, dataset_after_relief, dataset_after_pca) # list of datasets
kernel_list <- c("linear", "radial", "polynomial") # list of kernels 
c_list <- c(0.2, 1) # list of some error_cost values
for (dataset in dataset_list){
  for (kernel in kernel_list){
    for (c in c_list){
      iterate_folds_svm(dataset, k, c, kernel)
    }
  }
}

### Naive Bayes ###
print("======= NAIVE BAYES =======")
# No change of parameters was made
dataset_list <- list(HR, dataset_after_relief, dataset_after_pca)
for (dataset in dataset_list){
  iterate_folds_bayes(dataset, k)
}

### Neural Networks ###

# importing packages for dealing with neural networks and dummy variables
library(neuralnet)
library(dummies)

# turning categorical variables into dummy ones so that we can properly execute Neural Networks
HR_dummy <- dummy.data.frame(HR)
left <- HR_dummy$left
HR_dummy <- scale(HR_dummy[, c(1:6, 8:21)])
HR_dummy <- cbind(HR_dummy, left)

# shuffling dataset
HR_dummy <- HR_dummy[sample(nrow(HR_dummy)),]

# parameters to be varied in Neural Networks
num_neurons_list <- c(1, 2, 3) # number of neurons in hidden layer (Options: 1, 2, 3, 4, and so on)
learn_rate_list <- c(0.1, 0.5) # learning rate (Options: 0.1, 0.5, 1)

print("======= NEURAL NETWORKS =======")
# running Neural Network for dataset with ALL variables
# and printing average error, recall and precision
index_class <- 21 # index for "left" class
for (num_neurons in num_neurons_list){
  for (learn_rate in learn_rate_list){
    iterate_folds_nn(HR_dummy, k, num_neurons, learn_rate , index_class)
  }
}

# running NN for dataset with only variables selected by RELIEF
# and printing average error, recall and precision
index_class <- 6 
for (num_neurons in num_neurons_list){
  for (learn_rate in learn_rate_list){
    iterate_folds_nn(dataset_after_relief, k, num_neurons, learn_rate , index_class)
  }
}

# running Neural Network with PCA components
# and printing average error, recall and precision
index_class <- 7
for (num_neurons in num_neurons_list){
  for (learn_rate in learn_rate_list){
    iterate_folds_nn(dataset_after_pca, k, num_neurons, learn_rate, index_class)
  }
}
