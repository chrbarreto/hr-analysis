# address for downloading HR Analytics dataset
# https://www.kaggle.com/ludobenistant/hr-analytics/downloads/human-resources-analytics.zip

##### 1. Reading and viewing dataset #####

library(readr)
HR <- read_csv("HR.csv")

# shuffles dataset
HR <- HR[sample(nrow(HR)),]
View(HR)


##### 2. PCA #####

# discards non-numeric columns 
HR_without_chars <- HR[,c(1:8)]

# runs PCA after removing class and scaling dataset for data normalization
HR_princomp <- princomp(scale(HR_without_chars[,-7]))
print(HR_princomp$loadings)

# turns variables into PCA components
HR_pca <- predict(HR_princomp, newdata = HR_without_chars[,-7])

# removes last component, keeping a 85.7% variance
HR_pca <- HR_pca[,-7]

# gets class column from original dataset
left <- HR[,c(7)]

# merges dataset with class after PCA components calculation
dataset_after_pca <- cbind(HR_pca, left)

# shows resulting dataset 
View(dataset_after_pca)

##### 3. RELIEF #####

# If your application does not find FSelector package, 
# dyn.load parameter must be changed according to your JDK location
dyn.load('/Library/Java/JavaVirtualMachines/jdk1.8.0_112.jdk/Contents
         /Home/jre/lib/server/libjvm.dylib')
require(rJava)

library(FSelector)

# runs RELIEF algorithm for estimating variables' weights
weights <- relief(left ~ ., HR)

# selects top 5 more influent variables 
top_features <- cutoff.k(weights, 5)

# creates a new dataset including only variables selected by RELIEF 
# p.s.: RELIEF selected numerical variables at all times,
# which means that probably it does not work well with categorical variables 
dataset_after_relief <- HR[, c("satisfaction_level", "last_evaluation", 
                               "number_project", "average_montly_hours", "time_spend_company", "left")]

# shows dataset which includes only variables selected by RELIEF
View(dataset_after_relief)

##### 4. SVM #####

library(e1071)
# function for iterating over folds and estimating accuracy measures
iterate_folds_svm <- function(dataset, k, cost_input, kernel_type){
  precisionSum <- 0
  recallSum <- 0
  errorSum <- 0
  # partitions input dataset into k folds
  folds <- cut(seq(1, nrow(dataset)), breaks=k,labels=FALSE)
  for(i in 1:k){
    testIndexes <- which(folds==i,arr.ind=TRUE)
    # selects 1 fold for testing and k-1 for training
    testData <- dataset[testIndexes, ]
    trainData <- dataset[-testIndexes, ]
    
    # applies SVM
    fit <- svm(left~., data = trainData, kernel=kernel_type, scale = TRUE,
               cost=cost_input, type = "C")
    # applies model to training subset
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
    
    # shows results in console
    print(paste("Error: ", error))
    print(paste("Recall: ", recall))
    print(paste("Precision: ", precision))
    
    # adds calculated values to final results
    precisionSum <- precisionSum + precision
    recallSum <- recallSum + recall
    errorSum <- errorSum + error
  }
  # returns an average of the values obtained in all k iterations
  values <- list(errorSum/k, recallSum/k, precisionSum/k)
  print(paste("Average error: ", values[1]))
  print(paste("Average recall: ", values[2]))
  print(paste("Average precision: ", values[3]))
}

# parameters to be varied in SVM
k <- 5
dataset_list <- list(HR, dataset_after_relief, dataset_after_pca)
kernel_list <- c("linear", "radial", "sigmoid", "polynomial")
c_list <- c(0.1, 0.5, 1)
for (dataset in dataset_list){
  for (kernel in kernel_list){
    for (c in c_list){
      iterate_folds_svm(dataset, k, c, kernel)
    }
  }
}
##### 5. Neural Networks #####

# function for iterating over folds and estimating accuracy measures
iterate_folds_nn <- function(dataset, k, num_neurons_input, learn_input, index_class){
  precisionSum <- 0
  recallSum <- 0
  errorSum <- 0
  
  # generates a formula which contains all of the incoming variables to compare them against the "left" class
  n <- colnames(dataset)
  form <- as.formula(paste("left ~", paste(n[!n %in% "left"], collapse = " + ")))
  
  # partitions input dataset into k folds
  folds <- cut(seq(1,nrow(dataset)), breaks=k,labels=FALSE)
  for(i in 1:k){
    # selects 1 fold for testing and k-1 for training
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- dataset[testIndexes, ]
    trainData <- dataset[-testIndexes, ]
    
    # applies neural network
    fit <- neuralnet(form, data = trainData, hidden=num_neurons_input, 
                     learningrate = learn_input, err.fct ="ce",
                     linear.output = FALSE, stepmax=1e6)
    # applies model to training subset
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
    
    # shows results in console
    print(paste("Error: ", error))
    print(paste("Recall: ", recall))
    print(paste("Precision: ", precision))
    
    # adds calculated values to final results
    precisionSum <- precisionSum + precision
    recallSum <- recallSum + recall
    errorSum <- errorSum + error
  }
  # returns an average of the values obtained in all k iterations
  values <- list(errorSum/k, recallSum/k, precisionSum/k)
  print(paste("Average error: ", values[1]))
  print(paste("Average recall: ", values[2]))
  print(paste("Average precision: ", values[3]))
}

# imports packages for dealing with neural networks and dummy variables
library(neuralnet)
library(dummies)

# turns categorical variables into dummy ones so that we can properly execute Neural Networks
HR_dummy <- dummy.data.frame(HR)
left <- HR_dummy$left
HR_dummy <- scale(HR_dummy[, c(1:6, 8:21)])
HR_dummy <- cbind(HR_dummy, left)

# shuffles dataset
HR_dummy <- HR_dummy[sample(nrow(HR_dummy)),]

# parameters to be varied in Neural Networks
k <- 5 # number of folds
num_neurons_input <- 1 # number of neurons in hidden layer (Options: 1, 2, 3, 4, and so on)
learn_input <- 0.1 # learning rate (Options: 0.1, 0.5, 1)

# runs Neural Network for dataset with ALL variables
# and prints average error, recall and precision
index_class <- 21 # index for "left" class
iterate_folds_nn(HR_dummy, k, num_neurons_input, learn_input, index_class)

# runs SVM for dataset with only variables selected by RELIEF
# and prints average error, recall and precision
index_class <- 6 
iterate_folds_nn(dataset_after_relief, k, num_neurons_input, learn_input, index_class)

# runs Neural Network with PCA components
# and prints average error, recall and precision
index_class <- 7
iterate_folds_nn(dataset_after_pca, k, num_neurons_input, learn_input, index_class)


##### 6. Naive Bayes ######
library(e1071)
# function for iterating over folds and estimating accuracy measures
iterate_folds_bayes <- function(dataset, k){
  precisionSum <- 0
  recallSum <- 0
  errorSum <- 0
  # partitions input dataset into k folds
  folds <- cut(seq(1,nrow(dataset)), breaks=k,labels=FALSE)
  for(i in 1:k){
    testIndexes <- which(folds==i,arr.ind=TRUE)
    # selects 1 fold for testing and k-1 for training
    testData <- dataset[testIndexes, ]
    trainData <- dataset[-testIndexes, ]
    
    # applies model to training subset
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
    
    # shows results in console
    print(paste("Error: ", error))
    print(paste("Recall: ", recall))
    print(paste("Precision: ", precision))
    
    # adds calculated values to final results
    precisionSum <- precisionSum + precision
    recallSum <- recallSum + recall
    errorSum <- errorSum + error
  }
  # returns an average of the values obtained in all k iterations
  values <- list(errorSum/k, recallSum/k, precisionSum/k)
  print(paste("Average error: ", values[1]))
  print(paste("Average recall: ", values[2]))
  print(paste("Average precision: ", values[3]))
}

# Naive Bayes is faster than SVM and NN, so we can work with 10 folds
k <- 10
# No change of parametes was made
dataset_list <- list(HR, dataset_after_relief, dataset_after_pca)
for (dataset in dataset_list){
      iterate_folds_bayes(dataset, k)
}

