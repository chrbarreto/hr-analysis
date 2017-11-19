##### 1. Leitura e exibição do dataset #####
library(readr)
HR <- read_csv("HR.csv")
# reordena as instâncias de forma aleatória, para posteriormente manter a proporção de cada classe dentro dos folds
HR_random <- HR[sample(nrow(HR)),]
View(HR)

##### 2. Aplicação do PCA #####

# descarta as colunas nao numericas
HR_without_chars <- HR[,c(1:8)]

# executa o pca, retirando a coluna de classe, e normalizando os dados com scale()
HR_princomp <- princomp(scale(HR_without_chars[,-7]))
print(HR_princomp$loadings)

# transforma as variaveis em componentes pca, aplicando o loading
HR_transformado_pca <- predict(HR_princomp, newdata = HR_without_chars[,-7])

# descarta o ultimo componente, mantendo uma variacao de 85.7%
HR_transformado_pca <- HR_transformado_pca[,-7]

# obtem a coluna de classe do dataset original
left <- HR[,c(7)]

# faz merge do dataset transformado para incluir a classe
dataset_after_pca <- cbind(HR_transformado_pca, left)
View(dataset_after_pca)

##### 3. Aplicando o RELIEF #####

# Importa a biblioteca FSelector
library(FSelector)

# executa o algoritmo relief para calcular os pesos das características, 
# informando que left é a classe.
weights <- relief(left ~ ., HR)

# seleciona os 5 atributos de maior importância
top_features <- cutoff.k(weights, 5)
dataset_after_relief <- HR[, c(top_features)]

##### 4. Classificador SVM #####

library(e1071)
# Função para iterar os folds e calcular as medidas de acurácia
iterate_folds_svm <- function(dataset, k, cost_input, kernel_type){
  # inicializa algumas variáveis
  precisionSum <- 0
  recallSum <- 0
  errorSum <- 0
  # quebra o dataset em k folds
  folds <- cut(seq(1,nrow(dataset)), breaks=k,labels=FALSE)
  for(i in 1:k){
    testIndexes <- which(folds==i,arr.ind=TRUE)
    # separa 1 fold pra teste e o restante para treinamento 
    testData <- dataset[testIndexes, ]
    trainData <- dataset[-testIndexes, ]
    
    # Aplica-se SVM
    fit <- svm(left~., data = trainData, kernel=kernel_type, scale = TRUE, cost=cost_input, type = "C")
    print(fit)
    # Aplica o modelo obtido sobre a base de teste
    prediction <- predict(fit, testData, type="class")
    # --------------------------------------------
    
    # matriz de confusão
    tab <- table(prediction, testData$left)
    TP <- tab[c(2), c(2)] # true positive
    FP <- tab[c(2), c(1)] # false positive
    FN <- tab[c(1), c(2)] # false negative
    TN <- tab[c(1), c(1)] # true negative
    P <- TP + FN
    # cálculo de erro, precisão e recall para cada fold
    error <- (FN + FP) / (TP + FP + FN + TN)
    precision <- TP/(TP + FP)
    recall <- TP/(P)
    
    # exibe os valores no console
    print(paste("Erro: ", error))
    print(paste("Sensibilidade: ", recall))
    print(paste("Precisão: ", precision))
    
    # acumula os valores obtidos com cada iteração
    precisionSum <- precisionSum + precision
    recallSum <- recallSum + recall
    errorSum <- errorSum + error
  }
  # Retorna o valor médio obtido nas K iterações
  return(list(precisionSum/k, recallSum/k, errorSum/k))
}

# Parâmetros a serem variados para o SVM
k <- 5 # quantidade de folds, mantivemos 5 por causa do tempo de processamento
c <- 1 # custo. Utilizamos os valores 0.1, 0.5 e 1
kernel <- "radial" # trocar para cada tipo. Possibilidades: linear, sigmoid, polynomial e radial

# Executa o SVM com TODAS as características,  calcula e exibe erro, precisão, e recall médios 
values <- iterate_folds_svm(HR, k, c, kernel)
print(paste("Erro médio: ", values[3]))
print(paste("Sensibilidade média: ", values[2]))
print(paste("Precisão média: ", values[1]))

# Executa o SVM com características selecionadas pelo RELIEF,  calcula e exibe erro, precisão, e recall médios 
values <- iterate_folds_svm(dataset_after_relief, k, c, kernel)
print(paste("Erro médio: ", values[3]))
print(paste("Sensibilidade média: ", values[2]))
print(paste("Precisão média: ", values[1]))

# Executa o SVM com componentes selecionados usando PCA,  calcula e exibe erro, precisão, e recall médios 
values <- iterate_folds_svm(dataset_after_pca, k, c, kernel)
print(paste("Erro médio: ", values[3]))
print(paste("Sensibilidade média: ", values[2]))
print(paste("Precisão média: ", values[1]))

##### 5. Classificador Redes Neurais #####

# função iterate_folds modificada para NN
iterate_folds_nn <- function(dataset, k, cost_input, kernel_type){
  # inicializa algumas variáveis
  precisionSum <- 0
  recallSum <- 0
  errorSum <- 0
  # quebra o dataset em k folds
  folds <- cut(seq(1,nrow(dataset)), breaks=k,labels=FALSE)
  for(i in 1:k){
    testIndexes <- which(folds==i,arr.ind=TRUE)
    # separa 1 fold pra teste e o restante para treinamento 
    testData <- dataset[testIndexes, ]
    trainData <- dataset[-testIndexes, ]
    
    # Aplica-se neuralnet
    # TODO: ver se tem forma generica de fazer isso, pra servir pro PCA também 
    fit <- neuralnet( left ~ satisfaction_level + last_evaluation + number_project + 
                        average_montly_hours + time_spend_company + Work_accident +
                        promotion_last_5years + departmentaccounting + departmenthr + 
                        departmentIT + departmentmanagement + departmentmarketing + 
                        departmentproduct_mng + departmentRandD + departmentsales + 
                        departmentsupport + departmenttechnical + salaryhigh + salarylow + salarymedium,
                      data = trainData, hidden=num_neurons_input, learningrate = learn_input, err.fct ="ce",
                      linear.output = FALSE, stepmax=1e6)
    # Aplica o modelo obtido sobre a base de teste
    output <- compute(fit, testData[, -21])
    p <- output$net.result
    prediction <- ifelse(p>0.5, 1, 0)
    
    # matriz de confusão
    tab <- table(prediction, testData$left)
    TP <- tab[c(2), c(2)] # true positive
    FP <- tab[c(2), c(1)] # false positive
    FN <- tab[c(1), c(2)] # false negative
    TN <- tab[c(1), c(1)] # true negative
    P <- TP + FN
    # cálculo de erro, precisão e recall para cada fold
    error <- (FN + FP) / (TP + FP + FN + TN)
    precision <- TP/(TP + FP)
    recall <- TP/(P)
    
    # exibe os valores no console
    print(paste("Erro: ", error))
    print(paste("Sensibilidade: ", recall))
    print(paste("Precisão: ", precision))
    
    # acumula os valores obtidos com cada iteração
    precisionSum <- precisionSum + precision
    recallSum <- recallSum + recall
    errorSum <- errorSum + error
  }
  # Retorna o valor médio obtido nas K iterações
  return(list(precisionSum/k, recallSum/k, errorSum/k))
}



# cria variáveis dummy para poder executar o NN
HR_dummy <- dummy.data.frame(HR1)
View(HR_dummy)
left <- HR_dummy$left
HR_dummy <- scale(HR_dummyHR[, c(1:6, 8:21)])
HR_dummy <- cbind(HR_dummy, left)

# Parâmetros a serem variados para o NN
k <- 5 # quantidade de folds, mantivemos 5 por causa do tempo de processamento
num_neurons_input <- 1 # número de neurônios, variamos de 1 a 4 neurônios
learn_input <- 0.1 # learning rate. Utilizamos os seguintes valores: 0.1, 0.5 e 1

# Executa o NN com TODAS as características,  calcula e exibe erro, precisão, e recall médios 
values <- iterate_folds_nn(HR, k, c, kernel)
# TODO: completar o restante de NN, tem como fazer generico ou vamos ter que repetir a funcao iterate folds?


##### 6. Classificador naive bayes ######
library(e1071)
# função iterate_folds modificada para naive bayes
iterate_folds_bayes <- function(dataset, k){
  # inicializa algumas variáveis
  precisionSum <- 0
  recallSum <- 0
  errorSum <- 0
  # quebra o dataset em k folds
  folds <- cut(seq(1,nrow(dataset)), breaks=k,labels=FALSE)
  for(i in 1:k){
    testIndexes <- which(folds==i,arr.ind=TRUE)
    # separa 1 fold pra test e k-1 para treinamento 
    testData <- dataset[testIndexes, ]
    trainData <- dataset[-testIndexes, ]
    
    # --------------------------------------------
    # Aplica-se SVM/Redes Neurais/Naive Bayes aqui
    fit <- naiveBayes(left ~., data = dataset)
    print(fit)
    prediction <- predict(fit, testData)
    # --------------------------------------------
    
    # matriz de confusão
    tab <- table(prediction, testData$left)
    print(tab)
    TP <- tab[c(2), c(2)]
    FP <- tab[c(2), c(1)]
    FN <- tab[c(1), c(2)]
    TN <- tab[c(1), c(1)]
    P <- TP + FN
    # cálculo de erro, precisão e recall
    error <- (FN + FP) / (TP + FP + FN + TN)
    precision <- TP/(TP + FP)
    recall <- TP/(P)
    
    # exibe os valores no console
    print(paste("Erro: ", error))
    print(paste("Sensibilidade: ", recall))
    print(paste("Precisão: ", precision))
    
    # acrescenta-se aos valores totais
    precisionSum <- precisionSum + precision
    recallSum <- recallSum + recall
    errorSum <- errorSum + error
  }
  return(list(precisionSum/k, recallSum/k, errorSum/k))
}

# define o uso de 10 folds 
k <- 10

# Executa o naive bayes com TODAS as características,  calcula e exibe erro, precisão, e recall médios 
values <- iterate_folds_bayes(HR, k)
# calcula e exibe erro, precisão, e recall médios
print(paste("Erro médio: ", values[3]))
print(paste("Sensibilidade média: ", values[2]))
print(paste("Precisão média: ", values[1]))

# Executa o naive bayes com características selecionadas pelo RELIEF,  calcula e exibe erro, precisão, e recall médios 
values <- iterate_folds_bayes(dataset_after_relief, k)
print(paste("Erro médio: ", values[3]))
print(paste("Sensibilidade média: ", values[2]))
print(paste("Precisão média: ", values[1]))

# Executa o naive bayes com componentes selecionados usando PCA,  calcula e exibe erro, precisão, e recall médios 
values <- iterate_folds_bayes(dataset_after_pca, k)
print(paste("Erro médio: ", values[3]))
print(paste("Sensibilidade média: ", values[2]))
print(paste("Precisão média: ", values[1]))
