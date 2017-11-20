##### 1. Leitura e exibição do dataset #####

library(readr)
HR <- read_csv("HR.csv")
# reordena as instâncias de forma aleatória, para posteriormente manter a proporção de cada classe dentro dos folds
HR <- HR[sample(nrow(HR)),]
View(HR)


##### 2. Aplicação do PCA #####

# descarta as colunas nao numericas
HR_without_chars <- HR[,c(1:8)]

# executa o PCA retirando a coluna de classe e normalizando os dados com scale()
HR_princomp <- princomp(scale(HR_without_chars[,-7]))
print(HR_princomp$loadings)

# transforma as variaveis em componentes PCA, aplicando o loading
HR_transformado_pca <- predict(HR_princomp, newdata = HR_without_chars[,-7])

# descarta o ultimo componente, mantendo uma variacao de 85.7%
HR_transformado_pca <- HR_transformado_pca[,-7]

# obtem a coluna de classe do dataset original
left <- HR[,c(7)]

# faz merge do dataset transformado para incluir a classe
dataset_after_pca <- cbind(HR_transformado_pca, left)

# exibe o dataset com as componentes do PCA e a classe
View(dataset_after_pca)

##### 3. Aplicando o RELIEF #####

# talvez seja necessário executar as duas linhas abaixo caso o R não encontre o pacote FSelector
# o padrâmetro de dyn.load deve ser alterado conforme a localização de seu JDK
dyn.load('/Library/Java/JavaVirtualMachines/jdk1.8.0_112.jdk/Contents/Home/jre/lib/server/libjvm.dylib')
require(rJava)

# Importa a biblioteca FSelector
library(FSelector)

# executa o algoritmo RELIEF para calcular os pesos das características, informando que left é a classe.
weights <- relief(left ~ ., HR)

# seleciona os 5 atributos de maior importância
top_features <- cutoff.k(weights, 5)

# depois de algumas execuções do RELIEF, constatamos que as características mais influentes são as listadas abaixo
# seleciona os 5 atributos de maior importância (e incluímos a classe "left")
dataset_after_relief <- HR[, c("satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", 
                              "time_spend_company", "left")]

# exibe o dataset com características selecionadas pelo RELIEF
View(dataset_after_relief)

##### 4. Classificador SVM #####

library(e1071)
# Função para iterar os folds e calcular as medidas de acurácia
iterate_folds_svm <- function(dataset, k, cost_input, kernel_type){
  # inicializa algumas variáveis
  precisionSum <- 0
  recallSum <- 0
  errorSum <- 0
  # quebra o dataset em k folds
  folds <- cut(seq(1, nrow(dataset)), breaks=k,labels=FALSE)
  for(i in 1:k){
    testIndexes <- which(folds==i,arr.ind=TRUE)
    # separa 1 fold pra teste e o restante para treinamento 
    testData <- dataset[testIndexes, ]
    trainData <- dataset[-testIndexes, ]
    
    # Aplica-se SVM
    fit <- svm(left~., data = trainData, kernel=kernel_type, scale = TRUE, cost=cost_input, type = "C")
    # Aplica o modelo obtido sobre a base de teste
    prediction <- predict(fit, testData, type="class")
    # --------------------------------------------
    
    # matriz de confusão
    # a classe left é a sétima coluna no dataset 
    tab <- table(prediction, testData$left) 
    print(tab)
    TP <- ifelse(dim(tab)[1] == 1, 0, tab[c(2), c(2)])  # true positive
    FP <- ifelse(dim(tab)[1] == 1, 0, tab[c(2), c(1)]) # false positive
    FN <- tab[c(1), c(2)]  # false negative
    TN <- tab[c(1), c(1)]  # true negative
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
k <- 5 # Quantidade de folds: mantivemos 5 por causa do tempo de processamento
c <- 1 # Custo do erro: variamos entre os valores 0.1, 0.5 e 1
kernel <- "radial" # Kernel: trocamos para cada tipo. Possibilidades: linear, sigmoid, polynomial e radial

# Executa o SVM com TODAS as características, calcula e exibe erro, precisão, e recall médios 
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
iterate_folds_nn <- function(dataset, k, num_neurons_input, learn_input, index_class){
  # inicializa algumas variáveis
  precisionSum <- 0
  recallSum <- 0
  errorSum <- 0
  
  # Gera a fórmula que contém todas as características contra a classe ("left")
  n <- colnames(dataset)
  form <- as.formula(paste("left ~", paste(n[!n %in% "left"], collapse = " + ")))

  # quebra o dataset em k folds
  folds <- cut(seq(1,nrow(dataset)), breaks=k,labels=FALSE)
  for(i in 1:k){
    # separa-se 1 fold pra teste e o restante para treinamento 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- dataset[testIndexes, ]
    trainData <- dataset[-testIndexes, ]
    
    # Aplica-se neuralnet para todas as características
    fit <- neuralnet(form, data = trainData, hidden=num_neurons_input, learningrate = learn_input,
                     err.fct ="ce", linear.output = FALSE, stepmax=1e6)
    # Aplica o modelo obtido sobre a base de teste
    output <- compute(fit, testData[, -(index_class)])
    p <- output$net.result
    prediction <- ifelse(p>0.5, 1, 0)
    
    # matriz de confusão
    tab <- table(prediction, testData[, index_class])
    print(tab)
    TP <- ifelse(dim(tab)[1] == 1, 0, tab[c(2), c(2)])  # true positive
    FP <- ifelse(dim(tab)[1] == 1, 0, tab[c(2), c(1)]) # false positive
    FN <- tab[c(1), c(2)]  # false negative
    TN <- tab[c(1), c(1)]  # true negative
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

# importa pacotes para variáveis dummy e redes neurais
library(neuralnet)
library(dummies)

# cria variáveis dummy para poder executar o NN
HR_dummy <- dummy.data.frame(HR)
left <- HR_dummy$left
HR_dummy <- scale(HR_dummy[, c(1:6, 8:21)])
HR_dummy <- cbind(HR_dummy, left)

# reordena as instâncias de forma aleatória
HR_dummy <- HR_dummy[sample(nrow(HR_dummy)),]

# Parâmetros a serem variados para o NN
k <- 5 # quantidade de folds, mantivemos 5 por causa do tempo de processamento
num_neurons_input <- 1 # número de neurônios, variamos de 1 a 4 neurônios
learn_input <- 0.1 # learning rate. Utilizamos os seguintes valores: 0.1, 0.5 e 1

# Executa o NN com TODAS as características,  calcula e exibe erro, precisão, e recall médios 
index_class <- 21 # a classe left está na posição 21 neste caso
values <- iterate_folds_nn(HR_dummy, k, num_neurons_input, learn_input, index_class)
print(paste("Erro médio: ", values[3]))
print(paste("Sensibilidade média: ", values[2]))
print(paste("Precisão média: ", values[1]))

# Executa o NN com as características do RELIEF, calcula e exibe erro, precisão, e recall médios 
index_class <- 6 # a classe left está na posição 6 neste caso
values <- iterate_folds_nn(dataset_after_relief, k, num_neurons_input, learn_input, index_class)
print(paste("Erro médio: ", values[3]))
print(paste("Sensibilidade média: ", values[2]))
print(paste("Precisão média: ", values[1]))

# Executa o NN com os componentes do PCA, calcula e exibe erro, precisão, e recall médios 
index_class <- 7 # a classe left está na posição 7 neste caso
values <- iterate_folds_nn(dataset_after_pca, k, num_neurons_input, learn_input, index_class)
print(paste("Erro médio: ", values[3]))
print(paste("Sensibilidade média: ", values[2]))
print(paste("Precisão média: ", values[1]))


##### 6. Classificador Naive Bayes ######
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
    
    # Aplica-se Naive Bayes aqui
    fit <- naiveBayes(left ~., data = dataset)
    prediction <- predict(fit, testData)
    
    # matriz de confusão
    tab <- table(prediction, testData$left)
    print(tab)
    TP <- ifelse(dim(tab)[1] == 1, 0, tab[c(2), c(2)])  # true positive
    FP <- ifelse(dim(tab)[1] == 1, 0, tab[c(2), c(1)]) # false positive
    FN <- tab[c(1), c(2)]  # false negative
    TN <- tab[c(1), c(1)]  # true negative
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

# Executa o naive bayes com características selecionadas pelo RELIEF, calcula e exibe erro, precisão, e recall médios 
values <- iterate_folds_bayes(dataset_after_relief, k)
print(paste("Erro médio: ", values[3]))
print(paste("Sensibilidade média: ", values[2]))
print(paste("Precisão média: ", values[1]))

# Executa o naive bayes com componentes selecionados usando PCA, calcula e exibe erro, precisão, e recall médios 
values <- iterate_folds_bayes(dataset_after_pca, k)
print(paste("Erro médio: ", values[3]))
print(paste("Sensibilidade média: ", values[2]))
print(paste("Precisão média: ", values[1]))
