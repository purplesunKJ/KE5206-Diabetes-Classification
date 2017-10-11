library(nnet)
library(ROSE)
library(NeuralNetTools)
library(caret)

#read diabetes dataset
diabetes = read.csv("C:\\Users\\huangfuxing\\Documents\\NUS\\Computational intelligence\\CA1\\Diabetes.csv", header=FALSE)

#convert V9 to factor for BP 
diabetes$V9 <- as.factor(diabetes$V9)

size=nrow(diabetes)
length=ncol(diabetes)
index <- 1:size

## set the seed to make the partition reproductible
set.seed(1)
#75% data for training
positions <- sample(index, trunc(size * 0.75))

train <- diabetes[positions,]
test <- diabetes[-positions,1:length-1]
result = diabetes[-positions,]
result$V9 <- as.factor(result$V9)

#train and predict model
bpnet <- nnet(V9 ~ V1 + V2 + V8, data=train, size=50, maxit=10000)

predn<- predict(bpnet, test, type="class")
#create confusion matrix and calculate accuracy
table(true=result$V9, predicted=predn)
cm <- table(true=result$V9, predicted=predn)
sum(diag(cm))/sum(cm)
#True positive rate
TPR <- sensitivity(as.factor(predn), result$V9)
TPR
#True negative rate
TNR <- specificity(as.factor(predn), result$V9)
TNR

########################################################################
## Resampling the data to have a balanced "yes" and "no" default value##
########################################################################
diabetes_balanced <- ovun.sample(V9 ~ ., data = diabetes, method = "both", p=0.5, N=768, seed = 1)$data

size=nrow(diabetes_balanced)
length=ncol(diabetes_balanced)
index <- 1:size

## set the seed to make the partition reproductible
set.seed(1)
#75% data for training
positions <- sample(index, trunc(size * 0.75))

train <- diabetes_balanced[positions,]
test <- diabetes_balanced[-positions,1:length-1]
result = diabetes_balanced[-positions,]
result$V9 <- as.factor(result$V9)

#train and predict model
bpnet <- nnet(V9 ~ V1 + V2 + V8, data=train, size=50, maxit=10000, decay=.001)
pred1<- predict(bpnet, test, type="class")

#append BP predict to result table
result$pred1=as.factor(pred1)

#create confusion matrix and calculate accuracy
table(true=result$V9, predicted=result$pred1)
cm <- table(true=result$V9, predicted=pred1)
sum(diag(cm))/sum(cm)

#True positive rate
TPR <- sensitivity(as.factor(pred1), result$V9)
TPR
#True negative rate
TNR <- specificity(as.factor(pred1), result$V9)
TNR

##GRNN GRNN V1 + V2 + V8 as predictor

library(grnn)
result$pred2 = result[,length]
result$pred2 = -1

#extract only column 1, 2, 8 & 9
diabetes.grnn <- diabetes_balanced[,c(1,2,8,9)]
length=ncol(diabetes.grnn)

train <- diabetes.grnn[positions,]
test <- diabetes.grnn[-positions,1:length-1]

train$V9 <- as.integer(as.character(train$V9))

nn1 <- learn(train, variable.column=length)
nn1 <- smooth(nn1, sigma = 0.5)

for(i in 1:nrow(test))
{	
    vec <- as.matrix(test[i,])
    res <- guess(nn1, vec)
    
    if(is.nan(res))
    {
        cat("Entry ",i," Generated NaN result!\n")
    }
    else
    {
        result$pred2[i] <- res
    }
}

result$pred2=round(result$pred2)

table(true=result$V9, predicted=result$pred2)
cm <- table(true=result$V9, predicted=result$pred2)
sum(diag(cm))/sum(cm)

###GRNN USE ALL VARIABLES as predictor
result$pred2 = result[,length]
result$pred2 = -1

length=ncol(diabetes_balanced)

train <- diabetes_balanced[positions,]
test <- diabetes_balanced[-positions,1:length-1]

train$V9 <- as.integer(as.character(train$V9))

nn1 <- learn(train, variable.column=length)
nn1 <- smooth(nn1, sigma = 1.5)

for(i in 1:nrow(test))
{	
    vec <- as.matrix(test[i,])
    res <- guess(nn1, vec)
    
    if(is.nan(res))
    {
        cat("Entry ",i," Generated NaN result!\n")
    }
    else
    {
        result$pred2[i] <- res
    }
}

result$pred2=round(result$pred2)

table(true=result$V9, predicted=result$pred2)
cm <- table(true=result$V9, predicted=result$pred2)
sum(diag(cm))/sum(cm)

#True positive rate
TPR <- sensitivity(as.factor(result$pred2), result$V9)
TPR
#True negative rate
TNR <- specificity(as.factor(result$pred2), result$V9)
TNR

##PNN

library(pnn)

result$pred3 = result[,length]
result$pred3 = -1
nn2 <- smooth(learn(train,category.column = length), sigma = 1.5)
 
 
 for(i in 1:nrow(test))
 {             
     vec <- as.matrix(test[i,])
     res <- guess(nn2, vec)
     
     if(is.na(res))
     {
         cat("Entry ",i," Generated NaN result!\n")
     }
     else
     {
         result$pred3[i] <- res
     }
 }

result$pred3 <- as.numeric(as.character(result$pred3))

table(true=result$V9, predicted=result$pred3)
cm <- table(true=result$V9, predicted=result$pred3)
sum(diag(cm))/sum(cm)

#True positive rate
TPR <- sensitivity(as.factor(result$pred3), result$V9)
TPR
#True negative rate
TNR <- specificity(as.factor(result$pred3), result$V9)
TNR

##ensemble- voting strategy

#convert pred1 from factor to numeric
result$pred1 <- as.numeric(as.character(result$pred1))

 for(i in 1:nrow(test))
 {	 
     if(result$pred1[i] + result$pred2[i] + result$pred3[i] <= 1)
     {
         result$pred[i] <- 0
     }
     else if (result$pred1[i] + result$pred2[i] + result$pred3[i] > 1)

     {
         result$pred[i] <- 1
     }
 }

table(true=result$V9, predicted=result$pred)
cm <- table(true=result$V9, predicted=result$pred)
sum(diag(cm))/sum(cm)
#True positive rate
TPR <- sensitivity(as.factor(result$pred), result$V9)
TPR
#True negative rate
TNR <- specificity(as.factor(result$pred), result$V9)
TNR


##wine dataset
library(grnn)

#Read wine dataset
data = read.csv("C:\\Users\\huangfuxing\\Documents\\NUS\\Computational intelligence\\CA1\\winequality-white.csv", header=TRUE)
size=nrow(data)
length=ncol(data)
index <- 1:size

#set the seed to make the partition reproductible
set.seed(1)
#75% data for training
positions <- sample(index, trunc(size * 0.75))

train <- data[positions,]
test <- data[-positions,1:length-1]
result = data[-positions,]

#GRNN

#train the model
nn1 <- learn(train, variable.column=length)
nn1 <- smooth(nn1, sigma = 1)

#perform predict
for(i in 1:nrow(test))
{	
    vec <- as.matrix(test[i,])
    res <- guess(nn1, vec)
    
    if(is.nan(res))
    {
        cat("Entry ",i," Generated NaN result!\n")
    }
    else
    {
        result$pred1[i] <- res
    }
}

#append pred1 to result
result$pred1=round(result$pred1)

#create confusion matrix and calculate accuracy
table(true=result$quality, predicted=result$pred1)
cm <- table(true=result$quality, predicted=result$pred1)
sum(diag(cm))/sum(cm)


#PNN
library(pnn)
result$pred2 = result[,length]
result$pred2 = -1
nn2 <- smooth(learn(train,category.column = length), sigma = 0.2)
for(i in 1:nrow(test))
{             
    vec <- as.matrix(test[i,])
    res <- guess(nn2, vec)
    
    if(is.na(res))
    {
        cat("Entry ",i," Generated NaN result!\n")
    }
    else
    {
        result$pred2[i] <- res
    }
}
result$pred2 <- as.numeric(as.character(result$pred2))

table(true=result$quality, predicted=result$pred2)
 cm <- table(true=result$quality, predicted=result$pred2)
 sum(diag(cm))/sum(cm)

##BP NET
library(nnet)
library(neuralnet)
data$quality <- as.factor(data$quality)
train <- data[positions,]
test <- data[-positions,1:length-1]

bpnet <- nnet(quality ~ ., data=train, size=20, maxit=10000)
pred3<- predict(bpnet, test, type="class")

result$pred3=as.factor(pred3)
table(true=result$quality, predicted=result$pred3)

##ensemble NN

#convert pred3 from vector to numeric
result$pred3 <- as.numeric(as.character(result$pred3))

 for(i in 1:nrow(test))
 {	 
    if ( result$pred2[i] == -1)
      {
        pred.ave <- (result$pred1[i]  + result$pred3[i]) / 2
        result$pred[i] <- round(pred.ave)
      }
    else
     { pred.ave <- (result$pred1[i] + result$pred2[i] + result$pred3[i]) / 3  
       result$pred[i] <- round(pred.ave) 
     }
  }

table(true=result$quality, predicted=result$pred)

	