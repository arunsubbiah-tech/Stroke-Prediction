install.packages("pacman")
pacman::p_load(pacman,tidyverse)
library(caret)
library(e1071)
library(dplyr)
library(kernlab)
install.packages('ggfortify')
library(ggfortify)
install.packages("kknn")
library(kknn)
library(stats)
install.packages("ggplot2")
library(factoextra)
library(cluster)
library(ggplot2)

heart <- read.table("healthcare-dataset-stroke-data.csv",header = T,sep = ",",stringsAsFactors = T)
summary(heart)
#removing na values



heart[heart == 'N/A'] <- NA 
heart2 = na.omit(heart)
which(is.na(heart))
heart2$bmi = as.numeric(heart2$bmi)
summary(heart2)

#visualization

ggplot(data = heart2, aes(x = bmi)) + 
  geom_histogram(binwidth = 30)

ggplot(heart2, aes(x=as.factor(stroke), y=age)) + 
  geom_boxplot()

ggplot(heart2, aes(x=as.factor(stroke), fill=smoking_status))+
  geom_bar(position = position_dodge())

ggplot(heart2, aes(x=as.factor(stroke), fill=ever_married))+
  geom_bar(position = position_dodge())

ggplot(heart2, aes(x=as.factor(stroke), fill=Residence_type))+
  geom_bar(position = position_dodge())

ggplot(heart2, aes(x=as.factor(stroke), y=avg_glucose_level)) +
  geom_boxplot(fill='steelblue')

ggplot(heart2, aes(x=as.factor(stroke), fill=gender))+
  geom_bar(position = position_dodge())

ggplot(heart2, aes(x=as.factor(stroke), fill=work_type))+
  geom_bar(position = position_dodge())

ggplot(heart2, aes(x=as.factor(stroke), fill=as.factor(hypertension )))+
  geom_bar(position = position_dodge())

ggplot(heart2, aes(x=as.factor(stroke), fill=as.factor(heart_disease)))+
  geom_bar(position = position_dodge())

ggplot(heart2, aes(x = as.factor(stroke)))+
  geom_bar()

summary(heart2)
colnames(heart2)

plot(heart2$age,heart2$stroke)




#processing
#na values
#outliers
heart2$bmi = as.numeric(heart2$bmi)
heart2$hypertension = as.factor(heart2$hypertension)
heart2$heart_disease = as.factor(heart2$heart_disease)

ggplot(data = heart2, aes(x = avg_glucose_level)) + 
  geom_histogram()


bmi <- scale(heart2$bmi)
summary(bmi)
hist(bmi)


# breaks = 3 gives us 3 equal width bins
heart3 <- heart2 %>%
  mutate(glucose_level_factor = cut(avg_glucose_level, breaks = 3,
                        labels=c("low","medium","high")))
head(heart3)

# Mutate and store each
low <- heart3 %>% 
  filter(glucose_level_factor == 'low') %>% 
  mutate(avg_glucose_level = median(avg_glucose_level, na.rm = T))
medium <- heart3 %>% 
  filter(glucose_level_factor == 'medium') %>% 
  mutate(avg_glucose_level = median(avg_glucose_level, na.rm = T))
high <- heart3 %>% 
  filter(glucose_level_factor == 'high') %>% 
  mutate(avg_glucose_level = median(avg_glucose_level, na.rm = T))

# The resulting set for each pipeline is immutable and therefore need to be concatenated
# Tidyverse has a bind_rows function that helps us combine these separate sets
heart_copy <- bind_rows(list(low, medium, high))

summary(heart3)
summary(heart_copy)
summary(heart2)

remove(heart_copy)
ggplot(data = heart2, aes(x = avg_glucose_level)) + 
  geom_histogram(binwidth = 50)

ggplot(data = heart_copy, aes(x = avg_glucose_level)) + 
  geom_histogram()
head(heart_copy)

heart3$avg_glucose_level = heart_copy$avg_glucose_level
heart3<-heart3[,c(-13)]
view(heart_copy)
summary(heart3)


#connverting dummy variables 
heart_1 = heart3[,(-12)]
num_heart=dummyVars("~.", data=heart_1)
heart_dum=data.frame(predict(num_heart,newdata=heart_1))

summary(heart_dum)

#pca

heart.pca <- prcomp(heart_dum,center = T,scale. = T)
summary(heart.pca)

screeplot(heart.pca, type = "l") + title(xlab = "PCs")

heart_pca1 = as.data.frame(heart.pca$x)

heart_data1 = as.data.frame(heart.pca$x)
heart_data1$storke <- as.factor(heart2$stroke)

#visualization of data
ggplot(data = heart_data1, aes(x = PC1, y = PC2, col = storke)) + geom_point()+
  scale_color_manual(values=c('cornsilk3','cadetblue4'))







preproc <- preProcess(heart_dum, method=c("center", "scale"))
heart1 <- predict(preproc, heart_dum)


#HAC
dist_mat <- dist(heart_dum, method = 'euclidean')

hfit <- hclust(dist_mat, method = 'average')
plot(hfit)

fviz_nbclust(heart_dum, FUN = hcut, method = "wss")
fviz_nbclust(heart_dum, FUN = hcut, method = "silhouette")

h3 <- cutree(hfit, k=2)
fviz_cluster(list(data = heart_dum, cluster = h3))

heart_data1$Clusters = as.factor(h3)
# Plot and color by labels
ggplot(data = heart_data1, aes(x = PC1, y = PC2, col = Clusters)) + geom_point()




#kmeans
fviz_nbclust(heart1, kmeans, method = "wss")

fviz_nbclust(heart1, kmeans, method = "silhouette")


# Fit the data
fit_kmeans <- kmeans(heart1, centers = 10, nstart = 25)
# Display the kmeans object information
fit_kmeans

fviz_cluster(fit_kmeans, data = heart1)

view(heart1)

#classification

heart_dum$storke = as.factor(heart2$stroke)  
summary(heart_dum)


ctrl <- trainControl(method="cv", number = 10) 
heart_knn <- train(storke ~ ., data = heart_dum, 
                  method = "knn", 
                  trControl = ctrl, 
                  preProcess = c("center","scale"))
#Output of kNN fit

heart_pred_knn <- predict(heart_knn,heart_dum)
confusionMatrix(heart_dum$storke, heart_pred_knn)
heart_knn

#visualization

heart_data1$restul1 <- heart_pred_knn 
ggplot(heart_data1,aes(x=PC1,y=PC2,group=restul1))+
  geom_point(aes(color=restul1))+
  scale_color_manual(values=c('cornsilk3','orange3'))





#decision tree
hypers = rpart.control(minsplit = 5000, maxdepth = 4, minbucket = 2500)
heart_tree <- train(storke ~ ., data = heart_dum, method = "rpart1SE",control = hypers, trControl = ctrl)
heart_pred_tree <- predict(heart_tree,heart_dum)
confusionMatrix(heart_dum$storke, heart_pred_tree)

  
view(heart_dum)
summary(heart_dum)
colnames(heart_dum)

library(tidyverse)
library(rattle)
library(ggplot2)
library(pROC)




ctrl <- trainControl(method="cv", number = 10) 
heart_knn <- train(storke ~ ., data = heart_dum, 
                   method = "knn", 
                   trControl = ctrl, 
                   preProcess = c("center","scale"))
#Output of kNN fit

heart_pred_knn <- predict(heart_knn,heart_dum)
cm <- confusionMatrix(heart_dum$storke, heart_pred_knn)
# Store the byClass object of confusion matrix as a dataframe
metrics <- as.data.frame(cm$byClass)
# View the object
metrics
library(pROC)
# Get the precision value for each class
metrics %>% select(row("Precision"))


summary(heart_dum)

index = createDataPartition(y=heart_dum$storke, p=0.7, list=FALSE)
# Everything in the generated index list
train_pima = heart_dum[index,]
# Everything except the generated indices
test_pima = heart_dum[-index,]

# Set control parameter
train_control = trainControl(method = "cv", number = 10)
# Fit the model
knn <- train(storke ~., data = train_pima, method = "knn", trControl = train_control, tuneLength = 20)
# Evaluate fit
knn


# Evaluate the fit with a confusion matrix
pred_pima <- predict(knn, test_pima)
# Confusion Matrix
CM <- confusionMatrix(test_pima$storke, pred_pima)


# Store the byClass object of confusion matrix as a dataframe
metrics2 <- as.data.frame(CM$byClass)
# View the object
metrics2
library(pROC)
# Get the precision value for each class
metrics %>% select(row("Precision"))


library(pROC)
# Get class probabilities for KNN
pred_prob <- predict(knn, test_pima, type = "prob")
head(pred_prob)

# And now we can create an ROC curve for our model.
roc_obj <- roc((test_pima$storke), pred_prob[,1])
plot(roc_obj, print.auc=TRUE)