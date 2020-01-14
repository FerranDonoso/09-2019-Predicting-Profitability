#load libraries
if(!require(pacman)){
  install.packages("pacman")
  library(pacman)
}

pacman::p_load(lattice, ggplot2, caret, readr, corrplot, e1071, reshape2)

#load data sets
getwd()
existingproductattributes2017 <- read_csv("existingproductattributes2017.csv")
View(existingproductattributes2017)

newproductattributes2017 <- read_csv("newproductattributes2017.csv")
View(newproductattributes2017)

#search for duplicated rows
existingproductattributes2017$ID <- seq.int(nrow(existingproductattributes2017))

n_occur <- data.frame(table(existingproductattributes2017$x5StarReviews))

#extended warranty has the same product with multiple prices
duplicates <- existingproductattributes2017[existingproductattributes2017$x5StarReviews %in%
                                      n_occur$Var1[n_occur$Freq > 5], ]

#merge duplicated rows by price mean into a single row
duplicatedRows <- duplicates$ID
duplicates$Price <- mean(duplicates$Price)
duplicates <- duplicates[c(1), ]

#remove duplciated rows
existingproductattributes15 <- existingproductattributes2017[-c(duplicatedRows), ]

#add merged row
existingproductattributes15 <- rbind(existingproductattributes15, duplicates)

#getting to know the data
str(existingproductattributes2017)
str(newproductattributes2017)
summary(newproductattributes2017)
summary(existingproductattributes2017)

#check for NA - BestSellersRank
NAColumns <- colnames(existingproductattributes2017)[colSums(is.na(existingproductattributes2017)) > 0]
NAColumns
for (i in NAColumns) {
  print(sum(is.na(existingproductattributes2017[,i])))
}

#check for outliers
numerical <- unlist(lapply(existingproductattributes2017, is.numeric))
colnames <- colnames(existingproductattributes2017[ , numerical])

for (i in unique(colnames)) {
  print(boxplot(existingproductattributes2017[,i])$out)
}

#dummify product type
existingproductattributes2017$ProductType <- as.factor(existingproductattributes2017$ProductType)
existingproductattributes11 <- dummyVars(" ~ .", data = existingproductattributes2017)
existingproductattributes12 <- data.frame(predict(existingproductattributes11, newdata = existingproductattributes2017))
View(existingproductattributes12)

newproductattributes2017$ProductType <- as.factor(newproductattributes2017$ProductType)
newproductattributes11 <- dummyVars(" ~ .", data = newproductattributes2017)
newproductattributes12 <- data.frame(predict(newproductattributes11, newdata = newproductattributes2017))
View(newproductattributes12)

#correlation matrix
#remove BestSellersRank since has many NAs
existingproductattributes12$BestSellersRank <- NULL

corrData <- cor(existingproductattributes12)
corrData
corrplot(corrData)

#star features removed because of colinearity between them
#profit margin removed because its not a value that can be perceived by the final client
#productnum removed since it is an index
existingproductattributes12 <- subset(existingproductattributes12, select = -c(x5StarReviews,
                                                                               x3StarReviews,
                                                                               x1StarReviews,
                                                                               ProductNum,
                                                                               ProfitMargin))

newproductattributes12 <- subset(newproductattributes12, select = -c(x5StarReviews,
                                                                               x3StarReviews,
                                                                               x1StarReviews,
                                                                               ProductNum,
                                                                               ProfitMargin))

#volume prediction
#data split, 75% train, 25% test
set.seed(123);inTraining <- createDataPartition(existingproductattributes12$Volume, p = .75, list = FALSE)
training <- existingproductattributes12[inTraining,]
testing <- existingproductattributes12[-inTraining,]

#cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

#random forest
set.seed(123);rfFit1 <- train(Volume~., data = training, method = "rf", trControl = fitControl, tuneLength = 5)
rfFit1

PredictionsrfFit1 <- predict(rfFit1,testing)
postResample(PredictionsrfFit1, testing$Volume)

#predicted vs observed plot
plot(PredictionsrfFit1, type = "l",lwd = 2, col = "tomato1", ylab = "Volume", main = "Random Forest Model",
     ylim = c(0, 8000))
lines(testing$Volume, lwd = 2, col = "turquoise3")
legend(1, 7000, pch = c(19), col = c("tomato1", "turquoise3"), c("Predicted", "Observed"))

#linear model
set.seed(123);lmFit1 <- lm(Volume~., training)
summary(lmFit1)
varImp(lmFit1)

PredictionslmFit1 <- predict(lmFit1, testing)
postResample(PredictionslmFit1, testing$Volume)

#predicted vs observed plot
plot(PredictionslmFit1,type = "l",lwd = 2,col = "tomato1", ylab = "Volume", main = "Linear Model",
     ylim = c(-1000, 6000))
lines(testing$Volume, lwd = 2, col = "turquoise3")
legend(1, 5000, pch = c(19), col = c("tomato1", "turquoise3"), c("Predicted", "Observed"))

#svm
set.seed(123);svmFit1 <- svm(Volume~., data = training, fitted = TRUE)

PredictionssvmFit1 <- predict(svmFit1, testing)
postResample(predict(svmFit1, training), training$Volume)
postResample(PredictionssvmFit1, testing$Volume)

#predicted vs observed plot
plot(PredictionssvmFit1, type = "l", lwd = 2, col = "tomato1", ylab = "Volume", main = "SVM Model",
     ylim = c(0,6000))
lines(testing$Volume, lwd = 2, col = "turquoise3")
legend(1, 5000, pch = c(19), col = c("tomato1", "turquoise3"), c("Predicted", "Observed"))

#knn
set.seed(123);knnFit1 <- train(Volume~., data = training, method = "knn", trControl = fitControl)
knnFit1

PredictionsknnfFit1 <- predict(knnFit1,testing)
postResample(PredictionsknnfFit1, testing$Volume)

#predicted vs observed plot
plot(PredictionsknnfFit1, type = "l", lwd = 2, col = "tomato1", ylab = "Volume", main = "K-NN Model",
     ylim = c(0,6000))
lines(testing$Volume, lwd = 2, col = "turquoise3")
legend(1, 5000, pch = c(19), col = c("tomato1", "turquoise3"), c("Predicted", "Observed"))

#error metrics table to select the best model
#         RMSE      Rsquared        MAE 
#RF   882.1237625   0.6274362   327.8520000 
#LM   1067.5277951  0.6829554   602.0056874 
#SVM  416.6693836   0.6380058   292.7930599 
#KNN  412.5192027   0.6629773   249.2631579

#predictions on new products
newpPredictionssvmFit1 <- predict(svmFit1,newproductattributes12)
newpPredictionssvmFit1

#rounded predicted volume
roundednewpPredictionssvmFit1 <- round(newpPredictionssvmFit1, 0)
newproductattributes2017["PredictedVolume"] <- roundednewpPredictionssvmFit1

#rounded predicted profit
profitroundednewpPredictionssvmFit1 <- newproductattributes2017$ProfitMargin *
  newproductattributes2017$PredictedVolume * newproductattributes2017$Price
roundedprofitroundedPredictionsknnfFit1 <- round(profitroundednewpPredictionssvmFit1, 0)
newproductattributes2017["PredictedProfit"] <- roundedprofitroundedPredictionsknnfFit1


newproductattributes2017 <- subset(newproductattributes2017, select = -c(x5StarReviews, x4StarReviews,
                                                                         x3StarReviews, x2StarReviews,
                                                                         x1StarReviews, Volume,
                                                                         PositiveServiceReview,
                                                                         NegativeServiceReview,
                                                                         Recommendproduct,
                                                                         BestSellersRank, ShippingWeight,
                                                                         ProductDepth, ProductWidth,
                                                                         ProductHeight))
#export data frame to computer
write.csv(newproductattributes2017,"newproductattributes2017complete.csv")

#Total Predicted Profit Per Product Type [SVM]
ggplot(newproductattributes2017,
       aes(x = newproductattributes2017$ProductType,
           y = newproductattributes2017$PredictedProfit,
           fill = newproductattributes2017$ProductType)) +
  geom_bar(stat = "identity") +
  labs(fill = "Product Type" ,title = "Total Predicted Profit Per Product Type [SVM]") +
  scale_x_discrete(name = "Product Type") +
  scale_y_continuous(name = "Profit", breaks = seq(0,500000,50000))

#Total Predicted Volume Per Product Type [SVM]
ggplot(newproductattributes2017,
       aes(x = newproductattributes2017$ProductType,
           y = newproductattributes2017$PredictedVolume,
           fill = newproductattributes2017$ProductType)) +
  geom_bar(stat = "identity") +
  labs(fill = "Product Type" ,title = "Total Predicted Volume Per Product Type [SVM]") +
  scale_x_discrete(name = "Product Type") +
  scale_y_continuous(name = "Volume", breaks = seq(0,500000,50000))

#exploratory analysis
#Scatter plot - (4 Star/ 2 Star / Pos REviews / Neg Reviews) vs Volume
plot(existingproductattributes12$Volume,existingproductattributes12$x4StarReviews,
     xlab = "Volume", ylab = "4 Star Reviews", pch = (19),cex.lab = 1.5,
     main = "4 Star Reviews & Volume", cex.main = 1.5, frame.plot = FALSE, col = "turquoise3")
par(xpd = FALSE)
abline(lm(existingproductattributes12$x4StarReviews ~ existingproductattributes12$Volume),
       col = "tomato1", lwd = 3)

plot(existingproductattributes12$Volume,existingproductattributes12$x2StarReviews,
     xlab = "Volume", ylab = "2 Star Reviews", pch = (19), cex.lab = 1.5,
     main = "2 Star Reviews & Volume", cex.main = 1.5, frame.plot = FALSE, col = "turquoise3")
abline(lm(existingproductattributes12$x2StarReviews ~ existingproductattributes12$Volume),
       col = "tomato1", lwd = 3)

plot(existingproductattributes12$Volume,existingproductattributes12$PositiveServiceReview,
     xlab = "Volume", ylab = "Positive Service Reviews", pch = (19), cex.lab = 1.5,
     main = "Positive Service Reviews & Volume", cex.main = 1.5, frame.plot = FALSE,col = "turquoise3")
abline(lm(existingproductattributes12$PositiveServiceReview ~ existingproductattributes12$Volume),
       col = "tomato1", lwd = 3)

plot(existingproductattributes12$Volume,existingproductattributes12$NegativeServiceReview,
     xlab = "Volume", ylab = "Negative Service Reviews", pch = (19), cex.lab = 1.5,
     main = "Negative Service Reviews & Volume", cex.main = 1.5, frame.plot = FALSE,col = "turquoise3")
abline(lm(existingproductattributes12$NegativeServiceReview ~ existingproductattributes12$Volume),
       col = "tomato1", lwd = 3)

#barplot star reviews per product type
existingproductattributes16 <- existingproductattributes2017
View(existingproductattributes16)

existingproductattributes16 <- subset(existingproductattributes16, select = -c(ProductNum, PositiveServiceReview,
                                                                         NegativeServiceReview, Price,
                                                                         Recommendproduct,
                                                                         BestSellersRank,
                                                                         ShippingWeight,
                                                                         ProductDepth, ProductWidth,
                                                                         ProductHeight, Volume,
                                                                         ProfitMargin))

existingproductattributes17 <- melt(existingproductattributes16, id.vars = "ProductType")

ggplot(existingproductattributes17, aes(ProductType, value)) +
  geom_bar(aes(fill = variable), position = "dodge", stat = "identity") +
  labs(fill = "Service Reviews", title = "Service Reviews Per Product Type") +
  scale_x_discrete(name = "Product Type") + scale_y_continuous(name = "Number of Reviews")

#barplot pos and neg reviews per product type
existingproductattributes18 <- existingproductattributes2017
View(existingproductattributes18)

existingproductattributes18 <- subset(existingproductattributes18, select = -c(Price, Recommendproduct,
                                                                               BestSellersRank,
                                                                               ShippingWeight, ProductNum,
                                                                               ProductDepth, ProductWidth,
                                                                               ProductHeight, Volume,
                                                                               ProfitMargin, x5StarReviews,
                                                                               x4StarReviews, x3StarReviews,
                                                                               x2StarReviews, x1StarReviews))

existingproductattributes19 <- melt(existingproductattributes18, id.vars = "ProductType")

ggplot(existingproductattributes19, aes(ProductType, value)) +
  geom_bar(aes(fill = variable), position = "dodge", stat = "identity") +
  labs(fill = "Service Reviews", title = "Service Reviews Per Product Type") +
  scale_x_discrete(name = "Product Type") + scale_y_continuous(name = "Number of Reviews")

#existing volume
#search for duplicated rows
existingproductattributes2017$ID <- seq.int(nrow(existingproductattributes2017))

n_occur <- data.frame(table(existingproductattributes2017$x5StarReviews))

#extended warranty has the same product with multiple prices
duplicates <- existingproductattributes2017[existingproductattributes2017$x5StarReviews %in%
                                              n_occur$Var1[n_occur$Freq > 5], ]

#merge duplicated rows by price mean into a single row
duplicatedRows <- duplicates$ID
duplicates$Price <- mean(duplicates$Price)
duplicates <- duplicates[c(1), ]

#remove duplciated rows
existingproductattributes20 <- existingproductattributes2017[-c(duplicatedRows), ]

#add merged row
existingproductattributes20 <- rbind(existingproductattributes20, duplicates)

existingproductattributes24 <- existingproductattributes20

#sales by product type
existingproductattributes24 <- subset(existingproductattributes24, select = c(Volume, ProductType))

existingproductattributes26 <- aggregate(existingproductattributes24$Volume,
                                         by = list(Category = existingproductattributes24$ProductType),
                                         FUN = sum)
#plot sales by product type
ggplot(existingproductattributes26,
       aes(x = reorder(Category, -x),
           y = x)) +
  geom_bar(stat = "identity", fill = "tomato1") +
  labs(fill = "Product Type" ,title = "Blackwell Sales by Product Type") +
  scale_x_discrete(name = "") +
  scale_y_continuous(name = "", breaks = seq(0, 25000, 5000)) +
  theme(text = element_text(size = 19), axis.text.x = element_text(angle = 45, hjust = 1, size = 17),
        plot.title = element_text(hjust = 0.5), legend.text = element_text(size = 17),
        panel.grid  = element_blank(),panel.background = element_rect(fill = "transparent"),
        axis.ticks = element_blank(),legend.position = "none")



