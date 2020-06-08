# Load libraries
if ("pacman" %in% rownames(installed.packages()) == FALSE) {
  install.packages("pacman")
} else {
  pacman::p_load(lattice, ggplot2, caret, readr, corrplot, e1071, reshape2, rstudioapi,
                 dplyr, plotly, arules, arulesViz)
}

# Load data
current_path <- rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path))

Current_Products <- read_csv("existingproductattributes2017.csv")
New_Products <- read_csv("newproductattributes2017.csv")

Transactions <-
  read.transactions("ElectronidexTransactions2017.csv", format = "basket",
                    sep = ",", rm.duplicates = TRUE)
Items_Sheet <- read_csv("electroindexitems - Sheet.csv")

# Getting to know the data
str(Current_Products)
str(New_Products)
summary(Current_Products)
summary(New_Products)

# Search for duplicated rows
ReviewCols <- grep("Review|Type", names(Current_Products), value = TRUE)

idx <- duplicated(Current_Products[, c(ReviewCols)]) |
  duplicated(Current_Products[, c(ReviewCols)], fromLast = TRUE)

duplicated <- Current_Products[idx, ]

# Merge duplicated rows by price mean into a single row
duplicated$Price <- mean(duplicated$Price)
duplicated <- duplicated[c(1), ]

# Remove duplciated rows
Current_Products <- Current_Products[!idx, ]

# Add merged row
Current_Products <- rbind(Current_Products, duplicated)

# NA
NAColumns <- colnames(Current_Products)[colSums(is.na(Current_Products)) > 0]

for (i in NAColumns) {
  print(paste(i, sum(is.na(Current_Products[, i]))))
}

# Remove BestSellersRank since has many NAs
Current_Products$BestSellersRank <- NULL

# Dummify ProductType
Current_Products$ProductType <- as.factor(Current_Products$ProductType)
DummyC <- dummyVars(" ~ .", data = Current_Products, levelsOnly = TRUE)
Current_Products_Dummy <- data.frame(predict(DummyC, newdata = Current_Products))

New_Products$ProductType <- as.factor(New_Products$ProductType)
DummyN <- dummyVars(" ~ .", data = New_Products, levelsOnly = TRUE)
New_Products_Dummy <- data.frame(predict(DummyN, newdata = New_Products))

# Correlation matrix
corrData <- cor(Current_Products_Dummy)
corrplot(corrData, type = "lower", tl.col = "black", tl.srt = 10, tl.cex = 0.8)

# Feature selection
#star features removed because of colinearity between them
#profit margin removed because its not a value that can be perceived by the final client
#productnum removed since it is an index
Current_Products_Dummy <- subset(Current_Products_Dummy,
                                 select = -c(x5StarReviews, x3StarReviews, x1StarReviews,
                                             ProductNum, ProfitMargin))

New_Products_Dummy <- subset(New_Products_Dummy,
                             select = -c(x5StarReviews, x3StarReviews, x1StarReviews,
                                         ProductNum, ProfitMargin))

# Outliers
outliers <- boxplot(Current_Products_Dummy$Volume)$out
outliers

# Save rows with outliers
Current_Products_Dummy_Out <-
  Current_Products_Dummy[which(Current_Products_Dummy$Volume %in% outliers), ]

# Remove outliers for modeling
Current_Products_Dummy <-
  Current_Products_Dummy[-which(Current_Products_Dummy$Volume %in% outliers), ]

# Check for 0 variance rows
names(which(apply(Current_Products_Dummy, 2, var) == 0))

# Remove GameConsole since all it's observations where outliers
Current_Products_Dummy$GameConsole <- NULL
Current_Products_Dummy_Out$GameConsole <- NULL

# Modeling
# Volume prediction
# Define Train and Test sets
set.seed(123);inTraining <-
  createDataPartition(Current_Products_Dummy$Volume, p = .75, list = FALSE)

training <- Current_Products_Dummy[inTraining, ]
testing <- Current_Products_Dummy[-inTraining, ]

# Cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

# Random forest
set.seed(123);rfFit1 <-
  train(Volume ~., data = training, method = "rf", trControl = fitControl, metric = "MAE")

RF_Train <- round(postResample(predict(rfFit1, training), training$Volume), 2)

PredictionsrfFit1 <- predict(rfFit1, testing)
RF_Test <- round(postResample(PredictionsrfFit1, testing$Volume), 2)

# Svm
set.seed(123);svmFit1 <- svm(Volume ~., data = training, fitted = TRUE, kernel = "linear")

SVM_Train <- round(postResample(predict(svmFit1, training), training$Volume), 2)

PredictionssvmFit1 <- predict(svmFit1, testing)
SVM_Test <- round(postResample(PredictionssvmFit1, testing$Volume), 2)

# Knn
set.seed(123);knnFit1 <-
  train(Volume ~., data = training, method = "knn", trControl = fitControl)

KNN_Train <- round(postResample(predict(knnFit1, training), training$Volume), 2)

PredictionsknnfFit1 <- predict(knnFit1, testing)
KNN_Test <- round(postResample(PredictionsknnfFit1, testing$Volume), 2)

# Error metrics table
metrics <- data.frame(matrix(ncol = 5, nrow = 0))
colnames(metrics) <- c("RMSE", "Rsquared", "MAE", "Model", "Set")

metrics[nrow(metrics) + 1, ] <- c(RF_Train, "RF", "Train")
metrics[nrow(metrics) + 1, ] <- c(RF_Test, "RF", "Test")
metrics[nrow(metrics) + 1, ] <- c(SVM_Train, "SVM", "Train")
metrics[nrow(metrics) + 1, ] <- c(SVM_Test, "SVM", "Test")
metrics[nrow(metrics) + 1, ] <- c(KNN_Train, "KNN", "Train")
metrics[nrow(metrics) + 1, ] <- c(KNN_Test, "KNN", "Test")

metrics

# New products predictions
NewpPredictionsrfFit1 <- predict(rfFit1, New_Products_Dummy)

# Add predicted Volume
New_Products["Volume"] <- round(NewpPredictionsrfFit1, 0)

# Calculate predicted Profit
New_Products["Profit"] <-
  round(New_Products$ProfitMargin * New_Products$Volume * New_Products$Price, 0)

# New product predictions
New_Products_T <- subset(New_Products, select = c(ProductType, ProductNum, Price,
                                                  ProfitMargin, Volume, Profit))
New_Products_T <- New_Products_T[order(New_Products_T$Profit, decreasing = TRUE), ]

# Total predicted Profit per ProductType
ggplot(New_Products,
       aes(x = New_Products$ProductType,
           y = New_Products$Profit,
           fill = New_Products$ProductType)) +
  geom_bar(stat = "identity") +
  labs(fill = "Product Type", title = "Total Predicted Profit Per Product Type") +
  scale_x_discrete(name = "Product Type") +
  scale_y_continuous(name = "Profit in ???", breaks = seq(0, 120000, 20000)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Total predicted Volume per ProductType
ggplot(New_Products,
       aes(x = New_Products$ProductType,
           y = New_Products$Volume,
           fill = New_Products$ProductType)) +
  geom_bar(stat = "identity") +
  labs(fill = "Product Type" ,title = "Total Predicted Volume Per Product Type") +
  scale_x_discrete(name = "Product Type") +
  scale_y_continuous(name = "Volume", breaks = seq(0, 4500, 500)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Add outliers for descriptive plots
Current_Products_Dummy <- rbind(Current_Products_Dummy, Current_Products_Dummy_Out)

# Exploratory analysis
# Scatter plot - (4 Star/ 2 Star / Pos REviews / Neg Reviews) vs Volume
v1 <- c("x4StarReviews", "x2StarReviews", "PositiveServiceReview", "NegativeServiceReview")

for (i in v1) {
  x <- Current_Products_Dummy[[i]]
  plot(Current_Products_Dummy$Volume, x, pch = (19), col = "turquoise3", frame.plot = FALSE,
       xlab = "Volume", ylab = i, main = paste(i, "& Volume"), cex.lab = 1.5, cex.main = 1.5)
  abline(lm(x ~ Current_Products_Dummy$Volume),
         col = "tomato1", lwd = 3)
  }

# Barplot star reviews per ProductType
Current_Products_Star <- Current_Products[, c("ProductType", "x5StarReviews",
                                              "x4StarReviews", "x3StarReviews",
                                              "x2StarReviews", "x1StarReviews")] %>%
  melt(id.vars = "ProductType") %>%
  group_by(ProductType, variable) %>% 
  summarise(value = sum(value)) %>%
  mutate(freq = value / sum(value))

ggplot(Current_Products_Star, aes(ProductType, freq)) +
  geom_bar(aes(fill = variable), position = "dodge", stat = "identity") +
  labs(fill = "Star Reviews", title = "Star Reviews Per Product Type") +
  scale_x_discrete(name = "Product Type") +
  scale_y_continuous(name = "Frequency") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Barplot service reviews per ProductType
Current_Products_Review <- Current_Products[, c("ProductType", "PositiveServiceReview",
                                                "NegativeServiceReview")] %>%
  melt(id.vars = "ProductType") %>%
  group_by(ProductType, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(freq = value / sum(value))

ggplot(Current_Products_Review, aes(ProductType, freq)) +
  geom_bar(aes(fill = variable), position = "dodge", stat = "identity") +
  labs(fill = "Service Reviews", title = "Service Reviews Per Product Type") +
  scale_x_discrete(name = "Product Type") +
  scale_y_continuous(name = "Frequency") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Volume by ProductType
Current_Products_Sales <- subset(Current_Products, select = c(Volume, ProductType))

Current_Products_Sales <-
  aggregate(Current_Products_Sales$Volume,
            by = list(Category = Current_Products_Sales$ProductType), FUN = sum)

# Plot Volume by ProductType
ggplot(Current_Products_Sales, aes(x = reorder(Category, -x), y = x)) +
  geom_bar(stat = "identity", fill = "tomato1", colour = "black") +
  labs(fill = "Product Type", title = "Blackwell Sales by Product Type") +
  scale_x_discrete(name = "") +
  scale_y_continuous(name = "", breaks = seq(0, 25000, 5000)) +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5),
        panel.grid  = element_blank(),
        panel.background = element_rect(fill = "transparent"),
        axis.ticks = element_blank())

# Electronidex
# Getting to know the data
summary(Transactions)

# Plot transaction size
transactionSize <- size(Transactions)

h <- hist(transactionSize, col = "cadetblue", breaks = 30, xaxt = "n", ylim = c(0, 2500),
          xlab = "Transaction Size", main = "Transaction Size Histogram", las = 1)
axis(1, at = seq(0, 30, by = 1))
text(h$mids,h$counts,labels = h$counts, adj = c(0, -0.5), srt = 45)

# Plot item frequency
itemFrequencyPlot(Transactions, type = c("absolute"), topN = 10, main = "Most Sold Items",
                  ylab = "", col = "turquoise3", yaxt = "n")
axis(2, at = seq(0, 2500, by = 500), tick = FALSE, las = 2, line = -1)

# Generate association rules for items
Item_Rules <- apriori(Transactions,
                      parameter = list(supp = 0.004, conf = 0.65, minlen = 2))

Item_Rules <- sort(subset(Item_Rules, subset = lift > 2.7), by = "lift")

Item_Rules <- Item_Rules[!is.redundant(Item_Rules)]

# Plot item rules
plot(Item_Rules, method = "graph", control = list(type = "items"), shading = "confidence")

plot(Item_Rules, method = "paracoord", control = list(reorder = TRUE))

# Add ProductType
Items_Sheet <- Items_Sheet[order(Items_Sheet$Item), ]

Transactions_ProductType <- aggregate(Transactions, Items_Sheet$Category)

# Plot ProductType frequency
itemFrequencyPlot(Transactions_ProductType, type = c("absolute"), topN = 17,
                  main = "Electronidex Sales by Product Type", ylab = "",
                  col = "turquoise3", yaxt = "n")
axis(2, at = seq(0, 6000, by = 1000), tick = FALSE, las = 2, line = -1)

# Generate association rules for product types
ProductType_Rules <- apriori (Transactions_ProductType,
                              parameter = list(supp = 0.08, conf = 0.5, minlen = 2))

ProductType_Rules <- ProductType_Rules[!is.redundant(ProductType_Rules)]

# Plot product type rules
plot(ProductType_Rules, method = "graph", control = list(type = "items"))

plot(ProductType_Rules, method = "paracoord",
     control = list(reorder = TRUE), main = "Frequently sold together")
