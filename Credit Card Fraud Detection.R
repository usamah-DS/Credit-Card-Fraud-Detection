# Load required libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ROSE)) install.packages("ROSE", repos = "http://cran.us.r-project.org")
if(!require(pROC)) install.packages("pROC", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org") # For decision tree

library(tidyverse)
library(caret)
library(ROSE)
library(pROC)
library(rpart)

# Load the dataset
credit_data <- read.csv("C:/Users/usama/Downloads/Credit Card Fraud Detection/creditcard.csv")

# Inspect the dataset
str(credit_data)
summary(credit_data)

# Check for missing values
sum(is.na(credit_data))

# Explore the distribution of the target variable (fraudulent vs. non-fraudulent transactions)
credit_data %>%
  group_by(Class) %>%
  summarise(count = n()) %>%
  ggplot(aes(x = as.factor(Class), y = count, fill = as.factor(Class))) +
  geom_bar(stat = "identity") +
  labs(title = "Class Distribution (0 = Non-fraudulent, 1 = Fraudulent)", x = "Class", y = "Count")

# Handling class imbalance using ROSE
balanced_data <- ROSE(Class ~ ., data = credit_data, seed = 1)$data

# Verify the new class distribution
balanced_data %>%
  group_by(Class) %>%
  summarise(count = n()) %>%
  ggplot(aes(x = as.factor(Class), y = count, fill = as.factor(Class))) +
  geom_bar(stat = "identity") +
  labs(title = "Balanced Class Distribution", x = "Class", y = "Count")

# Split the data into training and testing sets (80% training, 20% testing)
set.seed(123)
train_index <- createDataPartition(balanced_data$Class, p = 0.8, list = FALSE)
train_data <- balanced_data[train_index, ]
test_data <- balanced_data[-train_index, ]

# Logistic Regression Model
log_model <- glm(Class ~ ., data = train_data, family = binomial)

# Predict on the test set using logistic regression
log_preds <- predict(log_model, test_data, type = "response")
log_preds_class <- ifelse(log_preds > 0.5, 1, 0)

# Evaluate Logistic Regression Model
log_cm <- confusionMatrix(as.factor(log_preds_class), as.factor(test_data$Class))
print(log_cm)

# ROC Curve for Logistic Regression
roc_curve_log <- roc(test_data$Class, log_preds)
plot(roc_curve_log, col = "blue", main = "ROC Curve for Logistic Regression")

# AUC for Logistic Regression
log_auc <- auc(roc_curve_log)
cat("Logistic Regression AUC:", log_auc, "\n")

# Decision Tree Model
tree_model <- rpart(Class ~ ., data = train_data, method = "class")

# Predict on the test set using the Decision Tree
tree_preds <- predict(tree_model, test_data, type = "class")

# Evaluate Decision Tree Model
tree_cm <- confusionMatrix(tree_preds, as.factor(test_data$Class))
print(tree_cm)

# ROC Curve for Decision Tree
tree_probs <- predict(tree_model, test_data, type = "prob")[, 2]
roc_curve_tree <- roc(test_data$Class, tree_probs)
plot(roc_curve_tree, col = "red", add = TRUE)

# AUC for Decision Tree
tree_auc <- auc(roc_curve_tree)
cat("Decision Tree AUC:", tree_auc, "\n")
