# Purchase-Prediction-Model-for-an-Advertising-Campaign-on-a-Social-Network
Purchase Prediction Model for an Advertising Campaign on a Social Network - Own Project Capstone HarvardX 125.9



```markdown
# Predictive Model Evaluation

This document provides a step-by-step analysis of a predictive model for a social network advertising campaign. The analysis involves data exploration, model training, performance evaluation, and visualization of results using R.

## Getting Started

Before we proceed, make sure to install and load the required R libraries:

```R
library(caTools)
library(class)
library(randomForest)
library(caret)
library(ggplot2)
```

## Data Retrieval

We begin by retrieving the dataset from a specified URL and loading it into R. The dataset comprises user demographic data and their purchasing behavior.

```R
# Define the direct download URL for the dataset
download_url <- "https://drive.google.com/uc?export=download&id=183CuUb08gcK5s3Sf1OToDTu-ZYn-89pX"

# Download the dataset file to the local directory
download.file(download_url, destfile = "Social_Network_Ads.csv", mode = "wb")

# Read the dataset into R
dataset <- read.csv('Social_Network_Ads.csv')
```

## Data Exploration

We begin by exploring the dataset to gain insights. This includes examining statistical summaries, data structure, correlation analysis, and visualizations of key variables.

### Summary Statistics

```R
summary(dataset)
```

### Data Structure

```R
str(dataset)
```

### Correlation Analysis

```R
cor(dataset[, c("Age", "EstimatedSalary", "Purchased")])
```

### Data Visualization

We create various visualizations to better understand the data:

- Age Distribution

```R
# Visualization of the age distribution
ggplot(dataset, aes(x = Age)) +
  geom_histogram(binwidth = 1, fill = "blue", color = "black") +
  ggtitle("Age Distribution") +
  xlab("Age") +
  ylab("Frequency")
```

- Estimated Salary Distribution

```R
# Visualization of the estimated salary distribution
ggplot(dataset, aes(x = EstimatedSalary)) +
  geom_histogram(binwidth = 5000, fill = "green", color = "black") +
  ggtitle("Estimated Salary Distribution") +
  xlab("Estimated Salary") +
  ylab("Frequency")
```

- Relationship Between Age and Estimated Salary

```R
# Scatter plot to visualize the relationship between age and estimated salary
ggplot(dataset, aes(x = Age, y = EstimatedSalary, color = factor(Purchased))) +
  geom_point() +
  ggtitle("Relationship Between Age and Estimated Salary") +
  xlab("Age") +
  ylab("Estimated Salary") +
  scale_color_manual(values = c("red", "green"), labels = c("Not Purchased", "Purchased"))
```

- Class Balance for the Target Variable 'Purchased'

```R
# Visualization of class balance for the target variable 'Purchased'
ggplot(dataset, aes(x = factor(Purchased))) +
  geom_bar(fill = "orange", color = "black") +
  ggtitle("Class Balance for Purchases") +
  xlab("Purchased (0 = No, 1 = Yes)") +
  ylab("Frequency")
```

- Gender Distribution

```R
# Visualization of gender distribution
ggplot(dataset, aes(x = Gender)) +
  geom_bar(fill = "purple", color = "black") +
  ggtitle("Gender Distribution") +
  xlab("Gender") +
  ylab("Frequency")
```

## Data Preprocessing

To prepare the data for modeling, we select relevant columns, encode the 'Purchased' variable as a factor, split the dataset into training and test sets, and perform feature scaling.

```R
# Select relevant columns: Age, EstimatedSalary, and Purchased
dataset <- dataset[3:5]

# Encode the 'Purchased' variable as a factor
dataset$Purchased <- factor(dataset$Purchased, levels = c(0, 1))

# Split dataset into Training and Test sets with a 75% split ratio
set.seed(123)  # Ensure reproducibility
split <- sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Apply feature scaling to the Age and EstimatedSalary columns
training_set[-3] <- scale(training_set[-3])
test_set[-3] <- scale(test_set[-3])
```

## Model Evaluation

We define an evaluation function to assess model performance using accuracy and Cohen's Kappa. Then, we train two models: K-Nearest Neighbors (K-NN) and Random Forest.

```R
# Define a function to evaluate and return model performance metrics

evaluate_model <- function(predictions, actual) {
  cm <- confusionMatrix(as.factor(predictions), as.factor(actual))
  return(list(accuracy = cm$overall['Accuracy'], kappa = cm$overall['Kappa']))
}

# Fit a K-Nearest Neighbors (KNN) model to the training data.

knn_pred <- knn(train = training_set[, -3], test = test_set[, -3], cl = training_set[, 3], k = 5)

# Fit a Random Forest classifier to the training data.

set.seed(123)  # Set seed again for consistency in random forest results.
rf_classifier <- randomForest(x = training_set[-3], y = training_set$Purchased, ntree = 500)

# Use the fitted Random Forest classifier to make predictions on the test set.

rf_pred <- predict(rf_classifier, newdata = test_set[-3])

# Evaluate the performance of the KNN model.

knn_performance <- evaluate_model(knn_pred, test_set$Purchased)

# Evaluate the performance of the Random Forest model in a similar manner.

rf_performance <- evaluate_model(rf_pred, test_set$Purchased)

# Print out accuracy and kappa statistics for both models
cat("K-NN Accuracy:", knn_performance$accuracy, "Kappa:", knn_performance$kappa, "\n")
cat("Random Forest Accuracy:", rf_performance$accuracy, "Kappa:", rf_performance$kappa, "\n")

# Determine and print which model is more efficient based on accuracy
efficient_model <- ifelse(knn_performance$accuracy > rf_performance$accuracy, "K-NN", 
                          ifelse(knn_performance$accuracy < rf_performance$accuracy, "Random Forest", "Both"))
cat("The more efficient model based on accuracy is:", efficient_model, "\n")

# Compare models based on Kappa statistic and print the result
best_kappa_model <- ifelse(knn_performance$kappa > rf_performance$kappa, "K-NN", 
                           ifelse(knn_performance$kappa < rf_performance$kappa, "Random Forest", "Both"))
cat("The model with the better Kappa score is:", best_kappa_model, "\n")

```

## Results

We visualize the classification results for both models using `ggplot2` and compare their performance metrics in a bar chart.

### K-NN Classification Results

```R
# Visualize K-NN classification results
print(plot_model_results(test_set, knn_pred, "K-NN"))
```

### Random Forest Classification Results

```R
# Visualize Random Forest classification results
print(plot
