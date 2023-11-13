# Load required libraries
library(caTools)
library(class)
library(randomForest)
library(caret)
library(ggplot2)

# Define the direct download URL for the dataset
download_url <- "https://drive.google.com/uc?export=download&id=183CuUb08gcK5s3Sf1OToDTu-ZYn-89pX"

# Download the dataset file to the local directory
download.file(download_url, destfile = "Social_Network_Ads.csv", mode = "wb")

# Read the dataset into R
dataset <- read.csv('Social_Network_Ads.csv')

# Data exploration

# Statistical summary of the dataset
summary(dataset)

# View the structure of the dataset
str(dataset)

# Correlation analysis for numerical variables
cor(dataset[, c("Age", "EstimatedSalary", "Purchased")])

# Visualization of the age distribution
ggplot(dataset, aes(x = Age)) +
  geom_histogram(binwidth = 1, fill = "blue", color = "black") +
  ggtitle("Age Distribution") +
  xlab("Age") +
  ylab("Frequency")

# Visualization of the estimated salary distribution
ggplot(dataset, aes(x = EstimatedSalary)) +
  geom_histogram(binwidth = 5000, fill = "green", color = "black") +
  ggtitle("Estimated Salary Distribution") +
  xlab("Estimated Salary") +
  ylab("Frequency")

# Scatter plot to visualize the relationship between age and estimated salary
ggplot(dataset, aes(x = Age, y = EstimatedSalary, color = factor(Purchased))) +
  geom_point() +
  ggtitle("Relationship Between Age and Estimated Salary") +
  xlab("Age") +
  ylab("Estimated Salary") +
  scale_color_manual(values = c("red", "green"), labels = c("Not Purchased", "Purchased"))

# Visualization of class balance for the target variable 'Purchased'
ggplot(dataset, aes(x = factor(Purchased))) +
  geom_bar(fill = "orange", color = "black") +
  ggtitle("Class Balance for Purchases") +
  xlab("Purchased (0 = No, 1 = Yes)") +
  ylab("Frequency")

# Visualization of gender distribution
ggplot(dataset, aes(x = Gender)) +
  geom_bar(fill = "purple", color = "black") +
  ggtitle("Gender Distribution") +
  xlab("Gender") +
  ylab("Frequency")

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

# Define a function to evaluate and return model performance metrics

evaluate_model <- function(predictions, actual) {
  cm <- confusionMatrix(as.factor(predictions), as.factor(actual))
  return(list(accuracy = cm$overall['Accuracy'], kappa = cm$overall['Kappa']))
}

# This function 'evaluate_model' is designed to assess the performance of a predictive model.
# It takes two arguments: 'predictions' which are the predicted values by the model,
# and 'actual' which are the true values to compare against.

# Inside the function, a confusion matrix is computed using the 'confusionMatrix' function.
# It requires both 'predictions' and 'actual' to be factors, hence the 'as.factor()' conversion.

# The confusion matrix 'cm' includes various performance metrics, but this function specifically extracts
# 'Accuracy' and 'Kappa' from the 'cm$overall' list. These metrics are commonly used to evaluate 
# the performance of classification models.

# 'Accuracy' measures the proportion of the total number of predictions that were correct.
# 'Kappa', also known as Cohen's Kappa, adjusts accuracy by accounting for the possibility of a correct prediction by chance.

# The function returns a list containing the 'accuracy' and 'kappa' metrics, allowing the user to easily access these values.
# This function is useful for quickly obtaining a performance summary of classification models in predictive analytics.


# Initialize the random number generator to a fixed value to ensure that each run of the script produces the same results.
# This is important for reproducibility, especially when using algorithms that involve a random component.
set.seed(123)

# Fit a K-Nearest Neighbors (KNN) model to the training data.
# 'training_set[, -3]' indicates using all columns except the third one as predictors.
# 'cl' argument represents the classes/target variable, which is the third column in the training set.
# 'k' is set to 5, which specifies that the model should consider the 5 nearest neighbors in making a prediction.
knn_pred <- knn(train = training_set[, -3], test = test_set[, -3], cl = training_set[, 3], k = 5)

# Fit a Random Forest classifier to the training data.
# Here we use 'randomForest()' function, with 'x' as the predictor variables (all except third column),
# and 'y' as the target variable 'Purchased' from the training set.
# 'ntree' specifies the number of trees to grow in the forest, here set to 500 for a robust model.
set.seed(123)  # Set seed again for consistency in random forest results.
rf_classifier <- randomForest(x = training_set[-3], y = training_set$Purchased, ntree = 500)

# Use the fitted Random Forest classifier to make predictions on the test set.
# 'newdata' parameter specifies the test set data (excluding the third column) on which to make predictions.
rf_pred <- predict(rf_classifier, newdata = test_set[-3])

# Evaluate the performance of the KNN model.
# 'knn_performance' will hold the accuracy and kappa statistics for the KNN predictions against actual data.
# 'test_set$Purchased' provides the actual target values from the test set for comparison.
knn_performance <- evaluate_model(knn_pred, test_set$Purchased)

# Evaluate the performance of the Random Forest model in a similar manner.
# 'rf_performance' will contain the performance metrics for the Random Forest predictions.
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

# Function to plot model results using ggplot2
plot_model_results <- function(test_set, predictions, model_name) {
  test_set$Predicted <- as.factor(predictions)
  ggplot(test_set, aes(x = Age, y = EstimatedSalary, color = Purchased, shape = Predicted)) +
    geom_point(alpha = 0.5) +
    scale_color_manual(values = c('red', 'green')) +
    scale_shape_manual(values = c(16, 17)) +
    labs(title = paste(model_name, "Classification Results"), x = "Age", y = "Estimated Salary",
         color = "Actual Class", shape = "Predicted Class") +
    theme_minimal()
}

# Create and print visualizations for both models
print(plot_model_results(test_set, knn_pred, "K-NN"))
print(plot_model_results(test_set, rf_pred, "Random Forest"))

# Combine the performance metrics into a data frame for plotting
performance_data <- data.frame(
  Model = c("K-NN", "Random Forest"),
  Accuracy = c(knn_performance$accuracy, rf_performance$accuracy),
  Kappa = c(knn_performance$kappa, rf_performance$kappa)
)

# Melt the data into a long format for ggplot2
library(reshape2)
performance_melted <- melt(performance_data, id.vars = "Model", variable.name = "Metric", value.name = "Value")

# Plot the performance metrics in a bar chart
performance_plot <- ggplot(performance_melted, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  scale_fill_manual(values = c("Accuracy" = "blue", "Kappa" = "green")) +
  labs(title = "Performance Metrics of K-NN and Random Forest Models", x = "Model", y = "Metric Value") +
  theme_minimal() +
  theme(legend.title = element_blank())

# Print the performance plot
print(performance_plot)
