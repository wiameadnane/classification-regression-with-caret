# Install necessary packages
install.packages("randomForest")
install.packages("caret")
install.packages("e1071") 
install.packages("pROC")

# Load required libraries
library(randomForest)
library(caret)
library(pROC)

# Load the dataset (replace with your file path)
alz_data <- read.csv("C:\\Users\\HP\\Downloads\\alzheimers_disease_data.csv", header = TRUE)

# Check the first few rows and structure
head(alz_data)
str(alz_data)

# Suppression des colomnes non significatives
alz_data <- alz_data[, !(names(alz_data) %in% c("PatientID", "Doctorincharge"))]

# Convertiriagnosis en factor
alz_data$Diagnosis <- as.factor(alz_data$Diagnosis)

# Gestion des valeurs manquantes
alz_data <- na.omit(alz_data)

# Split the dataset into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(alz_data$Diagnosis, p = 0.8, list = FALSE)
trainData <- alz_data[trainIndex, ]
testData <- alz_data[-trainIndex, ]

# Check the distribution of classes in the training data
table(trainData$Diagnosis)

# Train  Random Forest model
set.seed(123)  
rf_model <- randomForest(Diagnosis ~ ., data = trainData, ntree = 500, mtry = 3, importance = TRUE)

# Affichage du Random model summary
print(rf_model)

# Evaluate the model on the testing data
predictions <- predict(rf_model, testData)

# Generate a confusion matrix
conf_matrix <- confusionMatrix(predictions, testData$Diagnosis)
print(conf_matrix)

# ROC Curve (optional for binary classification tasks)
if (nlevels(trainData$Diagnosis) == 2) {
  # Get probabilities for the positive class
  prob_predictions <- predict(rf_model, testData, type = "prob")[, 2]
  roc_curve <- roc(testData$Diagnosis, prob_predictions)
  plot(roc_curve, main = "ROC Curve", col = "blue", lwd = 2)
  auc <- auc(roc_curve)
  print(paste("AUC:", auc))
}
# Hyperparameter tuning using caret
# Identify columns with one unique level
one_level_cols <- names(trainData)[sapply(trainData, function(x) length(unique(x)) <= 1)]

# Remove these columns from trainData and testData
trainData <- trainData[, !(names(trainData) %in% one_level_cols)]
testData <- testData[, !(names(testData) %in% one_level_cols)]

# Hyperparameter tuning using caret
set.seed(123)
control <- trainControl(method = "cv", number = 10)  # 10-fold cross-validation

tuned_rf <- train(
  Diagnosis ~ ., data = trainData,
  method = "rf",
  trControl = control,
  tuneGrid = expand.grid(mtry = c(2, 3, 4, 5))
)

# Print results
print(tuned_rf)

# Print the best parameters
print(tuned_rf$bestTune)

# Evaluate the tuned model on the test data
tuned_predictions <- predict(tuned_rf, testData)
tuned_conf_matrix <- confusionMatrix(tuned_predictions, testData$Diagnosis)
print(tuned_conf_matrix)

# Final tuned Random Forest summary
print(tuned_rf)
oob_error <- rf_model$err.rate[, "OOB"]

# Convertir l'erreur OOB en accuracy
oob_accuracy <- 1 - oob_error

# ----------------------------------------
# 8. Plot the Accuracy Curve
# ----------------------------------------
plot(
  oob_accuracy, type = "l", col = "blue", lwd = 2,
  xlab = "Nombre d'arbres (ntree)", ylab = "Accuracy OOB",
  main = "Courbe de l'Accuracy OOB en fonction du nombre d'arbres"
)

# Ajouter une grille pour une meilleure lisibilité
grid()

