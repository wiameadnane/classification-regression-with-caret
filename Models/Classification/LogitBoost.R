# Installer et charger les bibliothèques nécessaires
library(dplyr)
library(corrplot)
library(caret)

# Séparer les variables numériques
numeric_vars <- data_cleaned %>% select_if(is.numeric)

# Calculer les corrélations de Pearson
correlations <- sapply(numeric_vars, function(x) cor(x, as.numeric(data_cleaned$Diagnosis), method = "pearson"))
print("Corrélations avec Diagnosis :")
print(correlations)


# Filtrer les variables ayant une corrélation absolue significative
selected_vars <- names(correlations[abs(correlations) > 0.2])
print("Variables sélectionnées basées sur la corrélation :")
print(selected_vars)

# Garder uniquement les variables pertinentes et la colonne Diagnosis
data_reduced <- data_cleaned %>% select(all_of(selected_vars), Diagnosis)


set.seed(42)
trainIndex <- createDataPartition(data_reduced$Diagnosis, p = 0.6, list = FALSE)
data_train <- data_reduced[trainIndex, ]
data_test <- data_reduced[-trainIndex, ]

# Standardisation des données
preProc <- preProcess(data_train[, -ncol(data_train)], method = c("center", "scale"))
data_train_scaled <- predict(preProc, data_train)
data_test_scaled <- predict(preProc, data_test)


# Entraîner le modèle LogitBoost

model <- train(
  Diagnosis ~ ., 
  data = data_train_scaled,
  method = "LogitBoost",
  trControl = trainControl(method = "cv", number = 10, savePredictions = "final"),
  tuneLength = 5
)

# Afficher les résultats
print(model)
plot(model)

# Prédictions
y_pred_train <- predict(model, data_train_scaled)
y_pred_test <- predict(model, data_test_scaled)

# Matrices de confusion
conf_matrix_train <- confusionMatrix(y_pred_train, data_train_scaled$Diagnosis)
conf_matrix_test <- confusionMatrix(y_pred_test, data_test_scaled$Diagnosis)

print("Training : ")
print(conf_matrix_train)
print(" ")
print("Test : ")
print(conf_matrix_test)

# Courbe ROC pour l'ensemble de test
library(pROC)
probabilities <- predict(model, data_test_scaled, type = "prob")
roc_curve <- roc(data_test_scaled$Diagnosis, probabilities[, 2])
plot(roc_curve, col = "blue", main = "Courbe ROC")
cat("AUC :", auc(roc_curve))
