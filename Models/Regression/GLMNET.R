install.packages("glmnet")
# Charger les bibliothèques nécessaires
library(dplyr)   # Traitement des données
library(caret)   # Prétraitement et modèles ML
library(glmnet)  # Régression Lasso
library(ggplot2) # Visualisation

# Charger les données
data <- read.csv("Housing.csv")
summary(data)
str(data)

# 1. Transformation des variables catégorielles en variables fictives (dummy)
dummy_model <- dummyVars(" ~ .", data = data)
data_transformed <- data.frame(predict(dummy_model, newdata = data))

# 2. Suppression des variables à faible variance
nzv <- nearZeroVar(data_transformed)
if (length(nzv) > 0) {
  data_clean <- data_transformed[, -nzv]
} else {
  data_clean <- data_transformed
}

# 3. Sélection automatique des variables importantes via la corrélation
correlations <- cor(data_clean)
target_corr <- correlations[, "price"]
important_vars <- names(target_corr[abs(target_corr) > 0.1])
data_selected <- data_clean[, important_vars]

# 4. Division des données en ensembles d'entraînement et de test
set.seed(123)
splitIndex <- createDataPartition(data_selected$price, p = 0.8, list = FALSE)
trainData <- data_selected[splitIndex, ]
testData <- data_selected[-splitIndex, ]

# 5. Normalisation avancée avec transformation Yeo-Johnson
preProc <- preProcess(trainData[, -which(names(trainData) == "price")], 
                      method = c("center", "scale", "YeoJohnson"))
trainData_norm <- predict(preProc, trainData)
testData_norm <- predict(preProc, testData)

# Ajouter la variable cible après normalisation
trainData_norm$price <- trainData$price
testData_norm$price <- testData$price

# Préparer les matrices pour glmnet
x_train <- as.matrix(trainData_norm[, -which(names(trainData_norm) == "price")])
y_train <- trainData_norm$price
x_test <- as.matrix(testData_norm[, -which(names(testData_norm) == "price")])
y_test <- testData_norm$price

# 6. Modèle Lasso avec optimisation de lambda
set.seed(123)
grid_lasso <- expand.grid(alpha = 1, lambda = seq(0.001, 1, length = 50))

model_lasso <- train(price ~ ., data = trainData_norm, method = "glmnet",
                     trControl = trainControl(method = "cv", number = 10),
                     tuneGrid = grid_lasso)

# 7. Faire des prédictions sur l'ensemble de test
pred_lasso <- predict(model_lasso, newdata = testData_norm)

# 8. Calculer les performances
performance <- postResample(pred = pred_lasso, obs = y_test)
print(performance)

# 9. Importance des variables
var_imp <- varImp(model_lasso, scale = FALSE)

# Convertir en dataframe pour ggplot2
importance_df <- as.data.frame(var_imp$importance)
importance_df$Variables <- rownames(importance_df)

# Tracer l'importance des variables
ggplot(importance_df, aes(x = reorder(Variables, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Importance des variables pour le modèle Lasso",
       x = "Variables",
       y = "Importance") +
  theme_minimal()
