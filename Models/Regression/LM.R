# Charger les bibliothèques nécessaires
library(caret)
library(readr)
library(dplyr)
library(ggplot2)
library(reshape2)
library(corrplot)

# Chargement des données
data <- read.csv("Housing.csv")

# Explorer les premières lignes du dataset
head(data)

# Convertir les variables catégorielles (yes/no) en facteurs
data$mainroad <- factor(data$mainroad, levels = c("no", "yes"))
data$guestroom <- factor(data$guestroom, levels = c("no", "yes"))
data$basement <- factor(data$basement, levels = c("no", "yes"))
data$hotwaterheating <- factor(data$hotwaterheating, levels = c("no", "yes"))
data$airconditioning <- factor(data$airconditioning, levels = c("no", "yes"))
data$prefarea <- factor(data$prefarea, levels = c("no", "yes"))
data$furnishingstatus <- factor(data$furnishingstatus, levels = c("unfurnished", "semi-furnished", "furnished"))

# Vérifier les types des variables après conversion
str(data)

# Vérification et traitement des valeurs manquantes
if (any(is.na(data))) {
  cat("Il y a des valeurs manquantes, traitement des données...\n")
  
  # Imputer les valeurs manquantes avec la moyenne pour les variables numériques
  data <- data %>%
    mutate(across(where(is.numeric), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))
}

# Tracer la matrice de corrélation pour les variables numériques
num_vars <- c("price", "area", "bedrooms", "bathrooms", "stories", "parking")
correlation_matrix <- cor(data[, num_vars], use = "complete.obs")
corrplot(correlation_matrix, method = "color", type = "upper", addCoef.col = "black",
         tl.col = "black", tl.srt = 45, title = "Matrice de Corrélation", mar = c(0, 0, 1, 0))

# Normalisation des variables numériques
preProc <- preProcess(data[, num_vars], method = c("center", "scale"))
data[, num_vars] <- predict(preProc, data[, num_vars])

# Diviser le dataset en ensemble d'entraînement et de test
set.seed(123)  # Fixer la graine pour la reproductibilité
trainIndex <- createDataPartition(data$price, p = 0.8, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

# Entraîner le modèle de régression linéaire (lm)
model_lm <- lm(price ~ ., data = train_data)

# Résumé du modèle
summary(model_lm)

# Faire des prédictions sur l'ensemble de test
predictions <- predict(model_lm, newdata = test_data)

# Calculer les performances
performance <- postResample(pred = predictions, obs = test_data$price)
print(performance)

# Afficher un graphique des prédictions vs réels
plot(test_data$price, predictions,
     main = "Prédictions vs Réels",
     xlab = "Prix Réels",
     ylab = "Prédictions",
     col = "blue",
     pch = 16)
abline(0, 1, col = "red")  # Ajouter une ligne 1:1
