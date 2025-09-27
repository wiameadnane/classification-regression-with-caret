# Charger les bibliothèques nécessaires
library(caret)
library(readr)
library(dplyr)
library(ggplot2)
library(reshape2)
library(corrplot)
library(MASS)  # Charger le package MASS pour la régression robuste

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

# Vérification des valeurs manquantes
if (sum(is.na(data)) > 0) {
  cat("Il y a des valeurs manquantes, traitement des données...\n")
  
  # Exemple de traitement : Remplir les NA avec la médiane (pour les variables numériques)
  data <- data %>%
    mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))
}

# Filtrer uniquement les variables numériques pour la corrélation
num_vars <- c("price", "area", "bedrooms", "bathrooms", "stories", "parking")
data_numeric <- data[, num_vars]

# Vérifier qu'il n'y a pas de NA dans les données numériques
if (sum(is.na(data_numeric)) > 0) {
  stop("Des valeurs manquantes persistent dans les variables numériques.")
}

# Tracer la matrice de corrélation pour les variables numériques
correlation_matrix <- cor(data_numeric)
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

# Entraîner le modèle de régression robuste (rlm) directement avec MASS
model_rlm <- rlm(price ~ ., data = train_data)

# Résumé du modèle
summary(model_rlm)

# Faire des prédictions sur l'ensemble de test
predictions <- predict(model_rlm, newdata = test_data)

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
