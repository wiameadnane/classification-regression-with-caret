# Charger les bibliothèques nécessaires
library(caret)          # Pour la construction et l'évaluation du modèle
library(rpart)          # Pour construire l'arbre de décision
library(rpart.plot)     # Pour visualiser l'arbre de décision
library(ggplot2)        # Pour visualiser les résultats
library(reshape2)       # Pour réorganiser les données pour ggplot2

# Charger les données
data <- read.csv("C:/Users/aitte/Downloads/archive (1)/alzheimers_disease_data.csv")

# Renommer la colonne 'Diagnosis' pour éviter les erreurs
colnames(data)[colnames(data) == "Diagnosis"] <- "diagnosis"
data$diagnosis <- as.factor(data$diagnosis)

# Supprimer les colonnes inutiles
data <- data[, !(colnames(data) %in% c("PatientID", "DoctorInCharge", "SystolicBP", "Confusion"))]

# Vérifier les types des variables
str(data)

# Calculer la matrice de corrélation pour les variables numériques
numeric_data <- data[, sapply(data, is.numeric)]
correlation_matrix <- cor(numeric_data, use = "complete.obs")

# Afficher la matrice de corrélation
cat("\nMatrice de corrélation:\n")
print(correlation_matrix)

# Visualiser la matrice de corrélation
correlation_data <- melt(correlation_matrix)
colnames(correlation_data) <- c("Variable1", "Variable2", "Correlation")

ggplot(correlation_data, aes(x = Variable1, y = Variable2, fill = Correlation)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("Matrice de Corrélation") +
  xlab("") +
  ylab("")

# Normaliser les données (sauf la variable cible)
data_normalized <- scale(numeric_data)

# Appliquer l'ACP
pca <- prcomp(data_normalized, center = TRUE, scale. = TRUE)
summary(pca)  # Résumé pour voir la variance expliquée

# Sélectionner le nombre de composantes principales (par exemple, 95% de variance expliquée)
explained_variance <- cumsum(pca$sdev^2 / sum(pca$sdev^2))
num_components <- which(explained_variance >= 0.85)[1]
cat("Nombre de composantes principales sélectionnées :", num_components, "\n")

# Créer un nouveau dataset basé sur les composantes principales
pca_data <- as.data.frame(pca$x[, 1:num_components])
pca_data$diagnosis <- data$diagnosis  # Ajouter la variable cible

# Division des données en ensemble d'entraînement (80%) et de test (20%)
set.seed(123)  # Fixer une graine pour reproductibilité
index <- createDataPartition(pca_data$diagnosis, p = 0.8, list = FALSE)  # 80% pour l'entraînement
train_data <- pca_data[index, ]
test_data <- pca_data[-index, ]

# Paramètres pour l'entraînement du modèle
train_control <- trainControl(method = "cv", number = 10)  # 10-fold cross-validation

# Entraîner le modèle d'arbre de décision sans réglage des hyperparamètres
model <- train(diagnosis ~ ., data = train_data, method = "rpart", trControl = train_control)

# Afficher le résumé du modèle
cat("\nRésumé du modèle non ajusté:\n")
print(model)

# Précision d'entraînement
train_accuracy <- model$results$Accuracy
cat("\nPrécision d'entraînement (Train Accuracy) :", train_accuracy, "\n")

# Prédictions sur l'ensemble de test
predictions <- predict(model, newdata = test_data)

# Évaluer la performance du modèle (matrice de confusion)
conf_matrix <- confusionMatrix(predictions, test_data$diagnosis)
cat("\nMatrice de confusion pour le modèle non ajusté:\n")
print(conf_matrix)

# Précision sur l'ensemble de test
test_accuracy <- conf_matrix$overall['Accuracy']
cat("\nPrécision sur l'ensemble de test (Test Accuracy) :", test_accuracy, "\n")

# Visualiser l'arbre de décision du modèle non ajusté
cat("\nArbre de décision pour le modèle non ajusté:\n")
rpart.plot(model$finalModel)

# Entraîner le modèle d'arbre de décision avec réglage des hyperparamètres
model_tuned <- train(diagnosis ~ ., data = train_data, method = "rpart", trControl = train_control, tuneLength = 10)

# Afficher le modèle ajusté
cat("\nRésumé du modèle ajusté (tuning):\n")
print(model_tuned)

# Prédictions avec le modèle ajusté
predictions_tuned <- predict(model_tuned, newdata = test_data)

# Évaluer la performance du modèle ajusté (matrice de confusion)
conf_matrix_tuned <- confusionMatrix(predictions_tuned, test_data$diagnosis)
cat("\nMatrice de confusion pour le modèle ajusté:\n")
print(conf_matrix_tuned)

# Précision du modèle ajusté sur l'ensemble de test
test_accuracy_tuned <- conf_matrix_tuned$overall['Accuracy']
cat("\nPrécision du modèle ajusté (Tuned Model Test Accuracy) :", test_accuracy_tuned, "\n")

# Visualiser l'arbre de décision pour le modèle ajusté
cat("\nArbre de décision pour le modèle ajusté:\n")
rpart.plot(model_tuned$finalModel)

# Faire des prédictions sur l'ensemble de test avec le modèle d'arbre de décision ajusté
rpart_pred_test <- predict(model_tuned, newdata = test_data)

# Évaluer la performance du modèle sur l'ensemble de test en utilisant la matrice de confusion
cat("\nÉvaluation sur l'ensemble de test pour le modèle ajusté:\n")
confusionMatrix(rpart_pred_test, test_data$diagnosis)