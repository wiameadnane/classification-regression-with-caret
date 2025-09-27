# Charger les packages nécessaires
install.packages("ggcorrplot")
library(caret)
library(dplyr)
library(ggplot2)
library(corrplot)
library(ggcorrplot)

# Importer le dataset
house_data <- read.csv("C:\\Users\\HP\\Downloads\\Housing.csv")

# Fonction pour identifier les valeurs manquantes
getMissingValues <- function(data) {
  # Check for missing values in each column and store the result
  missing_values <- colSums(is.na(data))
  
  # Sort the columns by the number of missing values in descending order
  sorted_missing_values <- sort(missing_values, decreasing = TRUE)
  
  # Display the sorted missing values and their corresponding column names
  for (col_name in names(sorted_missing_values)) {
    cat("Column:", col_name, "\tMissing Values:", sorted_missing_values[col_name], "\n")
  }
}

getMissingValues(house_data)


# Conversion des variables catégorielles en numériques (One-Hot Encoding)
house_data_numeric <- dummyVars(" ~ .", data = house_data)
house_data_converted <- predict(house_data_numeric, newdata = house_data)
house_data_converted <- as.data.frame(house_data_converted)

# Ajouter la cible "price" après conversion
house_data_converted$price <- house_data$price

# Calculer la matrice de corrélation
cor_matrix <- cor(house_data_converted)

# Afficher la matrice de corrélation
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black",
         number.cex = 0.6, main = "Matrice de Corrélation - Optimisée")

# Identifier les variables ayant une faible corrélation avec "price"
seuil_corr <- 0.1

# Extraire les corrélations avec "price" et identifier celles inférieures au seuil
cor_with_price <- cor_matrix[,"price"]  # Extraire les corrélations avec price
low_corr_vars <- names(cor_with_price[abs(cor_with_price) < seuil_corr & names(cor_with_price) != "price"])

# Afficher les variables avec une faible corrélation avec "price"
cat("Variables avec faible corrélation avec 'price' (|corr| <", seuil_corr, "):\n")
print(low_corr_vars)

# Retirer les variables faiblement corrélées
house_data_optimized <- house_data_converted %>%
  select(-all_of(low_corr_vars))

# Visualisation : Distribution de "price"
ggplot(house_data, aes(x = price)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 30) +
  labs(title = "Distribution de la variable Price", x = "Price", y = "Fréquence") +
  theme_minimal()

# Division des données en Training et Test
set.seed(123)
index <- createDataPartition(house_data_optimized$price, p = 0.8, list = FALSE)
train_data <- house_data_optimized[index, ]
test_data <- house_data_optimized[-index, ]

# Entraîner un modèle Gradient Boosting Machines (GBM)
set.seed(123)
gbm_model <- train(price ~ ., 
                   data = train_data,
                   method = "gbm",
                   trControl = trainControl(method = "cv", number = 5),
                   verbose = FALSE)

# Résumé du modèle
print(gbm_model)



# Prédictions sur les données de test
gbm_predictions <- predict(gbm_model, newdata = test_data)

# Évaluation des performances
gbm_RMSE <- sqrt(mean((gbm_predictions - test_data$price)^2))
gbm_R2 <- cor(gbm_predictions, test_data$price)^2

cat("GBM RMSE :", gbm_RMSE, "\n")
cat("GBM R² :", gbm_R2, "\n")

# Graphique des prédictions vs valeurs réelles
ggplot(data.frame(Actual = test_data$price, Predicted = gbm_predictions), aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue", alpha = 0.6) +  # Points des prédictions vs réelles
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", size = 1) +  # Ligne y = x
  labs(title = "Prédictions vs Valeurs Réelles (GBM)",
       x = "Valeurs Réelles", y = "Prédictions") +
  theme_minimal()

# Charger les packages nécessaires
library(ggcorrplot)

# Sélectionner uniquement les colonnes numériques
numeric_data <- house_data %>% select_if(is.numeric)

# Calculer la matrice de corrélation
cor_matrix <- cor(numeric_data)

# Afficher la heatmap avec ggcorrplot
ggcorrplot(cor_matrix,
           hc.order = TRUE,
           type = "lower",
           lab = TRUE,
           colors = c("red", "white", "darkgreen"),
           title = "Heatmap des Corrélations (Colonnes Numériques)",
           ggtheme = theme_minimal())


