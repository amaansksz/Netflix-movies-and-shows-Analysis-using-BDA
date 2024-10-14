# Load necessary libraries
library(dplyr)
library(tidyr)
library(ggplot2)
library(pROC)
library(Metrics)
library(stringr)

# Set working directory (update the path accordingly)
setwd("C:/Users/amaan/Desktop/BDA mini project")


# Step 1: Read the dataset
netflix_data <- read.csv("titles.csv")

# Step 2: Create a binary classification for trending titles (TMDB popularity > 10)
netflix_data$trending <- ifelse(netflix_data$tmdb_popularity > 10, 1, 0)

# Step 3: Separate genres into multiple rows while keeping other columns
netflix_data_long <- netflix_data %>%
  separate_rows(genres, sep = ", ")  # Split genres into separate rows

# Diagnostic Step: Check unique genres present in the dataset
unique_genres <- unique(netflix_data_long$genres)
print("Unique genres before cleaning:")
print(unique_genres)  # Print the unique genres to check for any discrepancies

# Clean genre names: Remove unwanted characters and extra spaces
netflix_data_long$genres <- str_replace_all(netflix_data_long$genres, "\\[|\\]|'", "")  # Remove brackets and quotes
netflix_data_long$genres <- trimws(tolower(netflix_data_long$genres))  # Convert to lower case and trim whitespace

# Step 4: Define the top 10 genres (ensure correct casing)
top_genres <- c("action", "comedy", "drama", "thriller", "romance", 
                "horror", "scifi", "fantasy", "family", "documentary")

# Step 5: Filter to keep only rows with the top genres
top_genres_lower <- tolower(top_genres)  # Convert top genres to lower case
netflix_data_long <- netflix_data_long %>%
  filter(genres %in% top_genres_lower)  # Keep only rows that match the top genres

# Check the number of observations after filtering
cat("Number of observations after filtering:", nrow(netflix_data_long), "\n")

# Step 6: Create a summary table to combine genres back into one column if needed
netflix_data_summary <- netflix_data_long %>%
  group_by(trending, release_year, runtime, age_certification) %>%
  summarise(genres = paste(unique(genres), collapse = ", "), .groups = 'drop')

# Step 7: Handle missing values (if necessary)
filtered_data <- na.omit(netflix_data_summary)

# Step 8: Convert age_certification to a factor
filtered_data$age_certification <- as.factor(filtered_data$age_certification)

# Step 9: Fit the logistic regression model with the relevant dataset
model <- glm(trending ~ release_year + runtime + age_certification + genres, 
             data = filtered_data, family = binomial)

# Step 10: Summary of the model
summary(model)

# Step 11: Predict probabilities
predicted_probs <- predict(model, newdata = filtered_data, type = "response")

# Step 12: Create predictions based on a threshold (0.5)
predicted_classes <- ifelse(predicted_probs > 0.5, 1, 0)

# Step 13: Calculate ROC curve
roc_curve <- roc(filtered_data$trending, predicted_probs)  # Create ROC curve

# Step 14: Calculate AUC directly from the roc object
auc_value <- auc(filtered_data$trending, predicted_probs)

# Print AUC value
cat("AUC:", auc_value, "\n")

# Step 15: Plot ROC curve
plot(roc_curve, main = paste("ROC Curve - AUC:", round(auc_value, 2)))

# Step 16: Calculate MAE, MSE, R-squared, and MAPE
mae_value <- mean(abs(filtered_data$trending - predicted_classes))  # Mean Absolute Error
mse_value <- mean((filtered_data$trending - predicted_classes)^2)  # Mean Squared Error
r_squared <- 1 - (sum((filtered_data$trending - predicted_classes)^2) / 
                    sum((filtered_data$trending - mean(filtered_data$trending))^2))  # R-squared

# Avoid division by zero for MAPE
non_zero_indices <- which(filtered_data$trending != 0)
mape_value <- mean(abs((filtered_data$trending[non_zero_indices] - predicted_probs[non_zero_indices]) / filtered_data$trending[non_zero_indices])) * 100

# Step 17: Print metrics
cat("MAE:", mae_value, "\n")
cat("MSE:", mse_value, "\n")
cat("R-squared:", r_squared, "\n")
cat("MAPE:", mape_value, "\n")

# Step 18: Bar Plot of Trending Titles by Genre
ggplot(netflix_data_long, aes(x = genres, fill = as.factor(trending))) +
  geom_bar(position = "dodge") +  # Dodge to show bars side by side
  labs(x = "Genre", y = "Count", 
       title = "Count of Trending and Non-Trending Titles by Genre",
       fill = "Trending Status") +  # Legend title
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x labels for readability

# Step 19: Logistic Regression Plot (Predicted probabilities vs Release Year)
ggplot(filtered_data, aes(x = release_year, y = predicted_probs)) +
  geom_point(aes(color = as.factor(trending)), alpha = 0.6) +  # Scatter plot with points colored by trending status
  geom_smooth(method = "glm", method.args = list(family = "binomial"), se = FALSE, color = "blue") +  # Logistic regression line
  labs(x = "Release Year", y = "Predicted Probability", 
       title = "Logistic Regression: Predicted Probability vs Release Year",
       color = "Trending Status") +  # Title and labels
  theme_minimal()