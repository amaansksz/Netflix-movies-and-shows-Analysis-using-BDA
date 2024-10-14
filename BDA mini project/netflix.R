# Load necessary libraries
library(dplyr)
library(tidyr)
library(ggplot2)
library(randomForest)  # Random Forest library
library(pROC)          # For ROC curve
library(Metrics)       # For additional metrics
library(caret)         # For confusionMatrix
library(stringr)       # For string manipulation

# Set working directory (update the path accordingly)
setwd("C:/Users/amaan/Desktop/BDA mini project")  # Adjust the path as necessary

# Step 1: Read the dataset
netflix_data <- read.csv("titles.csv")

# Step 2: Create a binary classification for trending titles (TMDB popularity > 10)
netflix_data$trending <- ifelse(netflix_data$tmdb_popularity > 10, 1, 0)

# Step 3: Separate genres into multiple rows while keeping other columns
netflix_data_long <- netflix_data %>%
  separate_rows(genres, sep = ", ")  # Split genres into separate rows

# Clean genre names
netflix_data_long$genres <- str_replace_all(netflix_data_long$genres, "\\[|\\]|'", "")  # Remove brackets and quotes
netflix_data_long$genres <- trimws(tolower(netflix_data_long$genres))  # Convert to lower case and trim whitespace

# Step 4: Define the top 10 genres (ensure correct casing)
top_genres <- c("action", "comedy", "drama", "thriller", "romance", 
                 "horror", "scifi", "fantasy", "family", "documentary")

# Step 5: Filter to keep only rows with the top genres
netflix_data_long <- netflix_data_long %>%
  filter(genres %in% tolower(top_genres))  # Keep only rows that match the top genres

# Step 6: Create a summary table
netflix_data_summary <- netflix_data_long %>%
  group_by(trending, release_year, runtime, age_certification) %>%
  summarise(genres = paste(unique(genres), collapse = ", "), .groups = 'drop')

# Step 7: Handle missing values and convert age_certification to a factor
filtered_data <- na.omit(netflix_data_summary)
filtered_data$age_certification <- as.factor(filtered_data$age_certification)

# Step 8: Split the data into training and testing sets
set.seed(123)  # For reproducibility
train_index <- sample(1:nrow(filtered_data), 0.7 * nrow(filtered_data))  # 70% training data
train_data <- filtered_data[train_index, ]
test_data <- filtered_data[-train_index, ]

# Step 9: Train the Random Forest model
rf_model <- randomForest(as.factor(trending) ~ release_year + runtime + age_certification + genres, 
                         data = train_data, ntree = 100)  # Using 100 trees

# Step 10: Make predictions on the test set
predictions <- predict(rf_model, newdata = test_data)

# Step 11: Evaluate the model's performance
confusion <- confusionMatrix(as.factor(predictions), as.factor(test_data$trending))

# Print confusion matrix and accuracy
print(confusion)

# Calculate accuracy
accuracy <- confusion$overall['Accuracy']
cat("Accuracy:", accuracy, "\n")

# Step 12: Calculate ROC curve and AUC
predicted_probs_test <- predict(rf_model, newdata = test_data, type = "prob")[,2]  # Get probabilities for the positive class
roc_curve <- roc(test_data$trending, predicted_probs_test)  # Create ROC curve
auc_value <- auc(roc_curve)  # Calculate AUC from the ROC object

# Print AUC value
cat("AUC:", auc_value, "\n")

# Step 13: Plot ROC curve
plot(roc_curve, main = paste("ROC Curve - AUC:", round(auc_value, 2)))

# Optional: Feature importance
importance(rf_model)

# Optional: Plot the Random Forest model
plot(rf_model)
