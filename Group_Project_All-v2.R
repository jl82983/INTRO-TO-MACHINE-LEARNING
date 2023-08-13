# Load libraries
library(tree)
library(ISLR2)
library(rpart)
library(randomForest)
library(gbm)
library(tidyverse)
library(lubridate)
library(dplyr)
library(caret)
library(ggcorrplot)

# Load and clean the data
# We removed the VIN column and filtered by TX only in CSV
car_data <- read.csv("C:/Users/jran0/Desktop/STA 380 INTRO TO MACHINE LEARNING/Project1/true_car_listings_clean_TX.csv")

# Remove missing values
is.na(car_data)
car_data <- na.omit(car_data)

# Remove duplicate rows
car_data <- car_data[!duplicated(car_data), ]

# Take logs
car_data$Price <- log(car_data$Price)
car_data$Mileage <- log(car_data$Mileage)

# Convert data type to numeric
#car_data$Model <- as.numeric(factor(car_data$Model))
#car_data$Make <- as.numeric(factor(car_data$Make))
#car_data$City <- as.numeric(factor(car_data$City))

# Convert categorical variables to numeric using one-hot encoding
car_data_encoded <- car_data %>%
  select(Make, Model, Year, City, Mileage) %>%
  mutate(
    Make = as.numeric(factor(Make)),
    Model = as.numeric(factor(Model)),
    Year = as.numeric(Year),
    City = as.numeric(factor(City)),
    Mileage = as.numeric(Mileage)
  )

# Finding the correlation between the different independent variables
correlation_matrix <- cor(car_data_encoded)


# Creating a heat map to visualize the correlation matrix
ggcorrplot(correlation_matrix,
           hc.order = TRUE, # Automatically reorder the variables
           type = "upper",  # Show only the upper triangle of the matrix
           lab = TRUE,      # Show variable names as labels
           outline.color = "white", # Color of the cell outline
           # Width of the cell outline
           ggtheme = ggplot2::theme_minimal() # Choose a theme for the plot
)


# Split the data into training and test sets
set.seed(1)
train <- sample(1:nrow(car_data), nrow(car_data) / 2)
car_data.train <- car_data[train, ]
car_data.test <- car_data[-train, ]

# Fit a regression tree to the training set and plot, using all predictor variables
tree.cars <- tree(Price ~ ., data = car_data.train, mindev = 0.0001)
summary(tree.cars)

# First big tree is 49 before logging. 68 after logging
cat('first big tree size: \n')
print(length(unique(tree.cars$where)))

# We get a big ugly tree
plot(tree.cars)
text(tree.cars, pretty = 0)

# Test MSE: 149148776 before logging. 0.1834458 after logging
yhat <- predict(tree.cars, newdata = car_data.test)
mean((yhat - car_data.test$Price)^2)

# Perform cross-validation to find the optimal level of tree complexity
cv.car_data <- cv.tree(tree.cars)
plot(cv.car_data$size, cv.car_data$dev)
tree.min <- which.min(cv.car_data$dev)

# Prune the tree
prune.tree.cars <- prune.tree(tree.cars, best = 7)
plot(prune.tree.cars)
text(prune.tree.cars, pretty = 0)

# Prediction
yhat <- predict(prune.tree.cars, newdata = car_data.test)
mean((yhat - car_data.test$Price)^2)
mean((exp(yhat) - exp(car_data.test$Price))^2)


# Partition
par(mfrow = c(1, 2))
partition.tree(prune.tree.cars)



# Bagging
bag.cars <- randomForest(Price ~ ., data = car_data.train, mtry = 5, ntree = 100, importance = TRUE)
yhat.bag <- predict(bag.cars, newdata = car_data.test)


# Calculate MSE
mean((yhat.bag - car_data.test$Price)^2)
mean((exp(yhat.bag) - exp(car_data.test$Price))^2)

# Show plot
plot(yhat.bag, car_data.test$Price)
abline(0, 1)

# Importance - this is the result for MSE 0.09446372 
importance(bag.cars)

# Show the importance chart
varImpPlot(bag.cars)





# Random Forest
# MSE =  0.07588763, I got 0.07465817 the second time
rf.cars <- randomForest(Price ~ ., data = car_data.train, mtry = sqrt(5), ntree = 100, importance = TRUE)
yhat.rf <- predict(rf.cars, newdata = car_data.test)

# Calculate MSE
mean((yhat.rf - car_data.test$Price)^2)
mean((exp(yhat.rf) - exp(car_data.test$Price))^2)

#importance
importance(rf.cars)

# Show the imporatnce chart
varImpPlot(rf.cars)





# Boosting
# Turn variables into factors
car_data.train$City <- as.factor(car_data.train$City)
car_data.test$City <- as.factor(car_data.test$City)

car_data.train$Make <- as.factor(car_data.train$Make)
car_data.test$Make <- as.factor(car_data.test$Make)


# Consider Mileage + Year + City + Make since boosting only accept <=1024 levels of categorical variables
boost_car <- gbm(Price ~ Mileage + Year + City + Make,
                 data = car_data.train,
                 distribution = "gaussian",
                 n.trees = 100,
                 shrinkage = 0.01)

# See which variable is more important
summary(boost_car)

# Predict the Price on the test set (car_data.test)
predict_boost_car <- predict(boost_car,
                             newdata = car_data.test,
                             n.trees = 100,
                             type = "response")

# Calculate MSE
mean((predict_boost_car - car_data.test$Price)^2)
mean((exp(predict_boost_car) - exp(car_data.test$Price))^2)


# Show plot
plot(predict_boost_car, car_data.test$Price)
abline(0, 1)




# Linear Regression
lm_model <- lm(Price ~ Year + Mileage + City + Make + Model, data = car_data.train)

# Predictions
predictions <- predict(lm_model, newdata = car_data.train)


# R-square
r_squared <- summary(lm_model)$r.squared
cat("R-squared:", r_squared, "\n")

# Calculate MSE
actual_values <- car_data.test$Price
mean((predictions - actual_values)^2)
mean((exp(predictions) - exp(actual_values))^2)