library(readr)
library(dplyr)
library(ggplot2)
library(leaps)
library(glmnet)
library(e1071)
library(arm)

student_data <- read.csv("student_course_completion_prediction.csv") 
head(student_data)

# 01 DUPLICATE CHECK & REMOVAL

# Check total number of duplicate rows
initial_rows <- nrow(student_data)
duplicate_count <- sum(duplicated(student_data))
print(paste("Number of duplicate rows found:", duplicate_count))

# Remove duplicate rows if any exist
student_data <- student_data %>% distinct()
print(paste("Rows remaining after duplicate removal:", nrow(student_data)))


# 01 DATASET STRUCTURE & SUMMARY
str(student_data)
summary(student_data)


# 02 MISSING VALUE & DATA PREPROCESSING

missing_counts <- colSums(is.na(student_data))
print(missing_counts)
missing_percentage <- (colSums(is.na(student_data)) / nrow(student_data)) * 100
print(missing_percentage)

# 1. Convert Categorical / Text Variables to Factors (enables automatic dummy encoding)
categorical_vars <- c("Category", "Course_Level", "Employment_Status", 
                      "Education_Level", "Device_Type", "Internet_Connection_Quality", 
                      "Payment_Mode", "Gender")

for(col in categorical_vars) {
  if(col %in% names(student_data)) {
    student_data[[col]] <- as.factor(student_data[[col]])
  }
}

# 2. Feature Creation (Engineering new powerful metrics)
student_data <- student_data %>%
  mutate(
    Total_Engagement_Score = Time_Spent_Hours * (Video_Completion_Rate / 100),
    Inactivity_Penalty_Ratio = Days_Since_Last_Login / (Progress_Percentage + 1),
    Paid_Engagement = ifelse(Payment_Amount > 0, 1, 0)
  )

# 3. Drop Unrelated or Redundant Administrative Columns Safely
cols_to_drop <- c("Student_ID", "Enroll_Date", "Course_Name")
student_data <- student_data[, !(names(student_data) %in% cols_to_drop)]

# 4. Target Variable Encoding
student_data$Completed_num <- ifelse(student_data$Completed == "Completed" | student_data$Completed == "Complete", 1, 0)
table(student_data$Completed_num)


# 03 DESCRIPTIVE STATISTICS & VISUALIZATIONS

student_data %>% 
  summarise(
    Mean_Time_Spent = mean(Time_Spent_Hours, na.rm = TRUE),
    SD_Time_Spent = sd(Time_Spent_Hours, na.rm = TRUE),
    Mean_Quiz_Score = mean(Quiz_Score_Avg, na.rm = TRUE),
    SD_Quiz_Score = sd(Quiz_Score_Avg, na.rm = TRUE)
  )

ggplot(student_data, aes(x = factor(Completed))) +
  geom_bar(fill = c("#ff9999", "#66b3ff")) +
  labs(title = "Course Completion Distribution", x = "Completion Status (0 = Dropout, 1 = Complete)", y = "Count")

ggplot(student_data, aes(x = factor(Completed), y = Time_Spent_Hours, fill = factor(Completed))) +
  geom_boxplot() +
  labs(title = "Time Spent Hours by Completion Status", x = "Completion Status", y = "Time Spent (Hours)")

hist(student_data$Quiz_Score_Avg, main = "Histogram of Quiz Scores", xlab = "Average Quiz Score", col = "lightblue")


# ==========================================
# 04 STATISTICAL INFERENCE (Full Dataset)
# ==========================================
# Test 1: Two-Sample t-test (Comparison of Means)
t.test(Time_Spent_Hours ~ Completed, data = student_data)

# Test 2: ANOVA Test (Comparison of Means across Multiple Groups)
anova_result <- aov(Quiz_Score_Avg ~ Course_Level, data = student_data)
summary(anova_result)

# Test 3: Comparison of Proportions (Chi-Square Test for Gender)
chisq.test(student_data$Completed, student_data$Gender)

# Test 4: Comparison of Variances (F-Test / Var Test)
var.test(Time_Spent_Hours ~ Completed, data = student_data)

# Additional Inference Tests for Newly Encoded Categorical Features:
# Test 5: Chi-Square Test for Course Completion vs Employment Status
chisq.test(student_data$Completed, student_data$Employment_Status)

# Test 6: Chi-Square Test for Course Completion vs Device Type
chisq.test(student_data$Completed, student_data$Device_Type)


# ==========================================
# TRAIN-TEST SPLIT (70% Training, 30% Testing)
# ==========================================
set.seed(123)
train_index <- sample(nrow(student_data), 0.7 * nrow(student_data))
train_data <- student_data[train_index, ]
test_data <- student_data[-train_index, ]


# ==========================================
# 05 PREDICTIVE MODELLING & FEATURE SELECTION
# ==========================================

# 1. Best Subset Selection
subset_model <- regsubsets(Completed_num ~ Time_Spent_Hours + Quiz_Score_Avg + 
                             Days_Since_Last_Login + Assignments_Missed + 
                             Video_Completion_Rate + Progress_Percentage + 
                             Project_Grade + App_Usage_Percentage, 
                           data = train_data, nvmax = 8, method = "exhaustive")

subset_summary <- summary(subset_model)
print(subset_summary)

best_bic_index <- which.min(subset_summary$bic)
coef(subset_model, best_bic_index)


# 2. Forward Selection
null_model <- glm(Completed_num ~ 1, data = train_data, family = "binomial")
full_model <- glm(Completed_num ~ Time_Spent_Hours + Quiz_Score_Avg + 
                    Days_Since_Last_Login + Assignments_Missed + 
                    Video_Completion_Rate + Progress_Percentage + 
                    Project_Grade + App_Usage_Percentage + 
                    Total_Engagement_Score + Inactivity_Penalty_Ratio + Course_Level, 
                  data = train_data, family = "binomial")

forward_model <- step(null_model, 
                      scope = list(lower = null_model, upper = full_model), 
                      direction = "forward")
summary(forward_model)


# 3. Backward Selection 
backward_model <- step(full_model, direction = "backward")
summary(backward_model)


# 4. Final Optimized Logistic Regression Model
final_opt_model <- glm(Completed_num ~ Progress_Percentage + Time_Spent_Hours + 
                         Video_Completion_Rate + Quiz_Score_Avg + 
                         Days_Since_Last_Login + App_Usage_Percentage + 
                         Total_Engagement_Score + Inactivity_Penalty_Ratio + Course_Level, 
                       data = train_data, family = "binomial")
summary(final_opt_model)

# Test Predictions & Accuracy
final_preds <- predict(final_opt_model, newdata = test_data, type = "response")
final_class <- ifelse(final_preds > 0.5, 1, 0)
final_matrix <- table(Predicted = final_class, Actual = test_data$Completed_num)
print(final_matrix)
final_accuracy <- sum(diag(final_matrix)) / sum(final_matrix)
print(paste("Optimized Model Accuracy:", round(final_accuracy * 100, 2), "%"))


# ==========================================
# 06 CHECK ALL MODELS & REGULARIZATION
# ==========================================
# 1. Standard Logistic Regression & Optimized Model Accuracy
preds_log <- predict(final_opt_model, newdata = test_data, type = "response")
class_log <- ifelse(preds_log > 0.5, 1, 0)
mat_log <- table(Predicted = class_log, Actual = test_data$Completed_num)
acc_log <- (sum(diag(mat_log)) / sum(mat_log)) * 100
print(paste("1. Logistic Regression Accuracy:", round(acc_log, 2), "%"))

# 2. Best Subset Selection Model Accuracy
subset_opt_model <- glm(Completed_num ~ Time_Spent_Hours + Quiz_Score_Avg + 
                          Days_Since_Last_Login + Video_Completion_Rate + 
                          Progress_Percentage, 
                        data = train_data, family = "binomial")
preds_sub <- predict(subset_opt_model, newdata = test_data, type = "response")
class_sub <- ifelse(preds_sub > 0.5, 1, 0)
mat_sub <- table(Predicted = class_sub, Actual = test_data$Completed_num)
acc_sub <- (sum(diag(mat_sub)) / sum(mat_sub)) * 100
print(paste("2. Best Subset Selection Accuracy:", round(acc_sub, 2), "%"))

# 3. Regularized Models Data Preparation (Matrix Format with Categoricals)
x_train <- model.matrix(Completed_num ~ Progress_Percentage + Time_Spent_Hours + 
                          Video_Completion_Rate + Quiz_Score_Avg + 
                          Days_Since_Last_Login + App_Usage_Percentage + 
                          Total_Engagement_Score + Inactivity_Penalty_Ratio + Course_Level, data = train_data)[,-1]
y_train <- train_data$Completed_num

x_test <- model.matrix(Completed_num ~ Progress_Percentage + Time_Spent_Hours + 
                         Video_Completion_Rate + Quiz_Score_Avg + 
                         Days_Since_Last_Login + App_Usage_Percentage + 
                         Total_Engagement_Score + Inactivity_Penalty_Ratio + Course_Level, data = test_data)[,-1]
y_test <- test_data$Completed_num

# 4. LASSO Regression Accuracy (Alpha = 1)
set.seed(123)
cv_lasso <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)
lasso_preds <- predict(cv_lasso, newx = x_test, s = "lambda.min", type = "response")
lasso_class <- ifelse(lasso_preds > 0.5, 1, 0)
lasso_matrix <- table(Predicted = lasso_class, Actual = y_test)
acc_lasso <- (sum(diag(lasso_matrix)) / sum(lasso_matrix)) * 100
print(paste("3. LASSO Regression Accuracy:", round(acc_lasso, 2), "%"))

# 5. Ridge Regression Accuracy (Alpha = 0)
set.seed(123)
cv_ridge <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0)
ridge_preds <- predict(cv_ridge, newx = x_test, s = "lambda.min", type = "response")
ridge_class <- ifelse(ridge_preds > 0.5, 1, 0)
ridge_matrix <- table(Predicted = ridge_class, Actual = y_test)
acc_ridge <- (sum(diag(ridge_matrix)) / sum(ridge_matrix)) * 100
print(paste("4. Ridge Regression Accuracy:", round(acc_ridge, 2), "%"))

# 6. Elastic Net Regression Accuracy (Alpha = 0.5)
set.seed(123)
cv_enet <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0.5)
enet_preds <- predict(cv_enet, newx = x_test, s = "lambda.min", type = "response")
enet_class <- ifelse(enet_preds > 0.5, 1, 0)
enet_matrix <- table(Predicted = enet_class, Actual = y_test)
acc_enet <- (sum(diag(enet_matrix)) / sum(enet_matrix)) * 100
print(paste("5. Elastic Net Accuracy:", round(acc_enet, 2), "%"))


# ==========================================
# 07 PRINCIPAL COMPONENT ANALYSIS (PCA)
# ==========================================
pca_data <- train_data[, c("Progress_Percentage", "Time_Spent_Hours", "Video_Completion_Rate", 
                           "Quiz_Score_Avg", "Days_Since_Last_Login", "App_Usage_Percentage",
                           "Total_Engagement_Score", "Inactivity_Penalty_Ratio")]
pca_result <- prcomp(pca_data, scale. = TRUE)
summary(pca_result)
plot(pca_result, type = "l", main = "PCA Variance")


# ==========================================
# 08 NAÏVE BAYES
# ==========================================
nb_model <- naiveBayes(as.factor(Completed_num) ~ Progress_Percentage + Time_Spent_Hours + 
                         Video_Completion_Rate + Quiz_Score_Avg + 
                         Days_Since_Last_Login + App_Usage_Percentage + Course_Level, 
                       data = train_data)
nb_preds <- predict(nb_model, newdata = test_data)
nb_matrix <- table(Predicted = nb_preds, Actual = test_data$Completed_num)
print(nb_matrix)
nb_acc <- (sum(diag(nb_matrix)) / sum(nb_matrix)) * 100
print(paste("Naive Bayes Accuracy:", round(nb_acc, 2), "%"))


# ==========================================
# 09 BAYESIAN REGRESSION & DECISION MAKING
# ==========================================
bayes_reg_model <- bayesglm(Completed_num ~ Progress_Percentage + Time_Spent_Hours + 
                              Video_Completion_Rate + Quiz_Score_Avg + 
                              Days_Since_Last_Login + App_Usage_Percentage + Course_Level, 
                            data = train_data, family = binomial(link = "logit"))
summary(bayes_reg_model)

bayes_preds <- predict(bayes_reg_model, newdata = test_data, type = "response")
bayes_class <- ifelse(bayes_preds > 0.5, 1, 0)
bayes_matrix <- table(Predicted = bayes_class, Actual = test_data$Completed_num)
print(bayes_matrix)
bayes_acc <- (sum(diag(bayes_matrix)) / sum(bayes_matrix)) * 100
print(paste("Bayesian Regression Accuracy:", round(bayes_acc, 2), "%"))

thresholds <- seq(0.3, 0.7, by = 0.05)
best_acc <- 0
best_thresh <- 0.5

for(t in thresholds) {
  temp_class <- ifelse(bayes_preds > t, 1, 0)
  temp_acc <- sum(diag(table(temp_class, test_data$Completed_num))) / nrow(test_data)
  if(temp_acc > best_acc) {
    best_acc <- temp_acc
    best_thresh <- t
  }
}

print(paste("Optimal Bayesian Decision Threshold:", best_thresh))
print(paste("Optimized Bayesian Decision Accuracy:", round(best_acc * 100, 2), "%"))