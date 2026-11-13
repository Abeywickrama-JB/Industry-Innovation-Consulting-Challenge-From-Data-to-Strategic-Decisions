# ==========================================
# 01 LIBRARIES & DATA LOADING
# ==========================================
library(readr)
library(dplyr)
library(ggplot2)
library(leaps)
library(glmnet)
library(e1071)
library(arm)

student_data <- read.csv("student_course_completion_prediction.csv") 

# Check total number of duplicate rows & remove duplicates
initial_rows <- nrow(student_data)
duplicate_count <- sum(duplicated(student_data))
print(paste("Number of duplicate rows found:", duplicate_count))

student_data <- student_data %>% distinct()
print(paste("Rows remaining after duplicate removal:", nrow(student_data)))


# ==========================================
# 02 PREPROCESSING & FEATURE ENGINEERING & OUTLIER REMOVAL
# ==========================================
str(student_data)
summary(student_data)

# Check missing values
missing_counts <- colSums(is.na(student_data))
print(missing_counts)
missing_percentage <- (colSums(is.na(student_data)) / nrow(student_data)) * 100
print(missing_percentage)

# 1. Convert Categorical / Text Variables to Factors
categorical_vars <- c("Category", "Course_Level", "Employment_Status", 
                      "Education_Level", "Device_Type", "Internet_Connection_Quality", 
                      "Payment_Mode", "Gender")

for(col in categorical_vars) {
  if(col %in% names(student_data)) {
    student_data[[col]] <- as.factor(student_data[[col]])
  }
}

# 2. Filter Out Inactive Attempts (< 1 Hour)
student_data <- student_data[student_data$Time_Spent_Hours >= 1, ]

# 3. Feature Creation (Including Log Transformation)
student_data <- student_data %>%
  mutate(
    log_Time_Spent = log1p(Time_Spent_Hours),
    Total_Engagement_Score = Time_Spent_Hours * (Video_Completion_Rate / 100),
    Inactivity_Penalty_Ratio = Days_Since_Last_Login / (Progress_Percentage + 1),
    Paid_Engagement = ifelse(Payment_Amount > 0, 1, 0)
  )

# 4. Drop Unrelated Administrative Columns
cols_to_drop <- c("Student_ID", "Enroll_Date", "Course_Name")
student_data <- student_data[, !(names(student_data) %in% cols_to_drop)]

# 5. Target Variable Encoding
student_data$Completed_num <- ifelse(student_data$Completed == "Completed" | student_data$Completed == "Complete", 1, 0)

# 6. IQR-BASED OUTLIER REMOVAL (NEW SECTION)
num_cols_for_outliers <- c("Time_Spent_Hours", "Quiz_Score_Avg", "Days_Since_Last_Login", 
                           "Assignments_Missed", "Video_Completion_Rate", 
                           "Progress_Percentage", "Project_Grade", "App_Usage_Percentage")

rows_before_iqr <- nrow(student_data)

remove_outliers_iqr <- function(df, cols) {
  keep_mask <- rep(TRUE, nrow(df))
  for (col in cols) {
    if (col %in% names(df)) {
      Q1 <- quantile(df[[col]], 0.25, na.rm = TRUE)
      Q3 <- quantile(df[[col]], 0.75, na.rm = TRUE)
      IQR_val <- Q3 - Q1
      lower_bound <- Q1 - 1.5 * IQR_val
      upper_bound <- Q3 + 1.5 * IQR_val
      keep_mask <- keep_mask & (df[[col]] >= lower_bound & df[[col]] <= upper_bound)
    }
  }
  return(df[keep_mask, ])
}

student_data <- remove_outliers_iqr(student_data, num_cols_for_outliers)
cat("Rows removed by IQR outlier filter:", rows_before_iqr - nrow(student_data), "\n")
cat("Rows remaining after outlier filtering:", nrow(student_data), "\n")


# ==========================================
# 03 DESCRIPTIVE STATISTICS & VISUALIZATIONS
# ==========================================
student_data %>% 
  summarise(
    Mean_Time_Spent = mean(Time_Spent_Hours, na.rm = TRUE),
    SD_Time_Spent = sd(Time_Spent_Hours, na.rm = TRUE),
    Mean_Quiz_Score = mean(Quiz_Score_Avg, na.rm = TRUE),
    SD_Quiz_Score = sd(Quiz_Score_Avg, na.rm = TRUE)
  )

# 1. Target Variable Distribution Bar Chart
ggplot(student_data, aes(x = factor(Completed))) +
  geom_bar(fill = c("#ff9999", "#66b3ff")) +
  labs(title = "Course Completion Distribution", x = "Completion Status (0 = Dropout, 1 = Complete)", y = "Count")

# 2. Boxplot of Time Spent Hours by Completion
ggplot(student_data, aes(x = factor(Completed), y = Time_Spent_Hours, fill = factor(Completed))) +
  geom_boxplot() +
  labs(title = "Time Spent Hours by Completion Status (Post Outlier Removal)", x = "Completion Status", y = "Time Spent (Hours)")

# 3. Quiz Score Distribution
hist(student_data$Quiz_Score_Avg, main = "Histogram of Quiz Scores", xlab = "Average Quiz Score", col = "lightblue")

# 4. Faceted Histogram for Time Spent
ggplot(student_data, aes(x = Time_Spent_Hours, fill = factor(Completed_num))) +
  geom_histogram(bins = 30, color = "black", show.legend = FALSE) +
  facet_wrap(
    ~ Completed_num, 
    labeller = as_labeller(c("0" = "Dropout", "1" = "Completed"))
  ) +
  scale_fill_manual(values = c("#ff9999", "#66b3ff")) +
  labs(
    title = "Time Spent Hours Distribution by Completion Status",
    x = "Time Spent (Hours)",
    y = "Frequency"
  ) +
  theme_minimal()


# ==========================================
# 04 INFERENCE & HYPOTHESIS TESTING
# ==========================================
t.test(Time_Spent_Hours ~ Completed, data = student_data)
summary(aov(Quiz_Score_Avg ~ Course_Level, data = student_data))
chisq.test(student_data$Completed, student_data$Gender)
var.test(Time_Spent_Hours ~ Completed, data = student_data)
chisq.test(student_data$Completed, student_data$Employment_Status)
chisq.test(student_data$Completed, student_data$Device_Type)


# ==========================================
# 05 TRAIN-TEST SPLIT (70% Train, 30% Test)
# ==========================================
set.seed(123)
train_index <- sample(nrow(student_data), 0.7 * nrow(student_data))
train_data  <- student_data[train_index, ]
test_data   <- student_data[-train_index, ]


# ==========================================
# 06 MODELING & FEATURE SELECTION
# ==========================================
# 1. Best Subset Selection
subset_model <- regsubsets(Completed_num ~ log_Time_Spent + Quiz_Score_Avg + 
                             Days_Since_Last_Login + Assignments_Missed + 
                             Video_Completion_Rate + Progress_Percentage + 
                             Project_Grade + App_Usage_Percentage, 
                           data = train_data, nvmax = 8, method = "exhaustive")
subset_summary <- summary(subset_model)
best_bic_index <- which.min(subset_summary$bic)
coef(subset_model, best_bic_index)

# 2. Forward & Backward Selection
null_model <- glm(Completed_num ~ 1, data = train_data, family = "binomial")
full_model <- glm(Completed_num ~ log_Time_Spent + Quiz_Score_Avg + 
                    Days_Since_Last_Login + Assignments_Missed + 
                    Video_Completion_Rate + Progress_Percentage + 
                    Project_Grade + App_Usage_Percentage + 
                    Total_Engagement_Score + Inactivity_Penalty_Ratio + Course_Level, 
                  data = train_data, family = "binomial")

forward_model  <- step(null_model, scope = list(lower = null_model, upper = full_model), direction = "forward")
backward_model <- step(full_model, direction = "backward")

# 3. Optimized Logistic Regression Model
final_opt_model <- glm(Completed_num ~ Progress_Percentage + log_Time_Spent + 
                         Video_Completion_Rate + Quiz_Score_Avg + 
                         Days_Since_Last_Login + App_Usage_Percentage + 
                         Total_Engagement_Score + Inactivity_Penalty_Ratio + Course_Level, 
                       data = train_data, family = "binomial")

preds_log <- predict(final_opt_model, newdata = test_data, type = "response")
class_log <- ifelse(preds_log > 0.5, 1, 0)
mat_log   <- table(Predicted = class_log, Actual = test_data$Completed_num)
acc_log   <- (sum(diag(mat_log)) / sum(mat_log)) * 100

# 4. Best Subset Model Evaluation
subset_opt_model <- glm(Completed_num ~ log_Time_Spent + Quiz_Score_Avg + 
                          Days_Since_Last_Login + Video_Completion_Rate + 
                          Progress_Percentage, data = train_data, family = "binomial")
preds_sub <- predict(subset_opt_model, newdata = test_data, type = "response")
class_sub <- ifelse(preds_sub > 0.5, 1, 0)
mat_sub   <- table(Predicted = class_sub, Actual = test_data$Completed_num)
acc_sub   <- (sum(diag(mat_sub)) / sum(mat_sub)) * 100

# 5. Regularized Regression (LASSO, Ridge, Elastic Net)
x_train <- model.matrix(Completed_num ~ Progress_Percentage + log_Time_Spent + 
                          Video_Completion_Rate + Quiz_Score_Avg + 
                          Days_Since_Last_Login + App_Usage_Percentage + 
                          Total_Engagement_Score + Inactivity_Penalty_Ratio + Course_Level, data = train_data)[,-1]
y_train <- train_data$Completed_num

x_test <- model.matrix(Completed_num ~ Progress_Percentage + log_Time_Spent + 
                         Video_Completion_Rate + Quiz_Score_Avg + 
                         Days_Since_Last_Login + App_Usage_Percentage + 
                         Total_Engagement_Score + Inactivity_Penalty_Ratio + Course_Level, data = test_data)[,-1]
y_test <- test_data$Completed_num

# LASSO (Alpha = 1)
set.seed(123)
cv_lasso    <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)
lasso_preds <- predict(cv_lasso, newx = x_test, s = "lambda.min", type = "response")
lasso_class <- ifelse(lasso_preds > 0.5, 1, 0)
acc_lasso   <- (sum(diag(table(Predicted = lasso_class, Actual = y_test))) / length(y_test)) * 100

# Ridge (Alpha = 0)
set.seed(123)
cv_ridge    <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0)
ridge_preds <- predict(cv_ridge, newx = x_test, s = "lambda.min", type = "response")
ridge_class <- ifelse(ridge_preds > 0.5, 1, 0)
acc_ridge   <- (sum(diag(table(Predicted = ridge_class, Actual = y_test))) / length(y_test)) * 100

# Elastic Net (Alpha = 0.5)
set.seed(123)
cv_enet    <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0.5)
enet_preds <- predict(cv_enet, newx = x_test, s = "lambda.min", type = "response")
enet_class <- ifelse(enet_preds > 0.5, 1, 0)
acc_enet   <- (sum(diag(table(Predicted = enet_class, Actual = y_test))) / length(y_test)) * 100


# ==========================================
# 07 PCA, NAIVE BAYES & BAYESIAN REGRESSION
# ==========================================
# PCA
pca_data <- train_data[, c("Progress_Percentage", "log_Time_Spent", "Video_Completion_Rate", 
                           "Quiz_Score_Avg", "Days_Since_Last_Login", "App_Usage_Percentage",
                           "Total_Engagement_Score", "Inactivity_Penalty_Ratio")]
pca_result <- prcomp(pca_data, scale. = TRUE)
summary(pca_result)
plot(pca_result, type = "l", main = "PCA Variance")

# Naive Bayes
nb_model <- naiveBayes(as.factor(Completed_num) ~ Progress_Percentage + log_Time_Spent + 
                         Video_Completion_Rate + Quiz_Score_Avg + 
                         Days_Since_Last_Login + App_Usage_Percentage + Course_Level, 
                       data = train_data)
nb_preds <- predict(nb_model, newdata = test_data)
nb_acc   <- (sum(diag(table(Predicted = nb_preds, Actual = test_data$Completed_num))) / nrow(test_data)) * 100

# Bayesian Regression
bayes_reg_model <- bayesglm(Completed_num ~ Progress_Percentage + log_Time_Spent + 
                              Video_Completion_Rate + Quiz_Score_Avg + 
                              Days_Since_Last_Login + App_Usage_Percentage + Course_Level, 
                            data = train_data, family = binomial(link = "logit"))

bayes_preds <- predict(bayes_reg_model, newdata = test_data, type = "response")
bayes_class <- ifelse(bayes_preds > 0.5, 1, 0)
bayes_acc   <- (sum(diag(table(Predicted = bayes_class, Actual = test_data$Completed_num))) / nrow(test_data)) * 100

# Optimal Threshold Search
thresholds <- seq(0.3, 0.7, by = 0.05)
best_acc   <- 0
best_thresh <- 0.5

for(t in thresholds) {
  temp_class   <- factor(ifelse(bayes_preds > t, 1, 0), levels = c(0, 1))
  actual_class <- factor(test_data$Completed_num, levels = c(0, 1))
  temp_matrix  <- table(Predicted = temp_class, Actual = actual_class)
  temp_acc     <- sum(diag(temp_matrix)) / nrow(test_data)
  
  if(temp_acc > best_acc) {
    best_acc    <- temp_acc
    best_thresh <- t
  }
}


# ==========================================
# 08 VERIFICATION & SUMMARY OUTPUTS
# ==========================================
cat("\n=== DATASET SUMMARY ===\n")
cat("Total rows remaining:", nrow(student_data), "\n")
cat("Total variables:", ncol(student_data), "\n")
cat("Total missing values:", sum(is.na(student_data)), "\n")
cat("Missing values (%):", mean(is.na(student_data)) * 100, "%\n")

cat("\nTarget variable counts:\n")
print(table(student_data$Completed_num))

cat("\n=== HYPOTHESIS TESTING ===\n")
print(t.test(Time_Spent_Hours ~ Completed, data = student_data))
print(summary(aov(Quiz_Score_Avg ~ Course_Level, data = student_data)))
print(var.test(Time_Spent_Hours ~ Completed, data = student_data))
print(chisq.test(student_data$Completed, student_data$Gender))
print(chisq.test(student_data$Completed, student_data$Employment_Status))
print(chisq.test(student_data$Completed, student_data$Device_Type))

cat("\n=== FINAL ACCURACY COMPARISON ===\n")
cat("Standard Logistic Regression:", round(acc_log, 2), "%\n")
cat("Best Subset Regression:", round(acc_sub, 2), "%\n")
cat("LASSO Regression:", round(acc_lasso, 2), "%\n")
cat("Ridge Regression:", round(acc_ridge, 2), "%\n")
cat("Elastic Net Regression:", round(acc_enet, 2), "%\n")
cat("Naive Bayes:", round(nb_acc, 2), "%\n")
cat("Bayesian Regression:", round(bayes_acc, 2), "%\n")
cat("Optimal Bayesian Threshold:", best_thresh, "| Accuracy:", round(best_acc * 100, 2), "%\n")