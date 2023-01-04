#'*GROUP 6: FINAL PROJECT - R Code*
#'*Data Exploration - Visualizations, Summary and Structure*
#'*Method : Support Vector Machines*

##'*----------------------------------------------------------------------*
##'*Data Exploration*
##'*----------------------------------------------------------------------*
#Clearing workspace
rm(list=ls())

# Setting working directory
setwd("C:/Users/neela/Documents/Spring Quarter/Data Mining/Final Project")

# Loading relevant packages
library(caret)
library(caretEnsemble)
library(ggplot2)

# Reading data file
ER <- read.csv(file = "EmployeeRetention.csv",
               stringsAsFactors = FALSE)
# Checking structure
str(ER)

#summary of ER dataset
summary(ER)

#Distribution of attrition rate over age of employees
boxplot(formula = Age ~ Attrition, 
        data = ER,
        col = cm.colors(2))

#Distribution of attrition rate over no. of companies worked at
boxplot(formula = NumCompaniesWorked ~ Attrition, 
        data = ER,
        col = cm.colors(2))

#Distribution of attrition rate over monthly income
boxplot(formula = MonthlyIncome ~ Attrition, 
        data = ER,
        col = cm.colors(2))
# Preparing Target (Y) Variable
ER$Attrition <- as.factor(ER$Attrition)

#Distribution of attrition rate over job level
ggplot(data = ER, aes (JobLevel)) + geom_bar(aes(fill= Attrition) , position = "fill") + scale_y_continuous(labels = scales :: percent)

#Checking class imbalance
plot(ER$Attrition,
     main = "Attrition")

##'*----------------------------------------------------------------------*
##'*SUPPORT VECTOR MACHINE*
##'*----------------------------------------------------------------------*
##'*Data Pre-Processing*
##'*----------------------------------------------------------------------*

# Identifying Nominal (Unordered) Variables
noms <- c("Department", "Gender", "JobRole", "MaritalStatus", "OverTime",
          "EducationField")

# Factorizing nominal (Unordered) Variables
ER[ ,noms] <- lapply(X = ER[ ,noms],
                     FUN = factor)

# Ordinal (ordered) Factor Variables (removing Over18 as it has just 1 value)
# Identifying Ordinal Variables
# Putting all ordinal variables except 'BusinessTravel' in a vector 'ord_others'
# Factoring 'BusinessTravel' separately as it has text levels and factoring
# all the other ordinal variables together

##'* Ordinal Variables*
ords_others <- c("EnvironmentSatisfaction", "JobInvolvement",
                 "JobLevel", "JobSatisfaction", "PerformanceRating", "RelationshipSatisfaction",
                 "StockOptionLevel", "WorkLifeBalance", "Education")

ER[,ords_others] <- lapply(X =ER[,ords_others], FUN = factor, ordered = TRUE)

ER$BusinessTravel <- factor(x = ER$BusinessTravel ,
                            levels = c("Non-Travel", "Travel_Rarely", "Travel_Frequently"),
                            ordered = TRUE)

ords <- c("BusinessTravel", ords_others)

# Combining all ordinal variables in vector 'vars'
ords <- c("BusinessTravel", ords_others)

# Identifying all numerical Variables
# Removing target variable 'Attrition' as we're only including predictor variables
# Removing 'Over 18' variable as it has only one value for all observations
nums <- c("Age", "DistanceFromHome", "HourlyRate", "MonthlyIncome",
          "NumCompaniesWorked", "PercentSalaryHike", "TotalWorkingYears",
          "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole",
          "YearsSinceLastPromotion", "YearsWithCurrManager")

#Creating vector of all predictor variables
vars <- c(noms, ords, nums)

## SVM can handle redundant variables, but
## Missing values need to be handled, categorical
## Variables need to be binarized and rescaling should be done

# Checking for missing values
any(is.na(ER))

##'*There are no missing values in our data*

# Transforming Categorical Variables:
# Binarizing Nominal Variables and creating dummy variables for
# nominal variables with more than two levels

dep <- dummyVars(formula = ~ Department,
                 data = ER)
dep_dums <- predict(object = dep,
                    newdata = ER)

job <- dummyVars(formula = ~ JobRole,
                 data = ER)
job_dums <- predict(object = job,
                    newdata = ER)

ms <- dummyVars(formula = ~ MaritalStatus,
                data = ER)
ms_dums <- predict(object = ms,
                   newdata = ER)

ef <- dummyVars(formula = ~ EducationField,
                data = ER)
ef_dums <- predict(object = ef,
                   newdata = ER)

## Updated variables to use*

ER_dum <- data.frame(ER[ ,!names(ER) %in% c("Over18","Department", "JobRole","MaritalStatus",
                                            "EducationField") ],dep_dums, job_dums, ms_dums, ef_dums)

##'*checking for Outliers*

##'*Since there aren't any continuous numerical variable in our dataset*
##'* we are not going to check for outliers*

## Normalize Numeric Variables
# We will use min-max normalization to
# rescale our numeric variables during
# hyperparameter tuning.

## Training & Testing

# We use the createDataPartition() function
# from the caret package to create our
# training and testing sets using an
# 75/25 split.
set.seed(831) # initialize the random seed

# Generate the list of observations for the
# train dataframe
sub <- createDataPartition(y = ER_dum$Attrition,
                           p = 0.75,
                           list = FALSE)

# Create our train and test sets
train_svm <- ER_dum[sub, ]
test_svm <- ER_dum[-sub, ]

str(train_svm)
str(test_svm)

## Checking for Class Imbalance
# We can use the plot() function to create a barplot
plot(ER$Attrition,
     main = " Attrition Type", col= ("Orange"))

## Class imbalance exists. We will use weightd methid to deal with this issue
## Allocating higher weight to minority class
# We use the upSample() function in the caret

# allows case weighting.
target_var <- train_svm$Attrition # identify target variable

# Typically, we set the class weights to be
# inversely proportional ("balanced")
# If class-weights are needed
weights <- c(sum(table(target_var))/(nlevels(target_var)*table(target_var)))
weights
# If case-weights are needed
wghts <- weights[match(x = target_var,
                       table = names(weights))]


ctrl_svm <- trainControl(method = "repeatedcv",
                         number = 5,
                         repeats = 3,
                         search = "random")

##'*Removing the sampling argument in control object*

ctrl_svm$sampling <- NULL

## Initialize the random seed
set.seed(280)

SVMFit <- train(form = Attrition ~ .,
                data = train_svm,
                method = "svmRadial",
                preProcess = "range",
                trControl = ctrl_svm,
                tuneLength = 5,
                trace = FALSE,
                weights=wghts)
SVMFit

tuned.train.preds_svm <- predict(object = SVMFit,
                                 newdata = train_svm)

SVM_trtune_conf <- confusionMatrix(data = tuned.train.preds_svm,
                                   reference = train_svm$Attrition,
                                   positive = "Yes",
                                   mode = "everything")
SVM_trtune_conf

tuned.test.preds_svm <- predict(object = SVMFit,
                                newdata = test_svm)

SVM_tetune_conf <- confusionMatrix(data = tuned.test.preds_svm,
                                   reference = test_svm$Attrition,
                                   positive = "Yes",
                                   mode = "everything")
SVM_tetune_conf

# Overall
cbind(svm_train = SVM_trtune_conf$overall,
      svm_test = SVM_tetune_conf$overall)
# Class-Level
cbind(svm_train = SVM_trtune_conf$byClass,
      svm_test = SVM_tetune_conf$byClass)

plot(varImp(SVMFit),top = 20, main = "Variable Importance in SVM")
#'*-------------------------------------------------------------------'

##'*----------------------------------------------------------------------*
##'*Data Exploration Based On Variable Importance*
##'*----------------------------------------------------------------------*

# Distribution of Attrition over the most important ordinal, nominal and
# numerical variables

# Distribution of attrition rate over the variable 'OverTime'
p1 <- ggplot(data=ER) +
  aes(x=OverTime, y=Attrition) +
  geom_bar(stat="identity", fill="red")

# Distribution of attrition rate over job level
p2 <- ggplot(data=ER) +
  aes(x=JobLevel, y=Attrition) +
  geom_bar(stat="identity", fill="red")

# Distribution of attrition rate over monthly income
plot(ER$Attrition, ER$MonthlyIncome, 
     col = c("pink", "grey"),
     main = "Distribution of Monthly Income over Attrition",
     xlab = "Attrition",
     ylab = "Monthly Income")

#R.data file
save.image(file = "SVM Analysis_Group6.RData")

