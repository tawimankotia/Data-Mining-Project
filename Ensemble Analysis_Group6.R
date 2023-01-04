#'*GROUP 6: FINAL PROJECT - R Code*
#'*Data Exploration - Visualizations, Summary and Structure*
#'*Method : Ensemble Methods*

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
##'*Common Data Pre-Processing*
##'*----------------------------------------------------------------------*

# Nominal (Unordered) Factor Variables
noms <- c("Department", "Gender", "JobRole", "MaritalStatus", "OverTime", 
          "EducationField")

ER[ ,noms] <- lapply(X = ER[ ,noms], 
                     FUN = factor)

# Ordinal (ordered) Factor Variables (removing Over18 as it has just 1 value)

ER$BusinessTravel <- factor(x = ER$BusinessTravel, 
                levels = c("Non-Travel", "Travel_Rarely", "Travel_Frequently"),
                ordered = TRUE)

ords_others <- c("EnvironmentSatisfaction", "JobInvolvement", 
          "JobLevel", "JobSatisfaction", "PerformanceRating", 
          "RelationshipSatisfaction","StockOptionLevel", 
          "WorkLifeBalance", "Education")

ER[ ,ords_others] <- lapply(X = ER[ ,ords_others], 
                     FUN = factor)
ords <- c("BusinessTravel", ords_others)

# numerical Variables
nums <- names(ER)[!names(ER) %in% 
                              c(noms, ords,"Attrition", "Over18")]

#Column of all predictor variables
vars <- c(noms, ords, nums)

#summary of ER dataset with target variable
summary(ER[,c(vars, "Attrition")])

##'*----------------------------------------------------------------------*
##'*Data Pre-Processing - Ensemble Method*
##'*----------------------------------------------------------------------*

#train & test Split
sub <- createDataPartition(y = ER$Attrition,
                           p = 0.75,
                           list = FALSE)
train_em <- ER[sub, ]
test_em <- ER[-sub, ]

#Removing irrelevant variables
RFE_ctrl <- rfeControl(functions = rfFuncs, # RF-based
                       method = "cv", # k-fold CV
                       number = 5) # 5 fold
# Initializing random seed
set.seed(831)

# Using Recursive Feature Elimination (RFE) method
rfe_mod <- rfe(x = train_em[ ,vars], # predictor variables
               y = train_em$Attrition, # target variable
               sizes = 4:length(vars), # minimum of 4 variables in a classifier
               rfeControl = RFE_ctrl) # control object

# Viewing model output
rfe_mod 

# Checking optimal variables suggested by the model
rfe_mod$optVariables

# Substituting vars with optimal relevant variables found in previous step
vars <- c(rfe_mod$optVariables)

##'*----------------------------------------------------------------------*
##'*Model Fitting : Ensemble Methods*
##'*----------------------------------------------------------------------*

ctrl <- trainControl(method = "cv",
                     number = 5,
                     savePredictions = "final")

# Initializing random seed
set.seed(831)

#class imbalance - weights
model_weights <- ifelse(train_em$Attrition == "Yes",
                        (1/table(train_em$Attrition)[1]) * 0.5,
                        (1/table(train_em$Attrition)[2]) * 0.5)

# Removing the sampling argument in control object
ctrl$sampling <- NULL

# Training the ensemble models using caretList method
# Using class imbalance wieght for Bagging and RandomForest
# Not using weights for Boosting, as it applies weights to 
# misclassified records while training
ensemble_models <- caretList(x = train_em[ ,vars],
                             y = train_em$Attrition,
                             trControl = ctrl,
                             
                             tuneList = list(
                               bagging = caretModelSpec(method = "treebag",weights = model_weights),
                               randforest = caretModelSpec(method = "rf",
                                    tuneGrid = expand.grid(mtry = seq(from = 3,
                                                                        to = 28,
                                                                     by = 2)),weights = model_weights),
                               boosting = caretModelSpec(method = "adaboost",
                                                         tuneLength = 5)
                             ))

# Results of Bagging model
ensemble_models[["bagging"]]$results
# Results of Random Forest model
ensemble_models[["randforest"]]$results
# Results of Boosting model
ensemble_models[["boosting"]]$results

#combining and visualizing all our resampling results
results <- resamples(ensemble_models)

#Viewing summary to see the performance measures
summary(results)

##variable importance of all Ensemble Methods
# Bagging
plot(varImp(ensemble_models$bagging), main = "Variable Importance in Bagging")
# Random Forest
plot(varImp(ensemble_models$randforest), main = "Variable Importance in Rand Forest" )
# Boosting
plot(varImp(ensemble_models$boosting), main = "Variable Importance in Boosting Ensemble Method")


##'*----------------------------------------------------------------------*
##'*Model Performance : Ensemble Methods*
##'*----------------------------------------------------------------------*

## Making predictions using trained models 
ens_preds <- data.frame(predict(object = ensemble_models,
                                newdata = test_em),
                        stringsAsFactors = TRUE)

# Initializing a blank list ph
ph <- list()

# Assigning confusing matrix of all methods to list ph
for (i in 1:ncol(ens_preds)){
  ph[[i]] <- confusionMatrix(data = ens_preds[ ,i],
                             reference = test_em$Attrition,
                             positive = "Yes",
                             mode = "everything")
  names(ph)[[i]] <- colnames(ens_preds)[i]
}

# Overall performance of all ensemble methods
cbind(bag = ph[["bagging"]]$overall,
      rf = ph[["randforest"]]$overall,
      boost = ph[["boosting"]]$overall)


# Performance of all ensemble methods by class
cbind(bag = ph[["bagging"]]$byClass,
      rf = ph[["randforest"]]$byClass,
      boost = ph[["boosting"]]$byClass)

##'*----------------------------------------------------------------------*
##'*Goodness of Fit : Ensemble Methods*
##'*----------------------------------------------------------------------*

# Getting average training performance
t(matrix(colMeans(results$values[ ,-1]), 
         nrow = 3, 
         byrow = TRUE,
         dimnames = list(c("bag", "rf", "boost"), 
                         c("Accuracy", "Kappa"))))

# Getting average testing performance
cbind(bag = ph[["bagging"]]$overall,
      rf = ph[["randforest"]]$overall,
      boost = ph[["boosting"]]$overall)[1:2,]

##----------------------------------------------------------------------

#'*We chose boosting of all ensemble methods because it has better performance*
#'*measures than other methods here*

#R.data file
save.image(file = "Ensemble Analysis_Group6.RData")

