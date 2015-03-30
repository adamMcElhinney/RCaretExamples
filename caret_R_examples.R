library('caret')
library(mlbench)
library(pls)
# Taken from here: https://github.com/adam-m-mcelhinney/helpRFunctions
install.packages('devtools') # Only needed if you dont have this installed.
library(devtools)
install_github('adam-m-mcelhinney/helpRFunctions')
library(helpRFunctions)
library(RANN)

# Note to get the benefits of parallelization with R, all you have to do is
# register a cluster, just like in the parallel R presentation!
library('doParallel')
library('foreach')
stopCluster(cl)
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)
getDoParWorkers()


# Determine Rock or Metal (R/M) based off signals from 60 sonar frequencies
# Frequency values normalized between 0 - 1
# k = 61, n = 208
data(Sonar)
head(Sonar)
summary(Sonar)


# Let's introduce some missing values to demonstrate functionality
df <- Sonar
nMissing <- 15
for (i in 1:nMissing){
  rowRand <- round(runif(n = 1, 1, nrow(Sonar)))
  colRand <- round(runif(n = 1, 1, ncol(Sonar) -1))
  df[rowRand, colRand] <- NA
}

# Lets introduce some highly correlated values to demonstrate functionality
df[,'corVar'] <- df[,1] * runif(n= nrow(df), min = 0, max = .005)
cor(df[,'corVar'], df[,1])

# Let's introduce some near zero variance Variables
df[,'junkVar1'] <- rep(NA, nrow(df))
df[,'junkVar2'] <- rep(1, nrow(df))

summary(df)

# Caret can help us clean up these variables
# Identify variables with zero or near zero variance
# Use saveMetrics = TRUE to view the output
# Use saveMetrics = FALSE to create an index to use to remove the values
nzv <- nearZeroVar(df, saveMetrics= TRUE)
nzv

nzv <- nearZeroVar(df, saveMetrics= FALSE)
df <- df[, -nzv]

# Find highly correlated values
# Note that you can only feed in numerical variables
cor(df) # Gives error


# This is a wrapper function that I wrote to simplify this
df.clean <- rmv.highly.correlated(df, verbose = TRUE, cutoff = .7)




# ALWAYS specify a random seed to ensure consist results from person-to-person
# and iteration to iteration to iteration
set.seed(34543)


# Split the data into training and testing sets

### First create an index for row numbers for training set
trainIndx <- createDataPartition(
  y = df$Class
  # What percent in training and therefore what pct in testing?
  , p = .5
  , list = FALSE
  )
nrow(trainIndx)
nrow(df.clean)

# Then split the data based on the index numbers
train <- df.clean[trainIndx,]
test <-df.clean[-trainIndx,]
stopifnot(nrow(train) + nrow(test) == nrow(df.clean))

yTrain = train$Class
xTrain = train[, names(train)!='Class']
yTest = test$Class
xTest= test[, names(test)!='Class']


# "train" an imputation model by creating a preProcess object, then applying
# to the data set you wish to have imputed
impute <- preProcess(xTrain, method = 'knnImpute')
xTrain2 <- predict(impute, xTrain)
xTest2 <-  predict(impute, xTest)


# preProcess objects also center and scale data for algorithms that require this
centerScale <- preProcess(xTrain, method = c('center', 'scale'))
xTrain3 <- predict(impute, xTrain)
xTest3 <-  predict(impute, xTest)



# Best practice workflow for using Caret is as follows
# 1. Determine the model you want to fit, based off the list of models from
# Caret's website
# http://topepo.github.io/caret/modelList.html
# For our example, let's use partial least squares
# 2. Note the tuning parameters for your models.
# For our example, its ncomp
# 3. Review the documentation for your technique and understand what the tuning
# parameters do. For the tuning parameters, understand what a reasonable min, max
# and granularity is.
# For pls, reasonable ranges are 1 - 30
# 4. Fit a model using the technique you want to use, WITHOUT Caret. Caret does
# not always display all the errors for the underlying package, so this first
# run is simply to validate that the technique will execute in R. Debug any issues.


simpls.fit(X = xTrain, Y = yTrain, ncomp = 10)
# Note that the above gives an error
# From reading the documentation and iterative debugging, we figure out the
# following.
testModel <- simpls.fit(X = as.matrix(xTrain), Y = as.matrix(yTrain == "M"), ncomp = 10)
testModel # Notice the NAs, bc pls doesn't support missing values. Good to know
# and this may not be explicit if we just ran caret first



impute <- preProcess(xTrain, method = 'medianImpute')
xTrain <- predict(impute, xTrain)
xTest <-  predict(impute, xTest)


# Let's fit a partial least squares model without any control parameters
plsFit <- train(Class ~ .,
                data = train,
                method = "pls")

plsFit

# Notice that it tested 3 components. However, let's test more

# Model fitting is specified by a trainControl function
# This simply serves to store relevant information related to model fitting
# Many options, but typically you want repeatedcv
ctrl <- trainControl(method = 'repeatedcv'
                     , repeats = 5
                     # Do you want to save the output of each resample?
                     # This can be useful for diagnosing problems, but typically
                     # you do not want to save this.
                     , savePredictions = FALSE
                     # Do you want to compute the predicted class probabilities
                     # for classification values? If not, it will just compute
                     # the predicted class. Again, typically FALSE, but if you
                     # want to do some diagnostics, this can be useful.
                     , classProbs = TRUE
                     , summaryFunction = twoClassSummary
                     )
ctrl #View all the possible options and their defaults
?trainControl # Documentation of all outputs



plsFit <- train(x = xTrain,
                y = yTrain, # Note that you can specify the formula form,
                            # but its NOT recommended
                method = "pls",
                tuneLength = 15, # this says let's test 15 different combos
                                # of our tuning parameters. For pls, this
                                # means we'll test 15 different values for ncomp
                                # However, for different models it will test
                                # whatever tuning parameters are relevant
                                # for that model.
                trControl = ctrl,
                metric = "ROC" # This says that we want to select the tuning
                              # parameters to MAXIMIZE this metric. Note that
                              # if you want to minimize a metric, you need to
                              # specify that as an additional option.
                              # Also note that you can define this function
                              # IE Jung Ha's function!!
                )

plsFit
plot(plsFit) # View how your metric changes as function of your tuning parameter


# Note that you can extract the "best" model. Sometimes you will encounter
# errors if you try to operate on the result of the caret train object.
# Extracting the final model from caret and then operating on that can be better
finalModel <- plsFit$finalModel[1]
finalModel # This is now equivilent to having just fit the model using pls package directly with ncomp = 12



# View the variable importance
# This is a really useful sanity check of your models. Do these variables make sense?
v <- varImp(plsFit, Scale = TRUE)
v
plot(v)

# Model validation
# This autmatically gives you TONS of useful info for model validation
# If the predict function gives you issues, try calling predict from the
# final model itself, not through caret.
yScoredProb <- predict(plsFit, newdata = xTest, type = 'prob')
head(yScoredProb)
yScoredClass <- predict(plsFit, newdata = xTest, type = 'raw')
head(yScoredClass)



confusionMatrix(data = yScoredClass, reference = yTest)


### Exercise
# Use caret to fit a C5.0 model. Test 100 different models. Maximize ROC.
# Tell me what tuning parameter results in the "best" model.


c5Fit <- train(x = xTrain,
                y = yTrain,
                method = "C5.0",
                tuneLength = 100,
                trControl = ctrl,
                metric = "ROC"
                )
plot(c5Fit)
c5Fit