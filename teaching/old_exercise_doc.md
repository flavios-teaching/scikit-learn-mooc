# 2023-11-28-ds-sklearn exercises document

## Day 1
### Exercise: Machine learning concepts
Given a case study: pricing apartments based on a real estate website. We have thousands of house descriptions with their price. Typically, an example of a house description is the following:

“Great for entertaining: spacious, updated 2 bedroom, 1 bathroom apartment in Lakeview, 97630. The house will be available from May 1st. Close to nightlife with private backyard. Price ~$1,000,000.”

We are interested in predicting house prices from their description. One potential use case for this would be, as a buyer, to find houses that are cheap compared to their market value.

#### What kind of problem is it?

a) a supervised problem
b) an unsupervised problem
c) a classification problem
d) a regression problem

Select all answers that apply

#### What are the features?

a) the number of rooms might be a feature
b) the post code of the house might be a feature
c) the price of the house might be a feature

Select all answers that apply

#### What is the target variable?

a) the full text description is the target
b) the price of the house is the target
c) only house description with no price mentioned are the target

Select a single answer

#### What is a sample?

a) each house description is a sample
b) each house price is a sample
c) each kind of description (as the house size) is a sample

Select a single answer


### Exercise: Data exploration (15min,  in groups)

Imagine we are interested in predicting penguins species based on two of their body measurements: culmen length and culmen depth. First we want to do some data exploration to get a feel for the data.

The data is located in `../datasets/penguins_classification.csv`.

Load the data with Python and try to answer the following questions:
1. How many features are numerical? How many features are categorical?
2. What are the different penguins species available in the dataset and how many samples of each species are there?
3. Plot histograms for the numerical features
4. Plot features distribution for each class (Hint: use `seaborn.pairplot`).
5. Looking at the distributions you got, how hard do you think it will be to classify the penguins only using "culmen depth" and "culmen length"?

### 📝 Exercise (in breakout rooms): Adapting your first model
The goal of this exercise is to fit a similar model as we just did to get familiar with manipulating scikit-learn objects and in particular the `.fit/.predict/.score` API.

Before we used `model = KNeighborsClassifier()`. All scikit-learn models can be created without arguments. This is convenient because it means that you don’t need to understand the full details of a model before starting to use it.

One of the KNeighborsClassifier parameters is n_neighbors. It controls the number of neighbors we are going to use to make a prediction for a new data point.

#### 1. What is the default value of the n_neighbors parameter? 
Hint: Look at the documentation on the scikit-learn website or directly access the description inside your notebook by running the following cell. This will open a pager pointing to the documentation.
```python
from sklearn.neighbors import KNeighborsClassifier

KNeighborsClassifier?
```

#### 2. Create a KNeighborsClassifier model with n_neighbors=50
a. Fit this model on the data and target loaded above
b. Use your model to make predictions on the first 10 data points inside the data. Do they match the actual target values?
c. Compute the accuracy on the training data.
d. Now load the test data from "../datasets/adult-census-numeric-test.csv" and compute the accuracy on the test data.


### Exercise (in breakout rooms): Compare with simple baselines
The goal of this exercise is to compare the performance of our classifier in the previous notebook (roughly 81% accuracy with LogisticRegression) to some simple baseline classifiers. The simplest baseline classifier is one that always predicts the same class, irrespective of the input data.

What would be the score of a model that always predicts ' >50K'?

What would be the score of a model that always predicts ' <=50K'?

Is 81% or 82% accuracy a good score for this problem?

Use a DummyClassifier such that the resulting classifier will always predict the class ' >50K'. What is the accuracy score on the test set? Repeat the experiment by always predicting the class ' <=50K'.

Hint: you can set the strategy parameter of the DummyClassifier to achieve the desired behavior.

You can import DummyClassifier like this:
```python
from sklearn.dummy import DummyClassifier
```

### Exercise: Recap fitting a scikit-learn model on numerical data
#### 1. Why do we need two sets: a train set and a test set?

a) to train the model faster
b) to validate the model on unseen data
c) to improve the accuracy of the model

Select all answers that apply

#### 2. The generalization performance of a scikit-learn model can be evaluated by:

a) calling fit to train the model on the training set, predict on the test set to get the predictions, and compute the score by passing the predictions and the true target values to some metric function
b) calling fit to train the model on the training set and score to compute the score on the test set
c) calling cross_validate by passing the model, the data and the target
d) calling fit_transform on the data and then score to compute the score on the test set

Select all answers that apply

#### 3. When calling `cross_validate(estimator, X, y, cv=5)`, the following happens:

a) X and y are internally split five times with non-overlapping test sets
b) estimator.fit is called 5 times on the full X and y
c) estimator.fit is called 5 times, each time on a different training set
d) a Python dictionary is returned containing a key/value containing a NumPy array with 5 scores computed on the train sets
e) a Python dictionary is returned containing a key/value containing a NumPy array with 5 scores computed on the test sets

Select all answers that apply

#### 4. (optional) Scaling
We define a 2-dimensional dataset represented graphically as follows:
![](https://i.imgur.com/muvSbI6.png)

Question

If we process the dataset using a StandardScaler with the default parameters, which of the following results do you expect:

![](https://i.imgur.com/t5mTlVG.png)


a) Preprocessing A
b) Preprocessing B
c) Preprocessing C
d) Preprocessing D

Select a single answer

#### 5. (optional) Cross-validation allows us to:

a) train the model faster
b) measure the generalization performance of the model
c) reach better generalization performance
d) estimate the variability of the generalization score

Select all answers that apply

## Day 2

### Handling categorical data (Dani's part)

#### Ordinal encoding (everyone gives their answers in the collaborative doc):

Q1: Is ordinal encoding is appropriate for marital status? For which (other) categories in the adult census would it be appropriate? Why?
Q2: Can you think of another example of categorical data that is ordinal?
Q3: What problem arises if we use ordinal encoding on a sizing chart with options: XS, S, M, L, XL, XXL? (HINT: explore `ordinal_encoder.categories_`)
Q4: How could you solve this problem? (Look in documentation of OrdinalEncoder)
Q5: Can you think of an ordinally encoded variable that would not have this issue?

*Answers:*
A1: Only education (in fact, the encoder was already present in the data set as education-num), as this is the only one that can be expressed as an incremental feature
A2: Examples could be:
  - Alphabetized: US grading system: A, B, C, D, F
  - Not alphabetized: clothing sizes: XS, S, M, L, XL, XXL

A3: Would not be in correct order (it's alphabetized).
A4: top of documention will tell you to use `categories` argument with a list in the correct order
```
ordered_size_list = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
encoder_with_order = OrdinalEncoder(categories=ordered_size_list)
```
A5: US grading scheme is alphabetical to begin with (A,B,C,D,F)


#### Exercise: The impact of using integer encoding for with logistic regression (breakout rooms of 3-4 people):
first load the data:
```python=
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])
```

Q1: Use `sklearn.compose.make_column_selector` to automatically select columns containing strings
that correspond to categorical features in our dataset.

Q2: Define a scikit-learn pipeline composed of an `OrdinalEncoder` and a `LogisticRegression` classifier and
evaluate it using cross-validation.
*Note*: Because `OrdinalEncoder` can raise errors if it sees an unknown category at prediction time,  you can set the `handle_unknown="use_encoded_value"` and `unknown_value=-1` parameters. You can refer to the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)
for more details regarding these parameters.

Q3: Now, compare the generalization performance of our previous
model with a new model where instead of using an `OrdinalEncoder`, we will
use a `OneHotEncoder`. Repeat the model evaluation using cross-validation.
Compare the score of both models and conclude on the impact of choosing a
specific encoding strategy when using a linear model.


#### Quiz (if time permits): everyone answers in collaborative doc:
Select all true answers for each question
<!-- (how to remove colored formatting below?) -->

```{admonition} Question
Q1: How are categorical variables represented?
- a) a categorical feature is only represented by non-numerical data
- b) a categorical feature represents a finite number of values called categories
- c) a categorical feature can either be represented by numerical or
     non-numerical values
```

```{admonition} Question
Q2: An ordinal variable:
- a) is a categorical variable with a large number of different categories;
- b) can be represented by integers or string labels;
- c) is a categorical variable with a meaningful order.
```

```{admonition} Question
Q3: One-hot encoding:
- a) encodes each column with string-labeled values into a
     single integer-coded column
- b) transforms a numerical variable into a categorical variable
- c) creates one additional column for each possible category
- d) transforms string-labeled variables using a numerical representation
```

```{admonition} Question
Q4: Assume we have a dataset where each line describes a company. Which of the
following columns should be considered as a meaningful numerical feature to
train a machine learning model to classify companies:
- a) the sector of activity ("construction", "retail", "energy", "insurance"...)
- b) the phone number of the sales department
- c) the number of employees
- d) the profits of the last quarter
- e) the post code of the head quarters
```

#### Exercise: the impact of feature preprocessing on a pipeline that uses a decision-tree-based classifier
Again, load the data first:
```python=
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])
```

Q1: Measure the time and accuracy of the reference model (the model we used in the main notebook).
*Note*: remember that you can use the time module to measure elapsed time (Sven did this in his module this morning)  

Q2: Now write a similar pipeline that also scales the numerical features using `StandardScaler` (or similar). 
How does this compare to the reference?

Q3: Now let's see if we can improve the model using One-Hot encoding. Recreate the pipeline, but using one-hot encoding instead of ordinal encoding. 
How does this compare to the previous 2 pipelines?
*Note*: `HistGradientBoostingClassifier` does not yet support sparse input data. Use `OneHotEncoder(handle_unknown="ignore", sparse=False)` to force the use of a dense representation as a workaround.


## Day 3
### Exercise: overfitting and underfitting:

#### 1: A model that is underfitting:

a) is too complex and thus highly flexible
b) is too constrained and thus limited by its expressivity
c) often makes prediction errors, even on training samples
d) focuses too much on noisy details of the training set

Select all answers that apply

#### 2: A model that is overfitting:

a) is too complex and thus highly flexible
b) is too constrained and thus limited by its expressivity
c) often makes prediction errors, even on training samples
d) focuses too much on noisy details of the training set

Select all answers that apply

### Exercise: Train and test SVM classifier (30 min,  2 people / breakout room)

The aim of this exercise is to:
* train and test a support vector machine classifier through cross-validation;
* study the effect of the parameter gamma (one of the parameters controlling under/over-fitting in SVM) using a validation curve;
* determine the usefulness of adding new samples in the dataset when building a classifier using a learning curve. 

We will use blood transfusion dataset located in `../datasets/blood_transfusion.csv`. First take a data exploration to get familiar with the data.

You can then start off by creating a predictive pipeline made of:

* a [`sklearn.preprocessing.StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) with default parameter;
* a [`sklearn.svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

Script below will help you get started:

```python=
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

model = make_pipeline(StandardScaler(), SVC())
```

You can vary gamma between 10e-3 and 10e2 by generating samples on a logarithmic scale with the help of

```python=
gammas = np.logspace(-3, 2, num=30)
param_name = "svc__gamma"
```

To manipulate training size you could use:

```python=
train_sizes = np.linspace(0.1, 1, num=10)
```

### Exercise: bias vs variance
#### 1. Fitting a model with a high bias:

a) causes an underfitted model?
b) causes an overfitted model?
c) increases the sensitivity of the learned prediction function to a random resampling of the training set observations?
d) causes the learned prediction function to make systematic errors?

Select all answers that apply

#### 2. Fitting a high variance model:

a) causes an underfitted model?
b) causes an overfitted model?
c) increases the sensitivity of the learned prediction function to a random resampling of the training set observations?
d) causes the learned prediction function to make systematic errors?

Select all answers that apply

### Wrapping up exercise (1 hour, only if time):
https://inria.github.io/scikit-learn-mooc/overfit/overfit_wrap_up_quiz.html

## Day 4

### Exercise: Play with penguins data (1h, 3-4 people / breakout room ):
Apply what we have learned in this week and try out different models on the penguins dataset. 

The data is located in`../datasets/penguins.csv`. Awalys take a data exploration first to get familiar with the data.

Hints for available models:
```python=
# classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
```

Goals:
-  Recap and use what we have learned
-  Make your workflow for developing ML models