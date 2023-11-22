
# Predictive modeling pipeline

## Tabular data exploration

### Exercise M1.01. Exploring a dataset.

Work in groups of 2, with your neighbor. (15min) 

Imagine we are interested in predicting penguins species based on two of their body measurements: culmen length and culmen depth. First we want to do some data exploration to get a feel for the data.

The data is located in `../datasets/penguins_classification.csv`.

Load the data with Python and try to answer the following questions:
1. How many features are numerical? How many features are categorical?
2. What are the different penguins species available in the dataset and how many samples of each species are there?
3. Plot histograms for the numerical features
4. Plot features distribution for each class (Hint: use `seaborn.pairplot`).
5. Looking at the distributions you got, how hard do you think it will be to classify the penguins only using "culmen depth" and "culmen length"?

<mark> TODO -- Question for Sven </mark> what did he think about improving the exercise? changing the questions?
- perhaps drop the features distribution? -- seaborn
- but see below, there is a shorter way. show documentation? -- it's not the core.
- give them the command?

### Quiz M1.01 (ignored)

**1. In the code we previously wrote, we used pandas and specifically `adult_census = pd.read_csv("../datasets/adult-census.csv")` to:**
- a) load a comma-separated values file
- b) load a dataset already included in the pandas package
- c) load a file only containing the survey features
- d) load a file only containing the target of our classification problem: whether or not a person has a low or high income salary
- e) load a file containing both the features and the target for our classification problem

*Select all answers that apply*

**2. In the code, we used:**
- a) pandas to gain insights about the dataset
- b) pandas and seaborn to visually inspect the dataset
- c) numpy and scipy to perform numerical inspection (for instance using scipy.optimize.minimize)
- d) scikit-learn to fit some machine learning models

*Select all answers that apply*


**3. How is a tabular dataset organized?**
- a) a column represents a sample and a row represents a feature
- b) a column represents a feature and a row represents a sample
- c) the target variable is represented by a row
- d) the target variable is represented by a column

*Select all answers that apply*

**A categorical variable is:**
- a) a variable with only two different possible values
- b) a variable with continuous numerical values
- c) a variable with a finite set of possible values

*Select a single answer*


## Handling categorical data

### Additional exercise based on the material

#### Ordinal encoding (everyone gives their answers in the collaborative doc): [Flavio] 

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



### Exercise M1.04

header in exercise document:  "The impact of using integer encoding for with logistic regression (groups of 2, 15min)"

Work in groups of 2, with your neighbor. (15min) 

Goal: understand the impact of arbitrary integer encoding for categorical variables with linear classification such as logistic regression.

We keep using the `adult_census` data set already loaded in the code before. Recall that `target` contains the variable we want to predict and `data` contains the features.

If you need to re-load the data, you can do it as follows:

```python
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])
```


**(0) Select columns containing strings**
Use `sklearn.compose.make_column_selector` to automatically select columns containing strings that correspond to categorical features in our dataset.

**(1) Build a scikit-learn pipeline composed of an `OrdinalEncoder` and a `LogisticRegression` classifier**

You'll need the following, already loaded modules:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
```

Because OrdinalEncoder can raise errors if it sees an unknown category at prediction time, you can set the handle_unknown="use_encoded_value" and unknown_value parameters. You can refer to the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) for more details regarding these parameters.


**(2) Evaluate the model with cross-validation.**

You'll need the following, already loaded modules:

```python
from sklearn.model_selection import cross_validate

```

**(3) Repeat the previous steps using an `OneHotEncoder` instead of an `OrdinalEncoder`**

You'll need the following, already loaded modules:

```python
from sklearn.preprocessing import OneHotEncoder

```


We'll do the following steps
1. use the `OneHotEncoder` for preprocessing
2. assemble pipeline for `LogisticRegression`
3. use cross-validation to check the generalization performance, also relative to the `OneHotEncoder`


### Quiz M1.03: categorical and numerical variables (5 minutes in pairs; if time permits)[Flavio/Malte?]

Select all true answers for each question

**Q1: How are categorical variables represented?**
- a) a categorical feature is only represented by non-numerical data
- b) a categorical feature represents a finite number of values called categories
- c) a categorical feature can either be represented by numerical or
     non-numerical values

**Q2: An ordinal variable:**
- a) is a categorical variable with a large number of different categories;
- b) can be represented by integers or string labels;
- c) is a categorical variable with a meaningful order.


**Q3: One-hot encoding:**
- a) encodes each column with string-labeled values into a
     single integer-coded column
- b) transforms a numerical variable into a categorical variable
- c) creates one additional column for each possible category
- d) transforms string-labeled variables using a numerical representation


**Q4: Assume we have a dataset where each line describes a company. Which of the following columns should be considered as a meaningful numerical feature to train a machine learning model to classify companies:**
- a) the sector of activity ("construction", "retail", "energy", "insurance"...)
- b) the phone number of the sales department
- c) the number of employees
- d) the profits of the last quarter
- e) the post code of the head quarters


# Selecting the best model

## Validation and learning curves

### Exercise M2.01

The aim of this exercise is:
- train and test a support vector machine classifier through cross-validation
- study the effect of varying the parameter gamma of this classifier using a validation curve
- use a learning curve to determine whether we could expect lower test error when adding more samples

First, we need to load some data
```python
import pandas as pd

blood_transfusion = pd.read_csv("../datasets/blood_transfusion.csv")
data = blood_transfusion.drop(columns="Class")
target = blood_transfusion["Class"]
```

Here we use a support vector machine classifier (SVM). In its most simple form, a SVM classifier is a linear classifier behaving similarly to a logistic regression. Indeed, the optimization used to find the optimal weights of the linear model are different but we don’t need to know these details for the exercise.

Also, this classifier can become more flexible/expressive by using a so-called kernel that makes the model become non-linear. Again, no requirement regarding the mathematics is required to accomplish this exercise.

We will use an RBF kernel where a parameter gamma allows to tune the flexibility of the model.

**(1) Create a predictive pipeline**

Use the following ingredients
- `sklearn.preprocessing.StandardScaler` with default parameters
- `sklearn.svm.SVC`, setting the parameter `kernel` to `"rbf"` (which is the default)


**(2) Evaluate how well the model generalizes.**

Use the following
- the `ShuffleSplit` scheme for cross-validation
- pass this object to `sklearn.model_selection.cross_validate` as the `cv` parameter.



**(3) Evaluate the effect of the parameter gamma by using `sklearn.model_selection.ValidationCurveDisplay`**

As previously mentioned, the parameter `gamma` is one of the parameters controlling under/over-fitting in support vector machine with an RBF kernel.

Vary the `gamma` parameter between 10e-3 and 10e2 with this code: `np.logspace(-3, 2, num=30)`.

Use the following arguments for the `ValidationCurveDisplay`
- `scoring=None` 
- `param_name="svc__gamma"` (this is how the gamma parameter is stored in the present case)



**(4) Compute the learning curve (using `sklearn.model_selection.LearningCurveDisplay`) by computing the train and test scores for different training dataset size.**

Plot the train and test scores with respect to the number of samples.


### Quiz M2.02
in collaborative document: Quiz: over- and underfitting and learning curves (5 minutes, in pairs; if time-permitting)

**1. A model is overfitting when:**
- a) both the train and test errors are high
- b) train error is low but test error is high
- c) train error is high but the test error is low
- d) both train and test errors are low

*select a single answer*

**2. Assuming that we have a dataset with little noise, a model is underfitting when:**
- a) both the train and test errors are high
- b) train error is low but test error is high
- c) train error is high but the test error is low
- d) both train and test errors are low

*select a single answer*


**3. For a fixed training set, by sequentially adding parameters to give more flexibility to the model, we are more likely to observe:**
- a) a wider difference between train and test errors
- b) a reduction in the difference between train and test errors
- c) an increased or steady train error
- d) a decrease in the train error

*Select all answers that apply*

**4. For a fixed choice of model parameters, if we increase the number of labeled observations in the training set, are we more likely to observe:**
- a) a wider difference between train and test errors
- b) a reduction in the difference between train and test errors
- c) an increased or steady train error
- d) a decrease in the train error

*Select all answers that apply*

**5. Polynomial models with a high degree parameter:**
- a) always have the best test error (but can be slow to train)
- b) underfit more than linear regression models
- c) get lower training error than lower degree polynomial models
- d) are more likely to overfit than lower degree polynomial models

*Select all answers that apply*

**6. If we chose the parameters of a model to get the best overfitting/underfitting tradeoff, we will always get a zero test error.**
- a) True
- b) False

*Select a single answer*
















