# deep-learning-challenge
Module 21 Challenge
By Robin Ryan </br>
File deep_learning.ipynb was created using Google Colab


## **Module 21 Challenge overview**

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

    EIN and NAME—Identification columns
    APPLICATION_TYPE—Alphabet Soup application type
    AFFILIATION—Affiliated sector of industry
    CLASSIFICATION—Government organization classification
    USE_CASE—Use case for funding
    ORGANIZATION—Organization type
    STATUS—Active status
    INCOME_AMT—Income classification
    SPECIAL_CONSIDERATIONS—Special considerations for application
    ASK_AMT—Funding amount requested
    IS_SUCCESSFUL—Was the money used effectively

## **Instructions**
### **Step 1: Preprocess the Data**
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:

What variable(s) are the target(s) for your model?
What variable(s) are the feature(s) for your model?
Drop the EIN and NAME columns.

Determine the number of unique values for each column.

For columns that have more than 10 unique values, determine the number of data points for each unique value.

Use the number of data points for each unique value to pick a cutoff point to combine "rare" categorical variables together in a new value, Other, and then check if the replacement was successful.

Use pd.get_dummies() to encode categorical variables.

Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

### **Step 2: Compile, Train, and Evaluate the Model**
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.

Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

Create the first hidden layer and choose an appropriate activation function.

If necessary, add a second hidden layer with an appropriate activation function.

Create an output layer with an appropriate activation function.

Check the structure of the model.

Compile and train the model.

Create a callback that saves the model's weights every five epochs.

Evaluate the model using the test data to determine the loss and accuracy.

Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

### **Step 3: Optimize the Model**
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
Dropping more or fewer columns.
Creating more bins for rare occurrences in columns.
Increasing or decreasing the number of values for each bin.
Add more neurons to a hidden layer.
Add more hidden layers.
Use different activation functions for the hidden layers.
Add or reduce the number of epochs to the training regimen.
Note: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.

Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.

Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.

Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.

Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

### **Step 4: Write a Report on the Neural Network Model**
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

The report should contain the following:

 - Overview of the analysis: Explain the purpose of this analysis.

 - Results: Using bulleted lists and images to support your answers, address the following questions:

### **Data Preprocessing**

**What variable(s) are the target(s) for your model?**
 - IS_SUCCESSFUL </br>

**What variable(s) are the features for your model?**
 - AFFILIATION
 - CLASSIFICATION
 - USE_CASE
 - APPLICATION_TYPE
 - INCOME_AMT
 - ASK_AMT

**What variable(s) should be removed from the input data because they are neither targets nor features?**
 - EIN
 - Name

### **Compiling, Training, and Evaluating the Model**

**How many neurons, layers, and activation functions did you select for your neural network model, and why?**

**Model: "sequential_1"**
| Layer | Output Shape | Param # |
| ----------- | ----------- | ----------- |
| dense_3 (Dense) | (None, 80) | 3,920 |
| dense_4 (Dense) | (None, 30) | 2,430 |
| dense_5 (Dense) | (None, 1) | 31 |

**Were you able to achieve the target model performance?**
 - No, I was only able to achieve 73.6% Accurracy </br>

**What steps did you take in your attempts to increase model performance?**
 - I gradually added more features and adjusted the number of variables for each feature by adjusting cutoff values

### **Summary:** Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
In Summary, this model was not very accurate at predicting the selection of applicants for funding with the best chance of success in their ventures.  Unfortunately, I do not have any recommendations for different models.
I found this to be the most difficult module and really do not understand all the different models and their uses.


### **Step 5: Copy Files Into Your Repository**
Now that you're finished with your analysis in Google Colab, you need to get your files into your repository for final submission.

Download your Colab notebooks to your computer.

Move them into your Deep Learning Challenge directory in your local repository.

Push the added files to GitHub.

Alphabet Soup Charity Optimization Analysis
By Robin Ryan <br/>

Preprocess: 1st attempt

Supplied dataset - charity_data.csv
EIN	- drop column
NAME - drop column
APPLICATION_TYPE - Unique Values: 17
AFFILIATION	- Unique Values: 6
CLASSIFICATION - Unique Values: 71
USE_CASE - Unique Values: 	5
ORGANIZATION - Unique Values: 	4
STATUS - Unique Values: 	2
INCOME_AMT - Unique Values: 	9
SPECIAL_CONSIDERATIONS - Unique Values: 	2
ASK_AMT	- Unique Values: 8,747
IS_SUCCESSFUL - Unique Values: 2

More than 10 unique values:
APPLICATION_TYPE - Unique Values: 17
  T3	27037
  T4	1542
  T6	1216
  T5	1173
  T19	1065
  T8	737
  T7	725
  T10	528 -- Cutoff Value = 500
  T9	156
  T13	66
  T12	27
  T2	16
  T25	3
  T14	3
  T29	2
  T15	2
  T17	1
CLASSIFICATION - Unique Values: 71
C1000	17326
C2000	6074
C1200	4837
C3000	1918
C2100	1883 -- Cutoff value = 1800
C7000	777
C1700	287
C4000	194
C5000	116
C1270	114
C2700	104
C2800	95
C7100	75
C1300	58
C1280	50
C1230	36
C1400	34
C7200	32
C2300	32
C1240	30
C8000	20
C7120	18
C1500	16
C1800	15
C6000	15
C1250	14
C8200	11
C1238	10
C1278	10
C1235	9
C1237	9
C7210	7
C2400	6
C1720	6
C4100	6
C1257	5
C1600	5
C1260	3
C2710	3
C0	3
C3200	2
C1234	2
C1246	2
C1267	2
C1256	2
**Excluded values <1

ASK_AMT	- Unique Values: 8,747
5000	25398
10478	3
15583	3
63981	3
6725	3
...	...
5371754	1
30060	1
43091152	1
18683	1
36500179	1
** Did not use this as the majority of the data is at 5000

Kept the following columns:
IS_SUCCESSFUL	
APPLICATION_TYPE_Other	
APPLICATION_TYPE_T10	
APPLICATION_TYPE_T19	
APPLICATION_TYPE_T3	
APPLICATION_TYPE_T4	
APPLICATION_TYPE_T5	
APPLICATION_TYPE_T6	
APPLICATION_TYPE_T7	
APPLICATION_TYPE_T8	
CLASSIFICATION_C1000	
CLASSIFICATION_C1200	
CLASSIFICATION_C2000	
CLASSIFICATION_C2100	
CLASSIFICATION_C3000	
CLASSIFICATION_Other


Results:
268/268 - 1s - 4ms/step - accuracy: 0.6090 - loss: 0.6443
Loss: 0.6443251371383667, Accuracy: 0.6089795827865601


Optimization #1
Add in Ask Amount - cutoff count = >= 3
Change Classification cutoff to > 

Results:
268/268 - 1s - 2ms/step - accuracy: 0.6250 - loss: 0.6329
Loss: 0.6329272985458374, Accuracy: 0.6249562501907349

Optimization #2
Same as in #1 Plus:
Add in Affiliation Type cutoff > 100

Results:
268/268 - 0s - 2ms/step - accuracy: 0.7190 - loss: 0.5726
Loss: 0.5725576281547546, Accuracy: 0.7189504504203796

Optimization #3
Same as in # 2 Plus:
USE_CASE
ORGANIZATION
INCOME_AMT

Results:
268/268 - 0s - 2ms/step - accuracy: 0.7368 - loss: 0.5600
Loss: 0.5599625110626221, Accuracy: 0.7367929816246033