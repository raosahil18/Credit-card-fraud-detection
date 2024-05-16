#!/usr/bin/env python
# coding: utf-8

# ## CREDIT CARD FRAUD DETECTION CLASSIFIER

# ![1694776207408.png](attachment:1694776207408.png)

# # <p id="1" style="text-align: center; padding: 20px; background-color: #FDEBD0; border-radius: 10px; border: 2px solid #58D68D; font-size: 20px; color: #2E4053; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">1: Import Libraries</p>
# 

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os

import warnings
warnings.filterwarnings('ignore')


# # <p id="1" style="text-align: center; padding: 20px; background-color: #FDEBD0 ; border-radius: 10px; border: 2px solid #58D68D; font-size: 20px; color: #1A202C; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">2: Load Dataset </p>

# In[2]:


#changing the directory
os.chdir("C://Users//HP//OneDrive//Desktop//My Capstone project")


# In[3]:


# importing the test and train dataset
data = pd.read_csv("creditcard.csv")


# In[4]:


# making a copy of orignal dataset
df = data.copy()


# In[5]:


#pd.set_option("display.max_columns",31)


# In[6]:


# displaying first five rows of dataset
df.head()


# # <p id="1" style="text-align: center; padding: 20px; background-color: #FDEBD0; border-radius: 10px; border: 2px solid #58D68D; font-size: 20px; color: #1A202C; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">3: Data Prepration</p>
#  - 3.0 Data Preparation<br>
#  - 3.1 Understainding the data <br>
#  - 3.2 Descriptive Statistics <br>
#  - 3.3 Exploratory Data Analysis <br>
#  - 3.4 Missing value Treatment <br>
#  - 3.5 Outlier Treatment <br>
#  - 3.6 Encoding <br>
#  - 3.7 Feature Scaling <br>
#  - 3.8 Checking Imbalanced Data <br>
#  
#  - 4.0 Splitting the dataset<br>
#  - 5.0 Model Building<br>
#  - 5.1 Logistic Regression<br>
#  - 5.2 Decision Tree<br>
#  - 5.3 XGBoost<br>
#  - 5.4 ROC <br>
#  - 5.5 Hyperparameter Tunning<br>
#  - 6.0 Result

# # <p id="1" style="text-align: center; padding: 20px; background-color: #FDEBD0; border-radius: 10px; border: 2px solid #58D68D; font-size: 18px; color: #1A202C; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">3.1: Understanding the Data</p>

# In[7]:


# checking the dtypes of columns
df.info()


# `On using info() function we can check if any of the columns contain null values and also verify the datatypes of each columns. Looking at the output it can be seen that all the 31 columns have non-null values and datatpyes of each column is also correct.`

# In[8]:


# checking the No. of duplicate values in dataset
df.duplicated().sum()


# In[9]:


# dropping the duplicates in dataset
df.drop_duplicates(inplace = True)


# In[10]:


# checking the No. of duplicate values in dataset
df.duplicated().sum()


# <div style="background-color: #E8FEF0; padding: 20px; border-left: 4px solid #5e72e4;border-right: 4px solid #5e72e4; border-radius: 8px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.05);">
#     
# <b>Inference</b>:<br>
# <ul style="list-style-type: square; color: #004085;">
#     <li>The dataset contains <b>284807</b> entries.</li>
#     <li>Each entry represents a transaction as fraudulent or non-fraudulent.</li>
#     <li>There are <b>31</b> columns in the dataset.</li>
#     <li>The columns represent various features:</li>
#     <ul style="list-style-type: disc; color: #004085;">
#         <li>As, it contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, the original features and more background information about the data could not be provided. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'</li>
#         <li>'Time' contains the seconds elapsed between each transaction and the first transaction.</li>
#         <li>'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning.</li>
#         <li>'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.</li>
#     </ul>
#         <li>There are no missing values in columns.</li>
#     <li>There were duplicates <b>1081</b> in several rows and we have deopped them.</li>
#     <li>Now, the dataset contains <b>283726</b> entries.</li>
#     <li>The target variable is '<b>Class</b>', which represents the 1 in case of fraud and 0 for not fraud .</li>
# </ul>
#     
# </div>
# 

# In[11]:


#length of the dataset
len(df)


# In[12]:


# shape of the dataset
df.shape


# ###### The following are the names of the columns of the dataset. In total there are 31 columns.

# In[13]:


# No. of columns in dataset
df.columns


# In[14]:


# Count of total columns in dataset
len(df.columns)


# # <p id="1" style="text-align: center; padding: 20px; background-color: #FDEBD0; border-radius: 10px; border: 2px solid #58D68D; font-size: 18px; color: #1A202C; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">3.2: Descriptive Statistics</p>

# `The describe() function generates descriptive statistics that summarize the central tendency,dispersion and shape of a dataset's distribution, excluding NaN values.`

# In[15]:


# statistics summary for Numerical Features
df.describe().T


# <div style="background-color: #F5EEE6; padding: 10px 12px; border: 2px solid #cc0000; border-radius: 10px;">
#     <b>Inference</b>:<br>
#     <ul style="list-style-type: square; color: #004085;">
#         <li><b>Time:</b>
#             <ul style="list-style-type: disc; color: #004085;">
#                 <li>The average time in seconds elapsed between each transaction and the first transaction is approximately <b>94811 sec</b>, with a considerable standard deviation of <b>47481.04</b>.</li>
#                 <li>The maximum time reported in seconds is significantly high at <b>172792 sec</b>, which could represent outliers.</li>
#             </ul>
#         </li>
#         <li><b>Amount:</b>
#             <ul style="list-style-type: disc; color: #004085;">
#                 <li>The average amount is approximately <b>\$88.47</b>, with a standard deviation of <b>\$250.39</b>. This signifies the range of transaction Amount.</li>
#                 <li>Amount range from <b>\$0.00</b> to <b>\$25691.160</b>, with most falling between <b>\$5.60</b> and <b>\$77.51</b>. This indicates variability in transaction amount.</li>
#             </ul>
#         </li>
#     </ul>
# </div>
# 

# # <p id="1" style="text-align: center; padding: 20px; background-color: #FDEBD0; border-radius: 10px; border: 2px solid #58D68D ; font-size: 18px; color: #1A202C; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">3.3: Exploratory Data Analysis</p>

# #### Observe the distribution of classes with time

# In[16]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(16,8))

ax1.hist(df.Time[df.Class == 1], bins = 40, color= "#ff826e" ,edgecolor='black')
ax1.set_title('Fraud')

ax2.hist(df.Time[df.Class == 0], bins = 40, color = "#0101DF" ,edgecolor='black')
ax2.set_title('Non Fraud')

plt.xlabel('Time')
plt.ylabel('Number of Transactions')
plt.show()


# In[17]:


plt.figure(figsize=(15,8))
sns.barplot(data=df, y = 'Time',x='Class', palette = ["#33FF57", "#3366FF"])
plt.title('Transaction Time by Class')
plt.legend(["fraud not detected"])
plt.show()


# In[18]:


#Box plots for 'Amount' and 'Time' by Class
plt.figure(figsize=(12, 6))
sns.boxplot(x='Class', y='Amount', data=df, showfliers=False, palette = ["#FF5733", "#33FF57"])
plt.title('Transaction Amount by Class')
plt.legend(["fraud not detected"])
plt.show()


# #### Observe the distribution of classes with amount

# In[19]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(16,8))

ax1.hist(df.Amount[df.Class == 1], bins = 40, color = "#ff826e")
ax1.set_title('Fraud')

ax2.hist(df.Amount[df.Class == 0], bins = 40, edgecolor='black')
ax2.set_title('Non-Fraud')

plt.xlabel('Amount')
plt.ylabel('Number of Transactions')
plt.yscale('log')
plt.show()


# In[20]:


#Plots for Amount and Time variable
fig, ax = plt.subplots(1, 2, figsize=(16,8))

amount_val = df['Amount'].values
time_val = df['Time'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])

plt.show()


# In[21]:


# Barplot of "Class" variable to see how many transactions were fraud or not
count_classes = pd.value_counts(df['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')

plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()


# `Clearly the data is totally imbalanced!!`

# In[22]:


# Histogram of all columns
df.hist(figsize=(20,20),color='cyan')
plt.show()


# In[23]:


# Explore feature distributions
plt.figure(figsize=(25, 25))
for i in range(1, 29):  # Assuming V1 to V28 are the feature columns
    plt.subplot(7, 4, i)
    sns.histplot(df[f'V{i}'], bins=30, kde=True,color = 'blue')
    plt.title(f'Distribution of V{i}')
plt.tight_layout()
plt.show()


# # <p id="1" style="text-align: left; padding: 20px; background-color: #66b3ff; border-radius: 10px; border: 2px solid #58D68D; font-size: 18px; color: #1A202C; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);"> Correlation and Heatmap </p>

# In[25]:


# Correlation
df_corr = df.corr()
df_corr


# In[26]:


#heat map
plt.figure(figsize=(30,20))
sns.heatmap(df_corr, annot=True)
plt.show()


# # <p id="1" style="text-align: center; padding: 20px; background-color: #FDEBD0; border-radius: 10px; border: 2px solid #58D68D; font-size: 18px; color: #1A202C; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">3.4: Missing Value Treatment</p>

# In[24]:


# missing values in dataset
#pd.DataFrame(df.isnull().sum())
df.isnull().sum()/len(df)*100


# ### `There are no missing values in this dataset.`

# # <p id="1" style="text-align: center; padding: 20px; background-color: #FDEBD0; border-radius: 10px; border: 2px solid #58D68D; font-size: 18px; color: #1A202C; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">3.5: Outliers Treatment</p>

# `Not performing any outliers treatment for this dataset. Because all the columns are already PCA transformed, which assumed that the outlier values are taken care while transforming the data.`

# # <p id="1" style="text-align: center; padding: 20px; background-color: #FDEBD0; border-radius: 10px; border: 2px solid #58D68D; font-size: 18px; color: #1A202C; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">3.6: Encoding</p>

# <div style="background-color:#E8F0FE ; padding: 10px 12px; border: 2px solid #cc0000; border-radius: 10px;">
#     <p><strong>Categorization of Features for Encoding:</strong></p>
#     <ul style="list-style-type: square; color: #004085;">
#     <p>After analyzing the dataset, we can categorize the features into three groups:</p>
#     <ol>
#         <li><strong>No Encoding Needed:</strong> These are the features that do not require any form of encoding because they are already in a numerical format that can be fed into a model.</li>
#         <li><strong>One-Hot Encoding:</strong> This is required for nominal variables, which are categorical variables without any intrinsic order. One-hot encoding converts each unique value of the feature into a separate column with a 1 or 0, indicating the presence of that value.</li>
#         <li><strong>Label Encoding:</strong> This is used for ordinal variables, which are categorical variables with a meaningful order. Label encoding assigns a unique integer to each category in the feature, maintaining the order of the values.</li>
#     </ol>
#     </ul>
# </div>

# `Not performing any encoding for this dataset. Because all the columns are numerical format.`

# # <p id="1" style="text-align: center; padding: 20px; background-color: #FDEBD0; border-radius: 10px; border: 2px solid #58D68D; font-size: 18px; color: #1A202C; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">3.7: Feature Scaling</p>
# 

# `We need to scale the Amount and time variables using Standard Scaler and all other columns are already scaled by the PCA transformation.`

# In[27]:


# Standardization method
from sklearn.preprocessing import StandardScaler

# Instantiate the Scaler
scaler = StandardScaler()

# Fit the data into scaler and transform
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])


# # <p  style="text-align: center; padding: 14px; font-size: 20px; background-color: #FDEBD0; color: #1A202C; border: 2px solid #58D68D; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">3.8: Checking Imbalanced Data</p>

# In[28]:


# counts of categories in loan_status
df.Class.value_counts()


# In[29]:


plt.figure(figsize=(10,8))
ax = sns.countplot(x='Class', data=df, palette='crest')

# Add count values on top of each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.title('Distribution of Class')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# <div style="background-color: #F5EEE6; padding: 10px 12px; border: 2px solid #cc0000; border-radius: 10px;">
#     <ul style="list-style-type: square; color: #004085;">
#     The bar plot shows the number of fraud and genuine transaction in the dataset. Approximately 283253 of the class was genuine transaction, and 473 were fraud transactions. This indicates that there is high imbalance in the target variable.
#     To address this, we will use <b>SMOTE (Synthetic Minority Over-sampling Technique)</b>. SMOTE is a technique used to generate synthetic samples for the minority class in order to balance the class distribution in the dataset. By creating synthetic samples, SMOTE helps mitigate the impact of class imbalance and improves the performance of machine learning models in predicting the minority class.
#     </ul>
# </div>

# In[30]:


#find percentage of fraud/non_fraud records
fraud_percentage = (df.groupby('Class')['Class'].count() / df['Class'].count()) * 100

plt.figure(figsize=(10, 5))
plt.pie(fraud_percentage, labels=['Genuine', 'Fraud'], autopct='%0.3f%%', colors=['#FDEBD0','red'])
plt.title('Percentage of Fraud and Genuine Records')
plt.axis('equal')
plt.show()


# `Genuine transactions make up approximately 99.833% of the dataset, while fraudulent transactions represent only 0.167%.`

# In[31]:


# separating the data for analysis
gen = df[df.Class == 0]
fraud = df[df.Class == 1]


# In[32]:


# checking the shapes of genuine and fraud transaction
print(gen.shape)
print(fraud.shape)


# # <p  style="text-align: left; padding: 14px; font-size: 18px; color: #1A202C; background-color: #E8F0FE; border: 2px solid #58D68D; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);"> Handling Imbalanced Data</p>

# In[33]:


from imblearn.over_sampling import SMOTE
sm = SMOTE (sampling_strategy='minority', random_state=42)

# Fitting the model to generate the data.
oversampled_X, oversampled_Y = sm. fit_resample(df.drop('Class', axis=1), df['Class'])
oversampled = pd. concat([pd.DataFrame(oversampled_Y), pd.DataFrame(oversampled_X)],axis=1)


# In[34]:


# Calculating the percentage of each class
percentage = oversampled['Class'].value_counts(normalize=True) * 100

# Plotting the percentage of each class
plt.figure(figsize=(10, 5))
ax = sns.barplot(x=percentage.index, y=percentage, palette=['#7AA2E3', 'cyan'])
plt.title('Percentage of Genuine and Fraud')
plt.xlabel('Transaction')
plt.ylabel('Percentage (%)')
plt.xticks(ticks=[0, 1], labels=['Genuine','Fraud'])
plt.yticks(ticks=range(0,80,10))

# Displaying the percentage on the bars
for i, p in enumerate(percentage):
    ax.text(i, p + 0.5, f'{p:.2f}%', ha='center', va='bottom')

plt.show()


# In[35]:


oversampled.Class.value_counts()


# <div style="background-color: #F5EEE6; padding: 10px 12px; border: 2px solid #cc0000; border-radius: 10px;">
#     <ul style="list-style-type: square; color: #004085;">
#          <li><strong>Using Oversampling:</strong> Oversampling (SMOTE) is used instead of undersampling because the difference between the two classes( 0 and 1) is huge, so if undersampling  techniques is used, it will lead to losing of most of the sensitive information from the data. Therefore, to overcome that oversampling is used.</li>
#            </ul>
# </div>

# # <p id="1" style="text-align: center; padding: 20px; background-color: #FDEBD0; border-radius: 10px; border: 2px solid #58D68D ; font-size: 20px; color: #1A202C; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">4: Splitting the Training Dataset</p>

# In[36]:


#input and output variables
X=oversampled.drop(["Class"],axis=1)
y=oversampled["Class"]


# In[37]:


# split the data into train and train data
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True , random_state=42)


# In[38]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# # <p id="1" style="text-align: center; padding: 20px; background-color: #FDEBD0; border-radius: 10px; border: 2px solid #58D68D; font-size: 20px; color: #1A202C; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">5: Model Building</p>
# 
# - Logistic Regression Model<br>
# -  Decision Tree Model<br>
# - Xgboost

# # <p id="1" style="text-align: center; padding: 15px; background-color:  #FDEBD0; font-size: 18px; border-radius: 10px; color: #1A202C;border: 2px solid #58D68D;  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">5.1: Logistic Regression Model</p>

# In[39]:


# Importing Library
from sklearn.linear_model import LogisticRegression


# In[40]:


# Fitting the model
logistic=LogisticRegression()
logistic.fit(X_train,y_train)


# In[41]:


# Prdeicting the model
prediction=logistic.predict(X_test)
prediction


# In[42]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[43]:


# confusion matrix
cm1=confusion_matrix(y_test,prediction)
cm1


# In[44]:


cm = confusion_matrix(y_test, prediction)
labels = ['Genuine', 'Fraud']
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Logistic Regression')
plt.show()


# In[45]:


# calculating the accuracy
accuracy_score(y_test,prediction)


# In[46]:


# Classification Report
cl_report = classification_report(y_test, prediction, target_names = ['Genuine', 'Fraud'])
print("Logistic Regression Classification Report:")
print(cl_report)


# In[47]:


# ACCURACY OF Logistic Regression
from sklearn import metrics
accuracyList=[]
modelList=[]

# print the accuracy
print("Accuracy:",metrics.accuracy_score(y_test,prediction))
accuracyList.append(metrics.accuracy_score(y_test, prediction))
modelList.append("Logistic Regression")


# In[48]:


# lets get the precision and recall numbers using confusion matrix itself
CM=cm1
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
precisionList=[]
recallList=[]

print("precision",TP/(TP+FP))
print("recall", TP/(TP+FN))
precisionList.append(TP/(TP+FP))
recallList.append(TP/(TP+FN))


# In[49]:


print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(prediction , y_test))) 
print('Precision : {0:0.5f}'.format(metrics.precision_score(y_test , prediction)))
print('Recall : {0:0.5f}'.format(metrics.recall_score(y_test , prediction)))
# print('Confusion Matrix : \n', cnf_matrix)
print("\n")


# <div style="background-color: #C0D6E8 ; padding: 10px 12px; border: 2px solid #cc0000; border-radius: 10px;">
#     <h3>Logistic Regression Model Evaluation </h3>
#     <ul style="list-style-type: square; color: #004085;">
#     <table style="width:100%">
#         <tr>
#             <th>Metric</th>
#             <th>Value</th>
#             <th>Interpretation</th>
#         </tr>
#         <tr>
#             <td>Accuracy</td>
#             <td>94.52%</td>
#             <td>The model correctly predicted transactions  for 94.52% of the cases.</td>
#         </tr>
#         <tr>
#             <td>Precision (Fraud)</td>
#             <td>97.26%</td>
#             <td>Out of all class as fraud, only 97.26% were actually fraud.</td>
#         </tr>
#         <tr>
#             <td>Recall (Fraud)</td>
#             <td>91.66%</td>
#             <td>The model identified 94.66% of the actual fraud transaction.</td>
#         </tr>
#       </table>
#     
#    <p>
#        The evaluation of the Logistic Regression model in the class domain reveals its performance in predicting fraud transaction. While it achieved an accuracy of 94.52%, indicating overall correctness. </p>
#     </ul>
# 
# </div>
# 

# # <p id="1" style="text-align: center; padding: 15px; background-color:  #FDEBD0; font-size: 18px; border-radius: 10px; color: #1A202C; border: 2px solid #58D68D; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">5.2:  Decision Tree Model </p>

# In[50]:


# importing libarary
from sklearn.tree import DecisionTreeClassifier, plot_tree


# In[51]:


#Applying the Decision Tree on the training dataset
dt_model = DecisionTreeClassifier(random_state=0)
dt_model.fit(X_train,y_train)


# In[52]:


#Running the model on the test dataset
y_pred_ini = dt_model.predict(X_test)


# ### Check the accuracy of the model

# In[53]:


#Importing all the functions to for checking the accuracies
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score


# In[54]:


#Using accuracy score we are checking the accuracy on the testing dataset
accuracy_score(y_test,y_pred_ini)


# In[55]:


#Using confusion matrix we are checking the accuracy on the testing dataset
cm = confusion_matrix(y_test, y_pred_ini)
labels = ['Genuine', 'Fraud']
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Decision Tree')
plt.show()


# In[56]:


#Storing the predicted values of training dataset in y_pred_train
y_pred_train = dt_model.predict(X_train)


# In[57]:


#Checking the accuracy of training dataset 
accuracy_score(y_train,y_pred_train)


# In[58]:


#Checking the accuracy of testing dataset
accuracy_score(y_test,y_pred_ini)


# In[59]:


print('Accuracy :{0:0.5f}'.format(metrics.accuracy_score(y_pred_ini , y_test))) 
print('Precision : {0:0.5f}'.format(metrics.precision_score(y_test , y_pred_ini)))
print('Recall : {0:0.5f}'.format(metrics.recall_score(y_test , y_pred_ini)))
# print('Confusion Matrix : \n', cnf_matrix)


# In[60]:


# lets get the precision and recall numbers using confusion matrix itself
CM=cm
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
precisionList=[]
recallList=[]

print("precision",TP/(TP+FP))
print("recall", TP/(TP+FN))
precisionList.append(TP/(TP+FP))
recallList.append(TP/(TP+FN))


# <div style="background-color: #C0D6E8 ; padding: 10px 12px; border: 2px solid #cc0000; border-radius: 10px;">
#     <h3>Decision Tree Model Evaluation </h3>
#     <ul style="list-style-type: square; color: #004085;">
#     <table style="width:100%">
#         <tr>
#             <th>Metric</th>
#             <th>Value</th>
#             <th>Interpretation</th>
#         </tr>
#         <tr>
#             <td>Accuracy</td>
#             <td>99.82%</td>
#             <td>The model correctly predicted transactions for 99.82% of the cases.</td>
#         </tr>
#         <tr>
#             <td>Precision (Fraud)</td>
#             <td>99.73%</td>
#             <td>Out of all class as fraud, only 99.73% were actually fraud.</td>
#         </tr>
#         <tr>
#             <td>Recall (Fraud)</td>
#             <td>99.91%</td>
#             <td>The model identified 99.91% of the actual fraud transaction.</td>
#         </tr>
#       </table>
#     </ul>
# 
# </div>
# 

# # <p id="1" style="text-align: center; padding: 15px; background-color:  #FDEBD0; font-size: 18px; border-radius: 10px; color: #1A202C; border: 2px solid #58D68D; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">5.3:  XGBoosting Model</p>

# In[1]:


get_ipython().run_line_magic('pip', 'install xgboost')


# In[62]:


import xgboost as xgb


# In[63]:


#Define the model 
xgb_cal=xgb.XGBClassifier( n_estimators = 10)
xgb_cal


# In[64]:


# Fit  from the model
xgb_cal.fit(X_train,y_train)


# In[65]:


#predict the model
preds = xgb_cal.predict(X_test)
preds


# In[66]:


# ACCURACY OF Xgboost
from sklearn import metrics
# print the accuracy
print("Accuracy:",metrics.accuracy_score(y_test, preds))
accuracyList.append(metrics.accuracy_score(y_test, preds))
modelList.append("XGBoost")


# In[67]:


#import the confusion matrix from scikit learn
from sklearn.metrics import confusion_matrix
#create the confusion matrix
cm = confusion_matrix(y_test, preds)
cm


# In[68]:


# lets get the precision and recall numbers using confusion matrix itself
CM=cm
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]


print("precision",TP/(TP+FP))
print("recall", TP/(TP+FN))
precisionList.append(TP/(TP+FP))
recallList.append(TP/(TP+FN))


# <div style="background-color: #58D68D; padding: 10px 12px; border: 2px solid #cc0000; border-radius: 10px;">
#     <h3>XgBoost Model Evaluation </h3>
#     <ul style="list-style-type: square; color: #004085;">
#     <table style="width:100%">
#         <tr>
#             <th>Metric</th>
#             <th>Value</th>
#             <th>Interpretation</th>
#         </tr>
#         <tr>
#             <td>Accuracy</td>
#             <td>98.28%</td>
#             <td>The model correctly predicted transactions for 98.28% of the cases.</td>
#         </tr>
#         <tr>
#             <td>Precision (Fraud)</td>
#             <td>98.84%</td>
#             <td>Out of all transactions as fraud, only 98.84% were actually fraud.</td>
#         </tr>
#         <tr>
#             <td>Recall (Fraud)</td>
#             <td>97.72%</td>
#             <td>The model identified 97.72% of the actual fraud transactions.</td>
#         </tr>
#       </table>
#     </ul>
# </div>

# # <p id="1" style="text-align: center; padding: 15px; background-color:  #FDEBD0; font-size: 18px; border-radius: 10px; color: #1A202C; border: 2px solid #58D68D; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">5.4: ROC Curve</p>

# In[69]:


from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score


# In[70]:


lr_probs = logistic.predict_proba(X_test)[:, 1]
dt_model_curve = dt_model.predict_proba(X_test)[:, 1]
xgb_probs = xgb_cal.predict_proba(X_test)[:, 1]

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
knn_fpr, knn_tpr, _ = roc_curve(y_test, dt_model_curve)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs)

lr_auc = roc_auc_score(y_test, lr_probs)
dt_model_curve_auc = roc_auc_score(y_test, dt_model_curve)
xgb_auc = roc_auc_score(y_test, xgb_probs)


# In[71]:


plt.figure(figsize=(10, 6))
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.3f})')
plt.plot(knn_fpr, knn_tpr, label=f'DT (AUC = {dt_model_curve_auc:.3f})')
plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {xgb_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# - According to the analysis of **classification reports** and **AUC scores**, it's evident that the top-performing models for this problem are **Decision Tree** and **XGBoost**. These models demonstrate exceptional accuracy overall and exhibit strong capabilities in detecting fraud, as indicated by their high recall scores. Although **Logistic Regression** falls slightly behind in terms of fraud detection compared to the other two models, it still manages to achieve respectable overall accuracy.
# 
# - The **AUC scores** reveal that in the realm of distinguishing between fraudulent and genuine transactions, **XGBoost** leads with a remarkable score of **0.999**, closely trailed by **Decision Tree** at **0.998**. In contrast, **Logistic Regression** lags behind with a score of **0.989**. These findings emphasize the superior discriminatory prowess of **XGBoost** and **Decision Tree** models over **Logistic Regression** in this context.`
# 
# 
# - After analyzing the outcomes, we suggest considering either **XGBoost** or **Decision Tree** for addressing this issue. Further evaluation can help determine which model to prioritize, taking into account factors like computational capacity and the level of interpretability desired.`

# # <p id="1" style="text-align: center; padding: 15px; background-color:  #FDEBD0; font-size: 18px; border-radius: 10px; color: #1A202C; border: 2px solid #58D68D; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">5.5: Hyperparameter Tuning in XgBoost</p>

# In[72]:


# Initialize the XGBoost Classifier using optimal hyperparameters
xgb_tun = xgb.XGBClassifier(max_depth=6,
                        learning_rate=0.05,
                        n_estimators=200,  
                        min_child_weight=2,
                        scale_pos_weight=0.5,
                        subsample=0.9 ,
                        colsample_bytree=0.5,
                        colsample_bylevel=0.8 ,
                        reg_alpha=0.05 ,
                        reg_lambda=0.1 ,
                        max_delta_step=2 ,
                        gamma=0.1,
                        random_state=0)

# fit the XGBoost classifier
xgb_tun.fit(X_train, y_train)


# In[73]:


#predict the model
preds_tun = xgb_tun.predict(X_test)
preds_tun


# In[74]:


# checking Accuracy
accuracy_score(y_test,preds_tun)


# In[75]:


#import the confusion matrix from scikit learn
from sklearn.metrics import confusion_matrix
#create the confusion matrix
cm2 = confusion_matrix(y_test, preds_tun)
cm2


# In[76]:


# lets get the precision and recall numbers using confusion matrix itself
CM=cm2
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]


print("precision",TP/(TP+FP))
print("recall", TP/(TP+FN))


# <div style="background-color: #C0D6E8; padding: 10px 12px; border: 2px solid #cc0000; border-radius: 10px;">
#     <h3>XGBoost Model Evaluation (After Tuning)</h3>
#     <ul style="list-style-type: square; color: #004085;">
#     <table style="width:100%">
#         <tr>
#             <th>Metric</th>
#             <th>Value</th>
#             <th>Interpretation</th>
#         </tr>
#         <tr>
#             <td>Accuracy</td>
#             <td>99.75%</td>
#             <td>The model correctly predicted transactions for 99.75% of the classes.</td>
#         </tr>
#         <tr>
#             <td>Precision (Fraud)</td>
#             <td>99.84%</td>
#             <td>Out of all transactions predicted as fraud, only 99.84% were actually fraud.</td>
#         </tr>
#         <tr>
#             <td>Recall (Fraud)</td>
#             <td>99.66%</td>
#             <td>The model identified 99.66% of the actual fraud transactions.</td>
#         </tr>
#     </table>
#     </ul>
# </div>

# # <p id="1" style="text-align: center; padding: 15px; background-color: #FDEBD0; font-size: 18px; border-radius: 10px; color: #1A202C; border: 2px solid #58D68D; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">6: Result</p>

# 
# - After analyzing the outcomes, it's evident that leveraging **XGBoost** with **oversampled data** and **fine-tuned hyperparameters** yields superior results in identifying fraudulent transactions. This method consistently demonstrates high precision and recall rates across various folds, underscoring its resilience and accuracy in managing imbalanced datasets and delivering precise predictions.

# In[ ]:




