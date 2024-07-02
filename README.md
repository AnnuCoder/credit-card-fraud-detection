<center><img src="https://1.bp.blogspot.com/-d8quHeA23kA/XRH_AeTzRZI/AAAAAAAACYc/w4Qulnhld7g3msiQ17uxVaLwhOA_lQZdQCLcBGAs/s1600/credit%2Bcard.jpg"></center><br>

# Credit-Card-Fraud-Detection-Using-Machine-Learning
<br>

 # Introduction

A **credit card** is one of the most used financial products to make online purchases and payments. Though the Credit cards can be a convenient way to manage your finances, they can also be risky. Credit card fraud is the **unauthorized** use of someone else's credit card or credit card information to make purchases or withdraw cash.<br><br>
In a scenario such as this one, it is **extremely** important that credit card companies are able to easily recognize when a transaction is a result of a fraud or a genuine purchase, avoiding that customers end up being charged for items they did not acquire.<br><br>
In this project, I used the **scikit-learn** library to develop a prediction model that is able to learn and detect when a transaction is a fraud or a genuine purchase. I tested four different classification algorithms, **Decision Tree**, **Random Forest**, **Ada Boost Classifier** and **Gradient Boosting** to identify which one of them would achieve the best results with our dataset.

<br>

# Project goals

The main objective is to develop a prediction model that is able to learn and detect when a transaction is deriving from fraud or a genuine purchase. I intend to use some different classification algorithms and try to identify which one of them achieve the best results with our dataset.


<br>

# Data Source

The dataset was retrieved from the given link as mentioned below. It contains data on transactions made in 2013 by European credit card users in two days only. The dataset consists of 31 attributes and 284,807 rows. Twenty-eight attributes are numeric variables that, due to the confidentiality and privacy of the customers, have been transformed using PCA transformation; the three remaining attributes are ”Time”, which contains the elapsed seconds between the first and other transactions
of each Attribute, ”Amount” is the amount of each transaction, and the final attribute “Class” which contains binary variables where “1” is a case of fraudulent transaction, and “0” is not as case of fraudulent transaction.
<br>
<br>
<b>Dataset: </b>
<a href="creditcard.csv">creditcard.csv</a>

<br>

# Algorithm 
1. Random Forest
2. Decision Tree
3. Ada Boost
4. Gradient Boosting


<br>

# Evaluation Metrics for Classification Models<br>

When dealing with classification models, there are some evaluation metrics that we can use in order to see the efficiency of our models.<br><br>
One of those evaluation metrics is the **confusion matrix** which is a summary of predicted results compared to the actual values of our dataset. This is what a **confusion matrix** looks like for a binary classification problem:<br>
<center><img src= "https://miro.medium.com/max/1400/1*hbFaAWGBfFzlPys1TeSJuQ.png"></center><br><br>

**TP** is for **True Positive** and it shows the correct predictions of a model for a positive class.<br> 
**FP** is for **False Positive** and it shows the incorrect predictions of a model for a positive class.<br> 
**FN** is for **False Negative** and it shows the incorrect predictions of a model for a negative class.<br> 
**TN** is for **True Negative** and it shows the correct predictions of a model for a negative class.<br><br> 


### Accuracy <br>
Accuracy simply tells us the proportion of correct predictions.


### Precision <br> 
Precision tells us how frequently our model correctly predicts positives.

### Recall <br> 
Recall, which can also be referred to as *sensitivity*, can tell us how well our model predicts the class that we want to predict.

### F1 Score <br> 
Lastly, F1 Score is the harmonic mean of precision and recall.
<br>
<br>
# Future Work 
There are many ways to improve the model, such as using it on different datasets with various sizes and data types or by changing the data splitting ratio and viewing it from a different algorithm perspective. An example can be merging telecom datato calculate the location of people to have better knowledge of the location of the card owner while his/her credit card is being used; this will ease the detection because if the card owner is in Dubai and a transaction of his card was made in Abu Dhabi, it
will easily be detected as Fraud.

<br>

# Conclusion 
When we work with a **machine learning model**, we must always **know** for a fact **what it is that we're trying to get from that model**.<br><br>

In this project, our goal is to **detect fraudulent transactions when they occur**, and the model who best performed that task was the **Ada Boost Classifier** with a recall of 91.87%, correctly detecting 147 fraudulent transactions out of 160. However, it is also important to note that the Ada Boost classifier had the biggest number of false positives, that is, **1360 genuine transactions were mistakenly labeled as fraud**.<br><br>

A genuine purchase being incorrectly identified as a fraud could be a problem.<br><br>

In this scenario it is necessary to understand the business and make a few questions such as:<br><br>


- how cheap would a false positive be?<br><br>

- Would we keep the Ada Boost Classifier with the best performance in detecting frauds, while also detecting a lot of false positives or should we use the Random Forest Classifier, who also performed pretty well identified frauds (82.50% recall) and reduced the number of false positives (0.02% of genuine transactions flagged as fraud). But that would also imply in a larger number of fraudsters getting away with it and customers being mistakenly charged...<br><br>

These questions and a deeper understading of how the business works and how we want to approach solving a problem using machine learning are fundamental for a decision-making process to choose whether or not if we're willing to deal with a larger number of false positives to detect the largest amount of frauds as possible.
