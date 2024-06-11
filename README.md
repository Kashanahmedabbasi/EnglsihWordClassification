# EnglsihWordClassification
Introduction:
	Natural Language Processing (NLP) is a subfield of artificial intelligence that focuses on enabling computers to understand, interpret and generate human language. In recent years, NLP has become increasingly important in various industries such as healthcare, finance, and e-commerce. One of the fundamental tasks in NLP is language identification, which involves determining the language of a given text. In this report, we will focus on building a machine learning model for language identification using the Kaggle 47k English Words dataset.

The main objective of this report is to develop a model that can identify whether a given word is English or not. To achieve this, we will use the bag-of-words approach and train a logistic regression classifier on the dataset. We will evaluate the performance of the model using various metrics such as accuracy_score, classification_matrix, and confusion_matrix. In addition to the model building, we will also visualize the dataset using Seaborn and save the trained model using the pickle module.

The rest of this report is organized as follows. In the next section, we will describe the methodology for building the model and evaluating its performance. After that, we will present the results of our experiments and discuss their implications. Finally, we will conclude the report by summarizing our findings and outlining future research directions.
Methodology:
To Begin with import the libraries which will be use in our project. Pandas is used for    preprocessing of data and it’s an English word dataset so to convert English word into vector count vectorizer would be use. Moreover, for split data into train and test train_test_split is used. Furthermore, Logistic Regression is used for classification and rest of libraries for visualization and save model.
 


Data Preprocessing:
	First of all, load data from directory and convert into DataFrame for further preprocessing. Before we created the bag-of-words representation of the words, we performed some cleaning on the dataset to remove non-alphabetic characters and lowercase all characters. We also shuffled the dataset to remove any ordering bias in the data. 

 


 

To train the logistic regression model, we needed a target variable that indicates whether a word is English or not. We created this variable by checking if all characters in a word are the same or not. If all characters are the same, we assumed that the word is not English.
 

Data Visualization:
	For visualization, I used seaborn to check the word frequency in dataset. To check this, first convert vector into array and dataset is too large so only get first 10 values and visualize them.
 






Train Model:
	For training, first split data into train and test. For training I get 80 percent of dataset and 20 percent for testing. I trained the logistic regression model on the training set and evaluated its performance on the test set using metrics such as accuracy_score, classification_matrix, and confusion_matrix. 
 
 


Single Word Prediction:
	To predict if a single word is English or not, we trained the logistic regression model on the entire dataset and used the transform method of the CountVectorizer to transform a single word to its bag-of-words representation. We then used the trained model to predict if the word is English or not.
 

Save Model:
	With the help of pickle library I saved the model and now it’s ready to use for deployment.
	 


Model Selection:
	We chose logistic regression for this task because it is a simple yet effective classification algorithm that is commonly used for text classification tasks. It is also easy to interpret and can provide insight into which features (i.e., characters) are important for classification.

Conclusion:
	In conclusion, we have successfully created a machine learning model that can determine whether a given word is English or not using the Kaggle 479k English words dataset. We used the bag-of-words model and logistic regression algorithm to achieve this task and evaluated our model using standard classification metrics. The model achieved a high accuracy score, indicating its effectiveness in classifying English words.

