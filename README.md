# Predicting-Movie-Tags-with-Plot-Synopsis-Data
In this repository, I have done a detailed case study on predicting movie tags based on the movie plot summaries.

## What we did throughout this experiment:

The objective of this experiment was to suggest tags based on the movie plots collected from  IMDB and Wikipedia.

The dataset was collected from https://www.kaggle.com/cryptexcode/mpst-movie-plot-synopses-with-tags. The given dataset contains the movie name associated with a movie ID. Each movie contains a summary of the plots about the movie and the tags column contains information about the tags associated with each of the movies. There are a total of 14,828 movies and we have 71 unique tags spread across the entire dataset. The 'split' column in the dataset contains information about how the data is to be splitted in train, test and cross validation dataset.

The problem that we have is a multi-label classification problem. Multilabel classification assigns to each sample a set of target labels. This can be thought as predicting properties of a data-point that are not mutually exclusive, such as topics that are relevant for a document. A movie plot synopse may either have tags like horror, sad, violence, brutal or it may have all of these 4 tags. 

For building and evaluation of our machine learning models, we have chosen the micro averaged F1 score metric as our key performance indicator. It has been researched and found that micro averaged F1 score is the most ideal metric when we have a multi-label classification problem. As our secondary metrics we will also have weighted accuracy, hamming loss, weighted precision and weighted recall. 

<b>Micro-Averaged F1-Score (Mean F Score) </b>: 
The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:

<i>F1 = 2 * (precision * recall) / (precision + recall)</i><br>

In the multi-class and multi-label case, this is the weighted average of the F1 score of each class. <br>

<b>'Micro f1 score': </b><br>
Calculate metrics globally by counting the total true positives, false negatives and false positives. This is a better metric when we have class imbalance.
<br>

<b>'Macro f1 score': </b><br>
Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
<br>

https://www.kaggle.com/wiki/MeanFScore <br>
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html <br>
<br>
<b> Hamming loss </b>: The Hamming loss is the fraction of labels that are incorrectly predicted. <br>
https://www.kaggle.com/wiki/HammingLoss <br>


#### Data Loading Phase.

Since the dataset is given as a CSV file, we have used the popular pandas library to load the data. A very basic and high level information reveals that the dataset contains 14828 rows and 6 columns. The next thing we will do is to create an SQL DB from the given CSV file. This is done for ease op operation during the later stages of this Ipython notebook. 

A basic distribution plot reveals that there are 9489 training samples, 2966 test samples and 2373 samples for cross validation. It is also observed that out of the total data that is provided to us, almost 28% of the movie plot synopsis is collected from IMDB and almost 72% of them are collected from Wikipedia.

In this experiment, we have used Random Grid Search algorithm to optimize our hyperparameters. Due to this, we will combine the training and cross validation data into a single training set and perform K fold cross validation on it. We will use the test data (which should be unseen by the model) to evaluate our models performance. 

#### Checking for duplicate entries of rows.

A simple check reveals that there were 47 duplicate entries in the given dataset, that constitutes 0.32% of the entire data. We don't want these duplicate entries to affect our machine learning models in any way, hence we have removed them and created a new DB which doesn't contain any duplicated entries. 


#### Checking the number of times each movie appeared in the dataset

On simple EDA, it revealed that there are 14743 movies which occurred only once in the dataset, 32 movies occurred twice, there were 4 movies which occurred 3 times and only 1 movie which occurred 5 times.

#### Checking for the distribution of tags per movie

There are as many as 5133 movies which contains just a single tag, 2990 movies contains 2 tags, 1924 movies contains 3 tags. The maximum number of tags present in a movie is 24. That's massive! The average number of tags per movie in the entire dataset was roughly equal to 3 (2.98 to be precise). On checking the countplot of the distribution of tags per movie, we have seen that the distribution is highly skewed towards the left. There are an extremely high number of movies which contains 5 or less tags and there were very very low number of movies which contains more than 5 tags. There are almost 550 movies which contains 10 or more tags. 

#### Exploring the length of the movie plots

A high level statistics reveals that there the maximum length of movie plot summary was as high as 63959 characters and the lowest length being 442. The median length of all the movie plot synopsis consisted of 3825 characters. None of the movies synopsis contained any external reference, there was just one movie which contained html tags and almost 20 movies which contains greater than sign. Almost all the movies had punctuation marks and stopwords. Before building our machine learning models we have processed the dataset by removing stopwords, punctuations and also decontracted the occurrence of certain words like didn't to did not, shouldn't to should not etc. 

A simple distribution plot revealed that the median value of the length of the title texts were somewhere around 15 and there are extremely few movies which had it's length of plot synopsis greater than 20000 characters. 

#### Analysis of tags

There are a total of 71 unique tags present in the dataset. We have used a custom tokenize function which splits the tag data for each review based on 'comma' and also trim any whitespaces present before or after a tag occurs. Removing whitespaces was extremely important, or else it was giving the same tag twice - one with a whitespace and the other without it. For example without removing the whitespaces, 'absurd' and ' absurd' were considered two separate tags and we were getting a total of 140 odd tags. Only after we removed the whitespaces did we get the number of unique tags to be 71. 

Then we created a new dataframe which contains two columns - the tag name and the number of times each of these tags occurred in the entire dataset. On sorting this dataframe in descending order, we see 5 of the highest occurring tags are - murder, violence, flashback, romantic and cult. Tags like revenge, psychedelic and comedy closely followed the top 5 tags. This proves that most audience likes to watch movies which are related to murder, violence etc, hence more and more movies are made on this subjects. 

On plotting the distribution of the number of tags, we have also seen that almost 10 tags occurs more than 1000 times, almost 5 tags occurs more than 3000 times, 75% of the tags occurs less than 570 times and just 25% tags are present in the dataset hich occurs more than 570 times. 

We have also stored the tags which occurred more than 1000 times, 5000 times in separate lists. 

Key Observations from the analysis of tags: 

1. 75% of tags occurs less than 570 times across different movies.
2. 25% of tags occurs less than 119 times across different movies.
3. The maximum number of times a tag occurs in a movie is 5771
4. There are total 9 tags which are used more than 1000 times.
5. 1 tag is used more than 5000 times.
6. Most frequent tag (i.e. 'murder') is used 5771 times
7. Since some tags occur much more frequently than others, Micro-averaged F1-score is the appropriate metric for this problem.
8. Minimum number of tags in a movie plot is 1.
9. Maximum number of tags in a movie plot is 25
10. Average number of tags per movie was close to 3. 
11. 10551 movies had tags less than or equal to 3.
12. 11789 movies had tags less than or equal to 4.
13. 12705 movies had tags less than or equal to 5.
14. 13331 movies had tags less than or equal to 6.

#### Word Clouds of tags

The word cloud of all the tags revealed the same thing - which of the tags occurred with maximum number of occurrences across the dataset. In a word cloud, words which appears the most has bigger font compared to the one which occurs less. 


1. A look at the word cloud shows that "murder", "violence", flashback","romantic","cult" are the most frequently occurring tags in the movie synopses plots.

2. There are lots of tags which occurs less frequently like - 'brainwashing', 'alternate history', 'queer', 'clever', 'claustrophobic', 'whimsical', 'feel-good', 'blaxploitation', 'western', 'grindhouse film', 'magical realism', 'suicidal', 'autobiographical', 'christian film', 'non fiction'

#### K-Means clustering on Bag of Words representation of tags

Here, we have used K means clustering to get an idea which of the tags have a tendency to occur together. We have initialized the clusters centroids using the K-Means++ algorithm and selected the optimal number of clusters using the elbow method.

#### Data cleaning stage

In this stage, we have used custom functions with regex to clean and pre-process the movie plots. The custom functions does the following:

1. Remove HTML tags if any.
2. Remove punctuations, alphanumeric words and special characters.
3. Remove words which have character length less than equal to 2.
4. Convert all the words to lower case letters to avoid duplication of same words in two caps.
5. Remove the stopwords present in the movies.
6. Stem the words in the movie synopsis plots.

Once this step is done, we will create a new dataset which will contain the clean texts. We will build our machine learning models based on top of the cleaned texts. 

#### Building ML models. 

We will take the train and cross validation data and merge them to create one single training dataset. We will use this training data for cross validation as well. We will test our models using various featurization techniques and finally evaluate our models performance on the test set.

Before, we proceed to build our ML models, we need to encode the given tags. We will use binary bag of word vectors to convert the tags into binary numbers. This will be our dependent variable and we will build our machine learning models which would predict a binary vector.  


#### Featurizations

For all the models, we have used TFID features to vectorize the movie plot synopsis data. For the initial base models we have tried a simple Logistic Regression model, SGDClassifier with LogLoss and SGDClassifier with HingeLoss. We have used all these models with OneVsRestClassifier to build our ML models. We figured out that LogisticRegression seems to perform very well and achieved a greater micro average F1 score than the rest of the models. So we decided to stick to LogisticRegression for tuning our advance models.

For the baseline models, we have used all the 25 tags and a fixed value of the hyperparameter C/alpha. We have used various combinations of the word Ngram features. The different features we have used are TFIDF Unigrams, Bigrams, Trigrams and Ngrams. The best micro F1 score that we have obtained was 0.3264, which is at par with the performance of the models given in the actual research paper. 

To build more powerful models, we have actually taken the top 3, 4 and 5 tags resepectively to binarize our tag vectors and used them as a predictor to build our ML models. We did this because we have found out that the average number of tags associated with each movies were close to 3 tags. Hence, logically it should give us a good performance boost if we build our models using the top 3 tags instead of all the tags.

For word Ngram features with top 3 tags, the best micro averaged F1 score obtained was 0.5238 with a weighted recall value of 0.5. This is a significant improvement from the baseline models.

For top 4 tags and word ngram features, the best micro averaged F1 score that we have obtained was 0.5533 with a weighted recall score of 0.62. This is a much more significant improvement than our previous base models. 

The performance of our models dropped slightly when we actually took top 5 tags to vectorize our models. We have achieved a micro averaged F1 score of 0.5461 with a weighted recall score of 0.4613. 

For more advance features we have used character Ngram features to build our model. Just like word ngrams, character ngram features are used to generate Unigram, Bigram, Trigram, 4grams, 5grams and 6Grams features. We have also used char ngrams featurizations of 1-6 char ngrams, 3-4 char ngrams and 2-6 char ngram features. 

In the character ngrams featurization sections we have experimented and build our models using top 3, 4 and 5 tags. 

For the top 3 tags features, we have obtained the best micro averaged f1 score with char 6 grams models. The best micro average f1 score improved slightly from the previous word ngrams models and is no at 0.5776 and there was also a significant improvement in the weighted recall values - 0.6570. That's a massive improvement from both the baseline models as well as the word ngram models. 

For the top 4 tags models, the performance decreased slightly and the best micro averaged f1 score we got was close to 0.5639 with a weighted recall score of 0.6519.  

The performance of the models starts to decrease significantly when we featurize the tags data with top 5 tags. The best weighted f1 score achieved in this case was 0.53 with a recall score of 0.62.

#### Conclusion:

The TFIDF char ngrams features, as described in the research paper proved to be a surprisingly powerful feature which improved both the weighted f1 as well as the recall score. The accuracy values also seems to have improved with the these features. 

We have been able to achieve a micro averaged f1 score of 0.57 and a weighted recall of 0.65, which is a significant improvement from the models that were build in the actual research paper at https://www.aclweb.org/anthology/L18-1274. The highest value of F1 score they have reached was 0.37 whereas the machine learning models we have built has reached a maximum weighted f1 score of 0.57. That seems like a massive improvement. 

#### Future work:

Future versions of this work might include a many to many recurrent neural network to predict the tags, since we know recurrent neural networks are very robust in capturing sequential information. Also, we could further improve the models by adding more and more data to it. 14K data is very less to build a very powerful model. If we had more data and more computing power, I am sure we could actually get a very high micro averaged f1 score (more than 0.8). 

### References:

1. Research Paper: https://www.aclweb.org/anthology/L18-1274
2. Code References: https://www.appliedaicourse.com/
3. Ideas: https://en.wikipedia.org/wiki/Multi-label_classification
