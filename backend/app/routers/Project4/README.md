# Project4

Run:

```
streamlit run app.py --server.port 8604
```

# Pipeline
1. If you have no csv files, run data_preprocessor.py to generate dataset and preprocess it for model training.
2. If you have no data visualizations (visualizations folder), run sentiment_score.py to vizualize the data and assign value to the words.
3. If you have no saved pickle NLP model, run ner_model.py to initialize, train, evaluate, and save the model.

# Assignment Requirements
### Preprocess and Visualize the Data:
1. Perform a descriptive statistical analysis of the data and decide how to handle missing values. ✅
2. Store your data in a dataframe. ✅
3. Count the number of positive, negative, and neutral text items, as tagged by a score in one of the columns. ✅
4. Display your findings in a plot. ✅

### Build the Model:
1. Remove punctuation! ✅
2. Remove stop words (i.e., words that do not add a sentiment). ✅
3. Assign each word in every text element, with a sentiment score (use TfidVectorizer). ✅
4. Use a binary classification algorithm (e.g., logistic regression), which you can import from sklearn. ✅
5. Divide the data into a training set and testing set, with a ratio of 80:20. ✅
6. Fit the data set using the model. ✅
7. Compute the (accuracy) score of the model. ✅
8. Make Predictions: Enter several questions and assess the sentiment they convey. ❓

### Evaluate the Model:
1. Create a confusion matrix to assess the overall performance. ✅
2. Present the performance metrics and visualize the findings. ✅
3. Summarize the project, explaining to what extent it is suitable to perform sentiment analysis. ✅
