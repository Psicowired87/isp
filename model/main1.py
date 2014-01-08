import model

#Test 1

model.load_data('data/train.csv', 'data/test.csv')
train_tweets, test_tweets = model.preprocess_data()
X,y,test, feature_names = model.feature_extraction(train_tweets, test_tweets)
clfs = model.train2(X,y)


test_prediction = model.predict2(clfs, test)
model.saveResults('output/somePrediction.csv', test_prediction)