# https://www.kdnuggets.com/2019/01/build-api-machine-learning-model-using-flask.html
  
### Save Model as pickle file
import pickle
pickle.dump(clf, open('models/final_prediction.pickle', 'wb'))

# You can then open this pickle file later and call the function predict to get a prediction for new input data. 
# This is exactly what we will do in Flask.


