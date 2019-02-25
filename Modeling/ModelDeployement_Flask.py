# https://www.kdnuggets.com/2019/01/build-api-machine-learning-model-using-flask.html
# https://towardsdatascience.com/develop-a-nlp-model-in-python-deploy-it-with-flask-step-by-step-744f3bdd7776
# https://hackernoon.com/deploy-a-machine-learning-model-using-flask-da580f84e60c

### Save Model as pickle file
import pickle
pickle.dump(clf, open('models/final_prediction.pickle', 'wb'))

# You can then open this pickle file later and call the function predict to get a prediction for new input data. 
# This is exactly what we will do in Flask.


