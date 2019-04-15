# https://www.kdnuggets.com/2019/01/build-api-machine-learning-model-using-flask.html
# https://towardsdatascience.com/develop-a-nlp-model-in-python-deploy-it-with-flask-step-by-step-744f3bdd7776
# https://hackernoon.com/deploy-a-machine-learning-model-using-flask-da580f84e60c
# https://www.analyticsvidhya.com/blog/2017/09/machine-learning-models-as-apis-using-flask/
# https://towardsdatascience.com/designing-a-machine-learning-model-and-deploying-it-using-flask-on-heroku-9558ce6bde7b
# 

https://www.analyticsindiamag.com/5-python-libraries-to-package-and-deploy-machine-learning-models/


### Save Model as pickle file
import pickle
pickle.dump(clf, open('models/final_prediction.pickle', 'wb'))

# You can then open this pickle file later and call the function predict to get a prediction for new input data. 
# This is exactly what we will do in Flask.


