########## Flask ##########
# Deploying Keras Deep Learning models using Flask https://towardsdatascience.com/deploying-keras-deep-learning-models-with-flask-5da4181436a2
# Flask Videos https://www.youtube.com/playlist?list=PL-osiE80TeTs4UjLw5MM6OjgkjFeUxCYH
# https://www.kdnuggets.com/2019/01/build-api-machine-learning-model-using-flask.html
# https://towardsdatascience.com/develop-a-nlp-model-in-python-deploy-it-with-flask-step-by-step-744f3bdd7776
# https://medium.com/fintechexplained/flask-host-your-python-machine-learning-model-on-web-b598151886d
# https://www.analyticsvidhya.com/blog/2017/09/machine-learning-models-as-apis-using-flask/
# https://towardsdatascience.com/designing-a-machine-learning-model-and-deploying-it-using-flask-on-heroku-9558ce6bde7b
# https://www.analyticsindiamag.com/5-python-libraries-to-package-and-deploy-machine-learning-models/
# Deploy ML model using Flask - video - https://www.youtube.com/watch?v=UbCWoMf80PY
### Overview 
# 2 key components: WSGI & Jinja2
# First phase: Host ML model to your local machine
# Last phase: Host ML model from local to an external web server





### Save Model as pickle file
import pickle
pickle.dump(clf, open('models/final_prediction.pickle', 'wb'))

# You can then open this pickle file later and call the function predict to get a prediction for new input data. 
# This is exactly what we will do in Flask.



########## Jupyter - repl GUI ##########
# https://www.kdnuggets.com/2019/06/approaches-deploying-machine-learning-production.html

### Model Formats:
# Pickle - Python object is converted to a bitstream
# ONNX - Open Neural Network Exchange format
# PMML - Predictive model markup language







