### Links
* https://www.coursera.org/learn/natural-language-processing-tensorflow/home/welcome
* https://github.com/lmoroney/dlaicourse/tree/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP
* https://colab.research.google.com/drive/1znO9kRsbYRV79W0qkd4AGzTRlJexYRod

Different deep learning algos for NLP
* CNN - Convolutional Neural Network
* RNN - Recurrent Neural Network
* HAN - Hierarchical Attention Network

* Load & preprocess text: https://www.tensorflow.org/tutorials/load_data/text
* Whole text section: https://www.tensorflow.org/tutorials/text/word_embeddings


### Text Vectorization
* Pre-process text data: https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/TextVectorization
* Text Vectorization layer: https://towardsdatascience.com/you-should-try-the-new-tensorflows-textvectorization-layer-a80b3c6b00ee
</br>
* Standardize: cleaning of text
  * Default is lower_and_strip_punctuation 
  * For more control over standardization, you can pass your own Callable
    * Applying a custom standardization function example in Standardize section image of https://towardsdatascience.com/you-should-try-the-new-tensorflows-textvectorization-layer-a80b3c6b00ee
    * Any Callable can be passed to this Layer, but if you want to serialize this object you should only pass functions that are registered Keras serializables (see register keras serializable for more details).
    * When using a custom callable for standardize , the data received by the callable will be exactly as passed to this layer. The Callable should return a Tensor of the same shape as the input.
* Split: split into substring tokens	
* 





* GCP - coursera - Links
* Tensorflow tutorials: 
* https://www.tensorflow.org/tutorials/keras/text_classification
* Pre-processing layers : https://www.tensorflow.org/guide/keras/preprocessing_layers
 Look at official Tensorflow git account

* Jay Alammar
* Diff between various BERT models in hugging face
Abhishek Thakur - BERT for sentiment analysis code using BERT base uncased








