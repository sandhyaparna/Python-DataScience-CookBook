### Links
* Load & preprocess text: https://www.tensorflow.org/tutorials/load_data/text
* Whole text section: https://www.tensorflow.org/tutorials/text/word_embeddings
* Coursera Course work
  * https://www.coursera.org/learn/natural-language-processing-tensorflow/home/welcome
  * https://github.com/lmoroney/dlaicourse/tree/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP
  * https://colab.research.google.com/drive/1znO9kRsbYRV79W0qkd4AGzTRlJexYRod
* NLP Repo https://notebooks.quantumstat.com/
* AML 4 GCP https://github.com/sandhyaparna/GCP-GoogleCloudPlatform/tree/master/Labs
* Official NLP: https://github.com/tensorflow/models/tree/master/official/nlp Models: BERT, ALBERT, XLNet, Transformer, NHNet
  * https://colab.research.google.com/github/tensorflow/models/blob/master/official/colab/nlp/nlp_modeling_library_intro.ipynb
* Abhishek Thakur: Youtube videos; Data pre-processing for question & Answering; BERT for training sentiment model
  * https://github.com/abhishekkrthakur/bert-sentiment/
* AIEngineering https://www.youtube.com/watch?v=1aBPXWLftFs&t=379s
  * Code: https://github.com/srivatsan88/YouTubeLI
* NLP https://github.com/neomatrix369/awesome-ai-ml-dl/tree/master/natural-language-processing
* Hugging face
  * Preprocessing https://huggingface.co/transformers/preprocessing.html#
  * Tokenizers: https://huggingface.co/transformers/tokenizer_summary.html
  * Tokenization https://www.youtube.com/watch?v=0-wOZ2SXDOw
  * subword tokenization: Byte pair encoding https://www.youtube.com/watch?v=zjaRNfvNMTs 
* NLP Kaggle problems: https://twitter.com/abhi1thakur/status/1376483345790566402
* **Text embedding models on TFHub** https://tfhub.dev/s?module-type=text-embedding


### Tokenization
* https://huggingface.co/transformers/preprocessing.html
* BERTTOKENIZER, Byte pair, wordpiece and sentence piece are explained clearly https://huggingface.co/transformers/tokenizer_summary.html
* https://medium.com/@makcedward/how-subword-helps-on-your-nlp-model-83dd1b836f46
* Subword tokenization: Subword tokenization algorithms rely on the principle that frequently used words should not be split into smaller subwords, but rare words should be decomposed into meaningful subwords. 
* For instance "annoyingly" might be considered a rare word and could be decomposed into "annoying" and "ly"
* subword tokenization enables the model to process words it has never seen before, by decomposing them into known subwords
* BERT uses wword piece tokenizer
* WordPiece is the subword tokenization algorithm used for BERT, DistilBERT, and Electra. 
* XLNet uses sentence piece tokenizer
* Byte Pair Encoding: BPE relies on a pre-tokenizer that splits the training data into words. 

### Tensorflow Notebooks
* https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/text_classification.ipynb#scrollTo=51wNaPPApk1K
  * Data is extracted from url into folders. Then data is passed through 'tf.keras.preprocessing.text_dataset_from_directory' function to prepare data into suitable format
  * Function for a sneak peek into the data </br>
  for text_batch, label_batch in raw_train_ds.take(1): </br>
  for i in range(3): </br>
    print("Review", text_batch.numpy()[i]) </br>
    print("Label", label_batch.numpy()[i])   </br>
  * <br /> correspond to line breaks
  * 




### Models
* Summary of diff models https://huggingface.co/transformers/model_summary.html#autoencoding-models
* BERT https://huggingface.co/transformers/model_doc/bert.html
* ALBERT: https://huggingface.co/transformers/model_doc/albert.html
* Tensorflow checkpoints: https://huggingface.co/transformers/converting_tensorflow_models.html




Different deep learning algos for NLP:
* CNN - Convolutional Neural Network
* RNN - Recurrent Neural Network
* HAN - Hierarchical Attention Network








### Text Vectorization
* Pre-process text data: https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/TextVectorization
* Text Vectorization layer in detail for each argument: https://towardsdatascience.com/you-should-try-the-new-tensorflows-textvectorization-layer-a80b3c6b00ee
* Official Tensorflow's team colab example: https://colab.research.google.com/drive/1RvCnR7h0_l4Ekn5vINWToI9TNJdpUZB3
* Official parameters that can be sent into this Text Vectorization layer: https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/TextVectorization




* Tfrecord & Tfexample: https://www.tensorflow.org/tutorials/load_data/tfrecord
* GCP - coursera - Links
* Tensorflow tutorials: 
* https://www.tensorflow.org/tutorials/keras/text_classification
* Pre-processing layers : https://www.tensorflow.org/guide/keras/preprocessing_layers
 Look at official Tensorflow git account

* Jay Alammar
* Diff between various BERT models in hugging face
Abhishek Thakur - BERT for sentiment analysis code using BERT base uncased








