# Module Management System

_Engineering Management & Systems Engineering - ODU_

State of the art API that utilizes Machine Learning models to generate personalized degree paths for each student based on their respective experiences and learning styles.

> Throughout this document we refer to one of the supplementary services as _"client"_. To avoid confusion, treat the client service as a separate application that interacts with this API. The _"client"_ application both consumes and calls this API to present students with a user friendly way of seeing their calculated degree.

# Embedders

## GPT-3.5

Can be found in the [`gpt.ipynb`](https://github.com/odu-emse/mms/blob/dev/gpt.ipynb) file. _You must create and add your OpenAI API key to the `api.txt` file before you can successfully run the notebook._

## TF-IDF

Can be found in the [`cluster.ipynb`](https://github.com/odu-emse/mms/blob/dev/cluster.ipynb) and the [`models/classification.py`](https://github.com/odu-emse/mms/blob/dev/models/classification.py) files.

## Count Vectorizer

Can be found in the [`models/meta.py`](https://github.com/odu-emse/mms/blob/dev/models/meta.py) file.

## Collection

There is a collection of embedders that are implmented and compared within a single notebook. The notebook can be found in the [`embedders.ipynb`](https://github.com/odu-emse/mms/blob/dev/embedders.ipynb) file.

### BERT

The BERT section of the `embedders.ipynb` file is an implementation of the BERT (Bidirectional Encoder Representations from Transformers) model from the TensorFlow library. This embedder generates vector representations of words and sentences, which can be used for a variety of natural language processing tasks, including language translation and sentiment analysis. The BERT model is a pre-trained deep learning model that uses a transformer-based architecture to generate high-quality embeddings. The model can be fine-tuned on a specific task by adding a task-specific layer on top of the pre-trained BERT model. The BERT model has been shown to outperform other word and sentence embedding models on a variety of natural language processing tasks.

### Sent2Vec

The Sent2Vec section of the `embedders.ipynb` file is an implementation of a sentence embedding model that generates vector representations of sentences. This embedder is based on the skip-thoughts model and uses an encoder-decoder architecture to generate sentence embeddings. The encoder is a multi-layer bidirectional LSTM network that reads the input sentence and generates a fixed-length vector representation of the sentence. The decoder is another LSTM network that takes the sentence embedding as input and generates the surrounding sentences. The Sent2Vec model has been shown to outperform other sentence embedding models on a variety of natural language processing tasks, including sentiment analysis and text classification.

### Doc2Vec

The Doc2Vec section of the `embedders.ipynb` file is an implementation of the Doc2Vec model from the Gensim library. This embedder generates vector representations of documents, which can be used for a variety of natural language processing tasks, including document classification and information retrieval. The Doc2Vec model is an extension of the Word2Vec model, which generates vector representations of words. The Doc2Vec model adds an additional vector representation for each document, which is learned during the training process. The Doc2Vec model has been shown to outperform other document embedding models on a variety of natural language processing tasks.

### Word2Vec

The Word2Vec section of the `embedders.ipynb` file is an implementation of the Word2Vec model from the Gensim library. This embedder generates vector representations of words, which can be used for a variety of natural language processing tasks, including language translation and sentiment analysis. The Word2Vec model is a neural network-based model that learns vector representations of words by predicting the context in which they appear. The model can be trained on a large corpus of text data to generate high-quality word embeddings. The Word2Vec model has been shown to outperform other word embedding models on a variety of natural language processing tasks.

# Classification

## Supervised

The entire pipeline of supervised classification models can be found in the [`supervised.ipynb`](https://github.com/odu-emse/mms/blob/dev/supervised.ipynb) file.

## Unsupervised

To find unsupervised classification implementations, refer to the [`classification.ipynb`](https://github.com/odu-emse/mms/blob/dev/classification.ipynb) and the [`cluster.ipynb`](https://github.com/odu-emse/mms/blob/dev/cluster.ipynb) files.

# Utilities

## `seed.py`

The [`utils/seed.py`](https://github.com/odu-emse/mms/blob/dev/utils/seed.py) file contains the code that was used to seed the database with the initial data. It is not necessary to run this file as the database is already seeded. This file requires additional configuration of the _client_ application to work properly.

## `fetch.py`

The [`utils/fetch.py`](https://github.com/odu-emse/mms/blob/dev/utils/fetch.py) file contains the code that was used to retrieve the data from the managed database and store it in a local file. It is not necessary to run this file as the database is already seeded. This file requires additional configuration of the _client_ application to work properly.
