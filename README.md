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
