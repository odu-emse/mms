# Module Management System

_Engineering Management & Systems Engineering - ODU_

State of the art API that utilizes Machine Learning models to generate personalized degree paths for each student based on their respective experiences and learning styles.

> Throughout this document we refer to one of the supplementary services as _"client"_. To avoid confusion, treat the client service as a separate application that interacts with this API. The _"client"_ application both consumes and calls this API to present students with a user friendly way of seeing their calculated degree.

# Variations

## Embedding using GPT-3.5

Can be found in the `gpt.ipynb` file. _You must create and add your OpenAI API key to the `api.txt` file before you can successfully run the notebook._

## Embedding using TF-IDF vectorization

Can be found in the `cluster.ipynb` and the `models/classification.py` files.

## Embedding using count vectorization

Can be found in the `models/meta.py` file.
