# RecSys Diversity Analysis and Comparison

This repository contains the source code of the following research work:

* Is diversity optimization always suitable? Towards a better understanding of diversity within recommendation approaches

The aim of the research is to better understand how different types of recommender systems perform, in terms of the diversity criterion.

### Compared recommender systems

* CBF (Content-Based Filtering recommender based on item semantic similarities)
* DNN (Collaborative filtering recommender based on Deep Neural Networks)
* KGE (Hybrid recommender based on Knowledge Graph Embeddings)
* SVD (Collaborative filtering recommender based on Single Value Decomposition)
* UBCF (User Based Collaborative Filtering recommender)

### Code organization

This repository is organized in several subfolders.

#### Datasets

The experiments of the research work are based on three real-world rating datasets of different recommendation domains:

* Anime (anime)
* LibraryThing (book)
* MovieLens (movie)

Each of the three dataset folders contains the following data:

* The training set and the test set
* The mapping file containing the mapped items and their corresponding DBpedia URIs
* The learnt embeddings of entities and relations within two knowledge graphs:
  * ICKG (Item Content Knowledge Graph)
  * ICKG+UPKG (hybrid knowledge graph containing both item contents data (i.e. DBpedia properties) and user preferences data (i.e. ratings))

#### Experimental evaluations

For each of the 3 datasets and for each of the 5 recommenders, we created a script to run the experiments. Note that for the DNN recommender, we created a folder ('Exprimental_Evaluations/Recommenders/DNN'), which stores the architecture of the deep neural network. It is used to load the pre-trained DNN models. For other recommenders, their constructions are done while running the scripts.

### Run the exprimentations

Python 3.6 is needed to run the scripts.

#### Installation of packages
In addition, the following Python packages are also needed:

* The [Surprise](http://surpriselib.com/) package which is a Python scikit for building and analyzing recommender systems.
  * With pip
  ```python
  pip install numpy
  pip install scikit-surprise
  ```

  * With conda
  ```python
  conda install -c conda-forge scikit-surprise
  ```
* The [Pytorch](https://pytorch.org/get-started/locally/), an open source machine learning library.
  * With pip
  ```python
  pip3 install torch torchvision
  ```

  * With conda
  ```python
  conda install pytorch torchvision -c pytorch
    ```
#### Example for running the scripts.

For example, to run the code for the Anime dataset and the SVD recommender:

  ```Python
  python 'Exprimental_Evaluations/Anime/run_SVD.py'
  ```
This script will:
* Load training data and test data
* Construct the recommender based on the training data
* Using the constructed recommender to predict relevance scores for each user and each of the user's unrated items
* Run the greedy diversification optimization for both the classic and the personalised objective functions
* Print and save the results
