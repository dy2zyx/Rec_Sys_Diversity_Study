# RecSys Diversity Analysis and Comparison

This repository contains the source code of the following research work:

* Is diversity optimization always suitable? Towards a better understanding of diversity within recommendation approaches

The aim of the research is to better understand how different types of recommender systems perform, in terms of the diversity criterion.

### Compared recommender systems

* CBF (Content-Based Filtering model based on item semantic similarities)
* DNN (Collaborative filtering model based on Deep Neural Networks)
* KGE (Hybrid model based on Knowledge Graph Embeddings)
* SVD (Collaborative filtering model based on Single Value Decomposition)
* IBCF (Item-Based Collaborative Filtering model)
* TopPopular (Item popularity-based model)
* CBF-TopPopular (Hybride model based on the CBF and the TopPopular models)

### Code organization

This repository is organized in several subfolders.

#### Datasets

The experiments of the research work are based on three real-world rating datasets of different recommendation domains:

* Anime (anime)
* LibraryThing (book)
* MovieLens (movie)

Each of the three dataset folders contains the following data:

* The trainset (80%) and the testset (20%), containing lists of ratings in (user, item, rating) formats
* The mapping file containing the mapped items and their corresponding DBpedia URIs
* The learnt embeddings of entities and relations within two knowledge graphs:
  * ICKG (Item Content Knowledge Graph)
  * ICKG+UPKG (hybrid knowledge graph containing both item contents data (i.e. DBpedia properties) and user preferences data (i.e. ratings))

#### Hyperparamter Tuning

Each considered recommendation approach is optimized towards accuracy on each dataset before being used for the diversity analyse. The hyperparameters of each recommendation approach are tuned based on a validation set, containing 20% of the training set and the MAP metric is adopted for optimization. We run 20 interations of bayesian optimization for each recommender system on each dataset.

#### Experimental evaluations

For each of the 3 datasets and for each of the 7 recommenders, we created a script to run the experiments. The 'generate_candidates_{dataset_name}.py' script is used to generate, for each recommender and each dataset, a dictionary that stores "user -> list of condidate items (with predicted accuracy scores)". This is the input for our diversity study.

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

* The [BayesianOptimization](https://github.com/fmfn/BayesianOptimization), an open source python library for bayesian optimization
  * With pip
  ```python
  pip3 install bayesian-optimization
  ```

  * With conda
  ```python
  conda install -c conda-forge bayesian-optimization
    ```

#### Example for running the scripts.

For running examples, please refer to the 'NoteBook_Examples' folder.
