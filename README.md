<!--- BADGES: START --->
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-models-yellow)](https://huggingface.co/models?library=sentence-transformers)
[![GitHub - License](https://img.shields.io/github/license/UKPLab/sentence-transformers?logo=github&style=flat&color=green)][#github-license]
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sentence-transformers?logo=pypi&style=flat&color=blue)][#pypi-package]
[![PyPI - Package Version](https://img.shields.io/pypi/v/sentence-transformers?logo=pypi&style=flat&color=orange)][#pypi-package]
[![Docs - GitHub.io](https://img.shields.io/static/v1?logo=github&style=flat&color=pink&label=docs&message=sentence-transformers)][#docs-package]
<!-- [![PyPI - Downloads](https://img.shields.io/pypi/dm/sentence-transformers?logo=pypi&style=flat&color=green)][#pypi-package] -->

[#github-license]: https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE
[#pypi-package]: https://pypi.org/project/sentence-transformers/
[#conda-forge-package]: https://anaconda.org/conda-forge/sentence-transformers
[#docs-package]: https://www.sbert.net/
<!--- BADGES: END --->



# Semantic Search
Semantic search seeks to improve search accuracy by understanding the semantic meaning of the search query and the corpus to search over. Semantic search can also perform well given synonyms, abbreviations, and misspellings, unlike keyword search engines that can only find documents based on lexical matches.

# Background
The idea behind semantic search is to embed all entries in your corpus, whether they be sentences, paragraphs, or documents, into a vector space. At search time, the query is embedded into the same vector space and the closest embeddings from your corpus are found. These entries should have a high semantic similarity with the query.


![SemanticSearch](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/SemanticSearch.png) 


Sentence Transformers are used for Semantic Search.

# Sentence Transformers: Embeddings, Retrieval, and Reranking

This framework provides an easy method to compute embeddings for accessing, using, and training state-of-the-art embedding and reranker models. It can be used to compute embeddings using Sentence Transformer models ([quickstart](https://sbert.net/docs/quickstart.html#sentence-transformer)), to calculate similarity scores using Cross-Encoder (a.k.a. reranker) models ([quickstart](https://sbert.net/docs/quickstart.html#cross-encoder)) or to generate sparse embeddings using Sparse Encoder models ([quickstart](https://sbert.net/docs/quickstart.html#sparse-encoder)). This unlocks a wide range of applications, including [semantic search](https://sbert.net/examples/applications/semantic-search/README.html), [semantic textual similarity](https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html), and [paraphrase mining](https://sbert.net/examples/applications/paraphrase-mining/README.html).

A wide selection of over [15,000 pre-trained Sentence Transformers models](https://huggingface.co/models?library=sentence-transformers) are available for immediate use on 🤗 Hugging Face, including many of the state-of-the-art models from the [Massive Text Embeddings Benchmark (MTEB) leaderboard](https://huggingface.co/spaces/mteb/leaderboard). Additionally, it is easy to train or finetune your own [embedding models](https://sbert.net/docs/sentence_transformer/training_overview.html), [reranker models](https://sbert.net/docs/cross_encoder/training_overview.html) or [sparse encoder models](https://sbert.net/docs/sparse_encoder/training_overview.html) using Sentence Transformers, enabling you to create custom models for your specific use cases.

For the **full documentation**, see **[www.SBERT.net](https://www.sbert.net)**.

## Installation

We recommend **Python 3.9+**, **[PyTorch 1.11.0+](https://pytorch.org/get-started/locally/)**, and **[transformers v4.34.0+](https://github.com/huggingface/transformers)**.

**Install with pip**

```
pip install -U sentence-transformers
```

**Install with conda**

```
conda install -c conda-forge sentence-transformers
```

**Install from sources**

Alternatively, you can also clone the latest version from the [repository](https://github.com/UKPLab/sentence-transformers) and install it directly from the source code:

````
pip install -e .
```` 

**PyTorch with CUDA**

If you want to use a GPU / CUDA, you must install PyTorch with the matching CUDA Version. Follow
[PyTorch - Get Started](https://pytorch.org/get-started/locally/) for further details how to install PyTorch.

## Getting Started

See [Quickstart](https://www.sbert.net/docs/quickstart.html) in our documentation.

### Embedding Models

First download a pretrained embedding a.k.a. Sentence Transformer model.

````python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
````

Then provide some texts to the model.

````python
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# => (3, 384)
````

And that's already it. We now have numpy arrays with the embeddings, one for each text. We can use these to compute similarities.

````python
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])
````

### Reranker Models

First download a pretrained reranker a.k.a. Cross Encoder model.

```python
from sentence_transformers import CrossEncoder

# 1. Load a pretrained CrossEncoder model
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
```

Then provide some texts to the model.

```python
# The texts for which to predict similarity scores
query = "How many people live in Berlin?"
passages = [
    "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
    "Berlin has a yearly total of about 135 million day visitors, making it one of the most-visited cities in the European Union.",
    "In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.",
]

# 2a. predict scores for pairs of texts
scores = model.predict([(query, passage) for passage in passages])
print(scores)
# => [8.607139 5.506266 6.352977]
```

And we're good to go. You can also use [`model.rank`](https://sbert.net/docs/package_reference/cross_encoder/cross_encoder.html#sentence_transformers.cross_encoder.CrossEncoder.rank) to avoid having to perform the reranking manually:

```python
# 2b. Rank a list of passages for a query
ranks = model.rank(query, passages, return_documents=True)

print("Query:", query)
for rank in ranks:
    print(f"- #{rank['corpus_id']} ({rank['score']:.2f}): {rank['text']}")
"""
Query: How many people live in Berlin?
- #0 (8.61): Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.
- #2 (6.35): In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.
- #1 (5.51): Berlin has a yearly total of about 135 million day visitors, making it one of the most-visited cities in the European Union.
"""
```
### Sparse Encoder Models

First download a pretrained sparse embedding a.k.a. Sparse Encoder model.

```python

from sentence_transformers import SparseEncoder

# 1. Load a pretrained SparseEncoder model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# 2. Calculate sparse embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 30522] - sparse representation with vocabulary size dimensions

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[   35.629,     9.154,     0.098],
#         [    9.154,    27.478,     0.019],
#         [    0.098,     0.019,    29.553]])

# 4. Check sparsity stats
stats = SparseEncoder.sparsity(embeddings)
print(f"Sparsity: {stats['sparsity_ratio']:.2%}")
# Sparsity: 99.84%
```

## Pre-Trained Models

Hugging Face organization provides a large list of pretrained models for more than 100 languages. Some models are general purpose models, while others produce embeddings for specific use cases. 

* [Pretrained Sentence Transformer (Embedding) Models](https://sbert.net/docs/sentence_transformer/pretrained_models.html)
* [Pretrained Cross Encoder (Reranker) Models](https://sbert.net/docs/cross_encoder/pretrained_models.html)
* [Pretrained Sparse Encoder (Sparse Embeddings) Models](https://sbert.net/docs/sparse_encoder/pretrained_models.html)

## Training

This framework allows you to fine-tune your own sentence embedding methods, so that you get task-specific sentence embeddings. You have various options to choose from in order to get perfect sentence embeddings for your specific task. 

* Embedding Models
    * [Sentence Transformer > Training Overview](https://www.sbert.net/docs/sentence_transformer/training_overview.html)
    * [Sentence Transformer > Training Examples](https://www.sbert.net/docs/sentence_transformer/training/examples.html) or [training examples on GitHub](https://github.com/UKPLab/sentence-transformers/tree/master/examples/sentence_transformer/training).
* Reranker Models
    * [Cross Encoder > Training Overview](https://www.sbert.net/docs/cross_encoder/training_overview.html)
    * [Cross Encoder > Training Examples](https://www.sbert.net/docs/cross_encoder/training/examples.html) or [training examples on GitHub](https://github.com/UKPLab/sentence-transformers/tree/master/examples/cross_encoder/training).
* Sparse Embedding Models
    * [Sparse Encoder > Training Overview](https://www.sbert.net/docs/sparse_encoder/training_overview.html)
    * [Sparse Encoder > Training Examples](https://www.sbert.net/docs/sparse_encoder/training/examples.html) or [training examples on GitHub](https://github.com/UKPLab/sentence-transformers/tree/master/examples/sparse_encoder/training).

Some highlights across the different types of training are:
- Support of various transformer networks including BERT, RoBERTa, XLM-R, DistilBERT, Electra, BART, ...
- Multi-Lingual and multi-task learning
- Evaluation during training to find optimal model
- [20+ loss functions](https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html) for embedding models, [10+ loss functions](https://www.sbert.net/docs/package_reference/cross_encoder/losses.html) for reranker models and [10+ loss functions](https://www.sbert.net/docs/package_reference/sparse_encoder/losses.html) for sparse embedding models, allowing you to tune models specifically for semantic search, paraphrase mining, semantic similarity comparison, clustering, triplet loss, contrastive loss, etc.

## Application Examples

You can use this framework for:

- **Computing Sentence Embeddings**
  - [Dense Embeddings](https://www.sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html)
  - [Sparse Embeddings](https://www.sbert.net/examples/sparse_encoder/applications/computing_embeddings/README.html)

- **Semantic Textual Similarity** 
  - [Dense STS](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html)
  - [Sparse STS](https://www.sbert.net/examples/sparse_encoder/applications/semantic_textual_similarity/README.html)

- **Semantic Search**
  - [Dense Search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)  
  - [Sparse Search](https://www.sbert.net/examples/sparse_encoder/applications/semantic_search/README.html)

- **Retrieve & Re-Rank**
  - [Dense only Retrieval](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
  - [Sparse/Dense/Hybrid Retrieval](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)

- [Clustering](https://www.sbert.net/examples/sentence_transformer/applications/clustering/README.html)
- [Paraphrase Mining](https://www.sbert.net/examples/sentence_transformer/applications/paraphrase-mining/README.html)
- [Translated Sentence Mining](https://www.sbert.net/examples/sentence_transformer/applications/parallel-sentence-mining/README.html)
- [Multilingual Image Search, Clustering & Duplicate Detection](https://www.sbert.net/examples/sentence_transformer/applications/image-search/README.html)

and many more use-cases.

For all examples, see [examples/sentence_transformer/applications](https://github.com/UKPLab/sentence-transformers/tree/master/examples/sentence_transformer/applications).



## Symmetric vs. Asymmetric Semantic Search

A **critical distinction** for your setup is *symmetric* vs. *asymmetric semantic search*:
- For **symmetric semantic search** your query and the entries in your corpus are of about the same length and have the same amount of content. An example would be searching for similar questions: Your query could for example be *"How to learn Python online?"* and you want to find an entry like *"How to learn Python on the web?"*. For symmetric tasks, you could potentially flip the query and the entries in your corpus. 
   
- For **asymmetric semantic search**, you usually have a **short query** (like a question or some keywords) and you want to find a longer paragraph answering the query. An example would be a query like *"What is Python"* and you want to find the paragraph *"Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy ..."*. For asymmetric tasks, flipping the query and the entries in your corpus usually does not make sense.
  

It is critical **that you choose the right model** for your type of task.


For a simple example, see semantic_search.py:


## Elasticsearch
[Elasticsearch](https://www.elastic.co/elasticsearch/) has the possibility to [index dense vectors](https://www.elastic.co/what-is/vector-search) and to use them for document scoring. We can easily index embedding vectors, store other data alongside our vectors and, most importantly, efficiently retrieve relevant entries using [approximate nearest neighbor search](https://www.elastic.co/blog/introducing-approximate-nearest-neighbor-search-in-elasticsearch-8-0).

For further details, see semantic_search_quora_elasticsearch.py.

## OpenSearch
[OpenSearch](https://opensearch.org/) is a community-driven, open-source search engine that supports vector search capabilities. It allows you to index dense vectors and perform efficient similarity search using approximate nearest neighbor algorithms. OpenSearch can be used to implement both traditional keyword-based search (BM25) and semantic search, making it possible to compare and combine both approaches.

For an example implementation, see semantic_search_nq_opensearch.py, which shows how to use OpenSearch with the Natural Questions dataset, demonstrating both semantic search and BM25 search capabilities.

## Approximate Nearest Neighbor
Searching a large corpus with millions of embeddings can be time-consuming if exact nearest neighbor search is used. In that case, Approximate Nearest Neighbor (ANN) can be helpful. Here, the data is partitioned into smaller fractions of similar embeddings. This index can be searched efficiently and the embeddings with the highest similarity (the nearest neighbors) can be retrieved within milliseconds, even if you have millions of vectors. However, the results are not necessarily exact. It is possible that some vectors with high similarity will be missed.

For all ANN methods, there are usually one or more parameters to tune that determine the recall-speed trade-off. If you want the highest speed, you have a high chance of missing hits. If you want high recall, the search speed decreases.

Three popular libraries for approximate nearest neighbor are [Annoy](https://github.com/spotify/annoy), [FAISS](https://github.com/facebookresearch/faiss), and [hnswlib](https://github.com/nmslib/hnswlib/).

Examples:

- semantic_search_quora_hnswlib.py
- semantic_search_quora_annoy.py
- semantic_search_quora_faiss.py

## Retrieve & Re-Rank
For complex semantic search scenarios, a two-stage retrieve & re-rank pipeline is advisable:
![InformationRetrieval](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/InformationRetrieval.png)

For further details, see [Retrieve & Re-rank](https://github.com/UKPLab/sentence-transformers/blob/master/examples/sentence_transformer/applications/retrieve_rerank/README.md).

## Examples

A handful of common use cases:

### Similar Questions Retrieval
semantic_search_quora_pytorch.py [ [Colab version](https://colab.research.google.com/drive/12cn5Oo0v3HfQQ8Tv6-ukgxXSmT3zl35A?usp=sharing) ] shows an example based on the [Quora duplicate questions](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs) dataset. The user can enter a question, and the code retrieves the most similar questions from the dataset using `util.semantic_search`. As model, we use [distilbert-multilingual-nli-stsb-quora-ranking](https://huggingface.co/sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking), which was trained to identify similar questions and supports 50+ languages. Hence, the user can input the question in any of the 50+ languages. This is a **symmetric search task**, as the search queries have the same length and content as the questions in the corpus.

### Similar Publication Retrieval
semantic_search_publications.py [ [Colab version](https://colab.research.google.com/drive/12hfBveGHRsxhPIUMmJYrll2lFU4fOX06?usp=sharing) ] shows an example how to find similar scientific publications. As corpus, we use all publications that have been presented at the EMNLP 2016 - 2018 conferences. As search query, we input the title and abstract of more recent publications and find related publications from our corpus. We use the [SPECTER](https://huggingface.co/sentence-transformers/allenai-specter) model. This is a **symmetric search task**, as the paper in the corpus consists of title & abstract and we search for title & abstract.

### Question & Answer Retrieval
semantic_search_wikipedia_qa.py [ [Colab Version](https://colab.research.google.com/drive/11GunvCqJuebfeTlgbJWkIMT0xJH6PWF1?usp=sharing) ]: This example uses a model that was trained on the [Natural Questions dataset](https://huggingface.co/datasets/sentence-transformers/natural-questions). It consists of about 100k real Google search queries, together with an annotated passage from Wikipedia that provides the answer. It is an example of an **asymmetric search task**. As corpus, we use the smaller [Simple English Wikipedia](https://simple.wikipedia.org/wiki/Main_Page) so that it fits easily into memory.

retrieve_rerank_simple_wikipedia.ipynb[ [Colab Version](https://colab.research.google.com/github/UKPLab/sentence-transformers/blob/master/examples/sentence_transformer/applications/retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb) ]: This script uses the Retrieve & Re-rank strategy and is an example for an **asymmetric search task**. We split all Wikipedia articles into paragraphs and encode them with a bi-encoder. If a new query / question is entered, it is encoded by the same bi-encoder and the paragraphs with the highest cosine-similarity are retrieved. Next, the retrieved candidates are scored by a Cross-Encoder re-ranker and the 5 passages with the highest score from the Cross-Encoder are presented to the user. We use models that were trained on the [MS Marco Passage Reranking](https://github.com/microsoft/MSMARCO-Passage-Ranking/) dataset, a dataset with about 500k real queries from Bing search.


