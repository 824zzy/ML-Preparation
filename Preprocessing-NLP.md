# Data Preprocessing Techniques for Natural Language Processing

## 1. Bag of Words

The idea is to **represent each sentence as a bag of words**, disregarding grammar and paradigms, i.e., just the occurrence of words in a sentence defines the meaning of the sentence for the model.

For example: Given two sentence "I have a dog" and "You have a cat", the first sentence (“I have a dog”) representation becomes 1,1,1,1,0,0, while the second sentence (“You have a cat”) representation becomes 0,1,1,0,1,1.

Pros and Cons of Bag of Words:

1. Pros: Simple to implement, easy to understand.
2. Cons:
   1. If our input data is big, that would mean that the vocabulary size will also increase. This, in turn, makes our representation matrix much larger and makes computations very complex.
   2. Computational nightmare is the inclusion of many 0s in our matrix (i.e., a sparse matrix). A sparse matrix contains less information and wastes a lot of memory.
   3. The biggest disadvantage in Bag-of-Words is the complete inability to learn grammar and semantics.

## 2. Word2Vec
