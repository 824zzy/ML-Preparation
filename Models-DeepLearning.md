# Models

## Deep Learning Models

## Transformer

![mha](2022-09-08-11-01-25.png)

## BERT

BERT, which stands for Bidirectional Encoder Representations from Transformers, is based on Transformers, a deep learning model in which every output element is connected to every input element, and the weightings between them are dynamically calculated based upon their connection.

BERT is an open source machine learning framework for natural language processing (NLP). BERT is designed to help computers understand the meaning of ambiguous language in text by using surrounding text to establish context. The BERT framework was pre-trained using text from Wikipedia and can be fine-tuned with question and answer datasets.

BERT Base Model has 12 Layers and 110M parameters with 768 Hidden and equal embedding layers. This large size makes it very computationally heavy to train.

BERT Family:

1. ALBERT: A Lite BERT has 12 million parameters with 768 hidden layers and 128 embedding layers, the following 2 techniques are used
   1. Cross-layer parameter sharing: In this method, the parameter of only the first encoder is learnt and the same is used across all encoders.
   2. Factorized embedding layer parameterization: Instead of keeping the embedding layer at 768, the embedding layer is reduced by factorization to 128 layers.
2. RoBERTa: RoBERTa stands for “Robustly Optimized BERT pre-training Approach”. In many ways this is a better version of the BERT model. The key points of difference are as follows:
   1. Dynamic Masking: BERT uses static masking i.e. the same part of the sentence is masked in each Epoch. In contrast, RoBERTa uses dynamic masking, wherein for different Epochs different part of the sentences are masked. This makes the model more robust.
   2. Remove NSP Task: It was observed that the NSP task is not very useful for pre-training the BERT model. Therefore, the RoBERTa only with the MLM task.
   3. More data Points: BERT is pre-trained on “Toronto BookCorpus” and “English Wikipedia datasets” i.e. as a total of 16 GB of data. In contrast, in addition to these two datasets, RoBERTa was also trained on other datasets like CC-News (Common Crawl-News), Open WebText etc. The total size of these datasets is around 160 GB.
   4. Large Batch size: To improve on the speed and performance of the model, RoBERTa used a batch size of 8,000 with 300,000 steps. In comparison, BERT uses a batch size of 256 with 1 million steps.
3. ELECTRA: ELECTRA stands for “Efficiently Learning an Encoder that Classifies Token Replacements Accurately”. The model uses a generator-discriminator structure. Other than being the lighter version of BERT, ELECTRA has the following distinguishing features:
   1. Replaced Token Detection: Instead of MLM for pre-training, ELECTRA uses a task called “Replaced Token Detection” (RTD). In RTD, instead of masking the token, the token is replaced by a wrong token and the model is expected to classify, whether the tokens are replaced with wrong or not.
   2. No NSP pre-training is performed.

## T5

T5 is a text-to-text transformer model that was trained on a large corpus of text-to-text data. T5 is a successor to BERT and GPT-2, and is the first text-to-text transformer model that is trained on a large scale with a denoising objective. The changes compared to BERT include:

1. Adding a causal decoder to the bidirectional architecture.
2. Replacing the fill-in-the-blank cloze task with a mix of alternative pre-training tasks.

![T5 model structure](2022-09-08-11-11-34.png)
