# Synthetic-Experts
Approximate Generative AI with fine-tuned LLM for complex Classification Tasks

### Download the working paper from SSRN
["Creating Synthetic Experts with Generative AI"](https://papers.ssrn.com/abstract_id=4542949)

### Additional Ressources
[www.synthetic-experts.ai](http://www.synthetic-experts.ai)
This repo also includes a notebook that demonstrates how to label text with OpenAI's GPT4 via API (SyntheticExperts-Labels-from-Generative-AI.ipynb)

### Application: Identifying Marketing Mix Variabels (4P of Marketing) in Tweets
[HuggingFace Classifier](https://huggingface.co/dmr76/mmx_classifier_microblog_ENv02)

You can use this classifier to determine which of the 4P's of marketing, also known as marketing mix variables, a microblog post (e.g., Tweet) pertains to:

1. Product
2. Place
3. Price
4. Promotion

This classifier is a fine-tuned checkpoint of [cardiffnlp/twitter-roberta-large-2022-154m] (https://huggingface.co/cardiffnlp/twitter-roberta-large-2022-154m). 
It was trained on 15K Tweets that mentioned at least one of 699 brands. The Tweets were frist cleaned and then labeled using OpenAI's GPT4. 

Because this is a multi-label classification problem, we use binary cross-entropy (BCE) with logits loss for the fine-tuning. We basically combine a sigmoid layer with BCELoss in a single class. To obtain the probabilities for each label (i.e., marketing mix variable), you need to "push" the predictions through a sigmoid function. This is already done in the accompanying python notebook.

IMPORTANT: At the time of writing this description, Huggingface's pipeline did not support multi-label classifiers.

### Citation
Please cite the following reference if you use this model:
```
Ringel, Daniel, Creating Synthetic Experts with Generative Artificial Intelligence (July 15, 2023). Available at SSRN: https://ssrn.com/abstract=4542949
```
