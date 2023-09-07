# Synthetic-Experts
<b><i>Approximate Generative AI with fine-tuned LLM for complex Classification Tasks</b></i>

### Download the working paper from SSRN
["Creating Synthetic Experts with Generative AI"](https://papers.ssrn.com/abstract_id=4542949)

### Additional Ressources
- [www.synthetic-experts.ai](http://www.synthetic-experts.ai)  

## Notebooks in this Repo
<b><i>Using</b> the MMX Synthetic Expert</i>

- Quickly identify which marketing mix (MMX) variable a text pertains to, if any: `SyntheticExperts_Quickstart_MMXClassifier.ipynb`
- Batch processing for rapidly identifying MMX variables at scale: `SyntheticExperts_Predict_Texts_with_MMXClassifier.ipynb`
- Reveal differences in consumer sentiment across brands' MMX from texts and discover MMX-specific topics: `SyntheticExperts_MMX_Sentiment_x_Topics.ipynb`

<b><i>Creating</b> a Synthetic Expert</i>

- Label texts using generative AI by example of GPT4 via OpenAI's API: `SyntheticExperts-Labels-from-Generative-AI.ipynb`
- Approximate a powerful generative AI for a specific task by fine-tuning a foundational large language model: `SyntheticExperts_Train_MMX_Classifier.ipynb`

<b><i>Anonomyzing</b> texts with Synthetic Twins</i>

- Create replicas of texts that capture their idea and meaning but obfuscate identifying information with generative AI: `SyntheticExperts_Create_Synthetic_Twins_of_Texts.ipynb`

<b><i>GPU setup</b> on Apple M1 and M2 systems</i>

- Get your Apple notebook ready to create Synthetic Experts and rapidly identify constructs of interest in vast amounts of text: `Setup-MacBook-M2-Pytorch-TensorFlow-Apr2023.ipynb`

## Application: Identifying Marketing Mix Variabels (4P of Marketing) in Tweets
[MMX Classifier](https://huggingface.co/dmr76/mmx_classifier_microblog_ENv02)

You can use this classifier to determine which of the 4P's of marketing, also known as marketing mix variables, a microblog post (e.g., Tweet) pertains to:

1. Product
2. Place
3. Price
4. Promotion

This classifier is a fine-tuned checkpoint of [cardiffnlp/twitter-roberta-large-2022-154m] (https://huggingface.co/cardiffnlp/twitter-roberta-large-2022-154m). 
It was trained on 15K Tweets that mentioned at least one of 699 brands. The Tweets were frist cleaned and then labeled using OpenAI's GPT4. 

Because this is a multi-label classification problem, we use binary cross-entropy (BCE) with logits loss for the fine-tuning. We basically combine a sigmoid layer with BCELoss in a single class. To obtain the probabilities for each label (i.e., marketing mix variable), you need to "push" the predictions through a sigmoid function. This is already done in the accompanying python notebook.

**IMPORTANT:** At the time of writing this description, Huggingface's pipeline did not support multi-label classifiers.

### Quickstart
Check out this [Python Notebook](https://github.com/dringel/Synthetic-Experts/blob/main/SyntheticExperts_Quickstart_MMXClassifier.ipynb) to try out a MMX Synthetic Expert

### Demo Datasets
Demo datasets are available in the `Data` folder. The texts in these datasets are based on real Tweets that were rewritten by an AI. I call these data ***Synthetic Twins***.

>***Synthetic Twins*** correspond semantically in idea and meaning to original texts. However, wording, people, places, firms, brands, and products were changed by an AI. As such, ***Synthetic Twins*** mitigate, to some extent, possible privacy, and copyright concerns. If you'd like to learn more about ***Synthetic Twins***, another generative AI project by Daniel Ringel, then please get in touch! dmr@unc.edu  

You can ***create your own Synthetic Twins of texts*** with this Python notebook: `SyntheticExperts_Create_Synthetic_Twins_of_Texts.ipynb`, available as BETA version (still being tested) in this repo.

## Citation
Please cite the following reference if you use this model:
```
Ringel, Daniel, Creating Synthetic Experts with Generative Artificial Intelligence (July 15, 2023). Available at SSRN: https://ssrn.com/abstract=4542949
```
