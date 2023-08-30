'''
Helper Functions for using a Synthetic Expert on Twitter Data

Version 0.5  
Date: August 30, 2023    
Author: Daniel M. Ringel    
Contact: dmr@unc.edu

Daniel M. Ringel, Creating Synthetic Experts with Generative Artificial Intelligence (July 15, 2023).  
Available at SSRN: https://papers.ssrn.com/abstract_id=4542949.
'''

import pandas as pd
import numpy as np
import warnings
import torch
import re
from datetime import datetime
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

def parse_tweet(tweet):
    '''parses HTLM in Tweets'''
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # Turn all warnings into errors
        try:
            soup = BeautifulSoup(tweet, "html.parser")
            return soup.get_text() 
        except Warning:
            if "The input looks more like a filename than markup." in str(w[-1].message):
                return None
    return tweet

def clean_tweet(tweet):
    '''cleans-up Tweets by (1) removing URLS (replace with URL),
       (2) removing line breaks and leading periods and colons,
       (3) removing leading, trailing and duplicate spaces.
    '''
    tweet = re.sub(r"https?://\S+|www\.\S+", " URL ", tweet) 
    tweet = parse_tweet(tweet)
    if tweet is not None:
        tweet = re.sub(r"(?<=\S)\\n(?=\S)|(?<=\S)\n(?=\S)|\\n|\n|\\n+|\n+|\\n{2,}|\n{2,}", " ", tweet)
        tweet = re.sub(r'^[.:]+', '', tweet)
        tweet = re.sub(r" +", " ", tweet.strip())
    return tweet

def remove_joiners_commas_spaces(text):
    '''remove repeated commas and excessive spaces, and joiners'''
    # Remove multiple commas
    text = re.sub(r',+', ', ', text)
    text = re.sub(r'(\s*,\s*)+', ', text)
    # Remove specific Unicode characters including '⠀' (U+2800)
    text = re.sub(r'[\u200D\u200B\u2060\u00A0\u202F\uFEFF\u3000\u2800]', ' ', text)
    # Remove double spaces
    text = re.sub(r' +', ' ', text)
    return text.strip()

def predict_batch(texts, model, tokenizer, device, batch_sze=8):
    '''break texts into batches and predict'''
    all_probs = []
    sigmoid = torch.nn.Sigmoid()
    for i in range(0, len(texts), batch_sze):
        batch_texts = texts[i: i + batch_sze]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        outputs = model(**inputs)
        probabilities = sigmoid(outputs.logits).detach().cpu().numpy()
        all_probs.extend(probabilities)
    return np.array(all_probs)

def block_process(df, batch_size, model, tokenizer, device, t, id2label):
    '''break large dataframe with many tweets into batches to clean and predict'''
    print(f'{datetime.now().strftime("%H:%M:%S")} Starting block labeling:\n')
    batch_start = 0
    batch_end = batch_size
    while batch_start < len(df):
        df_batch = df.iloc[batch_start:batch_end].copy()
        
        # Preprocess Texts 
        df_batch.loc[:, "text"] = df_batch["text"].apply(clean_tweet)
        df_batch = df_batch.dropna(subset=["text"])
        
        # Predict
        probabilities = predict_batch(df_batch['text'].tolist(), model, tokenizer, device)
        binary_predictions = (probabilities >= t).astype(int)
        
        # Add probabilities and binary predictions to DataFrame
        for i, label in id2label.items():
            df_batch[f'Prob_{label}'] = probabilities[:, i]
        for i, label in id2label.items():
            df_batch[label] = binary_predictions[:, i]

        # Add a new column with the list of labels predicted as true
        bin_columns = [label for label in id2label.values()]
        df_batch['Labels'] = df_batch.apply(lambda row: [label for col, label in zip(bin_columns, id2label.values()) if row[label] == 1], axis=1)

        if batch_start == 0:
            df_out = df_batch
        else:
            df_out = pd.concat([df_out, df_batch])

        batch_start += batch_size
        batch_end += batch_size
        print(f'{datetime.now().strftime("%H:%M:%S")} --> Finished labeling {batch_start} Texts')

    return df_out

def apply_vader_sentiment(df, t):
    '''Get compound sentiment (i.e., polarity) for each Text'''
    # Initialize Sentiment Analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Get compound polarity score for all texts
    df['Sent_All'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

    # Get conditional polarity scores
    for label in ['Product', 'Place', 'Price', 'Promotion']:
        df[f'Sent_{label}'] = df.apply(
            lambda row: analyzer.polarity_scores(row['text'])['compound'] if row[label] == 1 else 'n.a.', 
            axis=1
        )

    # Get column name with maximum probability
    prob_cols = ['Prob_Product', 'Prob_Place', 'Prob_Price', 'Prob_Promotion']
    df['Sent_Max'] = df[prob_cols].idxmax(axis=1).str.replace('Prob_', '')
    df['Sent_Max'] = df.apply(lambda row: row['Sent_Max'] if row[row['Sent_Max']] >= t else 'n.a.', axis=1)

    return df

def get_summary(df, IN_file, MODEL_name, t):
    '''Summarize compund sentiment by marketing mix variable (each of the 4 Ps of Marketing). 
    Calculated sample standard deviation (unweighted and weighted) and range to capture how much sentiment of Ps deviates from overall sentiment.'''
    # Core metrics
    metrics = {
        'Brand': IN_file, 
        'Start_date': df['created_at'].min(), 
        'End_date': df['created_at'].max(),
        'Sampled_tweets': len(df),
        'Model': MODEL_name,
        'Threshold' : t,
        'Sentiment': round(df['Sent_All'].mean(), 3),
        'Sentiment_Std': round(df['Sent_All'].std(), 3)
    }
    
    # Counts for Product, Place, Price, Promotion
    for label in ['Product', 'Place', 'Price', 'Promotion']:
        metrics[f'Count_{label}'] = df[label].sum()
        metrics[f'S_{label}'] = round(df[df[label] == 1][f'Sent_{label}'].mean(), 3)
        metrics[f'S_{label}_Std'] = round(df[df[label] == 1][f'Sent_{label}'].std(), 3)
    
    # Metrics based on Sent_Max values
    max_means = []
    max_counts = []
    for label in ['Product', 'Place', 'Price', 'Promotion']:
        metrics[f'Max_{label}_count'] = (df['Sent_Max'] == label).sum()
        mean_value = round(df[df['Sent_Max'] == label][f'Sent_{label}'].mean(), 3)
        metrics[f'Max_{label}_mean'] = mean_value
        
        max_means.append(mean_value)
        max_counts.append(metrics[f'Max_{label}_count'])
        
        # Adding standard deviation for Max_Ps
        metrics[f'Max_{label}_Std'] = round(df[df['Sent_Max'] == label][f'Sent_{label}'].std(), 3)

    # Standard Deviation among 4P Sentiment
    sentiment_values = [metrics['S_Product'], metrics['S_Place'], metrics['S_Promotion'], metrics['S_Price']]
    count_values = [metrics['Count_Product'], metrics['Count_Place'], metrics['Count_Promotion'], metrics['Count_Price']]
    values = np.array(sentiment_values)
    weights = np.array(count_values)
    metrics['Std_dev'] = round(np.std(values, ddof=1), 3)   # Sample standard deviation
    weighted_mean = sum(values * weights) / sum(weights)
    metrics['Weighted_std_dev'] = round(np.sqrt(sum(weights * (values - weighted_mean)**2) / sum(weights)), 3)
    metrics['Range'] = round(values.max() - values.min(), 3)

    # Standard Deviation among Max_ Sentiment
    max_values = np.array(max_means)
    max_weights = np.array(max_counts)
    metrics['Max_std_dev'] = round(np.std(max_values, ddof=1), 3)   # Sample standard deviation
    weighted_mean_max = sum(max_values * max_weights) / sum(max_weights)
    metrics['Max_weighted_std_dev'] = round(np.sqrt(sum(max_weights * (max_values - weighted_mean_max)**2) / sum(max_weights)), 3)
    metrics['Range_max'] = round(max_values.max() - max_values.min(), 3)

    return pd.DataFrame([metrics])

def print_summary_line_by_line(summary_df):
    '''Conventiently print summary sentiment for a brand'''
    for column, value in summary_df.iloc[0].items():
        print(f"{column}: {value}")