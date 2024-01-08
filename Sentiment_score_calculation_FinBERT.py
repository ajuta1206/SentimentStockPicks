# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:45:37 2023

@author: Archana
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 19:45:29 2023

@author: Archana
"""

import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch

# Iterate over the range of file indices
for i in range(0, 5501, 500):
    
    # Load the earnings transcripts dataframe
    df = pd.read_csv(rf'D:\Users\Archana\Downloads\Earnings_Transcript\earnings_transcripts_{i}.csv', index_col=0)

    # Set the device for GPU processing
    device = torch.device('cuda:0')
    
    # Load the FinBERT model and tokenizer
    finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    
    nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer, device=device)

    # Function to get sentiment scores from earnings transcript
    def get_sentiment(transcript):
        chunks = transcript.split('. ')
        
        # Perform sentiment analysis on transcript chunks
        results = nlp(chunks)
        
        # Initialize variables for average scores and sentiment counts
        average_scores = {'Negative': 0, 'Positive': 0, 'Neutral': 0}
        sentiment_count = {'Negative': 0, 'Positive': 0, 'Neutral': 0}
        
        # Aggregate scores and counts
        for item in results:
            average_scores[item['label']] += item['score']
            sentiment_count[item['label']] += 1
        
        total_score = sum(average_scores.values())
        total_count = sum(sentiment_count.values())
        
        # Normalize scores and counts
        for key in average_scores:
            average_scores[key] /= total_score
        
        doc_sentiment_score = (sentiment_count['Positive'] - sentiment_count['Negative']) / total_count
        
        for key2 in sentiment_count:
            sentiment_count[key2] /= total_count
    
        return results, average_scores['Negative'], average_scores['Positive'], average_scores['Neutral'], sentiment_count['Negative'], sentiment_count['Positive'], sentiment_count['Neutral'], doc_sentiment_score
    
    df['Sentiment_Results'] = df['Earnings Transcript'].apply(get_sentiment)
    
    # Extract sentiment scores and counts into separate columns
    df[['Total_Sentiment_Results', 'Negative Sentiment Score', 'Positive Sentiment Score', 'Neutral Sentiment Score', 'Negative Sentiment Count', 'Positive Sentiment Count', 'Neutral Sentiment Count', 'Doc Sentiment Score']] = df['Sentiment_Results'].apply(lambda x: pd.Series([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]]))
    
    df = df.drop(['Sentiment_Results', 'Earnings Transcript'], axis=1)  
    
    # Save the dataframe with sentiment scores to a new CSV file
    df.to_csv(rf'D:\Users\Archana\Downloads\Earnings_Transcript\finBERT_sentiments_{i}.csv')
