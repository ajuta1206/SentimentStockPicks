# SentimentStockPicks
This repository contains code for analyzing sentiment in earnings transcripts and generating stock recommendations based on FinBERT, a pre-trained sentiment analysis model tailored for financial text.

# Title
Sentiment-Driven Stock Recommendation Tool

# Summary
This project is a stock recommendation tool utilizing BERT, a neural network model for natural language processing, to analyze and decode earnings call transcripts. The tool calculates sentiment scores from earnings transcript, providing retail investors with curated stock recommendations based on these scores, each quarter. 

# Problem Statement
Grasping insights from earnings calls can be challenging for retail investors. This stock recommendation tool addresses this by decoding and analyzing financial text using advanced NLP techniques, extracting sentiment from earnings call transcripts. This sentiment analysis forms the basis for sector-wise stock recommendations, providing investors with actionable insights after each earnings season.

# Project Details
# Data source: 
The data for this project is sourced from two key channels. 

Firstly, information on filing dates is extracted from 10-Q filings, this date is used as the start date for calculating stock returns and the end date for momentum. The return is calculated from market close to market close. The timing of the earnings call is also taken into consideration. the start date is one day before the earnings call if the earnings call was before the market close on the day it was made. 

Secondly, earnings call transcripts are obtained from https://site.financialmodelingprep.com/, from which the text and the year and quarter of the earnings call are extracted. 

# Stock Ranking System: 
The stock ranking system is designed by leveraging sentiment scores derived from FinBERT's sentence-level analysis of earnings call transcripts. Each sentence is processed as individual chunks, determining whether it carries positive, negative, or neutral sentiment. The overall sentiment score for the document is calculated using the formula: 
sentiment score= Total positive sentences-Total negative sentencesTotal sentences

Building upon these sentiment scores, I implemented a long-short sector-neutral quintile portfolio strategy. Within each sector, the top 20% of stocks with the highest sentiment scores are bought, while the bottom 20% are sold. This strategic approach ensures a balanced portfolio, maintaining a constant dollar amount for each transaction. The long-short strategy aims to be market-neutral, meaning it is designed to minimize exposure to overall market movements. This is achieved by holding both long (buy) and short (sell) positions simultaneously. Regardless of whether the market rises or falls, the impact on the portfolio is lessened.

# Web Application: 
For the final product, I developed a user-friendly website that empowers users to make informed investment decisions. After each earnings season, the website is systematically updated with a carefully curated list of stocks recommended for buying or selling in each sector. Suppose you're interested in discovering potential investments for the second quarter of 2023. In that case, you can effortlessly navigate to the sector-wise recommendations section. There, you'll find a comprehensive list of stocks, categorized as either buy or sell.

Moreover, if you have a specific company in mind, you can conveniently input its name or ticker symbol into the search bar. The tool then generates sentiment scores derived from the earnings call, presenting you with a well-rounded recommendation for the upcoming quarter. Whether it suggests it's an opportune time to buy, hold, or sell that particular stock, the website provides transparency and accessibility, catering specifically to retail investors.

# Performance Evaluation: 
Conducting backtesting over the past five years, I implemented a long-short sector-neutral quintile portfolio strategy. The results were remarkable, achieving annualized returns of 8.8% for mid and large-cap stocks and an impressive 14.3% for value sectors. This performance not only outpaces traditional long-only index models like the S&P 500 but also underscores the tool's efficacy in generating profitable investment strategies.

The notable aspect of the long-short strategy lies in its capacity to work on leverage. With the potential for returns to be 2.5-3 times higher, this strategy significantly enhances the overall profitability, further highlighting the tool's capability to provide investors with a competitive edge in the market.

The performance of the quintile portfolio for the past 5 years is shown below. 

![image](https://github.com/ajuta1206/SentimentStockPicks/assets/65238438/b70ff6d7-93f0-4275-b435-6236c132e9f6)

![image](https://github.com/ajuta1206/SentimentStockPicks/assets/65238438/8e6c984e-8239-44c3-9558-74c6388d6829)





