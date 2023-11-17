# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 17:12:18 2023
@author: Archana
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

def main():
    st.set_page_config(page_title="Investment Recommendations")
    
    current_dir = Path(__file__).parent
    image_path = current_dir / "logo.png"
    image = Image.open(image_path)
    st.image(image)

    st.markdown("<h2 style='text-align: center;'>Discover your ideal investments using stock sentiment analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Access Stock Recommendations Based on Sentiment and Market Momentum.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Get Started by Searching for a Stock</p>", unsafe_allow_html=True)
   
    st.write("\n\n")

    stock_input = st.text_input("Enter Stock Ticker or Company Name and press Enter", "")

    if stock_input:
        data = load_data()
        if data is not None:
            if not data.empty:
                data = data.sort_values(by=['Year', 'Quarter'], ascending=False)
                data['Doc Sentiment Score'] = pd.to_numeric(data['Doc Sentiment Score'], errors='coerce')
                found_stock = data[(data['Symbol'] == stock_input) & (data['Year'] == data['Year'].max()) & (data['Quarter'] == data['Quarter'].max())]
                if found_stock.empty:
                    previous_quarter = data[(data['Symbol'] == stock_input) & (data['Year'] == data['Year'].max()) & (data['Quarter'] == (data['Quarter'].max() - 1))]
                    if not previous_quarter.empty:
                        found_stock = previous_quarter

                if not found_stock.empty:
                    company_name = found_stock['Company Name'].values[0]
                    company_ticker = found_stock['Symbol'].values[0]
                    company_sector = found_stock['Sector'].values[0]
                    st.markdown(f"<h5>Company Name: {company_name}</h5>", unsafe_allow_html=True)
                    st.markdown(f"<h5>Stock Ticker: {company_ticker}</h5>", unsafe_allow_html=True)
                    
                    create_sentiment_bar_plot(found_stock)
                    
                    
                
                    sentiment_score = round(found_stock['Doc Sentiment Score'].values[0], 3)
                
                    data= data[data['Sector']==company_sector]
                    top_quantile = data['Doc Sentiment Score'].quantile(0.8)
                    bottom_quantile = data['Doc Sentiment Score'].quantile(0.2)

                    if sentiment_score >= top_quantile:
                        recommendation = "Buy"
                        recommendation_color = "green"
                        recommendation_text_color = "white"
                    elif sentiment_score <= bottom_quantile:
                        recommendation = "Sell"
                        recommendation_color = "red"
                        recommendation_text_color = "white"
                    else:
                        recommendation = "Hold"
                        recommendation_color = "orange"
                        recommendation_text_color = "black"

                    
                    st.markdown(
                        f"<div style='display: flex; justify-content: space-between;'>"
                        f"<div style='border: 1px solid black; background-color: transparent; padding: 10px; border-radius: 5px;'>"
                        f"Sentiment Score: {sentiment_score}"
                        "</div>"
                        f"<div style='background-color: {recommendation_color}; color: {recommendation_text_color}; font-weight: bold; padding: 10px; border-radius: 5px;'>"
                        f"Recommendation: {recommendation}"
                        "</div>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.warning(f"No matching stock found for '{stock_input}' in the latest year and quarter.")
            else:
                st.warning("Please enter a stock ticker to search")


    st.write("\n\n")
    st.write("\n\n")
    st.write("\n\n")

    st.markdown("<h2 style='text-align: center; font-size: 18px;'>Check out our featured recommendations</h2>", unsafe_allow_html=True)

    st.write("\n\n")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("Sector-Wise Stock Recommendations\nQ2 2023"):
            data = load_data()
            unique_sectors = data['Sector'].unique()
            #data = data[data['Sector'].str.isalpha()]

            # Display recommendations for each sector
            for sector_name in unique_sectors:
                selected_data = data[data['Sector'] == sector_name]
                sector = str(sector_name).upper()
                data.loc[data['Symbol'] == stock_input, 'Doc Sentiment Score'] = pd.to_numeric(data[data['Symbol'] == stock_input]['Doc Sentiment Score'], errors='coerce')
                #selected_data['Doc Sentiment Score'] = pd.to_numeric(selected_data['Doc Sentiment Score'], errors='coerce')

                if not selected_data.empty:
                    # Determine the recommendation based on quantiles of sentiment scores
                    top_quantile = selected_data['Doc Sentiment Score'].quantile(0.8)
                    bottom_quantile = selected_data['Doc Sentiment Score'].quantile(0.2)

                    # Filter the data for buy and sell recommendations
                    buy_recommendations = selected_data[selected_data['Doc Sentiment Score'] >= top_quantile][['Company Name', 'Symbol']]
                    sell_recommendations = selected_data[selected_data['Doc Sentiment Score'] <= bottom_quantile][['Company Name', 'Symbol']]
                    num_stocks = min(len(buy_recommendations), len(sell_recommendations))

                    # Select the top and bottom quantiles based on the number of stocks
                    buy_recommendations = buy_recommendations.head(num_stocks)
                    sell_recommendations = sell_recommendations.head(num_stocks)

                    st.markdown(f"<h2 style='text-align: center; font-size: 22px;'>Sector: {sector} Stock Recommendations</h2>", unsafe_allow_html=True)

                    # Display buy recommendations in a table with green index
                    st.markdown("<h3 style='font-size: 18px;'><b>Buy Recommendations</b></h3>", unsafe_allow_html=True)

                    buy_table = buy_recommendations.reset_index(drop=True)  
                    buy_table.index += 1  

                    buy_table_html = "<table>"
                    for row in buy_table.itertuples():
                        buy_table_html += f"<tr><td style='background-color: green;'>{row.Index}</td>"
                        buy_table_html += "".join(f"<td>{value}</td>" for value in row[1:])
                        buy_table_html += "</tr>"
                    buy_table_html += "</table>"
                    st.markdown(buy_table_html, unsafe_allow_html=True)

                    # Display sell recommendations in a table with red index
                    st.markdown("<h3 style='font-size: 18px;'><b>Sell Recommendations</b></h3>", unsafe_allow_html=True)

                    sell_table = sell_recommendations.reset_index(drop=True)  # Reset the index and drop the current index
                    sell_table.index += 1  # Start the index from 1

                    sell_table_html = "<table>"
                    for row in sell_table.itertuples():
                        sell_table_html += f"<tr><td style='background-color: red;'>{row.Index}</td>"
                        sell_table_html += "".join(f"<td>{value}</td>" for value in row[1:])
                        sell_table_html += "</tr>"
                    sell_table_html += "</table>"
                    st.markdown(sell_table_html, unsafe_allow_html=True)

                else:
                    st.warning(f"No data available for '{sector_name}' in Q3 2023.")

def load_data():
    csv_file_path = r'D:\Users\Archana\Downloads\The Data Incubator Files\Capstone Project\result_2023.csv'

    try:
        data = pd.read_csv(csv_file_path)
        return data
    except Exception as e:
        st.error(f"An error occurred while loading data: {str(e)}")
        return None

def create_sentiment_bar_plot(data):
    plt.figure(figsize=(8, 6))

    sentiment_counts = data[['Negative Sentiment Count', 'Positive Sentiment Count', 'Neutral Sentiment Count']]
    labels = ['Negative', 'Positive', 'Neutral']
    values = sentiment_counts.iloc[0].values  
    bar_width = 0.5

    plt.bar(labels, values, color=['red', 'green', 'blue'], width=bar_width)
    plt.title("Sentiment Plot")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")

    st.pyplot(plt)

if __name__ == '__main__':
    main()
