
# Starbucks Capstone Challenge 
[streamlit]https://starbucks-capstone-challenge-mahmoud-mousa.streamlit.app/ 
![Image Description](https://miro.medium.com/v2/resize:fit:1224/format:webp/1*lv_IE5h2CnUEVUOmK_dgpg.jpeg)

## Introduction
This project is my Capstone Challenge for Udacity’s Data Scientist Nanodegree. The project is in collaboration with Starbucks where we were given simulated data that mimics customer behavior on the Starbucks rewards app. The offer could be purely informational or it could include a discount such as BOGO (buy one get one free).

From the data we received, it appears that Starbucks sent 10 different offers to its customers via a variety of different channels.

## Datasets
For this project, we received 3 datasets:

1. Portfolio: dataset describing the characteristics of each offer type, including its offer type, difficulty, and duration.
2. Profile: dataset containing information regarding customer demographics including age, gender, income, and the date they created an account for Starbucks Rewards.
3. Transcript: dataset containing all the instances when a customer made a purchase, viewed an offer, received an offer, and completed an offer. It's important to note that if a customer completed an offer but never actually viewed the offer, then this does not count as a successful offer as the offer could not have changed the outcome.

## Project Overview
The purpose of this project is to complete an exploratory analysis to determine which demographic group responds best to which offer type. I will also create and compare different predictive models to evaluate which features contribute to a successful offer.

## Performance Metrics
The performance of each trained predictive model was measured using a test dataset. As this was a binary classification outcome, I used AUC, accuracy, f1 score, and confusion matrix as the performance metrics.
