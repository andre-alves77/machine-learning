# %% [markdown]
# <img src="https://github.com/FarzadNekouee/Retail_Customer_Segmentation_Recommendation_System/blob/master/image.png?raw=true" width="2400">

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:130%; text-align:left">
# 
# <h2 align="left"><font color=#ff6200>Problem:</font></h2>
# 
# 
# In this project, we delve deep into the thriving sector of __online retail__ by analyzing a __transactional dataset__ from a UK-based retailer, available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail). This dataset documents all transactions between 2010 and 2011. Our primary objective is to amplify the efficiency of marketing strategies and boost sales through __customer segmentation__. We aim to transform the transactional data into a customer-centric dataset by creating new features that will facilitate the segmentation of customers into distinct groups using the __K-means clustering__ algorithm. This segmentation will allow us to understand the distinct __profiles__ and preferences of different customer groups. Building upon this, we intend to develop a __recommendation system__ that will suggest top-selling products to customers within each segment who haven't purchased those items yet, ultimately enhancing marketing efficacy and fostering increased sales.
# 

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:130%; text-align:left">
# 
# <h2 align="left"><font color=#ff6200>Objectives:</font></h2>
# 
# 
# - **Data Cleaning & Transformation**: Clean the dataset by handling missing values, duplicates, and outliers, preparing it for effective clustering.
# 
#     
# - **Feature Engineering**: Develop new features based on the transactional data to create a customer-centric dataset, setting the foundation for customer segmentation.
# 
#     
# - **Data Preprocessing**: Undertake feature scaling and dimensionality reduction to streamline the data, enhancing the efficiency of the clustering process.
# 
#     
# - **Customer Segmentation using K-Means Clustering**: Segment customers into distinct groups using K-means, facilitating targeted marketing and personalized strategies.
# 
#     
# - **Cluster Analysis & Evaluation**: Analyze and profile each cluster to develop targeted marketing strategies and assess the quality of the clusters formed.
# 
#     
# - **Recommendation System**: Implement a system to recommend best-selling products to customers within the same cluster who haven't purchased those products, aiming to boost sales and marketing effectiveness.
# 

# %% [markdown]
# <a id="contents_tabel"></a>    
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:130%; text-align:left">
# 
# <h2 align="left"><font color=#ff6200>Table of Contents:</font></h2>
#     
# * [Step 1 | Setup and Initialization](#setup)
#     - [Step 1.1 | Importing Necessary Libraries](#libraries) 
#     - [Step 1.2 | Loading the Dataset](#load_dataset)
# * [Step 2 | Initial Data Analysis](#initial_analysis) 
#     - [Step 2.1 | Dataset Overview](#overview) 
#     - [Step 2.2 | Summary Statistics](#statistics) 
# * [Step 3 | Data Cleaning & Transformation](#data_cleaning)
#     - [Step 3.1 | Handling Missing Values](#missing_values)
#     - [Step 3.2 | Handling Duplicates](#duplicates)
#     - [Step 3.3 | Treating Cancelled Transactions](#InvoiceNo_cleaning)
#     - [Step 3.4 | Correcting StockCode Anomalies](#StockCode_cleaning)
#     - [Step 3.5 | Cleaning Description Column](#Description_cleaning)
#     - [Step 3.6 | Treating Zero Unit Prices](#UnitPrice_cleaning)
#     - [Step 3.7 | Outlier Treatment](#outlier_cleaning)
# * [Step 4 | Feature Engineering](#feature_engineering)
#     - [Step 4.1 | RFM Features](#rfm_features)
#         - [Step 4.1.1 | Recency (R)](#recency) 
#         - [Step 4.1.2 | Frequency (F)](#frequency)
#         - [Step 4.1.3 | Monetary (M)](#monetary)
#     - [Step 4.2 | Product Diversity](#product_diversity)
#     - [Step 4.3 | Behavioral Features](#behaviroal_features)
#     - [Step 4.4 | Geographic Features](#geographical_features)
#     - [Step 4.5 | Cancellation Insights](#cancellation_insights) 
#     - [Step 4.6 | Seasonality & Trends](#seasonality_trends) 
# * [Step 5 | Outlier Detection and Treatment](#outlier_detection)
# * [Step 6 | Correlation Analysis](#correlation)
# * [Step 7 | Feature Scaling](#scaling)
# * [Step 8 | Dimensionality Reduction](#pca)
# * [Step 9 | K-Means Clustering](#kmeans) 
#     - [Step 9.1 | Determining the Optimal Number of Clusters](#optimal_k) 
#         - [Step 9.1.1 | Elbow Method](#elbow)
#         - [Step 9.1.2 | Silhouette Method](#silhouette)
#     - [Step 9.2 | Clustering Model - K-means](#kmeans_model)
# * [Step 10 | Clustering Evaluation](#evaluation)  
#     - [Step 10.1 | 3D Visualization of Top Principal Components](#3d_visualization)
#     - [Step 10.2 | Cluster Distribution Visualization](#cluster_distributuion) 
#     - [Step 10.3 | Evaluation Metrics](#evaluations_metrics)
# * [Step 11 | Cluster Analysis and Profiling](#profiling)
#     - [Step 11.1 | Radar Chart Approach](#radar_chart)
#     - [Step 11.2 | Histogram Chart Approach](#histogram)
# * [Step 12 | Recommendation System](#recommendation_system)

# %% [markdown]
# <h2 align="left"><font color=#ff6200>Let's get started:</font></h2>

# %% [markdown]
# <a id="setup"></a>
# # <p style="background-color: #ff6200; font-family:calibri; color:white; font-size:140%; font-family:Verdana; text-align:center; border-radius:15px 50px;">Step 1 | Setup and Initialization</p>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <a id="libraries"></a>
# # <b><span style='color:#fcc36d'>Step 1.1 |</span><span style='color:#ff6200'> Importing Necessary Libraries</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# First of all, I will import all the necessary libraries that we will use throughout the project. This generally includes libraries for data manipulation, data visualization, and others based on the specific needs of the project:

# %%
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors as mcolors
from scipy.stats import linregress
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from tabulate import tabulate
from collections import Counter

%matplotlib inline

# %%
# Initialize Plotly for use in the notebook
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

# %%
# Configure Seaborn plot styles: Set background color and use dark grid
sns.set(rc={'axes.facecolor': '#fcf0dc'}, style='darkgrid')

# %% [markdown]
# <a id="load_dataset"></a>
# # <b><span style='color:#fcc36d'>Step 1.2 |</span><span style='color:#ff6200'> Loading the Dataset</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# Next, I will load the dataset into a pandas DataFrame which will facilitate easy manipulation and analysis:

# %%
df = pd.read_csv('/kaggle/input/ecommerce-data/data.csv', encoding="ISO-8859-1")

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:130%; text-align:left">
# 
# <h2 align="left"><font color=#ff6200>Dataset Description:</font></h2>
# 
# | __Variable__   | __Description__ |
# |     :---       |       :---      |      
# | __InvoiceNo__  | Code representing each unique transaction.  If this code starts with letter 'c', it indicates a cancellation. |
# | __StockCode__  | Code uniquely assigned to each distinct product. |
# | __Description__| Description of each product. |
# | __Quantity__   | The number of units of a product in a transaction. |
# | __InvoiceDate__| The date and time of the transaction. |
# | __UnitPrice__  | The unit price of the product in sterling. |
# | __CustomerID__ | Identifier uniquely assigned to each customer. |
# | __Country__    | The country of the customer. |
# 

# %% [markdown]
# <a id="initial_analysis"></a>
# # <p style="background-color: #ff6200; font-family:calibri; color:white; font-size:140%; font-family:Verdana; text-align:center; border-radius:15px 50px;">Step 2 | Initial Data Analysis</p>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# Afterward, I am going to gain a thorough understanding of the dataset before proceeding to the data cleaning and transformation stages.

# %% [markdown]
# <a id="overview"></a>
# # <b><span style='color:#fcc36d'>Step 2.1 |</span><span style='color:#ff6200'> Dataset Overview</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# First I will perform a preliminary analysis to understand the structure and types of data columns:

# %%
df.head(10)

# %%
df.info()

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Inferences:</font></h3>
#     
# The dataset consists of 541,909 entries and 8 columns. Here is a brief overview of each column:
# 
# - __`InvoiceNo`__: This is an object data type column that contains the invoice number for each transaction. Each invoice number can represent multiple items purchased in a single transaction.
#    
#     
# - __`StockCode`__: An object data type column representing the product code for each item. 
# 
#     
# - __`Description`__: This column, also an object data type, contains descriptions of the products. It has some missing values, with 540,455 non-null entries out of 541,909.
# 
#     
# - __`Quantity`__: This is an integer column indicating the quantity of products purchased in each transaction.
#    
# 
# - __`InvoiceDate`__: A datetime column that records the date and time of each transaction.
# 
#     
# - __`UnitPrice`__: A float column representing the unit price of each product.
# 
#     
# - __`CustomerID`__: A float column that contains the customer ID for each transaction. This column has a significant number of missing values, with only 406,829 non-null entries out of 541,909.
# 
#     
# - __`Country`__: An object column recording the country where each transaction took place.
# 
# From a preliminary overview, it seems that there are missing values in the `Description` and `CustomerID` columns which need to be addressed. The `InvoiceDate` column is already in datetime format, which will facilitate further time series analysis. We also observe that a single customer can have multiple transactions as inferred from the repeated `CustomerID` in the initial rows.
# 
# The next steps would include deeper data cleaning and preprocessing to handle missing values, potentially erroneous data, and to create new features that can help in achieving the project goals.

# %% [markdown]
# <a id="statistics"></a>
# # <b><span style='color:#fcc36d'>Step 2.2 |</span><span style='color:#ff6200'> Summary Statistics</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# Now, I am going to generate summary statistics to gain initial insights into the data distribution:

# %%
# Summary statistics for numerical variables
df.describe().T

# %%
# Summary statistics for categorical variables
df.describe(include='object').T

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Inferences:</font></h3>
# 
# 
# - __`Quantity`__:
#    - The average quantity of products in a transaction is approximately 9.55.
#    - The quantity has a wide range, with a minimum value of -80995 and a maximum value of 80995. The negative values indicate returned or cancelled orders, which need to be handled appropriately.
#    - The standard deviation is quite large, indicating a significant spread in the data. The presence of outliers is indicated by a large difference between the maximum and the 75th percentile values.
# 
#     
# - __`UnitPrice`__:
#    - The average unit price of the products is approximately 4.61.
#    - The unit price also shows a wide range, from -11062.06 to 38970, which suggests the presence of errors or noise in the data, as negative prices don't make sense.
#    - Similar to the Quantity column, the presence of outliers is indicated by a large difference between the maximum and the 75th percentile values.
#  
#     
# - __`CustomerID`__:
#    - There are 406829 non-null entries, indicating missing values in the dataset which need to be addressed.
#    - The Customer IDs range from 12346 to 18287, helping in identifying unique customers.
# 
#     
# - __`InvoiceNo`__:
#    - There are 25900 unique invoice numbers, indicating 25900 separate transactions.
#    - The most frequent invoice number is 573585, appearing 1114 times, possibly representing a large transaction or an order with multiple items.
# 
#     
# - __`StockCode`__:
#    - There are 4070 unique stock codes representing different products.
#    - The most frequent stock code is 85123A, appearing 2313 times in the dataset.
# 
#     
# - __`Description`__:
#    - There are 4223 unique product descriptions.
#    - The most frequent product description is "WHITE HANGING HEART T-LIGHT HOLDER", appearing 2369 times.
#    - There are some missing values in this column which need to be treated.
# 
#     
# - __`Country`__:
#    - The transactions come from 38 different countries, with a dominant majority of the transactions (approximately 91.4%) originating from the United Kingdom.

# %% [markdown]
# <a id="data_cleaning"></a>
# # <p style="background-color: #ff6200; font-family:calibri; color:white; font-size:140%; font-family:Verdana; text-align:center; border-radius:15px 50px;">Step 3 |  Data Cleaning & Transformation</p>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# This step encompasses a comprehensive cleaning and transformation process to refine the dataset. It includes addressing missing values, eliminating duplicate entries, correcting anomalies in product codes and descriptions, and other necessary adjustments to prepare the data for in-depth analysis and modeling.

# %% [markdown]
# <a id="missing_values"></a>
# # <b><span style='color:#fcc36d'>Step 3.1 |</span><span style='color:#ff6200'> Handling Missing Values</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# Initially, I will determine the percentage of missing values present in each column, followed by selecting the most effective strategy to address them:

# %%
# Calculating the percentage of missing values for each column
missing_data = df.isnull().sum()
missing_percentage = (missing_data[missing_data > 0] / df.shape[0]) * 100

# Prepare values
missing_percentage.sort_values(ascending=True, inplace=True)

# Plot the barh chart
fig, ax = plt.subplots(figsize=(15, 4))
ax.barh(missing_percentage.index, missing_percentage, color='#ff6200')

# Annotate the values and indexes
for i, (value, name) in enumerate(zip(missing_percentage, missing_percentage.index)):
    ax.text(value+0.5, i, f"{value:.2f}%", ha='left', va='center', fontweight='bold', fontsize=18, color='black')

# Set x-axis limit
ax.set_xlim([0, 40])

# Add title and xlabel
plt.title("Percentage of Missing Values", fontweight='bold', fontsize=22)
plt.xlabel('Percentages (%)', fontsize=16)
plt.show()

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# <h3 align="left"><font color=#ff6200>Handling Missing Values Strategy:</font></h3>
# 
# - __`CustomerID` (24.93% missing values)__
#    - The `CustomerID` column contains nearly a quarter of missing data. This column is essential for clustering customers and creating a recommendation system. Imputing such a large percentage of missing values might introduce significant bias or noise into the analysis.
#     
#    - Moreover, since the clustering is based on customer behavior and preferences, it's crucial to have accurate data on customer identifiers. Therefore, removing the rows with missing `CustomerID`s seems to be the most reasonable approach to maintain the integrity of the clusters and the analysis.
# 
#     
# - __`Description` (0.27% missing values)__
#    - The `Description` column has a minor percentage of missing values. However, it has been noticed that there are inconsistencies in the data where the same `StockCode` does not always have the same `Description`. This indicates data quality issues and potential errors in the product descriptions.
#     
#    - Given these inconsistencies, imputing the missing descriptions based on `StockCode` might not be reliable. Moreover, since the missing percentage is quite low, it would be prudent to remove the rows with missing `Description`s to avoid propagating errors and inconsistencies into the subsequent analyses.
#    
# By removing rows with missing values in the `CustomerID` and `Description` columns, we aim to construct a cleaner and more reliable dataset, which is essential for achieving accurate clustering and creating an effective recommendation system.
# 

# %%
# Extracting rows with missing values in 'CustomerID' or 'Description' columns
df[df['CustomerID'].isnull() | df['Description'].isnull()].head()

# %%
# Removing rows with missing values in 'CustomerID' and 'Description' columns
df = df.dropna(subset=['CustomerID', 'Description'])

# %%
# Verifying the removal of missing values
df.isnull().sum().sum()

# %% [markdown]
# <a id="duplicates"></a>
# # <b><span style='color:#fcc36d'>Step 3.2 |</span><span style='color:#ff6200'> Handling Duplicates</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# Next, I am going to recognize duplicate rows in the dataset:

# %%
# Finding duplicate rows (keeping all instances)
duplicate_rows = df[df.duplicated(keep=False)]

# Sorting the data by certain columns to see the duplicate rows next to each other
duplicate_rows_sorted = duplicate_rows.sort_values(by=['InvoiceNo', 'StockCode', 'Description', 'CustomerID', 'Quantity'])

# Displaying the first 10 records
duplicate_rows_sorted.head(10)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# <h3 align="left"><font color=#ff6200>Handling Duplicates Strategy:</font></h3>
#     
# In the context of this project, the presence of completely identical rows, including identical transaction times, suggests that these might be data recording errors rather than genuine repeated transactions. Keeping these duplicate rows can introduce noise and potential inaccuracies in the clustering and recommendation system. 
# 
# Therefore, I am going to remove these completely identical duplicate rows from the dataset. Removing these rows will help in achieving a cleaner dataset, which in turn would aid in building more accurate customer clusters based on their unique purchasing behaviors. Moreover, it would help in creating a more precise recommendation system by correctly identifying the products with the most purchases.

# %%
# Displaying the number of duplicate rows
print(f"The dataset contains {df.duplicated().sum()} duplicate rows that need to be removed.")

# Removing duplicate rows
df.drop_duplicates(inplace=True)

# %%
# Getting the number of rows in the dataframe
df.shape[0]

# %% [markdown]
# <a id="InvoiceNo_cleaning"></a>
# # <b><span style='color:#fcc36d'>Step 3.3 |</span><span style='color:#ff6200'> Treating Cancelled Transactions</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# To refine our understanding of customer behavior and preferences, we need to take into account the transactions that were cancelled. Initially, we will identify these transactions by filtering the rows where the `InvoiceNo` starts with "C". Subsequently, we will analyze these rows to understand their common characteristics or patterns:

# %%
# Filter out the rows with InvoiceNo starting with "C" and create a new column indicating the transaction status
df['Transaction_Status'] = np.where(df['InvoiceNo'].astype(str).str.startswith('C'), 'Cancelled', 'Completed')

# Analyze the characteristics of these rows (considering the new column)
cancelled_transactions = df[df['Transaction_Status'] == 'Cancelled']
cancelled_transactions.describe().drop('CustomerID', axis=1)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Inferences from the Cancelled Transactions Data:</font></h3>
# 
# - All quantities in the cancelled transactions are negative, indicating that these are indeed orders that were cancelled.
#     
#     
# - The `UnitPrice` column has a considerable spread, showing that a variety of products, from low to high value, were part of the cancelled transactions.
# 
# 

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Strategy for Handling Cancelled Transactions:</font></h3>
#     
# Considering the project's objective to cluster customers based on their purchasing behavior and preferences and to eventually create a recommendation system, it's imperative to understand the cancellation patterns of customers. Therefore, the strategy is to retain these cancelled transactions in the dataset, marking them distinctly to facilitate further analysis. This approach will:
# 
# - Enhance the clustering process by incorporating patterns and trends observed in cancellation data, which might represent certain customer behaviors or preferences.
#     
#     
# - Allow the recommendation system to possibly prevent suggesting products that have a high likelihood of being cancelled, thereby improving the quality of recommendations.
# 
# 

# %%
# Finding the percentage of cancelled transactions
cancelled_percentage = (cancelled_transactions.shape[0] / df.shape[0]) * 100

# Printing the percentage of cancelled transactions
print(f"The percentage of cancelled transactions in the dataset is: {cancelled_percentage:.2f}%")

# %% [markdown]
# <a id="StockCode_cleaning"></a>
# # <b><span style='color:#fcc36d'>Step 3.4 |</span><span style='color:#ff6200'> Correcting StockCode Anomalies</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# First of all, lets find the number of unique stock codes and to plot the top 10 most frequent stock codes along with their percentage frequency:

# %%
# Finding the number of unique stock codes
unique_stock_codes = df['StockCode'].nunique()

# Printing the number of unique stock codes
print(f"The number of unique stock codes in the dataset is: {unique_stock_codes}")

# %%
# Finding the top 10 most frequent stock codes
top_10_stock_codes = df['StockCode'].value_counts(normalize=True).head(10) * 100

# Plotting the top 10 most frequent stock codes
plt.figure(figsize=(12, 5))
top_10_stock_codes.plot(kind='barh', color='#ff6200')

# Adding the percentage frequency on the bars
for index, value in enumerate(top_10_stock_codes):
    plt.text(value, index+0.25, f'{value:.2f}%', fontsize=10)

plt.title('Top 10 Most Frequent Stock Codes')
plt.xlabel('Percentage Frequency (%)')
plt.ylabel('Stock Codes')
plt.gca().invert_yaxis()
plt.show()

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Inferences on Stock Codes:</font></h3>
# 
# - __Product Variety__: The dataset contains 3684 unique stock codes, indicating a substantial variety of products available in the online retail store. This diversity can potentially lead to the identification of distinct customer clusters, with preferences for different types of products.
# 
#     
# - __Popular Items__: A closer look at the top 10 most frequent stock codes can offer insights into the popular products or categories that are frequently purchased by customers.
# 
#     
# - __Stock Code Anomalies__: We observe that while most stock codes are composed of 5 or 6 characters, there are some anomalies like the code '__POST__'. These anomalies might represent services or non-product transactions (perhaps postage fees) rather than actual products. To maintain the focus of the project, which is clustering based on product purchases and creating a recommendation system, these anomalies should be further investigated and possibly treated appropriately to ensure data integrity.

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# To delve deeper into identifying these anomalies, let's explore the frequency of the number of numeric characters in the stock codes, which can provide insights into the nature of these unusual entries:

# %%
# Finding the number of numeric characters in each unique stock code
unique_stock_codes = df['StockCode'].unique()
numeric_char_counts_in_unique_codes = pd.Series(unique_stock_codes).apply(lambda x: sum(c.isdigit() for c in str(x))).value_counts()

# Printing the value counts for unique stock codes
print("Value counts of numeric character frequencies in unique stock codes:")
print("-"*70)
print(numeric_char_counts_in_unique_codes)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Inference:</font></h3>
# 
# The output indicates the following:
# 
# - A majority of the unique stock codes (3676 out of 3684) contain exactly 5 numeric characters, which seems to be the standard format for representing product codes in this dataset.
# 
#     
# - There are a few anomalies: 7 stock codes contain no numeric characters and 1 stock code contains only 1 numeric character. These are clearly deviating from the standard format and need further investigation to understand their nature and whether they represent valid product transactions.
# 
# Now, let's identify the stock codes that contain 0 or 1 numeric characters to further understand these anomalies:

# %%
# Finding and printing the stock codes with 0 and 1 numeric characters
anomalous_stock_codes = [code for code in unique_stock_codes if sum(c.isdigit() for c in str(code)) in (0, 1)]

# Printing each stock code on a new line
print("Anomalous stock codes:")
print("-"*22)
for code in anomalous_stock_codes:
    print(code)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# Let's calculate the percentage of records with these anomalous stock codes:    

# %%
# Calculating the percentage of records with these stock codes
percentage_anomalous = (df['StockCode'].isin(anomalous_stock_codes).sum() / len(df)) * 100

# Printing the percentage
print(f"The percentage of records with anomalous stock codes in the dataset is: {percentage_anomalous:.2f}%")

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# <h3 align="left"><font color=#ff6200>Inference:</font></h3>
# 
# Based on the analysis, we find that a very small proportion of the records, __0.48%__, have anomalous stock codes, which deviate from the typical format observed in the majority of the data. Also, these anomalous codes are just a fraction among all unique stock codes (__only 8 out of 3684__).
# 
# These codes seem to represent non-product transactions like "__BANK CHARGES__", "__POST__" (possibly postage fees), etc. Since they do not represent actual products and are a very small proportion of the dataset, including them in the analysis might introduce noise and distort the clustering and recommendation system.

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Strategy:</font></h3>
# 
# Given the context of the project, where the aim is to cluster customers based on their product purchasing behaviors and develop a product recommendation system, it would be prudent to exclude these records with anomalous stock codes from the dataset. This way, the focus remains strictly on genuine product transactions, which would lead to a more accurate and meaningful analysis.

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# Thus, the strategy would be to filter out and remove rows with these anomalous stock codes from the dataset before proceeding with further analysis and model development:

# %%
# Removing rows with anomalous stock codes from the dataset
df = df[~df['StockCode'].isin(anomalous_stock_codes)]

# %%
# Getting the number of rows in the dataframe
df.shape[0]

# %% [markdown]
# <a id="Description_cleaning"></a>
# # <b><span style='color:#fcc36d'>Step 3.5 |</span><span style='color:#ff6200'> Cleaning Description Column</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# First, I will calculate the occurrence count of each unique description in the dataset. Then, I will plot the top 30 descriptions. This visualization will give a clear view of the highest occurring descriptions in the dataset:

# %%
# Calculate the occurrence of each unique description and sort them
description_counts = df['Description'].value_counts()

# Get the top 30 descriptions
top_30_descriptions = description_counts[:30]

# Plotting
plt.figure(figsize=(12,8))
plt.barh(top_30_descriptions.index[::-1], top_30_descriptions.values[::-1], color='#ff6200')

# Adding labels and title
plt.xlabel('Number of Occurrences')
plt.ylabel('Description')
plt.title('Top 30 Most Frequent Descriptions')

# Show the plot
plt.show()

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Inferences on Descriptions:</font></h3>
# 
# - The most frequent descriptions are generally household items, particularly those associated with kitchenware, lunch bags, and decorative items.
#        
#     
# - Interestingly, all the descriptions are in uppercase, which might be a standardized format for entering product descriptions in the database. However, considering the inconsistencies and anomalies encountered in the dataset so far, it would be prudent to check if there are descriptions entered in lowercase or a mix of case styles.

# %%
# Find unique descriptions containing lowercase characters
lowercase_descriptions = df['Description'].unique()
lowercase_descriptions = [desc for desc in lowercase_descriptions if any(char.islower() for char in desc)]

# Print the unique descriptions containing lowercase characters
print("The unique descriptions containing lowercase characters are:")
print("-"*60)
for desc in lowercase_descriptions:
    print(desc)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Inference:</font></h3>
#     
# - Upon reviewing the descriptions that contain lowercase characters, it is evident that some entries are not product descriptions, such as "__Next Day Carriage__" and "__High Resolution Image__". These entries seem to be unrelated to the actual products and might represent other types of information or service details.

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Strategy:</font></h3>
# 
# - __Step 1__: Remove the rows where the descriptions contain service-related information like "__Next Day Carriage__" and "__High Resolution Image__", as these do not represent actual products and would not contribute to the clustering and recommendation system we aim to build.
# 
#     
# - __Step 2__: For the remaining descriptions with mixed case, standardize the text to uppercase to maintain uniformity across the dataset. This will also assist in reducing the chances of having duplicate entries with different case styles.
# 
# By implementing the above strategy, we can enhance the quality of our dataset, making it more suitable for the analysis and modeling phases of our project.

# %%
service_related_descriptions = ["Next Day Carriage", "High Resolution Image"]

# Calculate the percentage of records with service-related descriptions
service_related_percentage = df[df['Description'].isin(service_related_descriptions)].shape[0] / df.shape[0] * 100

# Print the percentage of records with service-related descriptions
print(f"The percentage of records with service-related descriptions in the dataset is: {service_related_percentage:.2f}%")

# Remove rows with service-related information in the description
df = df[~df['Description'].isin(service_related_descriptions)]

# Standardize the text to uppercase to maintain uniformity across the dataset
df['Description'] = df['Description'].str.upper()

# %%
# Getting the number of rows in the dataframe
df.shape[0]

# %% [markdown]
# <a id="UnitPrice_cleaning"></a>
# # <b><span style='color:#fcc36d'>Step 3.6 |</span><span style='color:#ff6200'> Treating Zero Unit Prices</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# In this step, first I am going to take a look at the statistical description of the `UnitPrice` column:

# %%
df['UnitPrice'].describe()

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Inference:</font></h3>
#     
# The minimum unit price value is zero. This suggests that there are some transactions where the unit price is zero, potentially indicating a free item or a data entry error. To understand their nature, it is essential to investigate these zero unit price transactions further. A detailed analysis of the product descriptions associated with zero unit prices will be conducted to determine if they adhere to a specific pattern:

# %%
df[df['UnitPrice']==0].describe()[['Quantity']]

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Inferences on UnitPrice: </font></h3>
# 
# - The transactions with a unit price of zero are relatively few in number (33 transactions).
#     
#     
# - These transactions have a large variability in the quantity of items involved, ranging from 1 to 12540, with a substantial standard deviation.
#     
#     
# - Including these transactions in the clustering analysis might introduce noise and could potentially distort the customer behavior patterns identified by the clustering algorithm.

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Strategy: </font></h3>
# 
# Given the small number of these transactions and their potential to introduce noise in the data analysis, the strategy should be to remove these transactions from the dataset. This would help in maintaining a cleaner and more consistent dataset, which is essential for building an accurate and reliable clustering model and recommendation system.

# %%
# Removing records with a unit price of zero to avoid potential data entry errors
df = df[df['UnitPrice'] > 0]

# %% [markdown]
# <a id="outlier_cleaning"></a>
# # <b><span style='color:#fcc36d'>Step 3.7 |</span><span style='color:#ff6200'> Outlier Treatment</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# In K-means clustering, the algorithm is sensitive to both the scale of data and the presence of outliers, as they can significantly influence the position of centroids, potentially leading to incorrect cluster assignments. However, considering the context of this project where the final goal is to understand customer behavior and preferences through K-means clustering, it would be more prudent to address the issue of outliers __after the feature engineering phase__ where we create a customer-centric dataset. At this stage, the data is transactional, and removing outliers might eliminate valuable information that could play a crucial role in segmenting customers later on. Therefore, we will postpone the outlier treatment and proceed to the next stage for now.

# %%
# Resetting the index of the cleaned dataset
df.reset_index(drop=True, inplace=True)

# %%
# Getting the number of rows in the dataframe
df.shape[0]

# %% [markdown]
# <a id="feature_engineering"></a>
# # <p style="background-color: #ff6200; font-family:calibri; color:white; font-size:140%; font-family:Verdana; text-align:center; border-radius:15px 50px;">Step 4 | Feature Engineering</p>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# In order to create a comprehensive customer-centric dataset for clustering and recommendation, the following features can be engineered from the available data:

# %% [markdown]
# <a id="rfm_features"></a>
# # <b><span style='color:#fcc36d'>Step 4.1 |</span><span style='color:#ff6200'> RFM Features</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# RFM is a method used for analyzing customer value and segmenting the customer base. It is an acronym that stands for:
# 
# - __Recency (R):__ This metric indicates how recently a customer has made a purchase. A lower recency value means the customer has purchased more recently, indicating higher engagement with the brand.
# 
#     
# - __Frequency (F):__ This metric signifies how often a customer makes a purchase within a certain period. A higher frequency value indicates a customer who interacts with the business more often, suggesting higher loyalty or satisfaction.
# 
#     
# - __Monetary (M):__ This metric represents the total amount of money a customer has spent over a certain period. Customers who have a higher monetary value have contributed more to the business, indicating their potential high lifetime value.
# 
#     
# Together, these metrics help in understanding a customer's buying behavior and preferences, which is pivotal in personalizing marketing strategies and creating a recommendation system.

# %% [markdown]
# <a id="recency"></a>
# ## <b><span style='color:#fcc36d'>Step 4.1.1 |</span><span style='color:#ff6200'> Recency (R)</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# In this step, we focus on understanding how recently a customer has made a purchase. This is a crucial aspect of customer segmentation as it helps in identifying the engagement level of customers. Here, I am going to define the following feature:
# 
# - __Days Since Last Purchas__: This feature represents the number of days that have passed since the customer's last purchase. A lower value indicates that the customer has purchased recently, implying a higher engagement level with the business, whereas a higher value may indicate a lapse or decreased engagement. By understanding the recency of purchases, businesses can tailor their marketing strategies to re-engage customers who have not made purchases in a while, potentially increasing customer retention and fostering loyalty.

# %%
# Convert InvoiceDate to datetime type
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Convert InvoiceDate to datetime and extract only the date
df['InvoiceDay'] = df['InvoiceDate'].dt.date

# Find the most recent purchase date for each customer
customer_data = df.groupby('CustomerID')['InvoiceDay'].max().reset_index()

# Find the most recent date in the entire dataset
most_recent_date = df['InvoiceDay'].max()

# Convert InvoiceDay to datetime type before subtraction
customer_data['InvoiceDay'] = pd.to_datetime(customer_data['InvoiceDay'])
most_recent_date = pd.to_datetime(most_recent_date)

# Calculate the number of days since the last purchase for each customer
customer_data['Days_Since_Last_Purchase'] = (most_recent_date - customer_data['InvoiceDay']).dt.days

# Remove the InvoiceDay column
customer_data.drop(columns=['InvoiceDay'], inplace=True)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# Now, __customer_data__ dataframe contains the __`Days_Since_Last_Purchase`__ feature:

# %%
customer_data.head()

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Note: </font></h3>
#     
# - I've named the customer-centric dataframe as __customer_data__, which will eventually contain all the customer-based features we plan to create.

# %% [markdown]
# <a id="frequency"></a>
# ## <b><span style='color:#fcc36d'>Step 4.1.2 |</span><span style='color:#ff6200'> Frequency (F)</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# In this step, I am going to create two features that quantify the frequency of a customer's engagement with the retailer:
# 
# - __Total Transactions__: This feature represents the total number of transactions made by a customer. It helps in understanding the engagement level of a customer with the retailer.
# 
#     
# 
# - __Total Products Purchased__: This feature indicates the total number of products (sum of quantities) purchased by a customer across all transactions. It gives an insight into the customer's buying behavior in terms of the volume of products purchased.
# 
#     
# These features will be crucial in segmenting customers based on their buying frequency, which is a key aspect in determining customer segments for targeted marketing and personalized recommendations.

# %%
# Calculate the total number of transactions made by each customer
total_transactions = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
total_transactions.rename(columns={'InvoiceNo': 'Total_Transactions'}, inplace=True)

# Calculate the total number of products purchased by each customer
total_products_purchased = df.groupby('CustomerID')['Quantity'].sum().reset_index()
total_products_purchased.rename(columns={'Quantity': 'Total_Products_Purchased'}, inplace=True)

# Merge the new features into the customer_data dataframe
customer_data = pd.merge(customer_data, total_transactions, on='CustomerID')
customer_data = pd.merge(customer_data, total_products_purchased, on='CustomerID')

# Display the first few rows of the customer_data dataframe
customer_data.head()

# %% [markdown]
# <a id="monetary"></a>
# ## <b><span style='color:#fcc36d'>Step 4.1.3 |</span><span style='color:#ff6200'> Monetary (M)</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# In this step, I am going to create two features that represent the monetary aspect of customer's transactions:
# 
# - __Total Spend__: This feature represents the total amount of money spent by each customer. It is calculated as the sum of the product of `UnitPrice` and `Quantity` for all transactions made by a customer. This feature is crucial as it helps in identifying the total revenue generated by each customer, which is a direct indicator of a customer's value to the business.
# 
#     
# - __Average Transaction Value__: This feature is calculated as the __Total Spend__ divided by the __Total Transactions__ for each customer. It indicates the average value of a transaction carried out by a customer. This metric is useful in understanding the spending behavior of customers per transaction, which can assist in tailoring marketing strategies and offers to different customer segments based on their average spending patterns.

# %%
# Calculate the total spend by each customer
df['Total_Spend'] = df['UnitPrice'] * df['Quantity']
total_spend = df.groupby('CustomerID')['Total_Spend'].sum().reset_index()

# Calculate the average transaction value for each customer
average_transaction_value = total_spend.merge(total_transactions, on='CustomerID')
average_transaction_value['Average_Transaction_Value'] = average_transaction_value['Total_Spend'] / average_transaction_value['Total_Transactions']

# Merge the new features into the customer_data dataframe
customer_data = pd.merge(customer_data, total_spend, on='CustomerID')
customer_data = pd.merge(customer_data, average_transaction_value[['CustomerID', 'Average_Transaction_Value']], on='CustomerID')

# Display the first few rows of the customer_data dataframe
customer_data.head()

# %% [markdown]
# <a id="product_diversity"></a>
# # <b><span style='color:#fcc36d'>Step 4.2 |</span><span style='color:#ff6200'> Product Diversity</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# In this step, we are going to understand the diversity in the product purchase behavior of customers. Understanding product diversity can help in crafting personalized marketing strategies and product recommendations. Here, I am going to define the following feature:
# 
# - __Unique Products Purchased__: This feature represents the number of distinct products bought by a customer. A higher value indicates that the customer has a diverse taste or preference, buying a wide range of products, while a lower value might indicate a focused or specific preference. Understanding the diversity in product purchases can help in segmenting customers based on their buying diversity, which can be a critical input in personalizing product recommendations.

# %%
# Calculate the number of unique products purchased by each customer
unique_products_purchased = df.groupby('CustomerID')['StockCode'].nunique().reset_index()
unique_products_purchased.rename(columns={'StockCode': 'Unique_Products_Purchased'}, inplace=True)

# Merge the new feature into the customer_data dataframe
customer_data = pd.merge(customer_data, unique_products_purchased, on='CustomerID')

# Display the first few rows of the customer_data dataframe
customer_data.head()

# %% [markdown]
# <a id="behaviroal_features"></a>
# # <b><span style='color:#fcc36d'>Step 4.3 |</span><span style='color:#ff6200'> Behavioral Features</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# In this step, we aim to understand and capture the shopping patterns and behaviors of customers. These features will give us insights into the customers' preferences regarding when they like to shop, which can be crucial information for personalizing their shopping experience. Here are the features I am planning to introduce:
# 
# - __Average Days Between Purchases__: This feature represents the average number of days a customer waits before making another purchase. Understanding this can help in predicting when the customer is likely to make their next purchase, which can be a crucial metric for targeted marketing and personalized promotions.
# 
#     
# - __Favorite Shopping Day__: This denotes the day of the week when the customer shops the most. This information can help in identifying the preferred shopping days of different customer segments, which can be used to optimize marketing strategies and promotions for different days of the week.
# 
#     
# - __Favorite Shopping Hour__: This refers to the hour of the day when the customer shops the most. Identifying the favorite shopping hour can aid in optimizing the timing of marketing campaigns and promotions to align with the times when different customer segments are most active.
# 
#     
# By including these behavioral features in our dataset, we can create a more rounded view of our customers, which will potentially enhance the effectiveness of the clustering algorithm, leading to more meaningful customer segments.

# %%
# Extract day of week and hour from InvoiceDate
df['Day_Of_Week'] = df['InvoiceDate'].dt.dayofweek
df['Hour'] = df['InvoiceDate'].dt.hour

# Calculate the average number of days between consecutive purchases
days_between_purchases = df.groupby('CustomerID')['InvoiceDay'].apply(lambda x: (x.diff().dropna()).apply(lambda y: y.days))
average_days_between_purchases = days_between_purchases.groupby('CustomerID').mean().reset_index()
average_days_between_purchases.rename(columns={'InvoiceDay': 'Average_Days_Between_Purchases'}, inplace=True)

# Find the favorite shopping day of the week
favorite_shopping_day = df.groupby(['CustomerID', 'Day_Of_Week']).size().reset_index(name='Count')
favorite_shopping_day = favorite_shopping_day.loc[favorite_shopping_day.groupby('CustomerID')['Count'].idxmax()][['CustomerID', 'Day_Of_Week']]

# Find the favorite shopping hour of the day
favorite_shopping_hour = df.groupby(['CustomerID', 'Hour']).size().reset_index(name='Count')
favorite_shopping_hour = favorite_shopping_hour.loc[favorite_shopping_hour.groupby('CustomerID')['Count'].idxmax()][['CustomerID', 'Hour']]

# Merge the new features into the customer_data dataframe
customer_data = pd.merge(customer_data, average_days_between_purchases, on='CustomerID')
customer_data = pd.merge(customer_data, favorite_shopping_day, on='CustomerID')
customer_data = pd.merge(customer_data, favorite_shopping_hour, on='CustomerID')

# Display the first few rows of the customer_data dataframe
customer_data.head()

# %% [markdown]
# <a id="geographical_features"></a>
# # <b><span style='color:#fcc36d'>Step 4.4 |</span><span style='color:#ff6200'> Geographic Features</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# In this step, we will introduce a geographic feature that reflects the geographical location of customers. Understanding the geographic distribution of customers is pivotal for several reasons:
# 
# - __Country__: This feature identifies the country where each customer is located. Including the country data can help us understand region-specific buying patterns and preferences. Different regions might have varying preferences and purchasing behaviors which can be critical in personalizing marketing strategies and inventory planning. Furthermore, it can be instrumental in logistics and supply chain optimization, particularly for an online retailer where shipping and delivery play a significant role.

# %%
df['Country'].value_counts(normalize=True).head()

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Inference: </font></h3>
#     
# Given that a substantial portion (__89%__) of transactions are originating from the __United Kingdom__, we might consider creating a binary feature indicating whether the transaction is from the UK or not. This approach can potentially streamline the clustering process without losing critical geographical information, especially when considering the application of algorithms like K-means which are sensitive to the dimensionality of the feature space.

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Methodology: </font></h3>
# 
# - First, I will group the data by `CustomerID` and `Country` and calculate the number of transactions per country for each customer.
# 
# - Next, I will identify the main country for each customer (the country from which they have the maximum transactions).
#     
# - Then, I will create a binary column indicating whether the customer is from the UK or not.
#     
# - Finally, I will merge this information with the `customer_data` dataframe to include the new feature in our analysis.

# %%
# Group by CustomerID and Country to get the number of transactions per country for each customer
customer_country = df.groupby(['CustomerID', 'Country']).size().reset_index(name='Number_of_Transactions')

# Get the country with the maximum number of transactions for each customer (in case a customer has transactions from multiple countries)
customer_main_country = customer_country.sort_values('Number_of_Transactions', ascending=False).drop_duplicates('CustomerID')

# Create a binary column indicating whether the customer is from the UK or not
customer_main_country['Is_UK'] = customer_main_country['Country'].apply(lambda x: 1 if x == 'United Kingdom' else 0)

# Merge this data with our customer_data dataframe
customer_data = pd.merge(customer_data, customer_main_country[['CustomerID', 'Is_UK']], on='CustomerID', how='left')

# Display the first few rows of the customer_data dataframe
customer_data.head()

# %%
# Display feature distribution
customer_data['Is_UK'].value_counts()

# %% [markdown]
# <a id="cancellation_insights"></a>
# # <b><span style='color:#fcc36d'>Step 4.5 |</span><span style='color:#ff6200'> Cancellation Insights</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# In this step, I am going to delve deeper into the cancellation patterns of customers to gain insights that can enhance our customer segmentation model. The features I am planning to introduce are:
# 
# - __Cancellation Frequency__: This metric represents the total number of transactions a customer has canceled. Understanding the frequency of cancellations can help us identify customers who are more likely to cancel transactions. This could be an indicator of dissatisfaction or other issues, and understanding this can help us tailor strategies to reduce cancellations and enhance customer satisfaction.
# 
#     
# - __Cancellation Rate__: This represents the proportion of transactions that a customer has canceled out of all their transactions. This metric gives a normalized view of cancellation behavior. A high cancellation rate might be indicative of an unsatisfied customer segment. By identifying these segments, we can develop targeted strategies to improve their shopping experience and potentially reduce the cancellation rate.
# 
# By incorporating these cancellation insights into our dataset, we can build a more comprehensive view of customer behavior, which could potentially aid in creating more effective and nuanced customer segmentation.
# 

# %%
# Calculate the total number of transactions made by each customer
total_transactions = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()

# Calculate the number of cancelled transactions for each customer
cancelled_transactions = df[df['Transaction_Status'] == 'Cancelled']
cancellation_frequency = cancelled_transactions.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
cancellation_frequency.rename(columns={'InvoiceNo': 'Cancellation_Frequency'}, inplace=True)

# Merge the Cancellation Frequency data into the customer_data dataframe
customer_data = pd.merge(customer_data, cancellation_frequency, on='CustomerID', how='left')

# Replace NaN values with 0 (for customers who have not cancelled any transaction)
customer_data['Cancellation_Frequency'].fillna(0, inplace=True)

# Calculate the Cancellation Rate
customer_data['Cancellation_Rate'] = customer_data['Cancellation_Frequency'] / total_transactions['InvoiceNo']

# Display the first few rows of the customer_data dataframe
customer_data.head()

# %% [markdown]
# <a id="seasonality_trends"></a>
# # <b><span style='color:#fcc36d'>Step 4.6 |</span><span style='color:#ff6200'> Seasonality & Trends</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# In this step, I will delve into the seasonality and trends in customers' purchasing behaviors, which can offer invaluable insights for tailoring marketing strategies and enhancing customer satisfaction. Here are the features I am looking to introduce:
# 
# - __Monthly_Spending_Mean__: This is the average amount a customer spends monthly. It helps us gauge the general spending habit of each customer. A higher mean indicates a customer who spends more, potentially showing interest in premium products, whereas a lower mean might indicate a more budget-conscious customer.
# 
#     
# - __Monthly_Spending_Std__: This feature indicates the variability in a customer's monthly spending. A higher value signals that the customer's spending fluctuates significantly month-to-month, perhaps indicating sporadic large purchases. In contrast, a lower value suggests more stable, consistent spending habits. Understanding this variability can help in crafting personalized promotions or discounts during periods they are expected to spend more.
# 
#     
# - __Spending_Trend__: This reflects the trend in a customer's spending over time, calculated as the slope of the linear trend line fitted to their spending data. A positive value indicates an increasing trend in spending, possibly pointing to growing loyalty or satisfaction. Conversely, a negative trend might signal decreasing interest or satisfaction, highlighting a need for re-engagement strategies. A near-zero value signifies stable spending habits. Recognizing these trends can help in developing strategies to either maintain or alter customer spending patterns, enhancing the effectiveness of marketing campaigns.
# 
# By incorporating these detailed insights into our customer segmentation model, we can create more precise and actionable customer groups, facilitating the development of highly targeted marketing strategies and promotions.
# 

# %%
# Extract month and year from InvoiceDate
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month

# Calculate monthly spending for each customer
monthly_spending = df.groupby(['CustomerID', 'Year', 'Month'])['Total_Spend'].sum().reset_index()

# Calculate Seasonal Buying Patterns: We are using monthly frequency as a proxy for seasonal buying patterns
seasonal_buying_patterns = monthly_spending.groupby('CustomerID')['Total_Spend'].agg(['mean', 'std']).reset_index()
seasonal_buying_patterns.rename(columns={'mean': 'Monthly_Spending_Mean', 'std': 'Monthly_Spending_Std'}, inplace=True)

# Replace NaN values in Monthly_Spending_Std with 0, implying no variability for customers with single transaction month
seasonal_buying_patterns['Monthly_Spending_Std'].fillna(0, inplace=True)

# Calculate Trends in Spending 
# We are using the slope of the linear trend line fitted to the customer's spending over time as an indicator of spending trends
def calculate_trend(spend_data):
    # If there are more than one data points, we calculate the trend using linear regression
    if len(spend_data) > 1:
        x = np.arange(len(spend_data))
        slope, _, _, _, _ = linregress(x, spend_data)
        return slope
    # If there is only one data point, no trend can be calculated, hence we return 0
    else:
        return 0

# Apply the calculate_trend function to find the spending trend for each customer
spending_trends = monthly_spending.groupby('CustomerID')['Total_Spend'].apply(calculate_trend).reset_index()
spending_trends.rename(columns={'Total_Spend': 'Spending_Trend'}, inplace=True)

# Merge the new features into the customer_data dataframe
customer_data = pd.merge(customer_data, seasonal_buying_patterns, on='CustomerID')
customer_data = pd.merge(customer_data, spending_trends, on='CustomerID')

# Display the first few rows of the customer_data dataframe
customer_data.head()

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# We've done a great job so far! We have created a dataset that focuses on our customers, using a variety of new features that give us a deeper understanding of their buying patterns and preferences.

# %%
# Changing the data type of 'CustomerID' to string as it is a unique identifier and not used in mathematical operations
customer_data['CustomerID'] = customer_data['CustomerID'].astype(str)

# Convert data types of columns to optimal types
customer_data = customer_data.convert_dtypes()

# %%
customer_data.head(10)

# %%
customer_data.info()

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# Let's review the descriptions of the columns in our newly created `customer_data` dataset:

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:130%; text-align:left">
# 
# <h2 align="left"><font color=#ff6200>Customer Dataset Description:</font></h2>
# 
# | __Variable__                       | __Description__ |
# |     :---                           |       :---      |
# | __CustomerID__                     | Identifier uniquely assigned to each customer, used to distinguish individual customers. |
# | __Days_Since_Last_Purchase__       | The number of days that have passed since the customer's last purchase. |
# | __Total_Transactions__             | The total number of transactions made by the customer. |
# | __Total_Products_Purchased__       | The total quantity of products purchased by the customer across all transactions. |
# | __Total_Spend__                    | The total amount of money the customer has spent across all transactions. |
# | __Average_Transaction_Value__      | The average value of the customer's transactions, calculated as total spend divided by the number of transactions. |
# | __Unique_Products_Purchased__      | The number of different products the customer has purchased. |
# | __Average_Days_Between_Purchases__ | The average number of days between consecutive purchases made by the customer. |
# | __Day_Of_Week__                    | The day of the week when the customer prefers to shop, represented numerically (0 for Monday, 6 for Sunday). |
# | __Hour__                           | The hour of the day when the customer prefers to shop, represented in a 24-hour format. |
# | __Is_UK__                          | A binary variable indicating whether the customer is based in the UK (1) or not (0). |
# | __Cancellation_Frequency__         | The total number of transactions that the customer has cancelled. |
# | __Cancellation_Rate__              | The proportion of transactions that the customer has cancelled, calculated as cancellation frequency divided by total transactions. |
# | __Monthly_Spending_Mean__          | The average monthly spending of the customer. |
# | __Monthly_Spending_Std__           | The standard deviation of the customer's monthly spending, indicating the variability in their spending pattern. |
# | __Spending_Trend__                 | A numerical representation of the trend in the customer's spending over time. A positive value indicates an increasing trend, a negative value indicates a decreasing trend, and a value close to zero indicates a stable trend. |
# 

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# We've done a great job so far! We have created a dataset that focuses on our customers, using a variety of new features that give us a deeper understanding of their buying patterns and preferences.
# 
# Now that our dataset is ready, we can move on to the next steps of our project. This includes looking at our data more closely to find any patterns or trends, making sure our data is in the best shape by checking for and handling any outliers, and preparing our data for the clustering process. All of these steps will help us build a strong foundation for creating our customer segments and, eventually, a personalized recommendation system.

# %% [markdown]
# <h3 align="left"><font color=#ff6200>Let's dive in!</font></h3>

# %% [markdown]
# <a id="outlier_detection"></a>
# # <p style="background-color: #ff6200; font-family:calibri; color:white; font-size:140%; font-family:Verdana; text-align:center; border-radius:15px 50px;">Step 5 | Outlier Detection and Treatment</p>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# In this section, I will identify and handle outliers in our dataset. Outliers are data points that are significantly different from the majority of other points in the dataset. These points can potentially skew the results of our analysis, especially in k-means clustering where they can significantly influence the position of the cluster centroids. Therefore, it is essential to identify and treat these outliers appropriately to achieve more accurate and meaningful clustering results.
# 
# Given the multi-dimensional nature of the data, it would be prudent to use algorithms that can detect outliers in multi-dimensional spaces. I am going to use the __Isolation Forest__ algorithm for this task. This algorithm works well for multi-dimensional data and is computationally efficient. It isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.
# 
# Let's proceed with this approach:

# %%
# Initializing the IsolationForest model with a contamination parameter of 0.05
model = IsolationForest(contamination=0.05, random_state=0)

# Fitting the model on our dataset (converting DataFrame to NumPy to avoid warning)
customer_data['Outlier_Scores'] = model.fit_predict(customer_data.iloc[:, 1:].to_numpy())

# Creating a new column to identify outliers (1 for inliers and -1 for outliers)
customer_data['Is_Outlier'] = [1 if x == -1 else 0 for x in customer_data['Outlier_Scores']]

# Display the first few rows of the customer_data dataframe
customer_data.head()

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# After applying the Isolation Forest algorithm, we have identified the outliers and marked them in a new column named `Is_Outlier`. We have also calculated the outlier scores which represent the anomaly score of each record. 
# 
# Now let's visualize the distribution of these scores and the number of inliers and outliers detected by the model:

# %%
# Calculate the percentage of inliers and outliers
outlier_percentage = customer_data['Is_Outlier'].value_counts(normalize=True) * 100

# Plotting the percentage of inliers and outliers
plt.figure(figsize=(12, 4))
outlier_percentage.plot(kind='barh', color='#ff6200')

# Adding the percentage labels on the bars
for index, value in enumerate(outlier_percentage):
    plt.text(value, index, f'{value:.2f}%', fontsize=15)

plt.title('Percentage of Inliers and Outliers')
plt.xticks(ticks=np.arange(0, 115, 5))
plt.xlabel('Percentage (%)')
plt.ylabel('Is Outlier')
plt.gca().invert_yaxis()
plt.show()

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Inference: </font></h3>
#     
# From the above plot, we can observe that about 5% of the customers have been identified as outliers in our dataset. This percentage seems to be a reasonable proportion, not too high to lose a significant amount of data, and not too low to retain potentially noisy data points. It suggests that our isolation forest algorithm has worked well in identifying a moderate percentage of outliers, which will be critical in refining our customer segmentation.

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Strategy: </font></h3>
# 
# Considering the nature of the project (customer segmentation using clustering), it is crucial to handle these outliers to prevent them from affecting the clusters' quality significantly. Therefore, I will separate these outliers for further analysis and remove them from our main dataset to prepare it for the clustering analysis. 
# 
# Let's proceed with the following steps:
# 
# - Separate the identified outliers for further analysis and save them as a separate file (optional).
# - Remove the outliers from the main dataset to prevent them from influencing the clustering process.
# - Drop the `Outlier_Scores` and `Is_Outlier` columns as they were auxiliary columns used for the outlier detection process.
# 
# Let's implement these steps:

# %%
# Separate the outliers for analysis
outliers_data = customer_data[customer_data['Is_Outlier'] == 1]

# Remove the outliers from the main dataset
customer_data_cleaned = customer_data[customer_data['Is_Outlier'] == 0]

# Drop the 'Outlier_Scores' and 'Is_Outlier' columns
customer_data_cleaned = customer_data_cleaned.drop(columns=['Outlier_Scores', 'Is_Outlier'])

# Reset the index of the cleaned data
customer_data_cleaned.reset_index(drop=True, inplace=True)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# We have successfully separated the outliers for further analysis and cleaned our main dataset by removing these outliers. This cleaned dataset is now ready for the next steps in our customer segmentation project, which includes scaling the features and applying clustering algorithms to identify distinct customer segments.

# %%
# Getting the number of rows in the cleaned customer dataset
customer_data_cleaned.shape[0]

# %% [markdown]
# <a id="correlation"></a>
# # <p style="background-color: #ff6200; font-family:calibri; color:white; font-size:140%; font-family:Verdana; text-align:center; border-radius:15px 50px;">Step 6 | Correlation Analysis</p>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# Before we proceed to KMeans clustering, it's essential to check the correlation between features in our dataset. The presence of __multicollinearity__, where __features are highly correlated__, can potentially affect the clustering process by not allowing the model to learn the actual underlying patterns in the data, as the features do not provide unique information. This could lead to clusters that are not well-separated and meaningful.
# 
# If we identify multicollinearity, we can utilize dimensionality reduction techniques like PCA. These techniques help in neutralizing the effect of multicollinearity by transforming the correlated features into a new set of uncorrelated variables, preserving most of the original data's variance. This step not only enhances the quality of clusters formed but also makes the clustering process more computationally efficient.

# %%
# Reset background style
sns.set_style('whitegrid')

# Calculate the correlation matrix excluding the 'CustomerID' column
corr = customer_data_cleaned.drop(columns=['CustomerID']).corr()

# Define a custom colormap
colors = ['#ff6200', '#ffcaa8', 'white', '#ffcaa8', '#ff6200']
my_cmap = LinearSegmentedColormap.from_list('custom_map', colors, N=256)

# Create a mask to only show the lower triangle of the matrix (since it's mirrored around its 
# top-left to bottom-right diagonal)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, k=1)] = True

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr, mask=mask, cmap=my_cmap, annot=True, center=0, fmt='.2f', linewidths=2)
plt.title('Correlation Matrix', fontsize=14)
plt.show()

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Inference: </font></h3>
# 
# Looking at the heatmap, we can see that there are some pairs of variables that have high correlations, for instance:
# 
# - `Monthly_Spending_Mean` and `Average_Transaction_Value`
#     
#     
# - `Total_Spend` and `Total_Products_Purchased`
# 
#     
# - `Total_Transactions` and `Total_Spend`
#     
#     
# - `Cancellation_Rate` and `Cancellation_Frequency`
#     
#     
# - `Total_Transactions` and `Total_Products_Purchased`
#  
#     
# These high correlations indicate that these variables move closely together, implying a degree of multicollinearity.

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# 
# Before moving to the next steps, considering the impact of multicollinearity on KMeans clustering, it might be beneficial to treat this multicollinearity possibly through dimensionality reduction techniques such as PCA to create a set of uncorrelated variables. This will help in achieving more stable clusters during the KMeans clustering process.

# %% [markdown]
# <a id="scaling"></a>
# # <p style="background-color: #ff6200; font-family:calibri; color:white; font-size:140%; font-family:Verdana; text-align:center; border-radius:15px 50px;">Step 7 | Feature Scaling</p>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# Before we move forward with the clustering and dimensionality reduction, it's imperative to scale our features. This step holds significant importance, especially in the context of distance-based algorithms like K-means and dimensionality reduction methods like PCA. Here's why:
# 
#   - __For K-means Clustering__: K-means relies heavily on the concept of '__distance__' between data points to form clusters. When features are not on a similar scale, features with larger values can disproportionately influence the clustering outcome, potentially leading to incorrect groupings.
#   
#     
#   - __For PCA__: PCA aims to find the directions where the data varies the most. When features are not scaled, those with larger values might dominate these components, not accurately reflecting the underlying patterns in the data.
# 

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Methodology: </font></h3>
#     
# Therefore, to ensure a balanced influence on the model and to reveal the true patterns in the data, I am going to standardize our data, meaning transforming the features to have a mean of 0 and a standard deviation of 1. However, not all features require scaling. Here are the exceptions and the reasons why they are excluded:
# 
# - __CustomerID__: This feature is just an identifier for the customers and does not contain any meaningful information for clustering.
#     
#     
# - __Is_UK__: This is a binary feature indicating whether the customer is from the UK or not. Since it already takes a value of 0 or 1, scaling it won't make any significant difference.
#     
#     
# - __Day_Of_Week__: This feature represents the most frequent day of the week that the customer made transactions. Since it's a categorical feature represented by integers (1 to 7), scaling it would not be necessary.
# 
#     
# I will proceed to scale the other features in the dataset to prepare it for PCA and K-means clustering.

# %%
# Initialize the StandardScaler
scaler = StandardScaler()

# List of columns that don't need to be scaled
columns_to_exclude = ['CustomerID', 'Is_UK', 'Day_Of_Week']

# List of columns that need to be scaled
columns_to_scale = customer_data_cleaned.columns.difference(columns_to_exclude)

# Copy the cleaned dataset
customer_data_scaled = customer_data_cleaned.copy()

# Applying the scaler to the necessary columns in the dataset
customer_data_scaled[columns_to_scale] = scaler.fit_transform(customer_data_scaled[columns_to_scale])

# Display the first few rows of the scaled data
customer_data_scaled.head()

# %% [markdown]
# <a id="pca"></a>
# # <p style="background-color: #ff6200; font-family:calibri; color:white; font-size:140%; font-family:Verdana; text-align:center; border-radius:15px 50px;">Step 8 | Dimensionality Reduction</p>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Why We Need Dimensionality Reduction? </font></h3>
# 
# - __Multicollinearity Detected__: In the previous steps, we identified that our dataset contains multicollinear features. Dimensionality reduction can help us remove redundant information and alleviate the multicollinearity issue.
# 
#     
# - __Better Clustering with K-means__: Since K-means is a distance-based algorithm, having a large number of features can sometimes dilute the meaningful underlying patterns in the data. By reducing the dimensionality, we can help K-means to find more compact and well-separated clusters.    
#    
#     
# - __Noise Reduction__: By focusing only on the most important features, we can potentially remove noise in the data, leading to more accurate and stable clusters.    
#    
#     
# - __Enhanced Visualization__: In the context of customer segmentation, being able to visualize customer groups in two or three dimensions can provide intuitive insights. Dimensionality reduction techniques can facilitate this by reducing the data to a few principal components which can be plotted easily.
#     
#     
# - __Improved Computational Efficiency__: Reducing the number of features can speed up the computation time during the modeling process, making our clustering algorithm more efficient.
# 
# 
# Let's proceed to select an appropriate dimensionality reduction method to our data.

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Which Dimensionality Reduction Method? </font></h3>
#   
#     
# In this step, we are considering the application of dimensionality reduction techniques to simplify our data while retaining the essential information. Among various methods such as KernelPCA, ICA, ISOMAP, TSNE, and UMAP, I am starting with **PCA (Principal Component Analysis)**. Here's why:
# 
# PCA is an excellent starting point because it works well in capturing linear relationships in the data, which is particularly relevant given the multicollinearity we identified in our dataset. It allows us to reduce the number of features in our dataset while still retaining a significant amount of the information, thus making our clustering analysis potentially more accurate and interpretable. Moreover, it is computationally efficient, which means it won't significantly increase the processing time.
# 
# However, it's essential to note that we are keeping our options open. After applying PCA, if we find that the first few components do not capture a significant amount of variance, indicating a loss of vital information, we might consider exploring other non-linear methods. These methods can potentially provide a more nuanced approach to dimensionality reduction, capturing complex patterns that PCA might miss, albeit at the cost of increased computational time and complexity.
# 
# 

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Methodology </font></h3>
# 
# I will apply PCA on all the available components and plot the cumulative variance explained by them. This process will allow me to visualize how much variance each additional principal component can explain, thereby helping me to pinpoint the optimal number of components to retain for the analysis:

# %%
# Setting CustomerID as the index column
customer_data_scaled.set_index('CustomerID', inplace=True)

# Apply PCA
pca = PCA().fit(customer_data_scaled)

# Calculate the Cumulative Sum of the Explained Variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Set the optimal k value (based on our analysis, we can choose 6)
optimal_k = 6

# Set seaborn plot style
sns.set(rc={'axes.facecolor': '#fcf0dc'}, style='darkgrid')

# Plot the cumulative explained variance against the number of components
plt.figure(figsize=(20, 10))

# Bar chart for the explained variance of each component
barplot = sns.barplot(x=list(range(1, len(cumulative_explained_variance) + 1)),
                      y=explained_variance_ratio,
                      color='#fcc36d',
                      alpha=0.8)

# Line plot for the cumulative explained variance
lineplot, = plt.plot(range(0, len(cumulative_explained_variance)), cumulative_explained_variance,
                     marker='o', linestyle='--', color='#ff6200', linewidth=2)

# Plot optimal k value line
optimal_k_line = plt.axvline(optimal_k - 1, color='red', linestyle='--', label=f'Optimal k value = {optimal_k}') 

# Set labels and title
plt.xlabel('Number of Components', fontsize=14)
plt.ylabel('Explained Variance', fontsize=14)
plt.title('Cumulative Variance vs. Number of Components', fontsize=18)

# Customize ticks and legend
plt.xticks(range(0, len(cumulative_explained_variance)))
plt.legend(handles=[barplot.patches[0], lineplot, optimal_k_line],
           labels=['Explained Variance of Each Component', 'Cumulative Explained Variance', f'Optimal k value = {optimal_k}'],
           loc=(0.62, 0.1),
           frameon=True,
           framealpha=1.0,  
           edgecolor='#ff6200')  

# Display the variance values for both graphs on the plots
x_offset = -0.3
y_offset = 0.01
for i, (ev_ratio, cum_ev_ratio) in enumerate(zip(explained_variance_ratio, cumulative_explained_variance)):
    plt.text(i, ev_ratio, f"{ev_ratio:.2f}", ha="center", va="bottom", fontsize=10)
    if i > 0:
        plt.text(i + x_offset, cum_ev_ratio + y_offset, f"{cum_ev_ratio:.2f}", ha="center", va="bottom", fontsize=10)

plt.grid(axis='both')   
plt.show()

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Conclusion </font></h3>
#     
# The plot and the cumulative explained variance values indicate how much of the total variance in the dataset is captured by each principal component, as well as the cumulative variance explained by the first n components.
# 
# Here, we can observe that:
# 
# - The first component explains approximately 28% of the variance.
# 
# - The first two components together explain about 49% of the variance.
# 
# - The first three components explain approximately 61% of the variance, and so on.
# 
#     
# To choose the optimal number of components, we generally look for a point where adding another component doesn't significantly increase the cumulative explained variance, often referred to as the "__elbow point__" in the curve.
# 
# From the plot, we can see that the increase in cumulative variance starts to slow down after the __6th component__ (which __captures about 81% of the total variance__).
# 
# Considering the context of customer segmentation, we want to retain a sufficient amount of information to identify distinct customer groups effectively. Therefore, retaining __the first 6 components__ might be a balanced choice, as they together explain a substantial portion of the total variance while reducing the dimensionality of the dataset.

# %%
# Creating a PCA object with 6 components
pca = PCA(n_components=6)

# Fitting and transforming the original data to the new PCA dataframe
customer_data_pca = pca.fit_transform(customer_data_scaled)

# Creating a new dataframe from the PCA dataframe, with columns labeled PC1, PC2, etc.
customer_data_pca = pd.DataFrame(customer_data_pca, columns=['PC'+str(i+1) for i in range(pca.n_components_)])

# Adding the CustomerID index back to the new PCA dataframe
customer_data_pca.index = customer_data_scaled.index

# %%
# Displaying the resulting dataframe based on the PCs
customer_data_pca.head()

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# Now, let's extract the coefficients corresponding to each principal component to better understand the transformation performed by PCA:

# %%
# Define a function to highlight the top 3 absolute values in each column of a dataframe
def highlight_top3(column):
    top3 = column.abs().nlargest(3).index
    return ['background-color:  #ffeacc' if i in top3 else '' for i in column.index]

# Create the PCA component DataFrame and apply the highlighting function
pc_df = pd.DataFrame(pca.components_.T, columns=['PC{}'.format(i+1) for i in range(pca.n_components_)],  
                     index=customer_data_scaled.columns)

pc_df.style.apply(highlight_top3, axis=0)

# %% [markdown]
# <a id="kmeans"></a>
# # <p style="background-color: #ff6200; font-family:calibri; color:white; font-size:140%; font-family:Verdana; text-align:center; border-radius:15px 50px;">Step 9 | K-Means Clustering</p>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# <h2 align="left"><font color=#ff6200>K-Means:</font></h2>
# 
# - __K-Means__ is an unsupervised machine learning algorithm that clusters data into a specified number of groups (K) by minimizing the __within-cluster sum-of-squares (WCSS)__, also known as __inertia__. The algorithm iteratively assigns each data point to the nearest centroid, then updates the centroids by calculating the mean of all assigned points. The process repeats until convergence or a stopping criterion is reached. 

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# <h2 align="left"><font color=#ff6200>Drawbacks of K-Means:</font></h2>
# 
# 
# Here are the main drawbacks of the K-means clustering algorithm and their corresponding solutions:
# 
# - 1️⃣ __Inertia is influenced by the number of dimensions__: The value of inertia tends to increase in high-dimensional spaces due to the curse of dimensionality, which can distort the Euclidean distances between data points.
# 
# > __Solution:__ Performing dimensionality reduction, such as __PCA__, before applying K-means to alleviate this issue and speed up computations.
#     
# ___ 
#     
# - 2️⃣ __Dependence on Initial Centroid Placement__: The K-means algorithm might find a local minimum instead of a global minimum, based on where the centroids are initially placed.
# 
# > __Solution:__ To enhance the likelihood of locating the global minimum, we can employ the __k-means++ initialization__ method.
#   
# ___ 
#     
# - 3️⃣ __Requires specifying the number of clusters__: K-means requires specifying the number of clusters (K) beforehand, which may not be known in advance.
# 
# > __Solution:__ Using methods such as the __elbow method__ and __silhouette analysis__ to estimate the optimal number of clusters.
#     
# ___     
#     
# - 4️⃣ __Sensitivity to unevenly sized or sparse clusters__: K-means might struggle with clusters of different sizes or densities.
# 
# > __Solution:__ Increasing the number of random initializations (n_init) or consider using algorithms that handle unevenly sized clusters better, like GMM or DBSCAN.
#     
# ___ 
#     
# - 5️⃣ __Assumes convex and isotropic clusters__: K-means assumes that clusters are spherical and have similar variances, which is not always the case. It may struggle with elongated or irregularly shaped clusters.
#     
# > __Solution:__ Considering using clustering algorithms that do not make these assumptions, such as DBSCAN or Gaussian Mixture Model (GMM).
#     
#     
# <img src="https://github.com/FarzadNekouee/Retail_Customer_Segmentation_Recommendation_System/blob/master/kmeans_drawbacks.jpg?raw=true" width="2400">
#     
# ___
#     
# Taking into account the aforementioned considerations, I initially applied PCA to the dataset. For the KMeans algorithm, I will set the `init` parameter to `k-means++` and `n_init` to `10`. To determine the optimal number of clusters, I will employ the elbow method and silhouette analysis. Additionally, it might be beneficial to explore the use of alternative clustering algorithms such as GMM and DBSCAN in future analyses to potentially enhance the segmentation results.

# %% [markdown]
# <a id="optimal_k"></a>
# # <b><span style='color:#fcc36d'>Step 9.1 |</span><span style='color:#ff6200'> Determining the Optimal Number of Clusters</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# To ascertain the optimal number of clusters (k) for segmenting customers, I will explore two renowned methods:
# 
# * __Elbow Method__
# 
# * __Silhouette Method__
# 
#     
# It's common to utilize both methods in practice to corroborate the results.

# %% [markdown]
# <a id="elbow"></a>
# ## <b><span style='color:#fcc36d'>Step 9.1.1 |</span><span style='color:#ff6200'> Elbow Method</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# <h3 align="left"><font color=#ff6200>What is the Elbow Method?</font></h3>
#     
# The Elbow Method is a technique for identifying the ideal number of clusters in a dataset. It involves iterating through the data, generating clusters for various values of k. The k-means algorithm calculates the sum of squared distances between each data point and its assigned cluster centroid, known as the __inertia__ or __WCSS__ score. By plotting the inertia score against the k value, we create a graph that typically exhibits an elbow shape, hence the name "__Elbow Method__". The __elbow point__ represents the k-value where the reduction in inertia achieved by increasing k becomes negligible, indicating the optimal stopping point for the number of clusters.

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# <h3 align="left"><font color=#ff6200>Utilizing the YellowBrick Library</font></h3>
# 
# In this section, I will employ the __YellowBrick__ library to facilitate the implementation of the __Elbow method__. YellowBrick, an extension of the Scikit-Learn API, is renowned for its ability to rapidly generate insightful visualizations in the field of machine learning.

# %%
# Set plot style, and background color
sns.set(style='darkgrid', rc={'axes.facecolor': '#fcf0dc'})

# Set the color palette for the plot
sns.set_palette(['#ff6200'])

# Instantiate the clustering model with the specified parameters
km = KMeans(init='k-means++', n_init=10, max_iter=100, random_state=0)

# Create a figure and axis with the desired size
fig, ax = plt.subplots(figsize=(12, 5))

# Instantiate the KElbowVisualizer with the model and range of k values, and disable the timing plot
visualizer = KElbowVisualizer(km, k=(2, 15), timings=False, ax=ax)

# Fit the data to the visualizer
visualizer.fit(customer_data_pca)

# Finalize and render the figure
visualizer.show();

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# <h3 align="left"><font color=#ff6200>Optimal k Value: Elbow Method Insights</font></h3>
# 
# The optimal value of k for the KMeans clustering algorithm can be found at the __elbow point__. Using the YellowBrick library for the Elbow method, we observe that the suggested optimal k value is __5__. However, __we don't have a very distinct elbow point in this case__, which is common in real-world data. From the plot, we can see that the inertia continues to decrease significantly up to k=5, indicating that __the optimum value of k could be between 3 and 7__. To choose the best k within this range, we can employ the __silhouette analysis__, another cluster quality evaluation method. Additionally, incorporating business insights can help determine a practical k value.

# %% [markdown]
# <a id="silhouette"></a>
# ## <b><span style='color:#fcc36d'>Step 9.1.2 |</span><span style='color:#ff6200'> Silhouette Method</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# <h3 align="left"><font color=#ff6200>What is the Silhouette Method?</font></h3>
#     
# The __Silhouette Method__ is an approach to find the optimal number of clusters in a dataset by evaluating the consistency within clusters and their separation from other clusters. It computes the __silhouette coefficient for each data point__, which measures how similar a point is to its own cluster compared to other clusters.
# 
# ____
#     
# <h3 align="left"><font color=#ff6200>What is the Silhouette Coefficient?</font></h3>
#     
# To determine the silhouette coefficient for a given point i, follow these steps:
# 
# * __Calculate a(i)__: Compute the average distance between point i and all other points within its cluster.
# * __Calculate b(i)__: Compute the average distance between point i and all points in the nearest cluster to its own.
# * __Compute the silhouette coefficient__, s(i), for point i using the following formula: 
#     
#     $$ s(i) = \frac{b(i) - a(i)}{\max(b(i), a(i))} $$
#     
# __Note:__ The silhouette coefficient quantifies the similarity of a point to its own cluster (cohesion) relative to its separation from other clusters. This value ranges from -1 to 1, with higher values signifying that the point is well aligned with its cluster and has a low similarity to neighboring clusters.    
# 
# ____
#     
# <h3 align="left"><font color=#ff6200>What is the Silhouette Score?</font></h3>
#     
# The __silhouette score__ is the __average silhouette coefficient__ calculated for all data points in a dataset. It provides an overall assessment of the clustering quality, taking into account both cohesion within clusters and separation between clusters. A higher silhouette score indicates a better clustering configuration.    
#     
# ____
#        
# <h3 align="left"><font color=#ff6200>What are the Advantages of Silhouette Method over the Elbow Method?</font></h3>
#     
# * The __Silhouette Method__ evaluates cluster quality by considering __both__ the __cohesion within clusters__ and their __separation__ from other clusters. This provides a more comprehensive measure of clustering performance compared to the __Elbow Method__, which only considers the __inertia__ (sum of squared distances within clusters).
# 
# 
# * The __Silhouette Method__ produces a silhouette score that directly quantifies the quality of clustering, making it easier to compare different values of k. In contrast, the __Elbow Method__ relies on the subjective interpretation of the elbow point, which can be less reliable in cases where the plot does not show a clear elbow.
# 
#     
# * The __Silhouette Method__ generates a visual representation of silhouette coefficients for each data point, allowing for easier identification of fluctuations and outliers within clusters. This helps in determining the optimal number of clusters with higher confidence, as opposed to the __Elbow Method__, which relies on visual inspection of the inertia plot.

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# <h3 align="left"><font color=#ff6200>Methodology</font></h3>
#     
# In the following analysis:
# 
# - I will initially choose a range of 2-6 for the number of clusters (k) based on the Elbow method from the previous section. Next, I will plot __Silhouette scores__ for each k value to determine the one with the highest score.
# 
# 
# - Subsequently, to fine-tune the selection of the most appropriate k, I will generate __Silhouette plots__ that visually display the __silhouette coefficients for each data point within various clusters__.
# 
# 
# The __YellowBrick__ library will be utilized once again to create these plots and facilitate a comparative analysis.

# %%
def silhouette_analysis(df, start_k, stop_k, figsize=(15, 16)):
    """
    Perform Silhouette analysis for a range of k values and visualize the results.
    """

    # Set the size of the figure
    plt.figure(figsize=figsize)

    # Create a grid with (stop_k - start_k + 1) rows and 2 columns
    grid = gridspec.GridSpec(stop_k - start_k + 1, 2)

    # Assign the first plot to the first row and both columns
    first_plot = plt.subplot(grid[0, :])

    # First plot: Silhouette scores for different k values
    sns.set_palette(['darkorange'])

    silhouette_scores = []

    # Iterate through the range of k values
    for k in range(start_k, stop_k + 1):
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, random_state=0)
        km.fit(df)
        labels = km.predict(df)
        score = silhouette_score(df, labels)
        silhouette_scores.append(score)

    best_k = start_k + silhouette_scores.index(max(silhouette_scores))

    plt.plot(range(start_k, stop_k + 1), silhouette_scores, marker='o')
    plt.xticks(range(start_k, stop_k + 1))
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette score')
    plt.title('Average Silhouette Score for Different k Values', fontsize=15)

    # Add the optimal k value text to the plot
    optimal_k_text = f'The k value with the highest Silhouette score is: {best_k}'
    plt.text(10, 0.23, optimal_k_text, fontsize=12, verticalalignment='bottom', 
             horizontalalignment='left', bbox=dict(facecolor='#fcc36d', edgecolor='#ff6200', boxstyle='round, pad=0.5'))
             

    # Second plot (subplot): Silhouette plots for each k value
    colors = sns.color_palette("bright")

    for i in range(start_k, stop_k + 1):    
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=0)
        row_idx, col_idx = divmod(i - start_k, 2)

        # Assign the plots to the second, third, and fourth rows
        ax = plt.subplot(grid[row_idx + 1, col_idx])

        visualizer = SilhouetteVisualizer(km, colors=colors, ax=ax)
        visualizer.fit(df)

        # Add the Silhouette score text to the plot
        score = silhouette_score(df, km.labels_)
        ax.text(0.97, 0.02, f'Silhouette Score: {score:.2f}', fontsize=12, \
                ha='right', transform=ax.transAxes, color='red')

        ax.set_title(f'Silhouette Plot for {i} Clusters', fontsize=15)

    plt.tight_layout()
    plt.show()

# %%
silhouette_analysis(customer_data_pca, 3, 12, figsize=(20, 50))

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# <h3 align="left"><font color=#ff6200>Guidelines to Interpret Silhouette Plots and Determine the Optimal K:</font></h3>
# 
# 
# To interpret silhouette plots and identify the optimal number of clusters (\( k \)), consider the following criteria:
# 
# - 1️⃣ __Analyze the Silhouette Plots__:
#    
#    * __Silhouette Score Width__:
#         - __Wide Widths (closer to +1)__: Indicate that the data points in the cluster are well separated from points in other clusters, suggesting well-defined clusters.
#         - __Narrow Widths (closer to -1)__: Show that data points in the cluster are not distinctly separated from other clusters, indicating poorly defined clusters.
#    
#    * __Average Silhouette Score__:
#         - __High Average Width__: A cluster with a high average silhouette score indicates well-separated clusters.
#         - __Low Average Width__: A cluster with a low average silhouette score indicates poor separation between clusters.
# 
# ____
#     
#     
# - 2️⃣ __Uniformity in Cluster Size__:
#    
#    2.1 __Cluster Thickness__:
#    - __Uniform Thickness__: Indicates that clusters have a roughly equal number of data points, suggesting a balanced clustering structure.
#    - __Variable Thickness__: Signifies an imbalance in the data point distribution across clusters, with some clusters having many data points and others too few.
# 
# ____
#     
#     
# - 3️⃣ __Peaks in Average Silhouette Score__:
#    - __Clear Peaks__: A clear peak in the __average__ silhouette score plot for a specific \( k \) value indicates this \( k \) might be optimal.
# 
# ____
#     
#     
# - 4️⃣ __Minimize Fluctuations in Silhouette Plot Widths__:
#    - __Uniform Widths__: Seek silhouette plots with similar widths across clusters, suggesting a more balanced and optimal clustering.
#    - __Variable Widths__: Avoid wide fluctuations in silhouette plot widths, indicating that clusters are not well-defined and may vary in compactness.
# 
# ____
#     
#     
# - 5️⃣ __Optimal Cluster Selection__:
#    - __Maximize the Overall Average Silhouette Score__: Choose the \( k \) value that gives the highest average silhouette score across all clusters, indicating well-defined clusters.
#    - __Avoid Below-Average Silhouette Scores__: Ensure most clusters have above-average silhouette scores to prevent suboptimal clustering structures.
# 
# ____
#     
#     
# - 6️⃣ __Visual Inspection of Silhouette Plots__:
#    - __Consistent Cluster Formation__: Visually inspect the silhouette plots for each \( k \) value to evaluate the consistency and structure of the formed clusters.
#    - __Cluster Compactness__: Look for more compact clusters, with data points having silhouette scores closer to +1, indicating better clustering.
# 

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# <h3 align="left"><font color=#ff6200>Optimal k Value: Silhouette Method Insights</font></h3>
# 
# Based on above guidelines and after carefully considering the silhouette plots, it's clear that choosing __\( k = 3 \)__ is the better option. This choice gives us clusters that are more evenly matched and well-defined, making our clustering solution stronger and more reliable.

# %% [markdown]
# <a id="kmeans_model"></a>
# # <b><span style='color:#fcc36d'>Step 9.2 |</span><span style='color:#ff6200'> Clustering Model - K-means</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# In this step, I am going to apply the K-means clustering algorithm to segment customers into different clusters based on their purchasing behaviors and other characteristics, using the optimal number of clusters determined in the previous step.
# 
# It's important to note that the K-means algorithm might assign different labels to the clusters in each run. To address this, we have taken an additional step to swap the labels based on the frequency of samples in each cluster, ensuring a consistent label assignment across different runs.

# %%
# Apply KMeans clustering using the optimal k
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=100, random_state=0)
kmeans.fit(customer_data_pca)

# Get the frequency of each cluster
cluster_frequencies = Counter(kmeans.labels_)

# Create a mapping from old labels to new labels based on frequency
label_mapping = {label: new_label for new_label, (label, _) in 
                 enumerate(cluster_frequencies.most_common())}

# Reverse the mapping to assign labels as per your criteria
label_mapping = {v: k for k, v in {2: 1, 1: 0, 0: 2}.items()}

# Apply the mapping to get the new labels
new_labels = np.array([label_mapping[label] for label in kmeans.labels_])

# Append the new cluster labels back to the original dataset
customer_data_cleaned['cluster'] = new_labels

# Append the new cluster labels to the PCA version of the dataset
customer_data_pca['cluster'] = new_labels

# %%
# Display the first few rows of the original dataframe
customer_data_cleaned.head()

# %% [markdown]
# <a id="evaluation"></a>
# # <p style="background-color: #ff6200; font-family:calibri; color:white; font-size:140%; font-family:Verdana; text-align:center; border-radius:15px 50px;">Step 10 | Clustering Evaluation</p>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# After determining the optimal number of clusters (which is 3 in our case) using elbow and silhouette analyses, I move onto the evaluation step to assess the quality of the clusters formed. This step is essential to validate the effectiveness of the clustering and to ensure that the clusters are __coherent__ and __well-separated__. The evaluation metrics and a visualization technique I plan to use are outlined below:
#     
# - 1️⃣ __3D Visualization of Top PCs__ 
# 
#     
# - 2️⃣ __Cluster Distribution Visualization__ 
#     
#     
# - 3️⃣ __Evaluation Metrics__ 
#     
#     * Silhouette Score
#     * Calinski Harabasz Score
#     * Davies Bouldin Score
#        
# ____  
#     
# **Note**: We are using the PCA version of the dataset for evaluation because this is the space where the clusters were actually formed, capturing the most significant patterns in the data. Evaluating in this space ensures a more accurate representation of the cluster quality, helping us understand the true cohesion and separation achieved during clustering. This approach also aids in creating a clearer 3D visualization using the top principal components, illustrating the actual separation between clusters.

# %% [markdown]
# <a id="3d_visualization"></a>
# # <b><span style='color:#fcc36d'>Step 10.1 |</span><span style='color:#ff6200'>  3D Visualization of Top Principal Components</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# In this part, I am going to choose the top 3 PCs (which capture the most variance in the data) and use them to create a 3D visualization. This will allow us to visually inspect the quality of separation and cohesion of clusters to some extent:

# %%
# Setting up the color scheme for the clusters (RGB order)
colors = ['#e8000b', '#1ac938', '#023eff']

# %%
# Create separate data frames for each cluster
cluster_0 = customer_data_pca[customer_data_pca['cluster'] == 0]
cluster_1 = customer_data_pca[customer_data_pca['cluster'] == 1]
cluster_2 = customer_data_pca[customer_data_pca['cluster'] == 2]

# Create a 3D scatter plot
fig = go.Figure()

# Add data points for each cluster separately and specify the color
fig.add_trace(go.Scatter3d(x=cluster_0['PC1'], y=cluster_0['PC2'], z=cluster_0['PC3'], 
                           mode='markers', marker=dict(color=colors[0], size=5, opacity=0.4), name='Cluster 0'))
fig.add_trace(go.Scatter3d(x=cluster_1['PC1'], y=cluster_1['PC2'], z=cluster_1['PC3'], 
                           mode='markers', marker=dict(color=colors[1], size=5, opacity=0.4), name='Cluster 1'))
fig.add_trace(go.Scatter3d(x=cluster_2['PC1'], y=cluster_2['PC2'], z=cluster_2['PC3'], 
                           mode='markers', marker=dict(color=colors[2], size=5, opacity=0.4), name='Cluster 2'))

# Set the title and layout details
fig.update_layout(
    title=dict(text='3D Visualization of Customer Clusters in PCA Space', x=0.5),
    scene=dict(
        xaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='PC1'),
        yaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='PC2'),
        zaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='PC3'),
    ),
    width=900,
    height=800
)

# Show the plot
fig.show()

# %% [markdown]
# <a id="cluster_distributuion"></a>
# # <b><span style='color:#fcc36d'>Step 10.2 |</span><span style='color:#ff6200'>  Cluster Distribution Visualization</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# I am going to utilize a bar plot to visualize the percentage of customers in each cluster, which helps in understanding if the clusters are balanced and significant:

# %%
# Calculate the percentage of customers in each cluster
cluster_percentage = (customer_data_pca['cluster'].value_counts(normalize=True) * 100).reset_index()
cluster_percentage.columns = ['Cluster', 'Percentage']
cluster_percentage.sort_values(by='Cluster', inplace=True)

# Create a horizontal bar plot
plt.figure(figsize=(10, 4))
sns.barplot(x='Percentage', y='Cluster', data=cluster_percentage, orient='h', palette=colors)

# Adding percentages on the bars
for index, value in enumerate(cluster_percentage['Percentage']):
    plt.text(value+0.5, index, f'{value:.2f}%')

plt.title('Distribution of Customers Across Clusters', fontsize=14)
plt.xticks(ticks=np.arange(0, 50, 5))
plt.xlabel('Percentage (%)')

# Show the plot
plt.show()

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Inference</font></h3>
#     
# The distribution of customers across the clusters, as depicted by the bar plot, suggests a fairly balanced distribution with clusters 0 and 1 holding around 41% of customers each and cluster 2 accommodating approximately 18% of the customers. 
# 
# This balanced distribution indicates that our clustering process has been largely successful in identifying meaningful patterns within the data, rather than merely grouping noise or outliers. It implies that each cluster represents a substantial and distinct segment of the customer base, thereby offering valuable insights for future business strategies.
# 
# Moreover, the fact that no cluster contains a very small percentage of customers, assures us that each cluster is significant and not just representing outliers or noise in the data. This setup allows for a more nuanced understanding and analysis of different customer segments, facilitating effective and informed decision-making.
# 

# %% [markdown]
# <a id="evaluations_metrics"></a>
# # <b><span style='color:#fcc36d'>Step 10.3 |</span><span style='color:#ff6200'> Evaluation Metrics</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# To further scrutinize the quality of our clustering, I will employ the following metrics:
# 
# - __Silhouette Score__: A measure to evaluate the separation distance between the clusters. Higher values indicate better cluster separation. It ranges from -1 to 1.
#     
#     
# - __Calinski Harabasz Score__: This score is used to evaluate the dispersion between and within clusters. A higher score indicates better defined clusters.
# 
#     
# - __Davies Bouldin Score__: It assesses the average similarity between each cluster and its most similar cluster. Lower values indicate better cluster separation.

# %%
# Compute number of customers
num_observations = len(customer_data_pca)

# Separate the features and the cluster labels
X = customer_data_pca.drop('cluster', axis=1)
clusters = customer_data_pca['cluster']

# Compute the metrics
sil_score = silhouette_score(X, clusters)
calinski_score = calinski_harabasz_score(X, clusters)
davies_score = davies_bouldin_score(X, clusters)

# Create a table to display the metrics and the number of observations
table_data = [
    ["Number of Observations", num_observations],
    ["Silhouette Score", sil_score],
    ["Calinski Harabasz Score", calinski_score],
    ["Davies Bouldin Score", davies_score]
]

# Print the table
print(tabulate(table_data, headers=["Metric", "Value"], tablefmt='pretty'))

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# <h3 align="left"><font color=#ff6200>Clustering Quality Inference</font></h3>
#     
#     
# - The __Silhouette Score__ of approximately 0.236, although not close to 1, still indicates a fair amount of separation between the clusters. It suggests that the clusters are somewhat distinct, but there might be slight overlaps between them. Generally, a score closer to 1 would be ideal, indicating more distinct and well-separated clusters.
# 
#     
# - The __Calinski Harabasz Score__ is 1257.17, which is considerably high, indicating that the clusters are well-defined. A higher score in this metric generally signals better cluster definitions, thus implying that our clustering has managed to find substantial structure in the data.
# 
#     
# - The __Davies Bouldin Score__ of 1.37 is a reasonable score, indicating a moderate level of similarity between each cluster and its most similar one. A lower score is generally better as it indicates less similarity between clusters, and thus, our score here suggests a decent separation between the clusters.
# 
# 
# In conclusion, the metrics suggest that the clustering is of good quality, with clusters being well-defined and fairly separated. However, there might still be room for further optimization to enhance cluster separation and definition, potentially by trying other clustering and dimensionality reduction algorithms.
# 
# 

# %% [markdown]
# <a id="profiling"></a>
# # <p style="background-color: #ff6200; font-family:calibri; color:white; font-size:140%; font-family:Verdana; text-align:center; border-radius:15px 50px;">Step 11 | Cluster Analysis and Profiling</p>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# In this section, I am going to analyze the characteristics of each cluster to understand the distinct behaviors and preferences of different customer segments and also profile each cluster to identify the key traits that define the customers in each cluster.

# %% [markdown]
# <a id="radar_chart"></a>
# # <b><span style='color:#fcc36d'>Step 11.1 |</span><span style='color:#ff6200'> Radar Chart Approach</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# First of all, I am going to create radar charts to visualize the centroid values of each cluster across different features. This can give a quick visual comparison of the profiles of different clusters.To construct the radar charts, it's essential to first compute the centroid for each cluster. This centroid represents the mean value for all features within a specific cluster. Subsequently, I will display these centroids on the radar charts, facilitating a clear visualization of the central tendencies of each feature across the various clusters:

# %%
# Setting 'CustomerID' column as index and assigning it to a new dataframe
df_customer = customer_data_cleaned.set_index('CustomerID')

# Standardize the data (excluding the cluster column)
scaler = StandardScaler()
df_customer_standardized = scaler.fit_transform(df_customer.drop(columns=['cluster'], axis=1))

# Create a new dataframe with standardized values and add the cluster column back
df_customer_standardized = pd.DataFrame(df_customer_standardized, columns=df_customer.columns[:-1], index=df_customer.index)
df_customer_standardized['cluster'] = df_customer['cluster']

# Calculate the centroids of each cluster
cluster_centroids = df_customer_standardized.groupby('cluster').mean()

# Function to create a radar chart
def create_radar_chart(ax, angles, data, color, cluster):
    # Plot the data and fill the area
    ax.fill(angles, data, color=color, alpha=0.4)
    ax.plot(angles, data, color=color, linewidth=2, linestyle='solid')
    
    # Add a title
    ax.set_title(f'Cluster {cluster}', size=20, color=color, y=1.1)

# Set data
labels=np.array(cluster_centroids.columns)
num_vars = len(labels)

# Compute angle of each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The plot is circular, so we need to "complete the loop" and append the start to the end
labels = np.concatenate((labels, [labels[0]]))
angles += angles[:1]

# Initialize the figure
fig, ax = plt.subplots(figsize=(20, 10), subplot_kw=dict(polar=True), nrows=1, ncols=3)

# Create radar chart for each cluster
for i, color in enumerate(colors):
    data = cluster_centroids.loc[i].tolist()
    data += data[:1]  # Complete the loop
    create_radar_chart(ax[i], angles, data, color, i)

# Add input data
ax[0].set_xticks(angles[:-1])
ax[0].set_xticklabels(labels[:-1])

ax[1].set_xticks(angles[:-1])
ax[1].set_xticklabels(labels[:-1])

ax[2].set_xticks(angles[:-1])
ax[2].set_xticklabels(labels[:-1])

# Add a grid
ax[0].grid(color='grey', linewidth=0.5)

# Display the plot
plt.tight_layout()
plt.show()

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# <h2 align="left"><font color=#ff6200>Customer Profiles Derived from Radar Chart Analysis</font></h2>
#     
# <h3 align="left"><font color=red>Cluster 0 (Red Chart):</font></h3>
# 
# 🎯 Profile: __Sporadic Shoppers with a Preference for Weekend Shopping__  
# 
# - Customers in this cluster tend to spend less, with a lower number of transactions and products purchased.  
# - They have a slight tendency to shop during the weekends, as indicated by the very high `Day_of_Week` value.  
# - Their spending trend is relatively stable but on the lower side, and they have a low monthly spending variation (low `Monthly_Spending_Std`).  
# - These customers have not engaged in many cancellations, showing a low cancellation frequency and rate.  
# - The average transaction value is on the lower side, indicating that when they do shop, they tend to spend less per transaction.  
# 
# ____
#     
# <h3 align="left"><font color=green>Cluster 1 (Green Chart):</font></h3>    
#  
# 🎯 Profile: __Infrequent Big Spenders with a High Spending Trend__  
#     
# - Customers in this cluster show a moderate level of spending, but their transactions are not very frequent, as indicated by the high `Days_Since_Last_Purchase` and `Average_Days_Between_Purchases`.  
# - They have a very high spending trend, indicating that their spending has been increasing over time.  
# - These customers prefer shopping late in the day, as indicated by the high `Hour` value, and they mainly reside in the UK.  
# - They have a tendency to cancel a moderate number of transactions, with a medium cancellation frequency and rate.  
# - Their average transaction value is relatively high, meaning that when they shop, they tend to make substantial purchases.  
# 
# ____
#     
# <h3 align="left"><font color=blue>Cluster 2 (Blue Chart):</font></h3>   
# 
# 🎯 Profile: __Frequent High-Spenders with a High Rate of Cancellations__
#     
# - Customers in this cluster are high spenders with a very high total spend, and they purchase a wide variety of unique products.  
# - They engage in frequent transactions, but also have a high cancellation frequency and rate.  
# - These customers have a very low average time between purchases, and they tend to shop early in the day (low `Hour` value).  
# - Their monthly spending shows high variability, indicating that their spending patterns might be less predictable compared to other clusters.  
# - Despite their high spending, they show a low spending trend, suggesting that their high spending levels might be decreasing over time.  

# %% [markdown]
# <a id="histogram"></a>
# # <b><span style='color:#fcc36d'>Step 11.2 |</span><span style='color:#ff6200'> Histogram Chart Approach</span></b>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
#     
# To validate the profiles identified from the radar charts, we can plot histograms for each feature segmented by the cluster labels. These histograms will allow us to visually inspect the distribution of feature values within each cluster, thereby confirming or refining the profiles we have created based on the radar charts.

# %%
# Plot histograms for each feature segmented by the clusters
features = customer_data_cleaned.columns[1:-1]
clusters = customer_data_cleaned['cluster'].unique()
clusters.sort()

# Setting up the subplots
n_rows = len(features)
n_cols = len(clusters)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3*n_rows))

# Plotting histograms
for i, feature in enumerate(features):
    for j, cluster in enumerate(clusters):
        data = customer_data_cleaned[customer_data_cleaned['cluster'] == cluster][feature]
        axes[i, j].hist(data, bins=20, color=colors[j], edgecolor='w', alpha=0.7)
        axes[i, j].set_title(f'Cluster {cluster} - {feature}', fontsize=15)
        axes[i, j].set_xlabel('')
        axes[i, j].set_ylabel('')

# Adjusting layout to prevent overlapping
plt.tight_layout()
plt.show()

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# The detailed insights from the histograms provide a more nuanced understanding of each cluster, helping in refining the profiles to represent the customer behaviors more accurately. Based on the detailed analysis from both the radar charts and the histograms, here are the refined profiles and titles for each cluster:

# %% [markdown]
# <img src="https://github.com/FarzadNekouee/Retail_Customer_Segmentation_Recommendation_System/blob/master/profiles.png?raw=true" width="2400">

# %% [markdown]
# <a id="recommendation_system"></a>
# # <p style="background-color: #ff6200; font-family:calibri; color:white; font-size:140%; font-family:Verdana; text-align:center; border-radius:15px 50px;">Step 12 | Recommendation System</p>
# ⬆️ [Tabel of Contents](#contents_tabel)

# %% [markdown]
# <div style="border-radius:10px; padding: 15px; background-color: #ffeacc; font-size:120%; text-align:left">
# 
# In the final phase of this project, I am set to develop a recommendation system to enhance the online shopping experience. This system will suggest products to customers based on the purchasing patterns prevalent in their respective clusters. Earlier in the project, during the customer data preparation stage, I isolated a small fraction (5%) of the customers identified as outliers and reserved them in a separate dataset called `outliers_data`.
# 
# Now, focusing on the core 95% of the customer group, I analyze the cleansed customer data to pinpoint the top-selling products within each cluster. Leveraging this information, the system will craft personalized recommendations, suggesting __the top three products__ popular within their cluster that they have not yet purchased. This not only facilitates targeted marketing strategies but also enriches the personal shopping experience, potentially boosting sales. For the outlier group, a basic approach could be to recommend random products, as a starting point to engage them.

# %%
# Step 1: Extract the CustomerIDs of the outliers and remove their transactions from the main dataframe
outlier_customer_ids = outliers_data['CustomerID'].astype('float').unique()
df_filtered = df[~df['CustomerID'].isin(outlier_customer_ids)]

# Step 2: Ensure consistent data type for CustomerID across both dataframes before merging
customer_data_cleaned['CustomerID'] = customer_data_cleaned['CustomerID'].astype('float')

# Step 3: Merge the transaction data with the customer data to get the cluster information for each transaction
merged_data = df_filtered.merge(customer_data_cleaned[['CustomerID', 'cluster']], on='CustomerID', how='inner')

# Step 4: Identify the top 10 best-selling products in each cluster based on the total quantity sold
best_selling_products = merged_data.groupby(['cluster', 'StockCode', 'Description'])['Quantity'].sum().reset_index()
best_selling_products = best_selling_products.sort_values(by=['cluster', 'Quantity'], ascending=[True, False])
top_products_per_cluster = best_selling_products.groupby('cluster').head(10)

# Step 5: Create a record of products purchased by each customer in each cluster
customer_purchases = merged_data.groupby(['CustomerID', 'cluster', 'StockCode'])['Quantity'].sum().reset_index()

# Step 6: Generate recommendations for each customer in each cluster
recommendations = []
for cluster in top_products_per_cluster['cluster'].unique():
    top_products = top_products_per_cluster[top_products_per_cluster['cluster'] == cluster]
    customers_in_cluster = customer_data_cleaned[customer_data_cleaned['cluster'] == cluster]['CustomerID']
    
    for customer in customers_in_cluster:
        # Identify products already purchased by the customer
        customer_purchased_products = customer_purchases[(customer_purchases['CustomerID'] == customer) & 
                                                         (customer_purchases['cluster'] == cluster)]['StockCode'].tolist()
        
        # Find top 3 products in the best-selling list that the customer hasn't purchased yet
        top_products_not_purchased = top_products[~top_products['StockCode'].isin(customer_purchased_products)]
        top_3_products_not_purchased = top_products_not_purchased.head(3)
        
        # Append the recommendations to the list
        recommendations.append([customer, cluster] + top_3_products_not_purchased[['StockCode', 'Description']].values.flatten().tolist())

# Step 7: Create a dataframe from the recommendations list and merge it with the original customer data
recommendations_df = pd.DataFrame(recommendations, columns=['CustomerID', 'cluster', 'Rec1_StockCode', 'Rec1_Description', \
                                                 'Rec2_StockCode', 'Rec2_Description', 'Rec3_StockCode', 'Rec3_Description'])
customer_data_with_recommendations = customer_data_cleaned.merge(recommendations_df, on=['CustomerID', 'cluster'], how='right')

# %%
# Display 10 random rows from the customer_data_with_recommendations dataframe
customer_data_with_recommendations.set_index('CustomerID').iloc[:, -6:].sample(10, random_state=0)

# %% [markdown]
# <div style="display: flex; align-items: center; justify-content: center; border-radius: 10px; padding: 20px; background-color: #ffeacc; font-size: 120%; text-align: center;">
# 
# <strong>🎯 If you need more information or want to explore the code, feel free to visit the project repository on <a href="https://github.com/FarzadNekouee/Retail_Customer_Segmentation_Recommendation_System">GitHub</a> 🎯</strong>
# </div>
# 

# %% [markdown]
# <h2 align="left"><font color='#ff6200'>Best Regards!</font></h2>


