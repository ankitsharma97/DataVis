import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
# Load the datasets

import random
seed = 42
np.random.seed(seed) 
random.seed(seed)


train = pd.read_csv("./data/DA2023_train.csv", low_memory=False)
train["Date"] = pd.to_datetime(train["Date"], dayfirst=True)

test = pd.read_csv("./data/DA2023_test.csv", low_memory=False)
test["Date"] = pd.to_datetime(test["Date"], dayfirst=True)

stores = pd.read_csv("./data/DA2023_stores.csv", low_memory=False)
stores = stores[stores.columns[:-2]]
stores_train = train.merge(stores, how="left", on="Store")

# Ensure the directory exists
os.makedirs('./media/img', exist_ok=True)

### Part 1.1 EDA for Naive Clustering

def boxplot(x, y, data, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(20, 7))
    box_plot = sns.boxplot(x=x, y=y, data=data)
    sns.set(rc={'figure.figsize': (20, 7)})
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.savefig(save_path)
    plt.close()

# Create boxplots
boxplot('Assortment', 'Sales', stores_train, 'Distribution of Sales by Assortment', 'Assortment', 'Sales', './media/img/boxplot.png')
boxplot('StoreType', 'Sales', stores_train, 'Distribution of Sales by Store Type', 'Store Type', 'Sales', './media/img/boxplot1.png')

### Part 1.2 - Naive Clustering

# Process data
monthly_sales = stores_train.groupby(['Store', stores_train['Date'].dt.to_period('M')])['Sales'].mean().unstack(level=0)
monthly_sales = monthly_sales.set_index(monthly_sales.index.to_timestamp())
monthly_sales_a = stores_train[stores_train['Assortment'] == 'a']['Store'].unique()
monthly_sales_b = stores_train[stores_train['Assortment'] == 'b']['Store'].unique()
monthly_sales_c = stores_train[stores_train['Assortment'] == 'c']['Store'].unique()

# Plotting Assortment A
plt.figure(figsize=(50, 20))
for column in monthly_sales:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_a:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='orange', linewidth=4, alpha=0.7)  # Highlight stores with Assortment 'a'
plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month", fontsize=40)
plt.ylabel("Sales", fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig('./media/img/Assortment.png')
plt.close()

# Plotting Assortment B
plt.figure(figsize=(50, 20))
for column in monthly_sales:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_b:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='red', linewidth=4, alpha=0.7)  # Highlight stores with Assortment 'b'
plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month", fontsize=40)
plt.ylabel("Sales", fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig('./media/img/Assortment1.png')
plt.close()

# Plotting Assortment C
plt.figure(figsize=(50, 20))
for column in monthly_sales:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_c:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='green', linewidth=4, alpha=0.7)  # Highlight stores with Assortment 'c'
plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month", fontsize=40)
plt.ylabel("Sales", fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig('./media/img/Assortment2.png')
plt.close()

# Process data by StoreType
monthly_sales_a = stores_train[stores_train['StoreType'] == 'a']['Store'].unique()
monthly_sales_b = stores_train[stores_train['StoreType'] == 'b']['Store'].unique()
monthly_sales_c = stores_train[stores_train['StoreType'] == 'c']['Store'].unique()
monthly_sales_d = stores_train[stores_train['StoreType'] == 'd']['Store'].unique()

# Plotting StoreType A
plt.figure(figsize=(50, 20))
for column in monthly_sales:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_a:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='orange', linewidth=4, alpha=0.7)  # Highlight stores with StoreType 'a'
plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month", fontsize=40)
plt.ylabel("Sales", fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig('./media/img/Assortment3.png')
plt.close()

# Plotting StoreType B
plt.figure(figsize=(50, 20))
for column in monthly_sales:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_b:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='blue', linewidth=4, alpha=0.7)  # Highlight stores with StoreType 'b'
plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month", fontsize=40)
plt.ylabel("Sales", fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig('./media/img/Assortment4.png')
plt.close()

# Plotting StoreType C
plt.figure(figsize=(50, 20))
for column in monthly_sales:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_c:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='green', linewidth=4, alpha=0.7)  # Highlight stores with StoreType 'c'
plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month", fontsize=40)
plt.ylabel("Sales", fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig('./media/img/Assortment5.png')
plt.close()

# Plotting StoreType D
plt.figure(figsize=(50, 20))
for column in monthly_sales:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_d:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='red', linewidth=4, alpha=0.7)  # Highlight stores with StoreType 'd'
plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month", fontsize=40)
plt.ylabel("Sales", fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig('./media/img/Assortment6.png')
plt.close()


### Part 1.3 - Time Series Clustering

def to_time_series_dataset(data):
    return np.array(data, dtype=np.float64)

# Create an empty DataFrame
info_df = pd.DataFrame(columns=["Store", "no_unique_date", "missing_dates"])

# Checking the missing data and dates
exhaust_dates = pd.date_range(start=train["Date"].min(), end=train["Date"].max())
for store in train["Store"].unique():
    cur_store_df = train[train.Store == store]
    uniq_date = cur_store_df["Date"].unique()
    no_uniq_date = len(uniq_date)
    missing_dates = exhaust_dates.difference(uniq_date)
    
    missing_dates = [date_obj.strftime('%Y%m%d') for date_obj in missing_dates] if len(missing_dates) > 0 else []
    
    cur_info_df = pd.DataFrame({"Store": [store], "no_unique_date": [no_uniq_date], "missing_dates": [missing_dates]})
    info_df = pd.concat([info_df, cur_info_df], ignore_index=True)

# Missing dates stores
missing_date_df = info_df[info_df.no_unique_date != len(train["Date"].unique())]
missing_date_stores = missing_date_df["Store"].values

print("Store with missing dates")
print(missing_date_stores)
print("\nValue counts of missing store")
print(missing_date_df["no_unique_date"].value_counts())
print("\nThe only one that does not have 758 missing")
print(missing_date_df[missing_date_df.no_unique_date != 758])

# Creating a copy of the train DataFrame
temp_train = train.copy()
temp_train.index = temp_train.Date

# Creation of new df to store 
ts_train_data = {}
for store in range(1, train["Store"].max() + 1):
    ts_train_data[store] = temp_train[temp_train.Store == store]["Sales"]

ts_train = pd.concat(ts_train_data, axis=1, keys=ts_train_data.keys())
ts_train.index = exhaust_dates

non_na_20130101 = ts_train.loc["2013-01-01", ts_train.loc["2013-01-01", :] != 0].index.values
train[(train.Date == pd.to_datetime("2013-01-01")) & pd.Series([store in non_na_20130101 for store in train.Store])]

# Replacing missing values for Store 988
ts_train.loc["2013-01-01", 988] = ts_train.loc["2013-01-01", :].mode().values[0]
ts_train[988]

missing_date_stores = list(np.delete(missing_date_stores, np.where(missing_date_stores == 988)))

# Preparing the ts dataset 
ts_train_758 = []
index_758 = []
ts_train_941 = []
index_941 = []

for store in range(1, train["Store"].max() + 1):
    if store in missing_date_stores:
        ts_train_758.append(ts_train[store].dropna().tolist())
        index_758.append(store) 
    else:
        ts_train_941.append(ts_train[store].tolist())
        index_941.append(store) 

# Creating time series dataset and scaling 
ts_train_758 = to_time_series_dataset(ts_train_758)
ts_train_758 = TimeSeriesScalerMeanVariance().fit_transform(ts_train_758)

ts_train_941 = to_time_series_dataset(ts_train_941)
ts_train_941 = TimeSeriesScalerMeanVariance().fit_transform(ts_train_941)

# Clustering for ts_train_758
km = TimeSeriesKMeans(n_clusters=2, verbose=True, random_state=seed, metric="dtw", max_iter=10, n_jobs=-1)
y_pred_758 = km.fit_predict(ts_train_758)
sz_758 = ts_train_758.shape[1]

plt.figure(figsize=(20, 10))
for yi in range(2):
    plt.subplot(2, 1, yi + 1)
    for xx in ts_train_758[y_pred_758 == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz_758)
    plt.ylim(-4, 4)
    plt.text(0.5, 1.02, 'Cluster %d' % (yi + 1), transform=plt.gca().transAxes)
    
plt.savefig('./media/img/cluster.png')

# Clustering for ts_train_941
km = TimeSeriesKMeans(n_clusters=2, verbose=True, random_state=seed, metric="dtw", max_iter=10, n_jobs=-1)
y_pred_941 = km.fit_predict(ts_train_941)
sz_941 = ts_train_941.shape[1]

plt.figure(figsize=(20, 15))
plt.title("DTW $k$-means for without missing dates")
for yi in range(2):
    plt.subplot(2, 1, yi + 1)
    for xx in ts_train_941[y_pred_941 == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz_941)
    plt.ylim(-4, 4)
    plt.text(0.5, 1.02, 'Cluster %d' % (yi + 1), transform=plt.gca().transAxes)
    
plt.savefig('./media/img/cluster1.png')

#creation of dataframe to store the clusters 
indexes = index_941 + index_758
y_pred_758_new = y_pred_758 + len(set(y_pred_941)) 
y_pred_all = np.append(y_pred_941, y_pred_758_new)
model_cluster = pd.DataFrame({"Store":indexes, "Cluster":y_pred_all})
print(model_cluster["Cluster"].value_counts().sort_index())
model_cluster.head()

#merge the cluster result to stores_train
stores_train_cluster = stores_train.merge(model_cluster, how = "left", on = "Store")
stores_train_cluster.head()

monthly_sales = stores_train_cluster.groupby(['Store',stores_train['Date'].dt.to_period('M')])['Sales'].mean().unstack(level=0)
monthly_sales = monthly_sales.set_index(monthly_sales.index.to_timestamp())
monthly_sales_a = stores_train_cluster[stores_train_cluster['Cluster']==0]['Store'].unique()
monthly_sales_b = stores_train_cluster[stores_train_cluster['Cluster']==1]['Store'].unique()
monthly_sales_c = stores_train_cluster[stores_train_cluster['Cluster']==2]['Store'].unique()
monthly_sales_d = stores_train_cluster[stores_train_cluster['Cluster']==3]['Store'].unique()
monthly_sales_e = stores_train_cluster[stores_train_cluster['Cluster']==4]['Store'].unique()

#### Time Series Clustering Result
plt.figure(figsize=(50,20))

for column in monthly_sales:
    plt.plot(monthly_sales.index,monthly_sales[column],marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_a:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='red', linewidth=4, alpha=0.7) #Here highlights the store with highest average sales, we may further use this plot to highlight certain stores' monthly sale in the future

plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month",fontsize=40)
plt.ylabel("Sale",fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig('./media/img/cluster2.png')

plt.figure(figsize=(50,20))

for column in monthly_sales:
    plt.plot(monthly_sales.index,monthly_sales[column],marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_b:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='orange', linewidth=4, alpha=0.7) #Here highlights the store with highest average sales, we may further use this plot to highlight certain stores' monthly sale in the future

plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month",fontsize=40)
plt.ylabel("Sale",fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig('./media/img/cluster3.png')

plt.figure(figsize=(50,20))

for column in monthly_sales:
    plt.plot(monthly_sales.index,monthly_sales[column],marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_c:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='green', linewidth=4, alpha=0.7) #Here highlights the store with highest average sales, we may further use this plot to highlight certain stores' monthly sale in the future

plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month",fontsize=40)
plt.ylabel("Sale",fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig('./media/img/cluster4.png')

plt.figure(figsize=(50,20))

for column in monthly_sales:
    plt.plot(monthly_sales.index,monthly_sales[column],marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_d:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='blue', linewidth=4, alpha=0.7) #Here highlights the store with highest average sales, we may further use this plot to highlight certain stores' monthly sale in the future

plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month",fontsize=40)
plt.ylabel("Sale",fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig('./media/img/cluster5.png')

plt.figure(figsize=(50,20))

for column in monthly_sales:
    plt.plot(monthly_sales.index,monthly_sales[column],marker='', color='grey', linewidth=1, alpha=0.4)
for column in monthly_sales_e:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='purple', linewidth=4, alpha=0.7) #Here highlights the store with highest average sales, we may further use this plot to highlight certain stores' monthly sale in the future

plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month",fontsize=40)
plt.ylabel("Sale",fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig('./media/img/cluster6.png')

#### The clusters of the missing records stores are similar, we should group them to one cluster


model_cluster['Cluster'] = model_cluster['Cluster'].replace(4,3)
model_cluster['Cluster'].head()

abnormal= monthly_sales.min().to_frame()
zero_abnormal = abnormal[abnormal[0]==0].index

plt.figure(figsize=(50,20))

for column in monthly_sales:
    plt.plot(monthly_sales.index,monthly_sales[column],marker='', color='grey', linewidth=1, alpha=0.4)
for column in zero_abnormal:
    plt.plot(monthly_sales.index, monthly_sales[column], marker='', color='black', linewidth=4, alpha=0.7) #Here highlights the store with highest average sales, we may further use this plot to highlight certain stores' monthly sale in the future

plt.title("Monthly Average Sales by Stores", fontsize=40)
plt.xlabel("Month",fontsize=40)
plt.ylabel("Sale",fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.savefig('./media/img/cluster7.png')




