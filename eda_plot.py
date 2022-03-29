import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

df_train = pd.read_csv('train.csv', sep=',')
df_test = pd.read_csv('test.csv', sep=',')
print(df_train.shape, df_test.shape)

# figure of outliers
# features = ['district','no_of_units','built_year','area_size']
data = pd.concat([df_train['price'], df_train['area_size']], axis=1)
fig = px.scatter(data, x='area_size', y='price')
fig.add_trace(go.Scatter(x=[11195,15000], y=[2035000,20350000], mode = 'markers',
                         marker_symbol = 'star',
                         marker_color = 'red',
                         marker_size = 15, name='outliers'))
fig.update_layout(
    xaxis_title="area_size",
    yaxis_title="price",
    font=dict(
        size=14
    )
)                         
fig.write_image('images/areasize-price.png')
# fig.show()

data = pd.concat([df_train['price'], df_train['no_of_units']], axis=1)
fig = px.scatter(data, x='no_of_units', y='price')
fig.add_trace(go.Scatter(x=[1129], y=[74800000], mode = 'markers',
                         marker_symbol = 'star',
                         marker_color = 'red',
                         marker_size = 15, name='outliers'))
fig.update_layout(
    xaxis_title="no_of_units",
    yaxis_title="price",
    font=dict(
        size=14
    )
)                         
fig.write_image('images/noofunits-price.png')
# fig.show()


data = pd.concat([df_train['price'], df_train['built_year']], axis=1)
fig = px.scatter(data, x='built_year', y='price')
fig.add_trace(go.Scatter(x=[1976, 2011], y=[60500000, 74800000], mode = 'markers',
                         marker_symbol = 'star',
                         marker_color = 'red',
                         marker_size = 15, name='outliers'))
fig.update_layout(
    xaxis_title="built_year",
    yaxis_title="price",
    font=dict(
        size=14
    )
)                         
fig.write_image('images/builtyear-price.png')
# fig.show()

data = pd.concat([df_train['price'], df_train['district']], axis=1)
fig = px.scatter(data, x='district', y='price')
fig.add_trace(go.Scatter(x=[4], y=[74800000], mode = 'markers',
                         marker_symbol = 'star',
                         marker_color = 'red',
                         marker_size = 15, name='outliers'))
fig.update_layout(
    xaxis_title="district",
    yaxis_title="price",
    font=dict(
        size=14
    )
)                         
fig.write_image('images/district-price.png')
# fig.show()
# df_train[(df_train['district']==4) & (df_train['price']==74800000)] # 2168,16294,19026
# df_train[(df_train['no_of_units']==1129) & (df_train['price']==74800000)] # 2168,16294,19026
# df_train[(df_train['built_year']==1976) & (df_train['price']==60500000)] # 22378
# df_train[(df_train['built_year']==2011) & (df_train['price']==74800000)] # 2168,16294,19026
# df_train[(df_train['area_size']==11195) & (df_train['price']==2035000)] # 13181
# df_train[(df_train['area_size']==15000) & (df_train['price']==20350000)] #4402
# df_train[(df_train['area_size'].isna())] # 2415, 12891

# Visualize missing values
f = plt.figure(figsize=(8,7))
ax1 = plt.subplot(1, 2, 1)
sns.set_style("white")
sns.set_color_codes(palette='deep')
missing = round(df_train.isnull().mean()*100,2)
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar(color="b")
ax1.set_ylabel("Percent of missing values", fontsize=16)
ax1.set_xlabel("Features", fontsize=14)
ax1.set_title("Percent missing data in train set", size=16)
sns.despine(trim=True, left=True)

ax2 = plt.subplot(1, 2, 2, sharey=ax1)
sns.set_style("white")
sns.set_color_codes(palette='deep')
missing = round(df_test.isnull().mean()*100,2)
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar(color="r")
ax2.set_ylabel("Percent of missing values", fontsize=16)
ax2.set_xlabel("Features", fontsize=14)
ax2.set_title("Percent missing data in test set", size=16)
sns.despine(trim=True, left=True)
plt.savefig('images/Missing.png', dpi=600, bbox_inches='tight')
# plt.show()

# bar/box plot 
indices = df_train['bedrooms'].str.contains('\+', na=False)
rooms = df_train.bedrooms[indices].to_numpy()
f = lambda x: int(x[0]) + int(x[2])
rooms = [f(r) for r in rooms]
df_train.loc[indices, 'bedrooms'] = rooms
df_train.loc[df_train['bedrooms'].notna(), 'bedrooms'] = df_train[df_train['bedrooms'].notna()].bedrooms.astype(str).astype(int)
df_train['bedrooms'] = df_train['bedrooms'].astype(float)

f = plt.figure(figsize=(14,6))
ax1 = plt.subplot(1, 2, 1)
var = 'bedrooms'
data = pd.concat([df_train['price'], df_train[var]], axis=1)
b1 = sns.barplot(x=var, y="price", data=data)
b1.set_xlabel("bedrooms",fontsize=16)
b1.set_ylabel("price",fontsize=16)

ax2 = plt.subplot(1, 2, 2, sharey=ax1)
var = 'bathrooms'
data = pd.concat([df_train['price'], df_train[var]], axis=1)
b2 = sns.barplot(x=var, y="price", data=data)
b2.set_xlabel("bathrooms",fontsize=16)
b2.set_ylabel("price",fontsize=16)
plt.savefig('images/bedrooms_bathrooms-bar.png', dpi=600, bbox_inches='tight')
# plt.show()

sns.set_style("white")
sns.set_color_codes(palette='deep')
f = plt.figure(figsize=(14,6))
ax1 = plt.subplot(1, 2, 1)
var = 'district'
data = pd.concat([df_train['price'], df_train[var]], axis=1)
b1 = sns.boxplot(x=var, y="price", data=data)
b1.set_xlabel("district",fontsize=16)
b1.set_ylabel("price",fontsize=16)

ax2 = plt.subplot(1, 2, 2, sharey=ax1)
var = 'region'
data = pd.concat([df_train['price'], df_train[var]], axis=1)
b2 = sns.boxplot(x=var, y="price", data=data)
b2.set_xlabel("region",fontsize=16)
b2.set_ylabel("price",fontsize=16)
plt.savefig('images/district_region.png', dpi=600, bbox_inches='tight')
# plt.show()

var = 'built_year'
data = pd.concat([df_train['price'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="price", data=data)
fig.set_xlabel("built_year",fontsize=16)
fig.set_ylabel("price",fontsize=16)
plt.xticks(rotation=90)
plt.savefig('images/built_year.png', dpi=600, bbox_inches='tight')
# plt.show()

# pairplot
sns.set(font_scale = 1.5)
cols = ['price', 'bedrooms','bathrooms','built_year','area_size']
sns.pairplot(df_train[cols], height = 2.5)
plt.savefig('images/pairplot.png', dpi=600, bbox_inches='tight')
# plt.show()

# visualization of log transformation
f = plt.figure(figsize=(12,10))
sns.set_style("white")
sns.set_color_codes(palette='deep')
ax1 = plt.subplot(2, 2, 1)
#Check the new distribution 
sns.distplot(df_train['price'], color="b");
ax1.set(ylabel="Frequency")
ax1.set(xlabel="Price")
ax1.set(title="Price distribution")

ax2 = plt.subplot(2, 2, 2)
res = stats.probplot(df_train['price'], plot=plt)

ax3 = plt.subplot(2, 2, 3)
sns.distplot(np.log(df_train['price']) , fit=norm, color="b");

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(np.log(df_train['price']))
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best',fontsize=11)
ax3.set(ylabel="Frequency")
ax3.set(xlabel="Price")
ax3.set(title="log(Price) distribution")

ax4 = plt.subplot(2, 2, 4)
res = stats.probplot(np.log(df_train['price']), plot=plt)
sns.despine(trim=True, left=True)
plt.tight_layout()
plt.savefig('images/price-dist.png', dpi=600, bbox_inches='tight')
# plt.show()

# correlation plot
df_train = pd.read_csv('train_concat_0326.csv', sep=',')
style.use('ggplot')
sns.set_style('whitegrid')
plt.subplots(figsize = (30,20))
data_corr = df_train.corr()
mask = np.zeros_like(data_corr, dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(data_corr, square = True, cmap=sns.diverging_palette(20, 220, n=200), mask = mask, linecolor = 'white', annot = True, center = 0)
plt.title("Heatmap of Features", fontsize = 30)
plt.savefig('images/corr.png', dpi=600, bbox_inches='tight')
# plt.show()