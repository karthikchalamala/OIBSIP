import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import plotly.express as px

df = pd.read_csv("Unemployment in india.csv")
print(df)

print(df.head())

print(df.describe())

print(df.isnull().sum())

df = df.rename(columns={
    df.columns[0]: 'State',
    df.columns[3]: 'EUR',
    df.columns[4]: 'EE',
    df.columns[5]: 'ELPR',
    df.columns[6]: 'Region'
})
print(df.head())

df["State"].unique()

df["Region"].unique()

df.groupby("Region").size()
region_stats = df.groupby(['Region'])[['EUR','EE','ELPR']].mean().reset_index()
region_stats = round(region_stats,2)
print(region_stats)

heat_maps = df[['EUR','EE','ELPR']]
heat_maps = heat_maps.corr()
plt.figure(figsize=(10,6))
sns.set_context('notebook',font_scale=1)
sns.heatmap(heat_maps,annot=True,cmap='summer')

df.columns = ["State","Date","Frequency","EUR","EE","ELPR","Region"]
plt.figure(figsize=(10,8))
plt.title("Unemployment_In_India")
sns.histplot(x="EUR",hue="Region",data=df)
plt.show()

region = df.groupby(["Region"])[["EUR","EE","ELPR"]].mean()
region = pd.DataFrame(region).reset_index()
fig = px.bar(region, x="Region",y="EUR",color="Region",title="Average Unemployment Rate by Region")
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()

unemployment = df[["State","Region","EUR"]]
fig = px.scatter(unemployment, x="Region", y="EUR", color="State",
                 title='Unemployment by Region (EUR)', height=800)
fig.show()
