#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 00:23:13 2026

@author: jamesshoenhair
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df_places = pd.read_csv(os.path.expanduser("~/Downloads/raw/PLACES_County_Data.csv"), low_memory=False)

print(df_places.shape)
print(df_places.columns.tolist())


print(df_places.shape)
print(df_places.dtypes)
df_places.head()

missing = df_places.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))


print(df_places['Category'].unique())
print(df_places['MeasureId'].unique())

measure_lookup = df_places[['MeasureId', 'Measure']].drop_duplicates()
print(measure_lookup.to_string())


df_places.groupby('MeasureId')['Data_Value'].describe()



health_measures = ['OBESITY', 'DIABETES', 'LPA']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, measure in zip(axes, health_measures):
    data = df_places[df_places['MeasureId'] == measure]['Data_Value'].dropna()
    sns.histplot(data, bins=40, kde=True, ax=ax)
    ax.set_title(f'Distribution: {measure}')
    ax.set_xlabel('Prevalence (%)')
    ax.set_xlim(0, 100)

plt.tight_layout()
plt.show()


obesity_by_state = df_places[df_places['MeasureId'] == 'OBESITY'] \
    .groupby('StateAbbr')['Data_Value'].mean() \
    .sort_values(ascending=False) \
    .head(10)

#Top 10 States

plt.figure(figsize=(10, 5))
sns.barplot(x=obesity_by_state.index, y=obesity_by_state.values)
plt.title('Top 10 States by Average County Obesity Rate')
plt.xlabel('State')
plt.ylabel('Avg Obesity Rate (%)')
plt.ylim(0, 100)
plt.tight_layout()
plt.show()


#CA County Specfic

ca_places = df_places[df_places['StateAbbr'] == 'CA']

ca_measures = ['OBESITY', 'DIABETES', 'LPA']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, measure in zip(axes, ca_measures):
    ca_data = ca_places[ca_places['MeasureId'] == measure] \
        .groupby('LocationName')['Data_Value'].mean() \
        .sort_values(ascending=False) \
        .head(20)
        
        
    sns.barplot(x=ca_data.values, y=ca_data.index, ax=ax)
    ax.set_title(f'California Counties: {measure}')
    ax.set_xlabel('Prevalence (%)')
    ax.set_ylabel('County')
    ax.set_xlim(0, 100)

plt.suptitle('Top 20 California Counties by Health Measure', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()


#CA Specific Correlation Between Health Outcomes

ca_wide = df_places[df_places['StateAbbr'] == 'CA'].pivot_table(
    index=['StateAbbr', 'LocationName'],
    columns='MeasureId',
    values='Data_Value'
).reset_index()

focus = ['OBESITY', 'DIABETES', 'LPA', 'CSMOKING', 'BPHIGH']
focus = [col for col in focus if col in ca_wide.columns]

plt.figure(figsize=(8, 6))
sns.heatmap(
    ca_wide[focus].corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    vmin=-1,
    vmax=1
)
plt.title('Correlation Between Health Outcomes\nCalifornia Counties')
plt.tight_layout()
plt.show()

print(f"California counties included: {ca_wide['LocationName'].nunique()}")
