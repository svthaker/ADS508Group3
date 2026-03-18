#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 00:05:22 2026

@author: jamesshoenhair
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


docs = os.path.expanduser("~/Documents")

df_main   = pd.read_csv(os.path.join(docs, "StateAndCountyData.csv"))
df_vars   = pd.read_csv(os.path.join(docs, "VariableList.csv"))

print(df_main.shape)
print(df_vars.shape)


print(df_vars['Category_Name'].unique())

health_vars = df_vars[df_vars['Category_Name'] == 'Health and Physical Activity']
print(health_vars[['Variable_Name', 'Variable_Code', 'Units']])

def lookup_var(code):
    result = df_vars[df_vars['Variable_Code'] == code]
    if not result.empty:
        return result[['Variable_Name', 'Category_Name', 'Units']].values[0]
    return "Not found"

print(lookup_var('PCT_DIABETES_ADULTS19'))


df_main.head()
df_main.info()
df_main.describe()

missing = df_main.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))
print(df_main[['State', 'County']].head(10))


print(df_main.columns.tolist())
print(df_vars['Variable_Code'].tolist())

def get_var(code):
    return df_main[df_main['Variable_Code'] == code][['State', 'County', 'Value']].copy()

def get_codes_for_category(category_name):
    return df_vars[df_vars['Category_Name'] == category_name]['Variable_Code'].tolist()

obesity = get_var('PCT_OBESE_ADULTS22')
print(obesity.describe())

health_codes = get_codes_for_category('Health and Physical Activity')
print(health_codes)

poverty   = get_var('POVRATE21').rename(columns={'Value': 'POVRATE21'})
diabetes  = get_var('PCT_DIABETES_ADULTS19').rename(columns={'Value': 'PCT_DIABETES_ADULTS19'})

scatter_df = poverty.merge(diabetes, on=['State', 'County'])

obesity = get_var('PCT_OBESE_ADULTS22')

plt.figure(figsize=(8, 5))
sns.histplot(obesity['Value'].dropna(), bins=40, kde=True)
plt.title('Distribution of Adult Obesity Rate by County (2022)')
plt.xlabel('Obesity Rate (%)')
plt.tight_layout()
plt.show()


obesity = get_var('PCT_OBESE_ADULTS22')

plt.figure(figsize=(8, 5))
sns.scatterplot(data=scatter_df, x='POVRATE21', y='PCT_DIABETES_ADULTS19', alpha=0.4)
plt.title('Poverty Rate vs. Adult Diabetes Rate')
plt.xlabel('Poverty Rate (%)')
plt.ylabel('Diabetes Rate (%)')
plt.tight_layout()
plt.show()

plt.xlim(0, 100)
plt.ylim(0, 100)

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(obesity['Value'].dropna(), bins=40, kde=True)
plt.title('Distribution of Adult Obesity Rate by County (2022)')
plt.xlabel('Obesity Rate (%)')
plt.xlim(0, 100)

plt.tight_layout()
plt.show()