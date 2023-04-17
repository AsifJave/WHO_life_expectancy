#!/usr/bin/env python
# coding: utf-8

# # Import and Data Reading

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('Life Expectancy Data.csv')


# # Understanding Data

# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.shape


# In[7]:


df.columns


# In[8]:


df.info()


# In[9]:


#How many countries are included in this data?
len(df.Country.unique())


# In[10]:


#What is the reference period?
df['Year'].agg(['min', 'max'])


# # Data Preparation

# In[11]:


df.isnull().sum()


# In[12]:


# Replacing Null Values with mean of the Data
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean',fill_value=None)
df['Life expectancy ']=imputer.fit_transform(df[['Life expectancy ']])
df['Adult Mortality']=imputer.fit_transform(df[['Adult Mortality']])
df['Alcohol']=imputer.fit_transform(df[['Alcohol']])
df['Hepatitis B']=imputer.fit_transform(df[['Hepatitis B']])
df[' BMI ']=imputer.fit_transform(df[[' BMI ']])
df['Polio']=imputer.fit_transform(df[['Polio']])
df['Total expenditure']=imputer.fit_transform(df[['Total expenditure']])
df['Diphtheria ']=imputer.fit_transform(df[['Diphtheria ']])
df['GDP']=imputer.fit_transform(df[['GDP']])
df['Population']=imputer.fit_transform(df[['Population']])
df[' thinness  1-19 years']=imputer.fit_transform(df[[' thinness  1-19 years']])
df[' thinness 5-9 years']=imputer.fit_transform(df[[' thinness 5-9 years']])
df['Income composition of resources']=imputer.fit_transform(df[['Income composition of resources']])
df['Schooling']=imputer.fit_transform(df[['Schooling']])


# In[13]:


df.isnull().sum()


# In[14]:


## to desrcibe the data
df.describe()


# # Exploratory Data Analysis and Visulization

# In[14]:


# TO CHECK LIFE EXPECTANCY IN DEVELOPING AND DEVELOPED COUNTRIES
fig, ax = plt.subplots(figsize=(8, 6))
ax.violinplot([df[df['Status'] == 'Developed']['Life expectancy '].dropna(),
               df[df['Status'] == 'Developing']['Life expectancy '].dropna()],
              showmedians=True, vert=False, widths=0.7, positions=[0, 1])
ax.set_yticks([0, 1])
ax.set_yticklabels(['Developed', 'Developing'])
ax.set_xlabel('Life expectancy')
ax.set_title('Life expectancy Based on Countries status')
plt.show()


# In[15]:


countries = df.loc[:, ['Country', 'Status']]
distinct_countries = countries.drop_duplicates(['Country'])
grouped_by_status = distinct_countries.groupby('Status').count()
grouped_by_status


# In[16]:


#Life Expectancy over World Map
import plotly.express as px
country_data = px.data.gapminder()
map_fig = px.scatter_geo(country_data,locations = 'iso_alpha', projection = 'orthographic', 
                         opacity = 0.8, color = 'country', hover_name = 'country', 
                         hover_data = ['lifeExp', 'year'],template = 'plotly_dark',title = '<b>Life Expectancy over the World Map')
map_fig.show()


# In[39]:


# Filter the data for the desired columns
#df_filtered = df[['Year', 'infant deaths ', 'Status']]

# Filter the data for the desired year range
#df_filtered = df_filtered[(df_filtered['Year'] >= 2000) & (df_filtered['Year'] <= 2015)]

# Remove any rows with missing values
#df_filtered = df_filtered.dropna()

# Filter the data for developed and developing countries
#df_developed = df_filtered[df_filtered['Status'] == 'Developed']
#df_developing = df_filtered[df_filtered['Status'] == 'Developing']

# Create a scatter plot with regression lines for both developed and developing countries
plt.figure(figsize = (10, 8))
sns.barplot(x = 'Year' , y = 'infant deaths' ,hue ='Status' ,data = df)
#sns.regplot(data=df_developed, x='Year', y='infant deaths ', scatter=False, label='Developed')
#sns.regplot(data=df_developing, x='Year', y='infant deaths ', scatter=False, label='Developing')
plt.title('Infant Deaths from 2000-2015 (Developed vs. Developing Countries)')
plt.xlabel('Year')
plt.ylabel('Infant Deaths')
plt.show()


# In[32]:


plt.figure(figsize = (10, 8))
sns.scatterplot(y = 'Adult Mortality',x = 'Country', data = df)


# In[24]:


# Filter the data for the desired columns
df_filtered = df[['Year', 'Life expectancy ', 'Status']]

# Filter the data for the desired year range
df_filtered = df_filtered[(df_filtered['Year'] >= 2000) & (df_filtered['Year'] <= 2015)]

# Remove any rows with missing values
df_filtered = df_filtered.dropna()

# Filter the data for developed and developing countries
df_developed = df_filtered[df_filtered['Status'] == 'Developed']
df_developing = df_filtered[df_filtered['Status'] == 'Developing']

# Create a scatter plot with regression lines for both developed and developing countries
sns.scatterplot(data=df_filtered, x='Year', y='Life expectancy ', hue='Status', alpha=0.5)
sns.regplot(data=df_developed, x='Year', y='Life expectancy ', scatter=False, label='Developed')
sns.regplot(data=df_developing, x='Year', y='Life expectancy ', scatter=False, label='Developing')
plt.title('Life Expectancy from 2000-2015 (Developed vs. Developing Countries)')
plt.xlabel('Year')
plt.ylabel('Life Expectancy')
plt.legend()
plt.show()


# In[24]:


# Filter the data for the desired columns
df_filtered = df[['Year', 'Adult Mortality', 'Status']]

# Filter the data for the desired year range
df_filtered = df_filtered[(df_filtered['Year'] >= 2000) & (df_filtered['Year'] <= 2015)]

# Filter the data for developed and developing countries
df_developed = df_filtered[df_filtered['Status'] == 'Developed']
df_developing = df_filtered[df_filtered['Status'] == 'Developing']

# Create a scatter plot with regression lines for both developed and developing countries
sns.scatterplot(data=df_filtered, x='Year', y='Adult Mortality', hue='Status', alpha=0.5)
sns.regplot(data=df_developed, x='Year', y='Adult Mortality', scatter=False, label='Developed')
sns.regplot(data=df_developing, x='Year', y='Adult Mortality', scatter=False, label='Developing')
plt.title('Adult Mortality from 2000-2015 (Developed vs. Developing Countries)')
plt.xlabel('Year')
plt.ylabel('Adult Mortality')
plt.legend()
plt.show()


# In[25]:


sns.scatterplot(data =df, x= df['Life expectancy '], y= df['Schooling'])


# In[30]:


# Create the correlation matrix
corr = df.corr()

# Set the figure size
plt.figure(figsize=(40, 40))

# Create a heatmap
sns.heatmap(corr, annot=True, cmap='Greens')

# Show the plot
plt.show()


# In[61]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame containing the data
# Replace the column names and DataFrame as per your specific data

# Create a boxplot with increased figure size
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(data=df, x='Year', y='percentage expenditure', hue='Status', ax=ax)

# Set the title and display the plot
plt.title('BAR plot of Percentage Expenditure by Year and Status')
plt.show()


# In[28]:


df.hist(figsize=(30,20))
plt.show()


# Here we can see, few features showing normal distribution while most have skewness. We will impute median for missing values in skewed distribution and mean for normal distribution, afterward find out the VIF.
# 
# Skewed distributed features: Adult mortality, infant deaths, Alcohol, percentage expenditure, Heptatitis B, Measles, under-five deaths, Polio, Diphtheria, HIV/AIDS, GDP, Population, thiness 1-19 years, thinness 5-9 years.
# 
# Normal distributed features: BMI, Total expenditure, Income composition of resources, Schooling

# In[33]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame containing the data
# Replace the column names and DataFrame as per your specific data

# Sort the DataFrame by the variable of interest (e.g. 'Adult Mortality')
sorted_df = df.sort_values(by='Adult Mortality', ascending=False)

# Get the top 10 countries based on 'Adult Mortality'
top_10_countries = sorted_df.head(40)

# Get the least 10 countries based on 'Adult Mortality'
least_10_countries = sorted_df.tail(40)

# Create a scatter plot for the top 10 countries
sns.scatterplot(x='Country', y='Adult Mortality', data=top_10_countries, size='GDP', hue='Status', style='Status')
plt.title('Top 10 Countries by Adult Mortality')
plt.xlabel('Country')
plt.ylabel('Adult Mortality')
plt.xticks(rotation=45)
plt.show()

# Create a scatter plot for the least 10 countries
sns.scatterplot(x='Country', y='Adult Mortality', data=least_10_countries, size='GDP', hue='Status', style='Status')
plt.title('Least 10 Countries by Adult Mortality')
plt.xlabel('Country')
plt.ylabel('Adult Mortality')
plt.xticks(rotation=90)
plt.show()


# In[26]:


import scipy.stats as stats

# Assume we have two groups: High Income Composition and Low Income Composition
group1 = df[df['Income composition of resources'] > df['Income composition of resources'].median()]
group2 = df[df['Income composition of resources'] <= df['Income composition of resources'].median()]

# Extract Polio vaccination coverage for each group
polio_coverage_group1 = group1['Polio']
polio_coverage_group2 = group2['Polio']

# Perform independent t-test assuming unequal variances
t_stat, p_value = stats.ttest_ind(polio_coverage_group1, polio_coverage_group2, equal_var=False)

# Print the results
print('T-statistic:', t_stat)
print('P-value:', p_value)

# Set significance level
alpha = 0.05

# Check if the p-value is less than the significance level
if p_value < alpha:
    print('Reject the null hypothesis. There is a significant difference in Polio vaccination coverage between countries with different income composition levels.')
else:
    print('Fail to reject the null hypothesis. There is no significant difference in Polio vaccination coverage between countries with different income composition levels.')


# In[18]:


# Group the data by country and calculate the mean healthcare expenditure and life expectancy
healthcare_expenditure = df.groupby('Country')['Total expenditure'].mean()
life_expectancy = df.groupby('Country')['Life expectancy '].mean()

# Create a scatter plot to visualize the relationship between healthcare expenditure and life expectancy
plt.figure(figsize=(10, 6))
sns.scatterplot(y=healthcare_expenditure, x=life_expectancy)
plt.xlabel('Total Expenditure on Health (% of GDP)')
plt.ylabel('Life Expectancy at Birth (years)')
plt.title('Relationship between Healthcare Expenditure and Life Expectancy')
plt.show()


# In[69]:


# Assuming 'df' is your DataFrame containing the data
# Replace the column names and DataFrame as per your specific data

# Create a line plot
plt.figure(figsize=(10, 6))
plt.scatter(df['percentage expenditure'], df['Life expectancy '], )
plt.xlabel('Health Expenditure')
plt.ylabel('Life Expectancy')
plt.title('Plot of Health Expenditure vs. Life Expectancy')
plt.grid(True)
plt.show()


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame containing the data
# Replace the column names and DataFrame as per your specific data

# Create a scatterplot
sns.scatterplot(data=df, x='Alcohol', y='Life expectancy ',  alpha=0.5)

# Set the title and labels
plt.title('Scatterplot of Alcohol vs. Life Expectancy')
plt.xlabel('Alcohol Consumption')
plt.ylabel('Life Expectancy')

# Display the plot
plt.show()


# In[71]:


# Assuming 'df' is your DataFrame containing the data
# Replace the column names as per your specific data

# Calculate correlation
correlation = df['Life expectancy '].corr(df['Population'])

# Print correlation coefficient
print("Correlation between Life Expectancy and GDP:", correlation)


# In[19]:


import pandas as pd



# Extract GDP and Population columns
gdp_pop_df = df[['GDP', 'Population']]

# Calculate correlation
correlation = gdp_pop_df['GDP'].corr(gdp_pop_df['Population'])

# Print the correlation coefficient
print("Correlation between GDP and Population:", correlation)


# # Hypothesis test

# # For Percentage Expenditure and Life Expectancy

# In[74]:


import scipy.stats as stats

# Assuming 'df' is your DataFrame containing the data
# Replace the column names as per your specific data

# Extract data for countries with higher and lower health expenditure
group1 = df[df['percentage expenditure'] == 'High']['Life expectancy ']
group2 = df[df['percentage expenditure'] == 'Low']['Life expectancy ']

# Perform two-sample t-test
t_stat, p_value = stats.ttest_ind(group1, group2)

# Compare p-value with significance level
alpha = 0.05  # Example significance level of 0.05
if p_value < alpha:
    print("Reject the null hypothesis: Countries with higher health expenditure have significantly higher life expectancy.")
else:
    print("Fail to reject the null hypothesis: No significant difference in life expectancy between countries with higher and lower health expenditure.")


# # For Polio in 2000 and 2015 (2)

# In[75]:


import numpy as np
from scipy.stats import ttest_rel

# Assuming 'df' is your DataFrame containing the polio vaccination coverage data
# Replace the column names and DataFrame as per your specific data

# Filter the data for years 2015 and 2000
polio_2015 = df[df['Year'] == 2015]['Polio']
polio_2000 = df[df['Year'] == 2000]['Polio']

# Perform paired t-test
t_statistic, p_value = ttest_rel(polio_2015, polio_2000)

# Print the results
print("Paired t-test results:")
print("t-statistic: ", t_statistic)
print("p-value: ", p_value)

# Set significance level (alpha)
alpha = 0.05

# Check if p-value is less than alpha for significance
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference in polio vaccination coverage between 2015 and 2000.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in polio vaccination coverage between 2015 and 2000.")


# # For Hepatitis in 2015 and 2000

# In[76]:


import numpy as np
from scipy.stats import ttest_rel

# Assuming 'df' is your DataFrame containing the hepatitis prevalence data
# Replace the column names and DataFrame as per your specific data

# Filter the data for years 2015 and 2000
hepatitis_2015 = df[df['Year'] == 2015]['Hepatitis B']
hepatitis_2000 = df[df['Year'] == 2000]['Hepatitis B']

# Perform paired t-test
t_statistic, p_value = ttest_rel(hepatitis_2015, hepatitis_2000)

# Print the results
print("Paired t-test results:")
print("t-statistic: ", t_statistic)
print("p-value: ", p_value)

# Set significance level (alpha)
alpha = 0.05

# Check if p-value is less than alpha for significance
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference in hepatitis prevalence between 2015 and 2000.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in hepatitis prevalence between 2015 and 2000.")


# # For Dipatheria in 2000 and 2015

# In[79]:


import numpy as np
from scipy.stats import ttest_rel

# Assuming 'df' is your DataFrame containing the diphtheria prevalence data
# Replace the column names and DataFrame as per your specific data

# Filter the data for years 2015 and 2000
diphtheria_2015 = df[df['Year'] == 2015]['Diphtheria ']
diphtheria_2000 = df[df['Year'] == 2000]['Diphtheria ']

# Perform paired t-test
t_statistic, p_value = ttest_rel(diphtheria_2015, diphtheria_2000)

# Print the results
print("Paired t-test results:")
print("t-statistic: ", t_statistic)
print("p-value: ", p_value)

# Set significance level (alpha)
alpha = 0.05

# Check if p-value is less than alpha for significance
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference in diphtheria prevalence between 2015 and 2000.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in diphtheria prevalence between 2015 and 2000.")


# # FOr Hepatitis-B in developing and Developed Countries (1)

# In[28]:


from scipy.stats import ttest_ind

# Assuming 'df' is your DataFrame containing the hepatitis prevalence data
# Replace the column names and DataFrame as per your specific data
da["Status"] = df.Status.replace({1: "Developed", 2: "Developing"})

# Filter the data for developing and developed countries separately for 2015 and 2000
hepatitis_2015_developing = df[(df['Year'] == 2015) & (df['Status'] == 'Developing')]['Hepatitis B']
hepatitis_2000_developing = df[(df['Year'] == 2000) & (df['Status'] == 'Developing')]['Hepatitis B']

hepatitis_2015_developed = df[(df['Year'] == 2015) & (df['Status'] == 'Developed')]['Hepatitis B']
hepatitis_2000_developed = df[(df['Year'] == 2000) & (df['Status'] == 'Developed')]['Hepatitis B']

# Perform independent t-test to compare means
t_statistic, p_value = ttest_ind(hepatitis_2015_developing, hepatitis_2000_developing)

# Print the results for developing countries
print("For Developing Countries:")
print("Mean Hepatitis B prevalence in 2015: ", hepatitis_2015_developing.mean())
print("Mean Hepatitis B prevalence in 2000: ", hepatitis_2000_developing.mean())
print("t-statistic: ", t_statistic)
print("p-value: ", p_value)

# Set significance level (alpha)
alpha = 0.05

# Check if p-value is less than alpha for significance
if p_value < alpha:
    print("Reject the null hypothesis. Hepatitis B prevalence was significantly less in 2015 than in 2000 for developing countries.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in Hepatitis B prevalence between 2015 and 2000 for developing countries.")

# Perform independent t-test to compare means for developed countries
t_statistic, p_value = ttest_ind(hepatitis_2015_developed, hepatitis_2000_developed)

# Print the results for developed countries
print("\nFor Developed Countries:")
print("Mean Hepatitis B prevalence in 2015: ", hepatitis_2015_developed.mean())
print("Mean Hepatitis B prevalence in 2000: ", hepatitis_2000_developed.mean())
print("t-statistic: ", t_statistic)
print("p-value: ", p_value)

# Check if p-value is less than alpha for significance
if p_value < alpha:
    print("Reject the null hypothesis. Hepatitis B prevalence was significantly less in 2015 than in 2000 for developed countries.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in Hepatitis B prevalence between 2015 and 2000 for developed countries.")


# In[34]:


# Check if p-value is less than alpha for significance
if p_value < alpha:
    print("Reject the null hypothesis. Hepatitis B prevalence was significantly less in 2015 than in 2000 for developing countries.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in Hepatitis B prevalence between 2015 and 2000 for developing countries.")

# Perform independent t-test to compare means for developed countries
t_statistic, p_value = ttest_ind(hepatitis_2015_developed, hepatitis_2000_developed)

# Print the results for developed countries
print("\nFor Developed Countries:")
print("Mean Hepatitis B prevalence in 2015: ", hepatitis_2015_developed.mean())
print("Mean Hepatitis B prevalence in 2000: ", hepatitis_2000_developed.mean())
print("t-statistic: ", t_statistic)
print("p-value: ", p_value)

# Set significance level (alpha)
alpha = 0.05

# Check if p-value is less than alpha for significance
if p_value < alpha:
    print("Reject the null hypothesis. Hepatitis B prevalence was significantly less in 2015 than in 2000 for developed countries.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in Hepatitis B prevalence between 2015 and 2000 for developed countries.")


# # For Polio VS Income Composition

# In[17]:


import scipy.stats as stats

# Assume we have two groups: High Income Composition and Low Income Composition
group1 = df[df['Income composition of resources'] > df['Income composition of resources'].median()]
group2 = df[df['Income composition of resources'] <= df['Income composition of resources'].median()]

# Extract Polio vaccination coverage for each group
polio_coverage_group1 = group1['Life expectancy ']
polio_coverage_group2 = group2['Life expectancy ']

# Perform independent t-test assuming unequal variances
t_stat, p_value = stats.ttest_ind(polio_coverage_group1, polio_coverage_group2, equal_var=False)

# Print the results
print('T-statistic:', t_stat)
print('P-value:', p_value)

# Set significance level
alpha = 0.05

# Check if the p-value is less than the significance level
if p_value < alpha:
    print('Reject the null hypothesis. There is a significant difference in Polio vaccination coverage between countries with different income composition levels.')
else:
    print('Fail to reject the null hypothesis. There is no significant difference in Polio vaccination coverage between countries with different income composition levels.')


# In[ ]:




