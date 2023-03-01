#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# In[ ]:


# Read in the data
data = pd.read_csv(r'C:\Users\Abhi\OneDrive\Desktop\Suicides in India 2001-2012.csv')


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# In[7]:


data.head()


# In[8]:


data.describe()


# In[13]:


# Get the counts of suicides by year
year_counts = data.groupby('Year')['Total'].sum()


# In[14]:



# Plot the counts of suicides by year
plt.plot(year_counts.index, year_counts.values)
plt.title('Suicides by Year')
plt.xlabel('Year')
plt.ylabel('Total Suicides')
plt.show()


# In[18]:


# Get the counts of suicides by state
state_counts = data.groupby('State')['Total'].sum().sort_values(ascending=False)

# Plot the top 5 states by suicide count
top5_states = state_counts.head(5)
plt.bar(top5_states.index, top5_states.values)
plt.title('Top 5 States by Suicide Count')
plt.xlabel('State')
plt.ylabel('Total Suicides')
plt.show()


# In[20]:


# Conduct hypothesis testing to compare suicide counts by gender
from scipy import stats
male_data = data[data['Gender'] == 'Male']['Total']
female_data = data[data['Gender'] == 'Female']['Total']
t, p = stats.ttest_ind(male_data, female_data)
print('t-statistic: ', t)
print('p-value: ', p)


# # The hypothesis test is comparing the suicide counts between two groups: male and female. The null hypothesis states that there is no significant difference in suicide counts between male and female populations, while the alternative hypothesis states that there is a significant difference in suicide counts between the two populations.
# 
# # The outcome of the hypothesis test shows a t-statistic of 9.455 and a p-value of 3.231e-21. The t-statistic measures the difference between the means of the two groups relative to the variation within the groups. The larger the absolute value of the t-statistic, the greater the difference between the two groups.
# 
# # The p-value represents the probability of obtaining the observed results or more extreme results if the null hypothesis is true. In this case, the p-value is very small (less than 0.05), indicating strong evidence against the null hypothesis. This means that we reject the null hypothesis and conclude that there is a significant difference in suicide counts between male and female populations.
# 
# # In summary, the outcome of the hypothesis test indicates that there is strong evidence to suggest that there is a significant difference in suicide counts between male and female populations, with males having a higher suicide rate than females.

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Read in the data
data = pd.read_csv(r'C:\Users\Abhi\OneDrive\Desktop\Suicides in India 2001-2012.csv')

# Convert the Year column to a datetime object
data['Year'] = pd.to_datetime(data['Year'], format='%Y')

# Create pivot table
pivot_table = pd.pivot_table(data, values='Total', index='Year', columns='Gender', aggfunc='sum')

# Rename the columns
pivot_table = pivot_table.rename(columns={'Male': 'Total Male Suicides', 'Female': 'Total Female Suicides'})

# Convert pivot table to a new DataFrame
suicide_data = pd.DataFrame(pivot_table)

# Fit the ARIMA models
model_male = sm.tsa.ARIMA(suicide_data['Total Male Suicides'], order=(1,1,1)).fit()
model_female = sm.tsa.ARIMA(suicide_data['Total Female Suicides'], order=(1,1,1)).fit()

# Forecast future values
forecast_years = pd.date_range(start='2013', end='2023', freq='Y')
forecast_male = model_male.predict(start=len(suicide_data), end=len(suicide_data)+len(forecast_years)-1)
forecast_female = model_female.predict(start=len(suicide_data), end=len(suicide_data)+len(forecast_years)-1)

# Combine the forecasts into a single DataFrame
forecasts = pd.DataFrame({'Year': forecast_years, 'Total Male Suicides': forecast_male.values, 'Total Female Suicides': forecast_female.values})

# Print the forecasts
print(forecasts)

# Evaluate the frequency of the models
print("Male model AIC: ", model_male.aic)
print("Male model BIC: ", model_male.bic)
print("Female model AIC: ", model_female.aic)
print("Female model BIC: ", model_female.bic)


# In[4]:


# Set the Year column as the index of the forecasts DataFrame
forecasts.set_index('Year', inplace=True)

# Convert the y-axis values to thousands
forecasts['Total Male Suicides'] /= 1000
forecasts['Total Female Suicides'] /= 1000

# Create a line graph of the forecasted values
plt.plot(forecasts.index, forecasts['Total Male Suicides'], label='Total Male Suicides')
plt.plot(forecasts.index, forecasts['Total Female Suicides'], label='Total Female Suicides')
plt.legend()
plt.title('Forecasted Total Suicides by Gender')
plt.xlabel('Year')
plt.ylabel('Total Suicides (in thousands)')
plt.show()


# In[21]:


# Select the 'Total Male Suicides' column as the dependent variable
y = suicide_data['Total Male Suicides']

# Create a variable for the independent variable (year) in datetime format
x = sm.add_constant(suicide_data.index.to_series().apply(lambda x: x.to_pydatetime().year))

# Fit the linear regression model
model = sm.OLS(y, x).fit()

# Print the summary of the model
print(model.summary())


# In[23]:


# Generate a scatter plot of the data and the regression line
sns.regplot(x=x.iloc[:, 1], y=y, line_kws={'color': 'red'})
plt.xlabel('Year')
plt.ylabel('Total Male Suicides')
plt.title('Linear Regression Model for Total Male Suicides in India')
plt.show()

sns.regplot(x=x.iloc[:, 1], y=y, line_kws={'color': 'blue'})
plt.xlabel('Year')
plt.ylabel('Total Female Suicides')
plt.title('Linear Regression Model for Total Male Suicides in India')
plt.show()
# Generate a residual plot
residuals = model.resid
sns.residplot(x=x.iloc[:, 1], y=residuals, lowess=True, line_kws={'color': 'red'})
plt.xlabel('Year')
plt.ylabel('Residuals')
plt.title('Residual Plot for Linear Regression Model')
plt.show()


# In[24]:


# Print the r-squared value and MSE
print('R-squared:', model.rsquared)
print('MSE:', model.mse_resid)


# The R-squared value of 0.969 indicates that about 96.9% of the variance in the total number of male suicides in India is explained by the linear regression model with year as the independent variable. This suggests that the model is a good fit for the data, and the year is a strong predictor of the total number of male suicides.
# 
# The MSE value of 148625692.11 represents the average of the squared differences between the observed and predicted values. This means that, on average, the model's predictions are off by approximately 12,189 suicides. While this may seem like a large number, it is important to consider the scale of the data being analyzed. The total number of male suicides in India is likely to be a large number, so an error of this magnitude may not be significant in the context of the data. However, it is still important to interpret the results in light of the specific goals and context of the analysis.

# In[ ]:




