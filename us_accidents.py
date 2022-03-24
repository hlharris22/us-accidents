# Principles of Data Analytics
# CPSC 5240
# CRN 20968
# Hunter Harris: zgt795
# Project: Part 2

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Store data as a DataFrame object
acc_df = pd.read_csv("US_Accidents_Dec20_updated.csv")

# Preliminary data inspection
print("----Initial Data Inspection----")
print(acc_df.info())

# Calculate time of delay (seconds) from start and end time - Convert times to datetime objects
acc_df[['Start_Time', 'End_Time']] = acc_df[['Start_Time', 'End_Time']].apply(pd.to_datetime)
acc_df['Delay_Time'] = acc_df['End_Time'] - acc_df['Start_Time']
acc_df['Delay_Time'] = acc_df['Delay_Time'] / np.timedelta64(1, 's')  # Convert time to second.
acc_df['Month'] = acc_df['Start_Time'].dt.month  # Add a column for month

# Drop rows where both columns contain null values. Can not determine precipitation
acc_df = acc_df.dropna(how='all', subset=['Precipitation(in)', 'Weather_Condition'])

# Create boolean column specifying if there is precipitation
acc_df['precipitation'] = acc_df['Weather_Condition'].str.contains('Rain|Snow|Drizzle|Mix|Ice|Sleet|Hail', case=False,
                                                                   regex=True)

# Set precipitation amount to 0 if no precipitation weather condition
acc_df.loc[(acc_df['Precipitation(in)'].isnull()) & (acc_df['precipitation'] == False), "Precipitation(in)"] = 0

# Remove features
acc_df.drop(acc_df.columns[[3, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 22, 24, 26, 27, 29, 44, 45, 46, 49]],
            axis=1,
            inplace=True)
print("----Post Feature Removal Data Inspection----")
print(acc_df.info())

# Check DataFrame for null values and remove rows
print("----Sum of null Values----")
print(acc_df.isnull().sum())
acc_df = acc_df.dropna(how='any', axis=0).reset_index()
print("----DataFrame info after dropping null values----")
print(acc_df.info())
print("----Post null value Removal Data Inspection----")
print(acc_df.isnull().sum())


# # Remove outliers
# # Function to remove outliers from a specific column
# def remove_outliers(column_name, data_frame):
#     q_1 = acc_df[column_name].quantile(0.25)
#     q_3 = acc_df[column_name].quantile(0.75)
#     iqr = q_3 - q_1
#     return data_frame[~((data_frame[column_name] < (q_1 - 1.5 * iqr)) | (data_frame[column_name] > (q_3 + 1.5 * iqr)))]
#
#
# # Remove outliers
# acc_df = remove_outliers('Distance(mi)', acc_df)
# acc_df = remove_outliers('Temperature(F)', acc_df)
# acc_df = remove_outliers('Humidity(%)', acc_df)
# acc_df = remove_outliers('Delay_Time', acc_df)
#
# acc_df = acc_df.reset_index(drop=True)


# Descriptive Statistics for quantitative attributes
# Function to add mean absolute deviation (MAD) and Mode to describe statistics
def describe(df):
    des_stats = df.describe()
    des_stats = des_stats.append(df.reindex(des_stats.columns, axis=1).agg(['mad'])).round(2)  # MAD
    des_stats = des_stats.append(df.mode()).round(2)  # Mode
    des_stats.rename(index={0: 'mode'}, inplace=True)
    des_stats.loc['amplitude'] = des_stats.loc['max'] - des_stats.loc['min']  # Amplitude
    des_stats.loc['iqr'] = des_stats.loc['75%'] - des_stats.loc['25%']  # Interquartile Range
    return des_stats


# Frequency Statistics
# Function to calculate frequency statistics
def frequency_stats(series):
    frequency_df = pd.DataFrame(series.value_counts())
    frequency_df.columns = ['Absolute Frequency']
    frequency_df.sort_index(inplace=True)
    frequency_df['Relative Frequency'] = 100 * (
        (frequency_df['Absolute Frequency'] / frequency_df['Absolute Frequency'].sum()).round(4))
    frequency_df['Absolute Cumulative Frequency'] = frequency_df['Absolute Frequency'].cumsum()
    frequency_df['Relative Cumulative Frequency'] = 100 * (
        (frequency_df['Absolute Cumulative Frequency'] / frequency_df['Absolute Frequency'].sum()).round(4))
    return frequency_df


# Display frequency statistics for severity
print(frequency_stats(acc_df['Severity']))

# Display descriptive statistics
statistics_df = describe(acc_df.iloc[:, [2, 6, 9, 10, 11, 12, 27]])
print(statistics_df)

# Normalize Values
acc_df_normalized = acc_df.iloc[:, [2, 6, 9, 10, 11, 12, 27]]
scaler = MinMaxScaler()
scaler.fit(acc_df_normalized)
scaled = scaler.transform(acc_df_normalized)
acc_df_normalized = pd.DataFrame(scaled, columns=acc_df_normalized.columns)

# Descriptive Bivariate Analysis
covariance_df = acc_df_normalized.cov().round(4)
print(covariance_df)
pearson_df = acc_df_normalized.corr().round(4)
print(pearson_df)

road_acc_df = acc_df.iloc[:, [2, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]]
print(road_acc_df)
checking = road_acc_df.groupby(['Severity'])['Sunrise_Sunset'].value_counts()
print(checking)

# Plots
sns.histplot(acc_df['Severity'], color='Orange')
plt.show()

sns.countplot(x='Month', hue="Severity", data=acc_df)
plt.show()

################################
# a_plot = sns.catplot(data=acc_df, x='Severity', kind='count', hue="Railway")
# a_plot.set(ylim=(0, 1500000))
# plt.show()
#
# a_plot = sns.catplot(data=acc_df, x='Severity', kind='count', hue="Amenity")
# a_plot.set(ylim=(0, 1500000))
# plt.show()
#
# a_plot = sns.catplot(data=acc_df, x='Severity', kind='count', hue="Bump")
# a_plot.set(ylim=(0, 1500000))
# plt.show()
#
# a_plot = sns.catplot(data=acc_df, x='Severity', kind='count', hue="Crossing")
# a_plot.set(ylim=(0, 1500000))
# plt.show()
#
# a_plot = sns.catplot(data=acc_df, x='Severity', kind='count', hue="Give_Way")
# a_plot.set(ylim=(0, 1500000))
# plt.show()
#
# a_plot = sns.catplot(data=acc_df, x='Severity', kind='count', hue="Junction")
# a_plot.set(ylim=(0, 1500000))
# plt.show()
#
# a_plot = sns.catplot(data=acc_df, x='Severity', kind='count', hue="No_Exit")
# a_plot.set(ylim=(0, 1500000))
# plt.show()
#
# a_plot = sns.catplot(data=acc_df, x='Severity', kind='count', hue="Roundabout")
# a_plot.set(ylim=(0, 1500000))
# plt.show()
#
# a_plot = sns.catplot(data=acc_df, x='Severity', kind='count', hue="Station")
# a_plot.set(ylim=(0, 1500000))
# plt.show()
#
# a_plot = sns.catplot(data=acc_df, x='Severity', kind='count', hue="Stop")
# a_plot.set(ylim=(0, 1500000))
# plt.show()
#
# a_plot = sns.catplot(data=acc_df, x='Severity', kind='count', hue="Traffic_Calming")
# a_plot.set(ylim=(0, 1500000))
# plt.show()
#
# a_plot = sns.catplot(data=acc_df, x='Severity', kind='count', hue="Traffic_Signal")
# a_plot.set(ylim=(0, 1500000))
# plt.show()
#
# a_plot = sns.catplot(data=acc_df, x='Severity', kind='count', hue="Turning_Loop")
# a_plot.set(ylim=(0, 1500000))
# plt.show()
#
# a_plot = sns.catplot(data=acc_df, x='Severity', kind='count', hue="Sunrise_Sunset")
# a_plot.set(ylim=(0, 1500000))
# plt.show()

############################################################
#
#
