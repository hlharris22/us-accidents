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
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# pd.set_option('float_format', '{:f}'.format)

# Store data as a DataFrame object
acc_df = pd.read_csv("US_Accidents_Dec20_updated.csv")

# Preliminary data inspection
print(acc_df.info())

# Calculate time of delay (seconds) - Convert times to datetime objects
acc_df[['Start_Time', 'End_Time']] = acc_df[['Start_Time', 'End_Time']].apply(pd.to_datetime)
acc_df['Delay_Time'] = acc_df['End_Time'] - acc_df['Start_Time']
acc_df['Delay_Time'] = acc_df['Delay_Time'] / np.timedelta64(1, 's')
print(acc_df.info())

# Drop rows that both columns contain null values. Can not determine precipitation
acc_df = acc_df.dropna(how='all', subset=['Precipitation(in)', 'Weather_Condition'])

# Create boolean column specifying if there is precipitation
acc_df['precipitation'] = acc_df['Weather_Condition'].str.contains('Rain|Snow|Drizzle|Mix|Ice|Sleet|Hail', case=False,
                                                                   regex=True)

# Set precipitation amount to 0 if no precipitation weather condition
acc_df.loc[(acc_df['Precipitation(in)'].isnull()) & (acc_df['precipitation'] == False), "Precipitation(in)"] = 0

# Remove features
acc_df.drop(acc_df.columns[[2, 3, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 22, 24, 26, 27, 29, 44, 45, 46, 48]],
            axis=1,
            inplace=True)
print(acc_df.info())

# Check DataFrame for null values and remove rows
print(acc_df.isnull().sum())
acc_df = acc_df.dropna(how='any', axis=0).reset_index()
print(acc_df.info())
print(acc_df.isnull().sum())

# Descriptive Statistics for quantitative attributes
print(acc_df.iloc[:, [2, 5, 8, 9, 10, 11, 26]].describe().round(2))


# # Remove outliers
#
# # Function to remove outliers from a specific column
# def remove_outliers(column_name, data_frame):
#     q_1 = acc_df[column_name].quantile(0.25)
#     q_3 = acc_df[column_name].quantile(0.75)
#     iqr = q_3 - q_1
#     return data_frame[~((data_frame[column_name] < (q_1 - 1.5 * iqr)) | (data_frame[column_name] > (q_3 + 1.5 * iqr)))]
#
#
# # Remove outliers
# acc_df = remove_outliers('Delay_Time', acc_df)
# acc_df = remove_outliers('Temperature(F)', acc_df)
# acc_df = acc_df.reset_index(drop=True)
#
# print(acc_df.iloc[:, [2, 5, 8, 9, 10, 11, 26]].describe().round(2))
#
# ################################
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
#
# ############################################################
#
#
# # Normalize Values
# acc_df_normalized = acc_df.iloc[:, [2, 5, 8, 9, 10, 11, 26]]
# scaler = MinMaxScaler()
# scaler.fit(acc_df_normalized)
# scaled = scaler.transform(acc_df_normalized)
# acc_df_normalized = pd.DataFrame(scaled, columns=acc_df_normalized.columns)
#
# def pearson_correlation(x, y, **kwags):
#     # Calculate Pearson Correlation
#     coef = np.corrcoef(x, y)[0][1]
#
#     # Make the label
#     label = r'$\rho$ = ' + str(round(coef, 2))
#
#     # Add the label to the plot
#     ax = plt.gca()  # Identifies instance of plot
#     ax.annotate(label, xy=(0.05, 0.95), size=12, xycoords=ax.transAxes)
#
#
# grid = sns.PairGrid(data=acc_df_normalized,
#                     vars=["Severity", "Distance(mi)", "Temperature(F)", "Humidity(%)", "Visibility(mi)",
#                           "Precipitation(in)", "Delay_Time"], height=2)
#
# # Map the plots to the locations
# grid = grid.map_upper(plt.scatter, color="Orange", s=5, alpha=0.7, edgecolors="Black", linewidth=0.25)
# grid = grid.map_upper(pearson_correlation)
# grid = grid.map_lower(plt.scatter, color="Orange", s=7, alpha=0.7, edgecolors="Black", linewidth=0.25)
# grid = grid.map_diag(plt.hist, bins=4, edgecolor='k', color='Orange')
# grid.fig.subplots_adjust(top=0.9)
# grid.fig.suptitle('League of Legends Game Statistics')
#
# # # Save figure
# # grid.savefig("League_Of_Legends_Game_Stats", dpi=1200, bbox_inches = 'tight')
#
# plt.show()
