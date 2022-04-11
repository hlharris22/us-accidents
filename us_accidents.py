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
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from numpy import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_validate, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from numba import jit, cuda
from sklearn.naive_bayes import GaussianNB

# Set pandas DataFrame options
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_columns', None)

# Store data as a DataFrame object
acc_df = pd.read_csv("US_Accidents_Dec20_updated.csv")

# Preliminary data inspection
print("----Initial Data Inspection----")
print(acc_df.info())

# Remove Duplicates
acc_df.drop_duplicates(subset=['Severity', 'Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng',
                               'Distance(mi)', 'Description', 'Number', 'Street', 'Side', 'City', 'County', 'State',
                               'Zipcode', 'Country', 'Timezone', 'Airport_Code', 'Weather_Timestamp', 'Temperature(F)',
                               'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Direction',
                               'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition', 'Amenity', 'Bump',
                               'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
                               'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop', 'Sunrise_Sunset',
                               'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight'
                               ],
                       keep='first',
                       inplace=True)
print("----Data Inspection after Removal of Duplicates----")
acc_df = acc_df.reset_index(drop=True)
print(acc_df.info())

# Calculate time of delay (seconds) from start and end time - Convert times to datetime objects
acc_df[['Start_Time', 'End_Time']] = acc_df[['Start_Time', 'End_Time']].apply(pd.to_datetime)
acc_df['Month'] = acc_df['Start_Time'].dt.month  # Add a column for month
acc_df['Year'] = acc_df['Start_Time'].dt.year  # Add a column for year

# Remove features
acc_df.drop(acc_df.columns[[0, 3, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 22, 26, 42]],
            axis=1,
            inplace=True)
acc_df.reset_index().drop(["index"], axis=1)
print("----Post Feature Removal Data Inspection----")
print(acc_df.info())

# Drop rows where both columns contain null values. Can not determine precipitation
acc_df.dropna(how='all', subset=['Precipitation(in)', 'Weather_Condition'], inplace=True)
print("----After precip and weather removal----")
acc_df = acc_df.reset_index().drop(["index"], axis=1)
print(acc_df.info())

# Create boolean column specifying if there is precipitation
acc_df['precipitation'] = acc_df['Weather_Condition'].str.contains('Rain|Snow|Drizzle|Mix|Ice|Sleet|Hail', case=False,
                                                                   regex=True)

# Set precipitation amount to 0 if no precipitation weather condition
acc_df.loc[(acc_df['Precipitation(in)'].isnull()) & (acc_df['precipitation'] == False), "Precipitation(in)"] = 0
print("----After Setting Precipitation(in) = 0----")
print(acc_df.info())

# Remove features weather_condition and precipitation
acc_df.drop(acc_df.columns[[12, 31]],
            axis=1,
            inplace=True)

# Check DataFrame for null values and remove rows
print("----Sum of null Values----")
print(acc_df.isnull().sum())
acc_df = acc_df.dropna(how='any', axis=0).reset_index().drop(["index"], axis=1)
print("----DataFrame info after dropping null values----")
print(acc_df.info())
print("----Post null value Removal Data Inspection----")
print(acc_df.isnull().sum())

# Convert boolean features to 0 and 1.
acc_df.iloc[:, 12:24] = acc_df.iloc[:, 12:24].astype(int)

# Convert Day and Night to Day = 0, Night = 1
acc_df['Sunrise_Sunset'].replace({"Day": 0, "Night": 1}, inplace=True)
acc_df['Civil_Twilight'].replace({"Day": 0, "Night": 1}, inplace=True)
acc_df['Nautical_Twilight'].replace({"Day": 0, "Night": 1}, inplace=True)
acc_df['Astronomical_Twilight'].replace({"Day": 0, "Night": 1}, inplace=True)
acc_df['Side'].replace({"L": 0, "R": 1}, inplace=True)

# Convert Severity to a binary ranking
acc_df['Severity'] = acc_df['Severity'].replace([1,2], 1)
acc_df['Severity'] = acc_df['Severity'].replace([3,4], 2)


# Remove outliers
# Function to remove outliers from a specific column
def remove_outliers(column_name, data_frame):
    q_1 = acc_df[column_name].quantile(0.25)
    q_3 = acc_df[column_name].quantile(0.75)
    iqr = q_3 - q_1
    lower = (q_1 - 1.5 * iqr)
    upper = (q_3 + 1.5 * iqr)
    return data_frame[~((data_frame[column_name] < lower) | (data_frame[column_name] > upper))]


# Function to print outliers
def print_outliers(column_name, data_frame):
    q_1 = data_frame[column_name].quantile(0.25)
    q_3 = data_frame[column_name].quantile(0.75)
    iqr = q_3 - q_1
    lower = (q_1 - 1.5 * iqr)
    upper = (q_3 + 1.5 * iqr)
    out_above_count = data_frame[data_frame[column_name] < lower].shape[0]
    out_below_count = data_frame[data_frame[column_name] > upper].shape[0]
    print(f"----{column_name}----", "\nLower Outlier: ", lower, "Upper Outlier: ", upper)
    print(f"Number of outliers below {lower}: ", out_above_count)
    print(f"Number of outliers above {upper}: ", out_below_count)


# Print outliers
print_outliers('Temperature(F)', acc_df)
print_outliers('Humidity(%)', acc_df)
print_outliers('Pressure(in)', acc_df)
print_outliers('Visibility(mi)', acc_df)
print_outliers('Wind_Speed(mph)', acc_df)
print_outliers('Precipitation(in)', acc_df)

# Remove outliers
acc_df = remove_outliers('Temperature(F)', acc_df)
acc_df = acc_df.reset_index(drop=True)


# Descriptive Statistics for quantitative features
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


# Display frequency statistics
print("----Frequency Statistics for Severity----")
freq_severity = frequency_stats(acc_df['Severity'])
print(freq_severity)
freq_severity.to_csv(r'Frequency_Statistics_Severity.csv', index=False)
print("----Frequency Statistics for Month----")
freq_month = frequency_stats(acc_df['Month'])
print(freq_month)
freq_month.to_csv(r'Frequency_Statistics_Month.csv', index=False)
print("----Frequency Statistics for Year----")
freq_year = frequency_stats(acc_df['Year'])
print(freq_year)
freq_year.to_csv(r'Frequency_Statistics_Year.csv', index=False)

# Display descriptive statistics
statistics_df = describe(acc_df.iloc[:, [0, 6, 7, 8, 9, 10, 11]])
print("----Descriptive Statistics for Quantitative Features----")
print(statistics_df)
statistics_df.to_csv(r'Descriptive_Statistics.csv', index=False)

# Normalize Values
acc_df.iloc[:, [6, 7, 8, 9, 10, 11]] = MinMaxScaler().fit_transform(acc_df.iloc[:, [6, 7, 8, 9, 10, 11]])
acc_df_normalized = acc_df.iloc[:, [0, 6, 7, 8, 9, 10, 11]]

# Descriptive Bivariate Analysis
covariance_df = acc_df_normalized.cov().round(4)
print("----Covariance----")
print(covariance_df)
covariance_df.to_csv(r'Covariance.csv', index=False)
pearson_df = acc_df_normalized.corr("pearson").round(4)
print("----Pearson Correlation----")
print(pearson_df)
pearson_df.to_csv(r'Pearson.csv', index=False)
spearman_df = acc_df_normalized.corr("spearman").round(4)
print("----Spearman Correlation----")
print(spearman_df)
spearman_df.to_csv(r'Spearman.csv', index=False)
road_acc_df = acc_df.iloc[:, [0, 4, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
spearman_df_2 = road_acc_df.corr("spearman").round(4)
print("----Spearman Correlation (Road Amenities)----")
print(spearman_df_2)
spearman_df_2.to_csv(r'Spearman_Road.csv', index=False)

# Plot Accident Severity count by Month
sns.countplot(x='Month', hue="Severity", data=acc_df)
plt.gcf().set_size_inches(12, 6)
plt.xlabel('Month', fontweight='bold', fontsize=14)
plt.ylabel('Count', fontweight='bold', fontsize=14)
plt.legend(title="Severity", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=10, shadow=True, borderpad=1)
plt.title("Accident Severity by Month", fontsize=15, fontweight='bold')
# Save figure
plt.savefig("Accident_Severity_Month", dpi=1200, bbox_inches='tight')
plt.show()

# Plot Accident count by State
sns.countplot(x='State', data=acc_df, order=acc_df['State'].value_counts().index)
plt.gcf().set_size_inches(16, 7)
plt.xlabel("State", fontweight='bold', fontsize=22)
plt.ylabel("Count", fontweight='bold', fontsize=22)
plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12)
plt.title("Accidents by State", fontsize=25, fontweight='bold')
# Save figure
plt.savefig("Accident_Count_State", dpi=1200, bbox_inches='tight')
plt.show()

# # Convert nominal values to
# state_one_hot = pd.get_dummies(acc_df.State, prefix='State')
# month_one_hot = pd.get_dummies(acc_df.Month, prefix='Month')
# year_one_hot = pd.get_dummies(acc_df.Year, prefix='Year')
# concat_frames = [acc_df.reset_index(drop=True), state_one_hot.reset_index(drop=True),
#                  month_one_hot.reset_index(drop=True), year_one_hot.reset_index(drop=True)]
# acc_df = pd.concat(concat_frames, axis=1)

# Convert nominal values to frequency values
acc_df["State_Freq"] = acc_df["State"].map(acc_df.groupby("State").size()/acc_df.shape[0])
acc_df["Year_Freq"] = acc_df["Year"].map(acc_df.groupby("Year").size()/acc_df.shape[0])
acc_df["Month_Freq"] = acc_df["Month"].map(acc_df.groupby("Month").size()/acc_df.shape[0])
print("-----State Frequency------")
print(acc_df['State_Freq'])

# Remove features
acc_df.drop(acc_df.columns[[1, 5, 28, 29]],
            axis=1,
            inplace=True)

# Inspect cleaned data
print("----Cleaned DataFrame----")
print(acc_df.info())
acc_df.to_csv(r'Cleaned_Data.csv', index=False)

# Feature Selection by wrapper technique
wrapper_y = acc_df['Severity']
wrapper_X = acc_df.drop(['Severity'], axis=1)

# Use DecisionTreeClassifier to inspect 1 to 10 features
sfs = SFS(DecisionTreeClassifier(), k_features=(1, 10), floating=False, forward=True, cv=0, n_jobs=-1, scoring="precision")
sfs.fit(wrapper_X, wrapper_y)

# Best Features found through forward selection
print(f'Forward Selection: {sfs.k_feature_names_}')
print(f'score = {sfs.k_score_}')
print(f'metric dictionary')

# Plot feature selection results
fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
plt.grid()
plt.show()

# Print Feature Selection metrics
metrics = sfs.get_metric_dict()
print("-----Wrapper Feature Selection Metrics------")
for value in metrics.values():
    print(value)

# Define Target attribute and predictors
ml_y = acc_df['Severity'].to_numpy()
ml_X = acc_df[['Start_Lat', 'Start_Lng', 'Humidity(%)', 'Pressure(in)', 'Wind_Speed(mph)', 'Civil_Twilight']].to_numpy()

# Define ml models
estimators = {'Decision Tree': DecisionTreeClassifier(),
              'Random Forest': RandomForestClassifier(),
              'Logistic Regression': LogisticRegression(solver='liblinear', class_weight='balanced'),
              'Gaussian Naive Bayes': GaussianNB(),
              }

for estimator_name, estimator_object in estimators.items():

    # Define Scores
    score = ['precision', 'recall', 'f1', 'roc_auc']
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('under', under), ('model', estimator_object)]
    pipeline = Pipeline(steps=steps)

    # Define F-fold details
    cv = KFold(n_splits=10, shuffle=True, random_state=11)

    # Run ml models
    scores = cross_validate(pipeline, ml_X, ml_y, scoring=score, cv=cv, n_jobs=-1)

    # predictions of the model and visualize.
    # fit model then produce labels.
    labels = estimator_object.predict(ml_X)
    print(labels)

    # Store ml models scores
    precision = mean(scores['test_precision'])
    recall = mean(scores['test_recall'])
    f1 = mean(scores['test_f1'])
    roc = mean(scores['test_roc_auc'])

    # Display scores
    print(f'-----{estimator_name}-----')
    print(f'Mean Precision: {precision}')
    print(f'Mean recall: {recall}')
    print(f'Mean f1: {f1}')
    print(f'Mean ROC AUC: {roc}\n')

