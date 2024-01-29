import os
import pandas as pd
import subprocess

# Drop 'missing_cbg' column
# Initially wanted to drop all empty columns, but no column completely empty
def merge_missing_cbg(directory):
  for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    df = pd.read_csv(file_path)
    for row_index, entire_row in df.iterrows():
      # if there is missing measurement, if there is then interpolate a new cbg value
      if entire_row['missing_cbg'] == 1:
        next_row_idx = row_index+1
        if pd.isna(df['cbg'][next_row_idx])== False:
          df.interpolate(method = 'linear', axis = 0)
        else:
          df.interpolate(method = 'ffill', axis = 0)
        df.at[row_index,'missing_cbg'] = 0
    # Check if all values in column 'missing_cbg' are 0
    if (df['missing_cbg'] != True).all():
      # Drop the column if all values are 0
      df.drop('missing_cbg', axis=1, inplace=True)
      #print(df.head())
      df.to_csv(filename, index=False)
    print('Dropped missing_cbg in'+ file_path)

## Drop hr column in 2020, both test and train, as NaNs
def check_hr_2020(directory):
  for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    df = pd.read_csv(file_path)
    if df['hr'].isnull().all():
      df.drop('hr', axis=1, inplace=True)
      #print(df.head())
      df.to_csv(filename, index=False)
    print('Dropped hr in'+ file_path)

#Drop rows that are missing any of '5minute_intervals_timestamp', 'cbg', 'gsr'
def drop_rows_too_empty(directory):
  for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    df = pd.read_csv(file_path)
    #Define in which columns to look for missing values, and see when they are all 3 nans
    # and drop those rows (but other columns might be not NaN)
    # you don't want rows without any of these 3 empty
    df.dropna(subset=['5minute_intervals_timestamp', 'cbg', 'gsr'], how='any',inplace=True)
    df.to_csv(filename, index=False)
    print('Dropped empty rows in'+ file_path)

#Finger, carbs, inputs have too many NaNs
def drop_all_nans(directory):
  for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    df = pd.read_csv(file_path)
    # drop rows cointaining NaNs
    df.dropna(axis=0, inplace = True)
    df.to_csv(filename, index=False)
    print('Dropped all NaNs in '+ file_path)

#go to /content
previous_directory = os.getcwd()

## Ohio2018_processed/train
# Change the current working directory to Ohio2018_processed/train
os.chdir("/content/Ohio Data/Ohio2018_processed/train")
# Print the current working directory
current_directory = subprocess.check_output("pwd", shell=True, text=True).strip()
print(current_directory)
directory = os.getcwd()
merge_missing_cbg(directory)
drop_rows_too_empty(directory)
drop_all_nans(directory)

## Ohio2018_processed/test
#Change back to /content
os.chdir(previous_directory)
# Change back to the previous working directory
os.chdir("/content/Ohio Data/Ohio2018_processed/test")
directory = os.getcwd()
merge_missing_cbg(directory)
drop_rows_too_empty(directory)
drop_all_nans(directory)

## Ohio2020_processed/train
#Change back to /content
os.chdir(previous_directory)
# Change back to the previous working directory
os.chdir("/content/Ohio Data/Ohio2020_processed/train")
directory = os.getcwd()
merge_missing_cbg(directory)
check_hr_2020(directory)
drop_rows_too_empty(directory)
drop_all_nans(directory)

## Ohio2020_processed/test
#Change back to /content
os.chdir(previous_directory)
# Change back to the previous working directory
os.chdir("/content/Ohio Data/Ohio2020_processed/test")
directory = os.getcwd()
merge_missing_cbg(directory)
check_hr_2020(directory)
drop_rows_too_empty(directory)
drop_all_nans(directory)
