# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:25:23 2023
@author: adila
"""
import pyreadstat
import zipfile
import tempfile
import pandas as pd
import pyreadstat
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from lightgbm.sklearn import LGBMRegressor

#Importing our WBES dataset

zip_path = 'assets.zip'
dta_filename = 'BEEPS_2009_2013_Panel.dta'

# Extract the .dta file from the zipped file
with zipfile.ZipFile(zip_path, 'r') as zip_file:
    zip_file.extract(dta_filename)

# Load the extracted .dta file using pyreadstat
dta_path = dta_filename  # Assumes the .dta file is in the current working directory
df, meta = pyreadstat.read_dta(dta_path, encoding='latin1')


#importing our GDP file
GDPper = pd.read_csv('ed4e6d79-d906-43c5-8054-80fd437963b2_Data.csv') 

#Select certain columns
selected_columns = ['Country Name', '2008 [YR2008]', '2009 [YR2009]','2012 [YR2012]','2013 [YR2013]'] 
GDPper = GDPper[selected_columns]

#Rename the columns
new_column_names = {'Country Name': 'country', '2008 [YR2008]': '2008', '2009 [YR2009]': '2009','2012 [YR2012]':'2012','2013 [YR2013]':'2013'}
GDPper.rename(columns=new_column_names, inplace=True)

#Convert from wide to long format
id_vars = ['country']  # Identifier variables to preserve
value_vars = ['2008', '2009', '2012', '2013']  # Columns to melt

# Reshape the DataFrame from wide to long format
GDPper_long = pd.melt(GDPper, id_vars=id_vars, value_vars=value_vars, var_name='year', value_name='GDP')

# Convert the 'year' column to numeric type in both DataFrames
df['year'] = pd.to_numeric(df['year'], errors='coerce')
GDPper_long['year'] = pd.to_numeric(GDPper_long['year'], errors='coerce')

# MERGING OUR DATA TOGETHER
#left join to keep our df variables
merged_data = pd.merge(df, GDPper_long, on=['country', 'year'], how='left')

#selecting what we need

columns_needed = ["country",
                  "year",
                  "GDP",
                  "d2",
                  "n3",
                  "j7a",
                  "j15",
                  "j12",
                  "j5",
                  "g4",
                  "c5",
                  "c14",
                  "c21_2009",
                  "j6",
                  "j2",
                  "j4",
                  "j14",
                  "j11",
                  "c4",
                  "c13",
                  "c20_2009",
                  "g3",
                  "d4",
                  "h7a",
                  "j30f",
                  "j30c",
                  "d30b",
                  "j30a",
                  "j30b",
                  "c30a",
                  "h30",
                  "b5",
                  "l1",
                  "l30b",
                  "b4",
                  "b7",
                  "b8",
                  "e11",
                  "i2a",
                  "i30",
                  ]


newdata = merged_data[columns_needed]

# making our names make sense

renaming = {"country":"country",
                  "year":"year",
                  "GDP":"GDPpercap",
                  "d2":"annualsales_1yrago",
                  "n3":"annualsales_3yrsago",
                  "j7a":"%sales",
                  "j15":"Bribe oper license",
                  "j12":"Bribe import license",
                  "j5":"Bribe tax",
                  "g4":"Bribe constr permit",
                  "c5":"Bribe electricity",
                  "c14":"Bribe water",
                  "c21_2009":"Bribe phone",
                  "j6":"% of contract",
                  "j2":"Mgmt time on buro",
                  "j4":"# of tax visits",
                  "j14":"Wait oper license",
                  "j11":"Wait imp license",
                  "c4":"Wait electricity",
                  "c13":"Wait water",
                  "c20_2009":"Wait phone",
                  "g3":"Wait constr permit",
                  "d4":"Wait customs",
                  "h7a":"Fairness of courts",
                  "j30f":"Obst corruption",
                  "j30c":"Obst bus license",
                  "d30b":"Obst customs",
                  "j30a":"Obst tax rates",
                  "j30b":"Obst tax admin",
                  "c30a":"Obst electricity",
                  "h30":"Obst courts",
                  "b5":"Year in oper",
                  "l1":"Firm size",
                  "l30b":"Low edu labor",
                  "b4":"Female",
                  "b7":"Mgr experience",
                  "b8":"ISO",
                  "e11":"Compete w/informal",
                  "i2a":"Pmts for security",
                  "i30":"Obst crime"
                  }


newdata = newdata.rename(columns=renaming)

#MULTIPLE IMPUTATION to correct missing values

unique_values = newdata.apply(lambda x: x.unique())

#Print the unique values for each column
for column, values in unique_values.iteritems():
    print(f"Column: {column}")
    print(f"Unique Values: {values}")
    print("---")


#Define the columns where negative numbers should be converted to NaN
columns_to_convert = ["%sales",
                      "annualsales_1yrago",
                      "annualsales_3yrsago",
                      "Bribe oper license",
                      "Bribe import license",
                      "Bribe tax",
                      "Bribe constr permit",
                      "Bribe electricity",
                      "Bribe water",
                      "Bribe phone",
                      "% of contract",
                      "Mgmt time on buro",
                      "# of tax visits",
                      "Wait oper license",
                      "Wait imp license",
                      "Wait electricity",
                      "Wait water",
                      "Wait phone",
                      "Wait constr permit",
                      "Wait customs",
                      "Fairness of courts",
                      "Obst corruption",
                      "Obst bus license",
                      "Obst customs",
                      "Obst tax rates",
                      "Obst tax admin",
                      "Obst electricity",
                      "Obst courts",
                      "Year in oper",
                      "Firm size",
                      "Low edu labor",
                      "Female",
                      "Mgr experience",
                      "ISO",
                      "Compete w/informal",
                      "Pmts for security",
                      "Obst crime"
                      ]

# Convert negative numbers to NaN in the specified columns
newdata[columns_to_convert] = newdata[columns_to_convert].where(newdata[columns_to_convert] >= 0, np.nan)

# checking how many values we do have missing

missing_values_count = newdata.isna().sum()
print("Missing Values Count:")
print(missing_values_count)



# Then we're going to try and see what our variable types are
for column in newdata.columns:
    column_type = newdata[column].dtype
    print(f"Column '{column}' has data type: {column_type}")

# we need to correct a few 

columns_to_numerfiy = ["GDPpercap",
                       "%sales",
                       "Bribe oper license",
                       "Bribe import license",
                       "Bribe tax",
                       "Bribe constr permit",
                       "Bribe electricity",
                       "Bribe water",
                       "Bribe phone",
                       "% of contract",
                       "# of tax visits",
                       "Wait oper license",
                       "Wait imp license",
                       "Wait electricity",
                       "Wait water",
                       "Wait phone",
                       "Wait constr permit",
                       "Wait customs",
                       "Pmts for security"
                       ]

# we convert everything to numeric so that we can to make it easier for our regressions later
for column in columns_to_numerfiy:
    newdata[column] = newdata[column].astype(float)  

# taking all number columns so that we can exclude the country column
numeric_columns = newdata.select_dtypes(include=['float', 'int']).columns

# Get the numerical columns from the dataset
numeric_data = newdata[numeric_columns]

# Perform imputation on the numerical dataset
imputer = IterativeImputer(estimator=lgb.LGBMRegressor(), max_iter=10)

#imputer = IterativeImputer(estimator=lgb.LGBMRegressor())
numeric_data_imputed = imputer.fit_transform(numeric_data)

# Create a DataFrame with the imputed numerical values
ndf_imputed = pd.DataFrame(numeric_data_imputed, columns=numeric_columns)

# Concatenate the imputed numerical columns with the original categorical columns
ndf_imputed = pd.concat([newdata[['country']], ndf_imputed], axis=1)

# just checking what our dataframe looks like now
ndf_imputed.head()

# fixing some impossible non integers

columns_to_round = ["Bribe oper license",
                    "Bribe import license",
                    "Bribe tax",
                    "Bribe constr permit",
                    "Bribe electricity",
                    "Bribe water",
                    "Bribe phone",
                    "Fairness of courts",
                    "Obst corruption",
                    "Obst bus license",
                    "Obst customs",
                    "Obst tax rates",
                    "Obst tax admin",
                    "Obst electricity",
                    "Obst courts",
                    "Year in oper",
                    "Firm size",
                    "Low edu labor",
                    "Female",
                    "Mgr experience",
                    "ISO",
                    "Compete w/informal",
                    "Obst crime"
                    ]
for column in columns_to_round:
    ndf_imputed[column] = ndf_imputed[column].round()

# fixing binaries

columns_to_fix_for_binaries = ["Bribe oper license",
                    "Bribe import license",
                    "Bribe tax",
                    "Bribe constr permit",
                    "Bribe electricity",
                    "Bribe water",
                    "Bribe phone",
                    "Female",
                    "ISO",
                    "Compete w/informal"
                    ]

for column in columns_to_fix_for_binaries:
    ndf_imputed[column] = ndf_imputed[column].replace({2: 1, 1: 0})
    
    
# Saving our dataframe as a csv 

ndf_imputed.to_csv("WBES.csv", index=False)


















