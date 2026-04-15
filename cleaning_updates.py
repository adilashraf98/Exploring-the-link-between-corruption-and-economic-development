# -*- coding: utf-8 -*-
"""
cleaning.py
-----------
Data preparation script for replication of Nur-tegin & Jakee (2019),
"Does Corruption Grease or Sand the Wheels of Development?"
The Quarterly Review of Economics and Finance, 75, 19-30.

This script:
  1. Loads firm-level panel data from the World Bank Enterprise Survey (WBES)
     covering Central Asia and Eastern Europe, 2008-2013.
  2. Merges country-level GDP per capita from the World Bank.
  3. Selects and renames variables to match the original study's codebook.
  4. Cleans miscoded values (negative entries treated as missing).
  5. Imputes missing values using an IterativeImputer with a LightGBM
     estimator, following a principled multiple imputation approach.
  6. Post-processes imputed values to restore valid ranges for discrete
     and binary variables.
  7. Outputs a clean CSV ready for regression analysis in regressions.py.

Author: Adil Ashraf Mayo
"""

import zipfile  
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyreadstat
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# Initial Configuration
# I am going to set up hard file paths, and variable names that I will need later on
# This makes it easier to change/alter these hard "settings" for later. 


# We begin by defining some important paths that we will use
ZIP_PATH = "assets.zip"
DTA_FILENAME = "BEEPS_2009_2013_Panel.dta"
GDP_PATH = "ed4e6d79-d906-43c5-8054-80fd437963b2_Data.csv"
OUTPUT_PATH = "WBES.csv"

# Highlighting the years covered by the WBES panel used in this replication
# These will be important when we're limiting our GDP dataset to these years
PANEL_YEARS = ["2008", "2009", "2012", "2013"]

# WBES variable codes mapped to human-readable names (following Table 1
# of Nur-tegin & Jakee 2019)
VARIABLE_RENAME = {
    "country":    "country",
    "year":       "year",
    "GDP":        "GDPpercap",
    "d2":         "annualsales_1yrago",
    "n3":         "annualsales_3yrsago",
    "j7a":        "%sales",
    "j15":        "Bribe oper license",
    "j12":        "Bribe import license",
    "j5":         "Bribe tax",
    "g4":         "Bribe constr permit",
    "c5":         "Bribe electricity",
    "c14":        "Bribe water",
    "c21_2009":   "Bribe phone",
    "j6":         "% of contract",
    "j2":         "Mgmt time on buro",
    "j4":         "# of tax visits",
    "j14":        "Wait oper license",
    "j11":        "Wait imp license",
    "c4":         "Wait electricity",
    "c13":        "Wait water",
    "c20_2009":   "Wait phone",
    "g3":         "Wait constr permit",
    "d4":         "Wait customs",
    "h7a":        "Fairness of courts",
    "j30f":       "Obst corruption",
    "j30c":       "Obst bus license",
    "d30b":       "Obst customs",
    "j30a":       "Obst tax rates",
    "j30b":       "Obst tax admin",
    "c30a":       "Obst electricity",
    "h30":        "Obst courts",
    "b5":         "Year in oper",
    "l1":         "Firm size",
    "l30b":       "Low edu labor",
    "b4":         "Female",
    "b7":         "Mgr experience",
    "b8":         "ISO",
    "e11":        "Compete w/informal",
    "i2a":        "Pmts for security",
    "i30":        "Obst crime",
}

# We also need to highlight the variables that will need to be 
# forced to be integers after our imputation.
# Continuous imputation can produce fractional values for inherently
# discrete variables (firm size, for example, is usually discrete).
DISCRETE_COLUMNS = [
    "Bribe oper license", "Bribe import license", "Bribe tax",
    "Bribe constr permit", "Bribe electricity", "Bribe water", "Bribe phone",
    "Fairness of courts",
    "Obst corruption", "Obst bus license", "Obst customs", "Obst tax rates",
    "Obst tax admin", "Obst electricity", "Obst courts",
    "Year in oper", "Firm size", "Low edu labor",
    "Female", "Mgr experience", "ISO", "Compete w/informal", "Obst crime",
]

# Binary variables are coded 1/2 in the raw WBES data (1 = No, 2 = Yes).
# We recode to the conventional 0/1 dummy encoding.
BINARY_COLUMNS = [
    "Bribe oper license", "Bribe import license", "Bribe tax",
    "Bribe constr permit", "Bribe electricity", "Bribe water", "Bribe phone",
    "Female", "ISO", "Compete w/informal",
]

# Columns where negative values are invalid and should be treated as missing.
# In the WBES codebook, negative entries typically encode non-applicability
# (for example, sometimes a firm may not seek a permit,
#  so wait time was not recorded).
# Following Nur-tegin & Jakee (2019), we treat these as missing rather than
# zero to avoid conflating non-experience with zero corruption exposure.
COLUMNS_TO_CLEAN = [
    "%sales", "annualsales_1yrago", "annualsales_3yrsago",
    "Bribe oper license", "Bribe import license", "Bribe tax",
    "Bribe constr permit", "Bribe electricity", "Bribe water", "Bribe phone",
    "% of contract", "Mgmt time on buro", "# of tax visits",
    "Wait oper license", "Wait imp license", "Wait electricity", "Wait water",
    "Wait phone", "Wait constr permit", "Wait customs", "Fairness of courts",
    "Obst corruption", "Obst bus license", "Obst customs", "Obst tax rates",
    "Obst tax admin", "Obst electricity", "Obst courts",
    "Year in oper", "Firm size", "Low edu labor", "Female", "Mgr experience",
    "ISO", "Compete w/informal", "Pmts for security", "Obst crime",
]

# Columns that need explicit float conversion before imputation.
# Most WBES variables are read as objects or categorical values 
# by pyreadstat due to embedded value labels in the .dta format.
# We highlight these variables here.
COLUMNS_TO_FLOAT = [
    "GDPpercap", "%sales",
    "Bribe oper license", "Bribe import license", "Bribe tax",
    "Bribe constr permit", "Bribe electricity", "Bribe water", "Bribe phone",
    "% of contract", "# of tax visits",
    "Wait oper license", "Wait imp license", "Wait electricity", "Wait water",
    "Wait phone", "Wait constr permit", "Wait customs", "Pmts for security",
]


# Now we move to our first major step

# STEP 1: Loading our WBES firm-level data

print("Begin loading WBES data")

# The raw WBES panel is distributed as a Stata .dta file inside a zip archive.
with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:
    zip_file.extract(DTA_FILENAME)

df, meta = pyreadstat.read_dta(DTA_FILENAME, encoding="latin1")

# We want to show how many observations we have now

print(f"  Loaded {len(df):,} firm-year observations.")


# STEP 2: Loading and reshaping GDP per capita data 

print("Begin Loading GDP per capita data")

gdp_raw = pd.read_csv(GDP_PATH)

# Retain only the country name and the four panel years
# our years list comes handy here

# I am going to map the messy formatting with clean labels

year_cols = {f"{y} [YR{y}]": y for y in PANEL_YEARS}
gdp_raw = gdp_raw[["Country Name"] + list(year_cols.keys())]

# Using a cool renaming method I discovered online
gdp_raw = gdp_raw.rename(columns={"Country Name": "country", **year_cols})

# Reshape from wide to long so we can merge on (country, year)
gdp_long = pd.melt(
    gdp_raw,
    id_vars=["country"],
    value_vars=PANEL_YEARS,
    var_name="year",
    value_name="GDP",
)

# Ensure year is numeric in both datasets before merging to avoid any
# funny business
df["year"] = pd.to_numeric(df["year"], errors="coerce")
gdp_long["year"] = pd.to_numeric(gdp_long["year"], errors="coerce")


# STEP 3: Merging our datasets and selecting key variables

print("We now merge our datasets together and select certain variables")

# Left join preserves all WBES observations; GDP will be NaN for any
# country-year not matched in the World Bank data.
merged = pd.merge(df, gdp_long, on=["country", "year"], how="left")

# Keep only the variables listed in the original study's codebook (Table 1)
# Another place where our "settings" are useful
raw_columns = list(VARIABLE_RENAME.keys())
data = merged[raw_columns].rename(columns=VARIABLE_RENAME).copy()

# Reminding ourselves of what the shape of our output looks like
print(f"  Working dataset: {data.shape[0]:,} rows x {data.shape[1]} columns.")


# STEP 4: Cleaning Miscoded Values

print("Dealing and clean coding negative values")

# Negative entries encode non-applicability in the WBES codebook and are
# not valid observations for our analysis. We replace them with NaN so that
# they are handled by imputation rather than distorting regression estimates.
data[COLUMNS_TO_CLEAN] = data[COLUMNS_TO_CLEAN].where(
    data[COLUMNS_TO_CLEAN] >= 0, other=np.nan
)

# We're going to see how many values are actually missing
missing_counts = data.isna().sum()
print("  Missing value counts after cleaning:")
print(missing_counts[missing_counts > 0].to_string())


# STEP 5: Convertying columns to numerics

print("We now convert some columns to numeric types for our imputation later")

# pyreadstat preserves Stata value labels as objects; explicit float
# conversion is required before imputation can proceed.
for col in COLUMNS_TO_FLOAT:
    data[col] = pd.to_numeric(data[col], errors="coerce") # "coerce" helps us display missing values as missing values


# STEP 6: Iterative imputations to handle missing data

print("As we described above, we now use imputations to manage the vast missing data we have")

# We use sklearn's IterativeImputer with a LightGBM estimator. Unlike the
# original authors' Mersenne Twister multiple imputation, this approach
# conditions each variable's imputed values on all other variables through
# iterative rounds, allowing for nonlinear relationships and interactions.
# A fixed random_state ensures reproducibility too!
#
# Limitation: IterativeImputer produces a single completed dataset and does
# not propagate between-imputation uncertainty into downstream standard
# errors. This is noted in the accompanying research memo.


# Collecting all our numeric data now to put into out imputation machine
numeric_cols = data.select_dtypes(include=["float", "int"]).columns
numeric_data = data[numeric_cols]

imputer = IterativeImputer(
    estimator=lgb.LGBMRegressor(random_state=42, verbose=-1),
    max_iter=10,
    random_state=42,
)
imputed_array = imputer.fit_transform(numeric_data)

# Reconstruct a DataFrame with the original column names
imputed_df = pd.DataFrame(imputed_array, columns=numeric_cols)

# Re-attach the country string column (excluded from imputation)
imputed_df = pd.concat([data[["country"]].reset_index(drop=True), imputed_df], axis=1)

print("  Imputation complete.")


# STEP 7: Post processing our imputed values (they may have problems)

print("We conduct some additional treatments to our imputed values")

# Imputation produces continuous values for variables that are inherently
# discrete. We round these back to valid integer values.
for col in DISCRETE_COLUMNS: # again good thing to define this early on.
    imputed_df[col] = imputed_df[col].round().astype(int)

# The WBES encodes binary responses as 1 (No) / 2 (Yes). After rounding,
# we recode to the standard 0/1 dummy convention used in the regressions.
for col in BINARY_COLUMNS:
    imputed_df[col] = imputed_df[col].map({1: 0, 2: 1})


# STEP 8: Saving our ouput

print(f"We now save our dataset to {OUTPUT_PATH}")

imputed_df.to_csv(OUTPUT_PATH, index=False)

print(f"Final dataset: {imputed_df.shape[0]:,} rows x {imputed_df.shape[1]} columns.")
print(imputed_df.head())
