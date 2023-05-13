# -*- coding: utf-8 -*-
"""
Created on Thu May 11 19:17:26 2023

@author: adila
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tabulate import tabulate

# Read CSV file into a DataFrame
df = pd.read_csv("WBES.csv")

# Creating our first dependent variable 

# Apply logarithmic transformation to the sales columns

df['ln_sales1'] = np.log(np.where(np.isfinite(df['annualsales_1yrago']), df['annualsales_1yrago'], 1))
df['ln_sales3'] = np.log(np.where(np.isfinite(df['annualsales_3yrsago']), df['annualsales_3yrsago'], 1))


# Create the new variable based on the formula: 1/3 * (ln(sales3) - ln(sales1))
df['Sales Growth'] = (1/3) * (df['ln_sales3'] - df['ln_sales1'])
df.dropna(subset=['Sales Growth'], inplace=True)
df = df[~np.isinf(df['Sales Growth'])]

# Reset the index of the DataFrame
df = df.reset_index(drop=True)

#Annual %age change in sales
df['Annual %age change sales'] = ((df['annualsales_1yrago'] - df['annualsales_3yrsago'])/df['annualsales_1yrago'])*(1/3)

# Applying the logarithmin transformation to the GDP column too 
df['ln_GDPpercap'] = np.log(df['GDPpercap'])

# Making our years operated variable better
df["Year in oper"] = 2013 - df["Year in oper"]

# Creating the interaction terms 

corruption_vars = ["Bribe oper license",
                   "Bribe import license",
                   "Bribe tax",
                   "Bribe constr permit",
                   "Bribe electricity",
                   "Bribe water",
                   "Bribe phone"]
 
obstacle_vars = ["Wait oper license",
                 "Wait imp license",
                 "# of tax visits",
                 "Wait constr permit",
                 "Wait electricity",
                 "Wait water",
                 "Wait phone"]

# Demeaning the corruption and obstacle variables
for var in corruption_vars + obstacle_vars:
    df[var] = df[var] - df[var].mean()

for i in range(len(corruption_vars)):
    # Get the variable names at the current index
    var1 = corruption_vars[i]
    var2 = obstacle_vars[i]

    # Create a new column name for the interaction term
    new_column_name = f'{var1}*{var2}'

    # Calculate the interaction term by multiplying the corresponding values
    df[new_column_name] = df[var1] * df[var2]

# Checking our columns first    
# regression 
print(df.columns.tolist())

# NOW WE CAN PERFORM OUR REGRESSIONS

#Setting up our regression summary image generation function

def save_regression_summary_image(results, filename):
    # Convert the summary table to a string
    summary_table = results.summary().tables[1]
    summary_str = tabulate(summary_table, headers='firstrow')

    # Set the figure and paper size
    fig = plt.figure(figsize=(8, 10))
    fig.set_size_inches(8, 10)
    fig.set_dpi(300)

    # Save the summary as an image
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.text(0, 0.5, summary_str, va='center', fontfamily='monospace')
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.5)
    
# defining our two dependent variables

ysalesgrowth = df['Sales Growth']
ypercentsales = df['Annual %age change sales']

"""
 Defining our independent variables, we first include everything
"""
independent_variables = ['%sales', 
                         'Bribe oper license', 
                         'Bribe import license', 
                         'Bribe tax', 
                         'Bribe constr permit', 
                         'Bribe electricity', 
                         'Bribe water', 
                         'Bribe phone', 
                         '% of contract', 
                         'Mgmt time on buro', 
                         '# of tax visits', 
                         'Wait oper license', 
                         'Wait imp license', 
                         'Wait electricity', 
                         'Wait water', 
                         'Wait phone', 
                         'Wait constr permit', 
                         'Wait customs', 
                         'Fairness of courts', 
                         'Obst corruption', 
                         'Obst bus license', 
                         'Obst customs', 
                         'Obst tax rates', 'Obst tax admin', 'Obst electricity', 
                         'Obst courts', 'Year in oper', 'Firm size', 'Low edu labor', 
                         'Female', 'Mgr experience', 'ISO', 'Compete w/informal', 
                         'Pmts for security', 'Obst crime', 
                         'Bribe oper license*Wait oper license', 
                         'Bribe import license*Wait imp license', 'Bribe tax*# of tax visits', 
                         'Bribe constr permit*Wait constr permit', 
                         'Bribe electricity*Wait electricity', 'Bribe water*Wait water', 
                         'Bribe phone*Wait phone', 'ln_GDPpercap']

# Encode the 'country' variable using one-hot encoding
encoded_countries = pd.get_dummies(df['country'], prefix='country')

# Concatenate the encoded countries with the other independent variables
X = pd.concat([df[independent_variables], encoded_countries], axis=1)

# Add a constant column for the intercept
X = sm.add_constant(X)

# Perform robust regression using statsmodels
robust_model_1 = sm.RLM(ysalesgrowth, X, M=sm.robust.norms.HuberT())
robust_results_1 = robust_model_1.fit()

# Print the regression results
print(robust_results_1.summary())

save_regression_summary_image(robust_results_1, 'regression_summary_reg1_robust_yislnsalesgrowth.png')

#repeating the same for our second dependent variable
#using the same variable names to not overburden python's limited ram on this PC
robust_model_2 = sm.RLM(ypercentsales, X, M=sm.robust.norms.HuberT())
robust_results_2 = robust_model_2.fit()
print(robust_results_2.summary())
save_regression_summary_image(robust_results_2, 'regression_summary_reg2_robust_yispercentsalesgrowth.png')


# Regression number 3
# Fixed Effects and OLS
# we use the same dependent and independent variables as our last regression
# but we try fitting it to a non-robust OLS regression
# Perform fixed effects regression using statsmodels
model_3 = sm.OLS(ysalesgrowth, sm.add_constant(df[independent_variables]))
model_3 = model_3.fit(cov_type='cluster', cov_kwds={'groups': df['country']})

# Print the regression results
print(model_3.summary())
save_regression_summary_image(model_3, 'regression_summary_reg3_FE_OLS_yislnsalesgrowth.png')

model_4 = sm.OLS(ypercentsales, sm.add_constant(df[independent_variables]))
model_4 = model_4.fit(cov_type='cluster', cov_kwds={'groups': df['country']})
print(model_4.summary())
save_regression_summary_image(model_4, 'regression_summary_reg4_FE_OLS_yispercentsalesgrowth.png')

# the OLS model does not work well
# almost all of our coefficients are insignificant 

#Fixed Effects + Robust

# Add fixed effects (dummy variables) for countries
fixed_effects = pd.get_dummies(df['country'], prefix='country', drop_first=True)

# Concatenate the fixed effects with the independent variables
X_with_fixed_effects = pd.concat([X, fixed_effects], axis=1)

# Perform fixed effects and robust regression using statsmodels
robust_model_5 = sm.RLM(ysalesgrowth, X_with_fixed_effects, M=sm.robust.norms.HuberT())
robust_results_5 = robust_model_5.fit()

# Print the regression results
print(robust_results_5.summary())

save_regression_summary_image(robust_results_5, 'regression_summary_reg5_FE_robust_yislnsalesgrowth.png')

robust_model_6 = sm.RLM(ypercentsales, X_with_fixed_effects, M=sm.robust.norms.HuberT())
robust_results_6 = robust_model_6.fit()

# Print the regression results
print(robust_results_6.summary())

save_regression_summary_image(robust_results_6, 'regression_summary_reg6_FE_robust_yispercentsalesgrowth.png')



#Let's adjust our independent variables and remove insignificant coefficients

independent_variables = ['Bribe constr permit', 
                         'Bribe electricity', 
                         '% of contract', 
                         'Mgmt time on buro', 
                         'Wait oper license', 
                         'Wait water', 
                         'Wait phone', 
                         'Wait constr permit', 
                         'Obst corruption', 
                         'Obst bus license', 
                         'Obst customs', 
                         'Obst tax rates', 'Obst tax admin',
                         'Obst courts', 'Year in oper', 'Firm size', 'Low edu labor', 
                         'Female', 'ISO', 'Compete w/informal', 
                         'Pmts for security', 'Obst crime',  
                         'Bribe constr permit*Wait constr permit',
                         'ln_GDPpercap']

X = pd.concat([df[independent_variables], encoded_countries], axis=1)

# Add a constant column for the intercept
X = sm.add_constant(X)

# Add fixed effects (dummy variables) for countries
X_with_fixed_effects = pd.concat([X, fixed_effects], axis=1)

# Concatenate the fixed effects with the independent variables
robust_model_7 = sm.RLM(ysalesgrowth, X_with_fixed_effects, M=sm.robust.norms.HuberT())
robust_results_7 = robust_model_7.fit()

# Print the regression results
print(robust_results_7.summary())
save_regression_summary_image(robust_results_7, 'regression_summary_reg7_FE_robust_adjindvar_yislnsalesgrowth.png')

robust_model_8 = sm.RLM(ypercentsales, X_with_fixed_effects, M=sm.robust.norms.HuberT())
robust_results_8 = robust_model_8.fit()

# Print the regression results
print(robust_results_8.summary())
save_regression_summary_image(robust_results_8, 'regression_summary_reg8_FE_robust_adjindvar_yispercentsalesgrowth.png')


#visualizing our final model's residuals and coefficients
def plot_regression_results(model, coef_plot_filename, residual_plot_filename):

    # Exclude country variables from coefficient plot
    excluded_variables = ["country_Albania", "country_Armenia", "country_Azerbaijan", "country_Belarus","country_Bih","country_Bulgaria","country_Croatia",	"country_Czech",	"country_Estonia",	"country_Georgia",	"country_Hungary",	"country_Kazakhstan",	"country_Kosovo",	"country_Kyrgyzstan",	"country_Latvia",	"country_Lithuania",	"country_Macedonia",	"country_Moldova",	"country_Mongolia",	"country_Montenegro",	"country_Poland",	"country_Romania",	"country_Russia",	"country_Serbia",	"country_Slovakia",	"country_Slovenia",	"country_Tajikistan",	"country_Turkey",	"country_Ukraine",	"country_Uzbekistan"] 

    # Coefficient Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    coefficients = model.params.drop('const')
    coefficients = coefficients.drop(excluded_variables)
    coef_ci = model.conf_int().drop('const')
    coef_ci = coef_ci.drop(excluded_variables)
    coef_df = pd.DataFrame({'Coefficient': coefficients, 'CI_lower': coef_ci.iloc[:, 0], 'CI_upper': coef_ci.iloc[:, 1]})
    coef_df.plot(kind='bar', y=['Coefficient'], yerr=coef_df[['Coefficient', 'CI_lower', 'CI_upper']].values.T, ax=ax)
    ax.set_xlabel('Independent Variables')
    ax.set_ylabel('Coefficient')
    ax.set_title('Coefficient Plot with Confidence Intervals')
    plt.xticks(rotation=45, ha='right', fontsize=8)  # Set rotation angle, alignment, and font size of x-axis labels
    plt.tight_layout()
    plt.savefig(coef_plot_filename, dpi=300)  # Set the DPI to 300 for higher resolution
    plt.close()

    # Residual Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    standardized_residuals = model.resid / np.sqrt(np.mean(model.resid**2))
    ax.scatter(model.fittedvalues, standardized_residuals)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Standardized Residuals')
    ax.set_title('Residual Plot')
    plt.tight_layout()
    plt.savefig(residual_plot_filename, dpi=300)  # Set the DPI to 300 for higher resolution
    plt.close()


# making our actual plots for both our final model specifications

plot_regression_results(robust_results_7,"coefplot_reg7.png","resplot_reg7.png")

plot_regression_results(robust_results_8,"coefplot_reg8.png","resplot_rep8.png")








