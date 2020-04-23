import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from fbprophet import Prophet
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from datetime import datetime
import time
import requests


### DATASETS
hiv_death_rates = pd.read_csv("hiv-death-rates-by-age.csv")
number_people_hiv = pd.read_csv("number-of-people-living-with-hiv.csv")
share_population = pd.read_csv("share-of-population-infected-with-hiv-ihme.csv")
death_newcases = pd.read_csv("deaths-and-new-cases-of-hiv.csv")
share_women = pd.read_csv("share-of-women-among-the-population-living-with-hiv.csv")
children_orphaned = pd.read_csv("number-of-children-orphaned-from-aids.csv")
antiretroviral_coverage = pd.read_csv("antiretroviral-therapy-coverage-among-people-living-with-hiv.csv")

### MERGING

#number_people_hiv column values
prevalence = number_people_hiv["Prevalence - HIV/AIDS - Sex: Both - Age: All Ages (Number) (people with HIV)"]
#Store prevalence in death_newcases dataset
death_newcases["Number of people living with HIV"] = prevalence
#Prevalence in age 15-49 percent
prevalence_15_49 = share_population["Prevalence - HIV/AIDS - Sex: Both - Age: 15-49 years (Percent) (%)"]
#Store prevalence_15_49 in death_newcases dataset
death_newcases["Prevalence age 15-49 percent"] = prevalence_15_49
#Adding HIV rates by age columns into death_newcases
death_rates_age = hiv_death_rates.iloc[:, 3:-1]
hiv_data = pd.concat([death_newcases, death_rates_age], axis =1)

### PREPROCESSING
hiv_data = hiv_data.round(3)
#Dropping col
hiv_data.drop("Code", axis=1, inplace=True)
#Renaming columns
hiv_data.columns = ["Country", "Year", "Death", "New_infections",
                    "Living with HIV (x10)", "Living with HIV", "HIV_Age 15_49",
                    "HIV under 5(per 100,000)", "HIV all ages(per 100,000)", "HIV 70+(per 100,000)",
                    "HIV age 5_14(per 100,000)", "HIV age 15_49(per 100,000)", "HIV age 50_69(per 100,000)"]
hiv_data.columns = hiv_data.columns.str.replace(" ", "_").str.lower()
#Dropping unnecessary col
hiv_data.drop('living_with_hiv_(x10)', axis=1, inplace=True)

################ GDP AS "merged_hiv_gdp ######################

# gdp1 = pd.read_csv("gdp1.csv")
#
# columns = ['Country Name','1990', '1991', '1992', '1993', '1994', '1995',
#        '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
#        '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
#        '2014', '2015', '2016', '2017']
# gdp1 = gdp1[columns]
# #Unique countries in gdp dataset
# hiv_countries = list(hiv_data.country.unique())
# gdp1_countries = list(gdp1["Country Name"].unique())
# mask1 = [e in gdp1_countries for e in hiv_countries]
#
# ##TURNING INTO LIST
# list_countries1 = list(pd.Series(hiv_countries)[mask1])
#
# years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996',
#        '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005',
#        '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
#        '2015', '2016', '2017']
# ### TURN INTO DATASET
# adam_test = gdp1[gdp1["Country Name"].isin(list_countries1)]
# list_of_dicts = []
# # (Country, year, gdp)
# for current_country in adam_test['Country Name'].unique():
#     for current_year in years:
#         temp_dict = {}
#         temp_dict['country'] = current_country
#         temp_dict['year'] = current_year
#         temp_dict['gdp'] = adam_test[(adam_test['Country Name'] == current_country)][current_year].iloc[0]
#         list_of_dicts.append(temp_dict)
# adam_df = pd.DataFrame(list_of_dicts)
#
# ### MERGING GDP AND HIV
# adam_df["year"] = adam_df["year"].astype(int)
# merged_hiv_gdp = hiv_data.merge(adam_df, on=['country', 'year'], how='left')

### THIS DOES THE FOLLOWING FUNCTION
################################################################
# #### MERGED DB
# merged = merged_hiv_gdp.copy()
# ### DROP GDP
# merged.drop("gdp", axis=1, inplace=True)
#
# ### WANT NEXT INFECTIONS
# merged["next_infections"] = merged.groupby("country")["new_infections"].shift(-1)
#
# ### COUNTRY DUMMIES
# country_dummies = pd.get_dummies(merged["country"])
# ### MERGING COUNTRIES TO DB
# merged = pd.concat([merged, country_dummies], axis=1)
#
# ###TEST SET 2017
# test_merged = merged[merged["year"] == 2017]
# ###TRAIN SET NOT 2017
# train_merged = merged[merged["year"] != 2017]
# ###TRAIN LABEL
# train_label = train_merged["next_infections"]
#
# #Drop next infections TRAIN
# train_merged.drop(["next_infections", "country"], axis=1, inplace=True)
# #Drop next infections TEST
# test_merged.drop(["next_infections", "country"], axis=1, inplace=True)
###################### TO PREDICT ##############################

#### MERGED DB

######## AREA POP CONTINENT #########
sub_continents = ['Andean Latin America', 'Australasia', 'Caribbean', 'Central Asia',
       'Central Europe',
       'Central Europe, Eastern Europe, and Central Asia',
       'Central Latin America', 'Central Sub-Saharan Africa', 'East Asia',
       'Eastern Europe', 'Eastern Sub-Saharan Africa', 'Latin America and Caribbean', 'North Africa and Middle East', 'North America',
       'Oceania', 'South Asia', 'Southeast Asia',
       'Southeast Asia, East Asia, and Oceania', 'Southern Latin America',
       'Southern Sub-Saharan Africa', 'Sub-Saharan Africa',
       'Tropical Latin America',
       'Western Europe', 'Western Sub-Saharan Africa', 'World']
income_list = ['High SDI',
       'High-income', 'High-income Asia Pacific', 'High-middle SDI', 'Low SDI', 'Low-middle SDI',
       'Middle SDI']

def merge_data_pop(data):
    import Scrapped_datasets as scrap
    ### CALL SCRAPPED DB
    scrap_countries = scrap.scrap_countries()
    ### MERGE
    merged = data.merge(scrap_countries, on="country", how='left')
    ##### 3 datasets
    continent = merged[merged.country.isin(sub_continents)]
    income = merged[merged.country.isin(income_list)]
    data = merged[~merged.country.isin(income_list) & ~merged.country.isin(sub_continents)]
    data = data[data["country"] != "United States Virgin Islands"]
    #### DROP COLS FOR continent and income
    continent.drop(['area_km', 'continent'], axis=1, inplace=True)
    income.drop(['area_km', 'continent'], axis=1, inplace=True)
    data.drop(['area_km'], axis=1, inplace=True)
    return data, continent, income

def merge_data_gdp(data):
    gdp = pd.read_csv("gdp1.csv")
    columns = ['Country Name', '1990', '1991', '1992', '1993', '1994', '1995',
               '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
               '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
               '2014', '2015', '2016', '2017']
    gdp = gdp[columns]
    ###Unique countries in gdp dataset
    hiv_countries = list(data.country.unique())
    gdp_countries = list(gdp["Country Name"].unique())
    ###DOES IT MATCH?
    mask = [e in gdp_countries for e in hiv_countries]
    ##TURNING INTO LIST
    list_countries = list(pd.Series(hiv_countries)[mask])
    years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996',
             '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005',
             '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
             '2015', '2016', '2017']
    ### TURN INTO DATASET
    adam_test = gdp[gdp["Country Name"].isin(list_countries)]
    list_of_dicts = []
    # (Country, year, gdp)
    for current_country in adam_test['Country Name'].unique():
        for current_year in years:
            temp_dict = {}
            temp_dict['country'] = current_country
            temp_dict['year'] = current_year
            temp_dict['gdp'] = adam_test[(adam_test['Country Name'] == current_country)][current_year].iloc[0]
            list_of_dicts.append(temp_dict)
    adam_df = pd.DataFrame(list_of_dicts)

    ### MERGING GDP AND HIV
    adam_df["year"] = adam_df["year"].astype(int)
    merged_hiv_gdp = data.merge(adam_df, on=['country', 'year'], how='left')
    return merged_hiv_gdp

######### SPLITTING DATASET BASED ON ISSUE #########

def train_label_test(data, column, years):
    #### MERGED DB
    merged = data.copy()
    #### NEXT YEAR
    new_column = column + "1"
    #### WHAT YEARS
    what_years = list(merged.year.unique())[-years:]
    ### WANT NEXT INFECTIONS
    merged[new_column] = merged.groupby("country")[column].shift(-years)
    ### COUNTRY DUMMIES
    country_dummies = pd.get_dummies(merged["country"])
    ##YEAR DUMMIES
    year_dummies = pd.get_dummies(merged["year"])

    if 'continent' in merged.columns:
        continent_dummies = pd.get_dummies(merged["continent"])
        ### MERGING COUNTRIES TO DB
        merged = pd.concat([merged, continent_dummies, year_dummies, country_dummies], axis=1)
    else:
        merged = pd.concat([merged, year_dummies, country_dummies], axis=1)
    ###TEST SET 2017
    test_merged = merged[merged["year"].isin(what_years)]
    ###TRAIN SET NOT 2017
    train_merged = merged[~merged["year"].isin(what_years)]
    ###TRAIN LABEL
    train_label = train_merged[new_column]
    ### Drop next infections TRAIN
    train_merged.drop([new_column], axis=1, inplace=True)
    ### Drop next infections TEST
    test_merged.drop([new_column], axis=1, inplace=True)
    return train_merged, train_label, test_merged

#### MERGE PREDICTION DATA, LABEL, TEST(YEARS), PREDICTION
model_xgb = xgb.XGBRegressor(objective="reg:squarederror")
continents = ['asia', 'europe', 'africa', 'oceania', 'north_america', 'south_america']
def predict_regression(x, y, e, model, since, years_to_predict, by_group=None):
    model_lasso = Lasso()
    if by_group:
        #Stacks group: dataframe
        #Merges X, Y
        data = pd.concat([x, y], axis=1)
        #Dictionary group: prediction
        final_datasets = {}
        for group in x[by_group].unique():
            x_variable = pd.DataFrame(data[data[by_group] == group])
            x_variable = x_variable[x_variable["year"] >= since]
            x_variable = x_variable.drop([by_group, 'year'], axis=1)
            test_var = pd.DataFrame(e[e[by_group] == group])
            test_var = test_var[test_var["year"] >= since]
            test_var = test_var.drop([by_group, 'year'], axis=1)
            if by_group == "country":
                x_variable = x_variable.drop('continent', axis=1)
                test_var = test_var.drop('continent', axis=1)
            if by_group == "continent":
                x_variable = x_variable.drop('country', axis=1)
                test_var = test_var.drop('country', axis=1)
            #x and y variables
            x_var = x_variable.iloc[:, :-1]
            y_var = x_variable.iloc[:, -1]
            ### X + E
            e_shape = test_var.shape[0]
            merge_x_e = pd.concat([x_var, test_var], axis=0)
            scale = StandardScaler()
            merge_xe_scaled = pd.DataFrame(scale.fit_transform(merge_x_e), columns=merge_x_e.columns)
            x_scaled_ = merge_xe_scaled.iloc[:-e_shape, :]
            e_scaled_ = merge_xe_scaled.iloc[-e_shape:, :]

            x_train, x_test, y_train, y_test = train_test_split(x_scaled_, y_var, test_size=0.20)
            ####FIT
            model.fit(x_train, y_train)
            prediction = model.predict(e_scaled_)
            #Stacking prediction to variables
            x_var["prediction"] = y_var
            test_var["prediction"] = prediction
            ## LASSO
            model_lasso.fit(x_train, y_train)
            lasso = model_lasso.predict(e_scaled_)
            x_var["lasso"] = y_var
            test_var["lasso"] = lasso
            ## RIDGE
            model_xgb.fit(x_train, y_train)
            xgb = model_xgb.predict(e_scaled_)
            x_var["xgb"] = y_var
            test_var["xgb"] = xgb

            merged_back = pd.concat([x_var, test_var], axis=0)
            if by_group == "continent":
                #Gets columns to merge back df
                columns = merged_back.iloc[:, :10]
                continents1 = merged_back.iloc[:, 10:16].idxmax(axis=1).rename("continent")
                years = merged_back.iloc[:, 16:44].idxmax(axis=1).rename("year")
                countries = merged_back.iloc[:, 44:-3].idxmax(axis=1).rename("country")
                prediction = merged_back.iloc[:, -3]
                lasso = merged_back.iloc[:, -2]
                xgb = merged_back.iloc[:, -1]
                #linear = merged_back.iloc[:, -1]
                future_year = pd.DataFrame(years + years_to_predict)["year"].rename("year_predicted")
                #Final dataset for each group. To dictionary, group: dataframe
                final_data = pd.concat([countries, continents1, future_year, prediction, lasso, xgb, years, columns], axis=1)
                final_datasets[group] = final_data
            if by_group == "country":
                columns = merged_back.iloc[:, :10]
                continents1 = merged_back.iloc[:, 10:16].idxmax(axis=1).rename("continent")
                years = merged_back.iloc[:, 16:44].idxmax(axis=1).rename("year")
                countries = merged_back.iloc[:, 44:-3].idxmax(axis=1).rename("country")
                prediction = merged_back.iloc[:, -3]
                lasso = merged_back.iloc[:, -2]
                xgb = merged_back.iloc[:, -1]
                #linear = merged_back.iloc[:, -1]
                future_year = pd.DataFrame(years + years_to_predict)["year"].rename("year_predicted")
                # Final dataset for each group. To dictionary, group: dataframe
                final_data = pd.concat([countries, continents1, future_year, prediction, lasso, xgb, years, columns], axis=1)
                final_datasets[group] = final_data
        #Merge dataframes with predictions
        merge_final = pd.concat(final_datasets.values())
        merge_final = merge_final.sort_values(["country", "year"])
        return merge_final


def top_countries_list(data, group, number):
    ### GET TOP COUNTRIES LIST
    top_countries_continent = []
    for i in data[group].unique():
        dataset = data[data[group] == i]
        top_countries = (list(dataset.groupby("country")["prediction"]
                              .mean().sort_values(ascending=True).head(number).reset_index()["country"]))
        for i in top_countries:
            top_countries_continent.append(i)
    return top_countries_continent


def predict_regression_prophet(data, issue, years_to_predict, top_countries, by_group=None):
    data = data[data.country.isin(top_countries)]
    if by_group:
        #Dictionary group: prediction
        final_dataset = {}
        for group in data[by_group].unique():
            country_data = data[data[by_group] == group]
            which_country = country_data.country.iloc[0]
            which_continent = country_data.continent.iloc[0]
            country_var = country_data[["year", issue]]
            country_var = country_var.rename(columns={'year':'ds', issue:"y"})
            count = country_var["y"].copy()
            country_var['y'] = np.log(country_var["y"])
            #country_var['y'], lam = boxcox(country_var['y']) ####
            country_var['ds'] = country_var["ds"].astype(str)
            country_var['ds'] = pd.to_datetime(country_var["ds"])
            country_var['y_orig'] = count
            ####FIT
            m = Prophet(seasonality_mode='multiplicative')
            m.fit(country_var)
            future = m.make_future_dataframe(periods=years_to_predict, freq="Y")
            predictions = m.predict(future)
            country_pred_df = predictions.copy()
            if by_group == "country":
                country_pred_df["continent"] = which_continent
                country_pred_df["country"] = group
            if by_group == "continent":
                country_pred_df["continent"] = group
                country_pred_df["country"] = which_country

            country_pred_df['yhat'] = np.exp(country_pred_df['yhat'])
            country_pred_df['yhat_lower'] = np.exp(country_pred_df['yhat_lower'])
            country_pred_df['yhat_upper'] = np.exp(country_pred_df['yhat_upper'])
            #country_pred_df[['yhat', 'yhat_upper', 'yhat_lower']] = country_pred_df[['yhat', 'yhat_upper', 'yhat_lower']].apply(lambda x: inv_boxcox(x, lam)) ####
            country_var["y_box"] = country_var['y']
            country_var["y"] = count
            ##Arrange
            forecast = country_pred_df.set_index('ds')
            country_ind = country_var.set_index('ds')

            country_ind.index = pd.to_datetime(country_ind.index)
            connect_date = country_ind.index[-2]
            mask = (forecast.index > connect_date)
            predict_df = forecast.loc[mask]
            viz_df = country_ind.join(predict_df[['yhat', 'yhat_lower', 'yhat_upper']], how='outer')
            if by_group == "country":
                viz_df["continent"] = which_continent
                viz_df["country"] = group
            if by_group == "continent":
                viz_df["continent"] = group
                viz_df["country"] = which_country
            final_dataset[group] = viz_df
        #Merge dataframes with predictions
        merge_final = pd.concat(final_dataset.values())
        merge_final = merge_final.sort_values(["country", "ds"])
        return merge_final


def plot_time(data, prediction, years_to_predict, hue):
    data["predicted_year"] = data["year"] + years_to_predict
    plt.figure(figsize=(10, 8))
    sns.lineplot(data=data, y=prediction,x='predicted_year', hue=hue)
    plt.tick_params(labelsize=20, pad=10)
    plt.legend(fontsize=20)
    plt.xlabel("Year", fontsize=25, labelpad=10)
    plt.ylabel("New Infections", fontsize=25, labelpad=10)
    plt.title("New Infections trough Time", fontsize=35, y=1.025)
    return

def plot_time_group(data, issue, prediction, years_to_predict, number, group, hue):
    data["predicted_year"] = data["year"] + years_to_predict
    issue = issue.replace("_", " ")
    for i in data[group].unique():
        plt.figure(figsize=(12, 10))
        dataset = data[data[group] == i]
        top_countries = list(
            dataset.groupby("country")["prediction"]
                .mean().sort_values().tail(number).reset_index()["country"])
        dataset = dataset[dataset["country"].isin(top_countries)]
        sns.set_context("poster", rc={"lines.linewidth": 4.0})
        sns.lineplot(data=dataset, y=prediction, x=list(dataset['predicted_year']),
                     hue=hue, palette="Set2")
        plt.tick_params(labelsize=20, pad=12)
        plt.legend(fontsize=25)
        plt.xlabel("Predicted year", fontsize=25, labelpad=20)
        plt.xticks(rotation=25)
        plt.ylabel(issue, fontsize=25, labelpad=20)
        plt.title("{} by time".format(issue), fontsize=35, y=1.025)
    return


def plot_prophet(data, issue, group, prediction):
    for i in data[group].unique():
        data_country = data[data[group] == i]
        fig, ax1 = plt.subplots(figsize=(14, 10))
        issue = issue.replace("_", " ").capitalize()
        year = "Year"
        country = i.capitalize()
        ax1.grid(b=True, which='major', color='lightblue', linestyle='dotted')
        ax1.plot(data_country.y)
        ax1.plot(data_country.yhat, color='black', linestyle=':')
        ax1.fill_between(data_country.index, data_country['yhat_upper'], data_country['yhat_lower'], alpha=0.5,
                         color='darkgray')
        ax1.set_title('{}: {} by {}'.format(country, issue, year))
        ax1.set_ylabel("{}".format(issue))
        ax1.set_xlabel('{}'.format(year))
        ax1.legend(("Observed", "Predicted"))
        fig;
    return


def plot_prophet_and_regression(data_prophet, data_reg, issue, group, prediction, model):
    data = data_prophet
    for i in data[group].unique():
        data_country = data[data[group]==i]
        data_reg_country = data_reg[data_reg[group]==i]

        data_reg_country['ds'] = data_reg_country["year_predicted"].astype(str)
        data_reg_country['ds'] = pd.to_datetime(data_reg_country["ds"])
        data_reg_country = data_reg_country.set_index('ds')
        fig, ax1 = plt.subplots(figsize=(14, 10))
        issue = issue.replace("_", " ").capitalize()
        year = "Year"
        country = i.capitalize()
        #sns.lineplot(data=data_reg_country, y=prediction, x=list(dataset['predicted_year']),hue=hue, palette="Set2")

        ax1.grid(b=True, which='major', color='lightblue', linestyle='dotted')
        # FROM REG
        ax1.plot(data_reg_country[prediction])
        # LASSO
        ax1.plot(data_reg_country['lasso'])
        # RIDGE
        ax1.plot(data_reg_country['xgb'])
        # LINEAR
        #ax1.plot(data_reg_country['linear'])

        # FROM PROPHET
        ax1.plot(data_country.y)
        ax1.plot(data_country.yhat, color='black', linestyle=':')
        ax1.fill_between(data_country.index, data_country['yhat_upper'], data_country['yhat_lower'], alpha=0.5,
                         color='darkgray')
        ax1.set_title('{}: {} by {}'.format(country, issue, year))
        ax1.set_ylabel("{}".format(issue))
        ax1.set_xlabel('{}'.format(year))
        ax1.legend(("ElasticNet regression", "Lasso regression", "XGB Regression", "Observed", "Prophet Prediction"))
        fig;
    return

countries = ["Brazil", "Mexico"]
def filtered_datasets(data, country):
    child = d.children_orphaned
    ar_cov = d.antiretroviral_coverage
    ar_cov = ar_cov.rename({"Year": "year", 'Entity':'country'}, axis=1)
    gdp_merged = d.merge_data_gdp(data)
    orphan = child[child["Entity"].isin(country)]
    orphan = orphan.rename({"Year": "year", 'Entity':'country'}, axis=1)
    ar_coverage = ar_cov[ar_cov["country"].isin(country)]
    dataset = gdp_merged.merge(ar_coverage, how='left', on=['year','country'])
    dataset = dataset.merge(orphan, how='left', on=['year','country'])
    dataset.drop(["Code_x", "Code_y"], axis=1, inplace=True)
    return dataset
# dataset = filtered_datasets(mex_bra_data, countries)
# dataset = dataset.dropna()