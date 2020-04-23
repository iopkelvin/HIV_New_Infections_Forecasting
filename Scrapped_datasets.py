from __future__ import print_function, division
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
requests.__path__


def scrap_countries():
    url = 'http://www.geonames.org/countries/'
    response = requests.get(url)
    page = response.text
    #WebScrap Beautifulsoup
    soup = BeautifulSoup(page, "html.parser")

    values = [entry.text for entry in soup.find_all('td')][2:]

    columns = ['code', 'code1', 'numeric', 'fips', 'country', 'capital', 'area_km', 'population', 'continent']


    def value_to_dict(values, columns):
        dictionary = defaultdict(list)
        cols = len(columns)

        for idx, article_value in enumerate(values):
            if idx % cols == 0:
                for i in range(cols):
                    dictionary[columns[i]].append(values[idx + i])
        return dictionary

    replaced = {"EU": "europe", "AS": "asia",
                "NA": "north_america", "AF": "africa",
                "AN": "antarctica", "SA": "south_america", "OC": "oceania"}

    countries = pd.DataFrame(value_to_dict(values, columns))
    #### ADD continent
    countries["continent"] = countries.continent.replace(replaced)
    #### FILTER
    countries = countries.iloc[:, 4:]
    #### DROP CAPITAL
    countries.drop("capital", axis=1, inplace=True)

    ###### REPLACE MISSING COUNTRIES #######
    replace_countries = {"Cabo Verde": "Cape Verde",
                         "Czechia": "Czech Republic",
                         "Eswatini": "Swaziland",
                         "Ivory Coast": "Cote d'Ivoire",
                          "DR Congo": "Democratic Republic of Congo",
                         "North Macedonia": "Macedonia",
                         "Micronesia": "Micronesia (country)",
                          "St Vincent and Grenadines": "Saint Vincent and the Grenadines",
                          "São Tomé and Príncipe": "Sao Tome and Principe",
                          "Timor-Leste": "Timor"}
    #countries = countries.replace({'country': replace_countries})
    countries["country"] = countries.country.replace(replace_countries)

    ### ADD UK ###
    en = {'country':"England", 'area_km':'130,395', 'population':'53,107,169', 'continent':'europe'}
    ir = {'country':"Northern Ireland", 'area_km':'13,843', 'population':'1,851,621', 'continent':'europe'}
    sc = {'country':"Scotland", 'area_km':'78,772', 'population':'5,373,000', 'continent':'europe'}
    wa = {'country':"Wales", 'area_km':'20,779', 'population':'3,099,086', 'continent':'europe'}
    countries = countries.append([en, ir, sc, wa], ignore_index=True)

    ##### DROP POPULATION COUNT #######
    countries.drop("population", axis=1, inplace=True)

    ##### INDEX UNNECESSARY #####
    not_index = [  4,   7,   9,  14,  15,  26,  30,  34,  39,  45,  51,  54,  55,
                 67,  73,  75,  81,  82,  84,  88,  91,  96,  97, 105, 107, 112,
                121, 125, 130, 139, 142, 149, 151, 153, 162, 164, 170, 171, 176,
                181, 182, 186, 189, 200, 202, 205, 212, 215, 217, 221, 228, 233,
                237, 240, 241, 244, 246, 248]

    ### FILTER DF
    countries = countries[~countries.index.isin(not_index)]
    countries["area_km"] = countries.area_km.str.replace(",","").astype(float)
    return countries

def scrap_cases_rta():
    url = 'https://apps.who.int/gho/athena/data/GHO/HIV_0000000001,HIV_0000000009,HIV_ARTCOVERAGE.html?profile=ztable'
    response = requests.get(url)
    page = response.text
    # WebScrap Beautifulsoup
    soup = BeautifulSoup(page, "html.parser")

    # Parsing and creading list of strings
    article_values = [entry.text for entry in soup.find_all('td')]

    # Columns in dataset
    columns = ["GHO", "PUBLISH_STATE", "YEAR", "REGION", "COUNTRY", "DISPLAY_VALUE", "NUMERIC_VALUE", "LOW_RANGE",
               "HIGH_RANGE", "COMMENT"]

    article_dict = defaultdict(list)
    for idx, article_value in enumerate(article_values):
        if idx % 10 == 0:
            for i in range(9):
                article_dict[columns[i]].append(article_values[idx + i])

    # Putting dict into DF
    dataset = pd.DataFrame(article_dict)
    # standardizing
    dataset.columns = dataset.columns.str.lower()
    # Sorting by GHO
    dataset = dataset.sort_values("gho")
    # Sorting by Year
    dataset = dataset.sort_values("year")

    ####### DROP EMPTY ROWS
    dataset = dataset.replace("", np.nan).dropna()
    # TURN COLS INTO FLOAT
    dataset[['numeric_value', 'low_range', 'high_range']] = dataset[
        ['numeric_value', 'low_range', 'high_range']].replace(" ", "").astype(float)

    return dataset