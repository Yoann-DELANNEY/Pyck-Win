#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json
import pandas as pd
import numpy as np


# In[6]:


path = "D:\\Documents\\Recherche d'emplois\\Formation\Datascientest\\Projet\\Datas\\"


# In[7]:


def imports(saison,path=''):
    """
    Renvoie un dataframe comprenant la fusion des résultats des matchs et des cotes des bookmakers
    
    path   : String du chemin allant jusqu'au dossier contenant les dossiers seasonXX-XX.
             Non obligatoire si le fichier de donn&es se trouve dans le même dossier que l'exécutable.
             
    season : String contenant les 2 derniers digits des années de la saison à analyser
        exemple : season = "1718" # Pour la saison 2017-2018
    
    """
    
#     
    new_path_saison = path + "season" + season[:1] + "-" + season[2:] + "/season_match_stats.json"
    data_saison = json.load(open(new_path_saison))
    
    table = []
    for index, value in data_saison.items():
        ligne = [index] + list(value.values())  
        table.append(ligne)
    
    df_saison = pd.DataFrame(table, columns = ['id_match', 'home_team_id', 'away_team_id', 'home_team_name', 'away_team_name', 'date_string', 'half_time_score', 'full_time_score'])
    df_saison.head()
    
    
#     
    new_path_match = path + "season" + season[:1] + "-" + season[2:] + "season_stats.json"    
    data_match = json.load(open(new_path_match,encoding='utf-8'))
    
    table = []
    for index, value in data_match.items():
        home = dict(list(value.values())[0])
        ligne = [index] + list(home['team_details'].values())  
        table.append(ligne)
    
    df_match_home = pd.DataFrame(table, columns = ['id_match', 'team_id_home', 'team_name_home', 'team_rating_home', 'date'])

    compteur = 0
    for index, value in data_match.items():
        home = dict(list(value.values())[0])
    
        for i,j in home['aggregate_stats'].items() :
            df_match_home.loc[compteur, i] = j
         
        compteur += 1
    
    table = []

    for index, value in data_match.items():
        away = dict(list(value.values())[1])
        ligne = [index] + list(away['team_details'].values())  
        table.append(ligne)
    
    df_match_away = pd.DataFrame(table, columns = ['id_match', 'team_id_away', 'team_name_away', 'team_rating_away', 'date'])

    compteur = 0
    for index, value in data_match.items():
        away = dict(list(value.values())[1])
    
        for i,j in away['aggregate_stats'].items() :
            df_match_away.loc[compteur, i] = j
         
        compteur += 1
        
        df = pd.merge(df_match_home,df_match_away,how='left',on='id_match')
        
        
        
        bookmakers = pd.read_csv('season-1617_bookmakers.csv')
        
    return df_stats, bookmakers


# In[8]:


def preprocess(df_stats,bookmakers):
    

    data_saison = json.load(open("season_match_stats.json"))
    
    table = []
    for index, value in data_saison.items():
        ligne = [index] + list(value.values())  
        table.append(ligne)
    
    df_saison = pd.DataFrame(table, columns = ['id_match', 'home_team_id', 'away_team_id', 'home_team_name', 'away_team_name', 'date_string', 'half_time_score', 'full_time_score'])
    df_saison.head()
    
    

    data_match = json.load(open("season_stats.json",encoding='utf-8'))
    
    table = []
    for index, value in data_match.items():
        home = dict(list(value.values())[0])
        ligne = [index] + list(home['team_details'].values())  
        table.append(ligne)
    
    df_match_home = pd.DataFrame(table, columns = ['id_match', 'team_id_home', 'team_name_home', 'team_rating_home', 'date'])

    compteur = 0
    for index, value in data_match.items():
        home = dict(list(value.values())[0])
    
        for i,j in home['aggregate_stats'].items() :
            df_match_home.loc[compteur, i] = j
         
        compteur += 1
    
    table = []

    for index, value in data_match.items():
        away = dict(list(value.values())[1])
        ligne = [index] + list(away['team_details'].values())  
        table.append(ligne)
    
    df_match_away = pd.DataFrame(table, columns = ['id_match', 'team_id_away', 'team_name_away', 'team_rating_away', 'date'])

    compteur = 0
    for index, value in data_match.items():
        away = dict(list(value.values())[1])
    
        for i,j in away['aggregate_stats'].items() :
            df_match_away.loc[compteur, i] = j
         
    compteur += 1
        
    df_match = pd.merge(df_match_home,df_match_away,how='left',on='id_match')       
    bookmakers = pd.read_csv('season-1617_bookmakers.csv')

    #Changement de format de date_x et Date pour fusionner 
    df_match['date_x'] = pd.to_datetime(df_match['date_x'])
    bookmakers['Date'] = pd.to_datetime(bookmakers['Date'])
    #dftest = pd.merge(df,bookmakers,how='left',left_on=['team_name_home','date_x','team_name_away'],right_on=['HomeTeam','Date','AwayTeam'])
    df = pd.merge(df_match,bookmakers,how='inner',left_on=['team_name_home','date_x','team_name_away'],right_on=['HomeTeam','Date','AwayTeam'])

    df=df.replace(np.nan, 0)



    #On Dichotomise la variable team_name_home
    dummy = pd.get_dummies (df['team_name_home'])
    df = pd.concat( [df, dummy ], axis = 1)
    df.drop( ['team_name_home'], axis = 1)

    #drop des pronostics du bookmakers pinnacle
    df.drop(['PSH'],axis =1)
    df.drop(['PSD'],axis =1)
    df.drop(['PSA'],axis =1)

    

    return df


# In[9]:


def train_pipe(df):
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier # À voir
    from pipeline import Pipeline
    
# Séparation en variables explicatives et variable cible
    features = df['']
    target = df.drop('',1)
    

# Train test split en gardant l'ordre des matchs
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target,
                                                    test_size = 0.2,
                                                    shuffle=False,
                                                    random_state = 777)
    
    model = RandomForestClassifier(n_estimators = 1000,
                                max_depth=4,
                                n_jobs=-1)

    model = Pipeline(steps=[('normalisation',sc),
                         ('model',rf)])

    model.fit(X_train,y_train)
    
    return model


# In[10]:


def bet(mise, model):
    
    
    return match_result, match_score, gain_perte


# In[14]:





# In[ ]:




