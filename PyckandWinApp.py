import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn import preprocessing
import socket

os.chdir(r"C:\Users\Olivier\Desktop\proj\Streamlit")

#Fonction de simulation d'un pari
def gain_paris(matchs, resultats, montants):
    #La fonction prend en entrée une liste de matchs, une liste de résultats prédits, et une liste de montants pariés
    df['benefices'] = 0.0
    benefices = 0
    for match, resultat, montant in zip(matchs, resultats, montants):
        #Itération simultanées des éléments des trois listes 
        if df.loc[match]['FTR'] == resultat:
            #bénéfice du pari si le résultat prédit est bon 
            benefice = (df.loc[match][dic_resultat[resultat]] - 1)*montant
        #Perte du montant parié si le résultat prédit est faux
        else : benefice = -1*montant
        #On agrège le bénéfice du pari au bénéfices totaux
        benefices += benefice
        
        df.at[match, 'benefices'] += benefice
    #Calcul du pourcentage gagné/perdu totaux
    pourcentage = benefices*100/sum(montants)
    
    return st.write('Montant parié :', sum(montants)), st.write('Gains/pertes :', benefices), st.write('Taux de rendement :', pourcentage, ' %')

def undummify(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df

st.title('Pyck & Win')

text = '''
---
'''

st.markdown(text)
st.write("Liste des matchs de la saison")
dataset_container = st.sidebar.beta_expander("Configuration", True)
with dataset_container:
    dataset = st.selectbox("Choisir une saison", ("Saison 14-15", "Saison 15-16", "Saison 16-17","Saison 17-18"))
with dataset_container:
    models = st.selectbox("Choisir un modèle", ("SVM", "RandomForest", "XGBoost"))
    
if dataset == "Saison 14-15":
    df = pd.read_csv("bookmakers14-15.csv", index_col=0)    
elif dataset == "Saison 15-16":
    df = pd.read_csv("bookmakers15-16.csv", index_col=0)
elif dataset == "Saison 16-17":
    df = pd.read_csv("bookmakers16-17.csv", index_col=0)
elif dataset == "Saison 17-18":
    df = pd.read_csv("bookmakers17-18.csv", index_col=0)
    
st.write(df.head(400))
features = df.drop(['date','bk_H','bk_D','bk_A','Prob_bk_H','Prob_bk_D','Prob_bk_A','FTR'],axis=1)
target = df['FTR']

X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target,
                                                    test_size = 0.2,
                                                    shuffle=False,
                                                    random_state = 777)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)



# SVM
if models=="SVM": 
        clf=SVC(probability=True)
        parametres = {'C':[0.1,1,10], 'kernel':['rbf','linear', 'poly'], 'gamma':[0.001, 0.01, 0.1]}
        model = GridSearchCV(estimator=clf, param_grid=parametres)
        grille = model.fit(X_train_scaled,y_train)
        score = model.score(X_test_scaled,y_test)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
    
# RF
if models=="RandomForest":
    
        model = RandomForestClassifier(n_estimators = 1000,
                                max_depth=4,
                                n_jobs=-1)

        model.fit(X_train_scaled,y_train)
        score = model.score(X_test_scaled,y_test)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
    
# XGBoost
if models=="XGBoost":
        train = xgb.DMatrix(data=X_train_scaled, label=y_train)
        test = xgb.DMatrix(data=X_test_scaled, label=y_test)
        params = {'booster': 'gblinear', 'learning_rate': 0.001, 'objective': 'multi:softprob', 'num_class':3}

        model = xgb.train(params=params, dtrain=train, num_boost_round=700, evals=[(train, 'train'), (test, 'eval')]);
        bst_cv = xgb.cv(params=params,
                    dtrain=train,
                    num_boost_round=100,
                    nfold=3,
                    early_stopping_rounds=60)
        test_matrix = xgb.DMatrix(data=X_test_scaled, label=y_test)
        y_pred = model.predict(test_matrix)
        y_pred_proba = model.predict(test_matrix).astype(float)


            
if dataset == "Saison 14-15":
    
    seuil_pari = .6
    df['Prob_H'] = 0.0
    df['Prob_D'] = 0.0
    df['Prob_A'] = 0.0

    try:
        df.loc[244:]['Prob_H'] = y_pred_proba[:,0]
    except:
        pass
    df.loc[244:]['Prob_D'] = y_pred_proba[:,1]
    df.loc[244:]['Prob_A'] = y_pred_proba[:,2]


    df['bet_H?'] = np.where((df['Prob_H']>df['Prob_bk_H']) & (df['Prob_H']>seuil_pari), 1 , 0)
    df['bet_H?'] = np.where((df['Prob_H']-df['Prob_bk_H']>0.15) & (df['Prob_H']>seuil_pari), 3 , df['bet_H?'])
    df['bet_D?'] = np.where((df['Prob_D']>df['Prob_bk_D']) & (df['Prob_D']>seuil_pari), 1 , 0)
    df['bet_D?'] = np.where((df['Prob_D']-df['Prob_bk_D']>0.15) & (df['Prob_D']>seuil_pari), 3 , df['bet_D?'])
    df['bet_A?'] = np.where((df['Prob_A']>df['Prob_bk_A']) & (df['Prob_A']>seuil_pari), 1 , 0)
    df['bet_A?'] = np.where((df['Prob_A']-df['Prob_bk_A']>0.15) & (df['Prob_A']>seuil_pari), 3 , df['bet_A?'])
    df['montant'] = df['bet_H?'] + df['bet_D?'] + df['bet_A?']
    df = df.fillna(0)    

#Dictionnaire pour attribuer un chiffre à un resultat de pari
    dic_resultat = {0: 'bk_H', 1: 'bk_D', 2: 'bk_A'}
#Simulation de paris
    matchs = []
    for i in range(len(df.index)):
        matchs.extend([i,i,i])

    resultats = []
    for i in range(len(df.index)):
        resultats.extend([0,1,2])  
    montants = []
    for i in range(len(df.index)):
        montants.extend(df.loc[i][['bet_H?', 'bet_D?', 'bet_A?']])

    gain_paris(matchs, resultats, montants)
    
elif dataset == "Saison 15-16":
    seuil_pari = .6
    df['Prob_H'] = 0.0
    df['Prob_D'] = 0.0
    df['Prob_A'] = 0.0
    try:
        df.loc[273:]['Prob_H'] = y_pred_proba[:,0]
    except:
        pass
    df.loc[273:]['Prob_D'] = y_pred_proba[:,1]
    df.loc[273:]['Prob_A'] = y_pred_proba[:,2]

    df['bet_H?'] = np.where((df['Prob_H']>df['Prob_bk_H']) & (df['Prob_H']>seuil_pari), 1 , 0)
    df['bet_H?'] = np.where((df['Prob_H']-df['Prob_bk_H']>0.15) & (df['Prob_H']>seuil_pari), 3 , df['bet_H?'])
    df['bet_D?'] = np.where((df['Prob_D']>df['Prob_bk_D']) & (df['Prob_D']>seuil_pari), 1 , 0)
    df['bet_D?'] = np.where((df['Prob_D']-df['Prob_bk_D']>0.15) & (df['Prob_D']>seuil_pari), 3 , df['bet_D?'])
    df['bet_A?'] = np.where((df['Prob_A']>df['Prob_bk_A']) & (df['Prob_A']>seuil_pari), 1 , 0)
    df['bet_A?'] = np.where((df['Prob_A']-df['Prob_bk_A']>0.15) & (df['Prob_A']>seuil_pari), 3 , df['bet_A?'])
    df['bet_A?'] = np.where((df['Prob_A']>df['Prob_bk_A']) & (df['Prob_A']>seuil_pari), 1 , 0)
    df = df.fillna(0)
    
#Dictionnaire pour attribuer un chiffre à un resultat de pari
    dic_resultat = {0: 'bk_H', 1: 'bk_D', 2: 'bk_A'}
#Simulation de paris
    matchs = []
    for i in range(len(df.index)):
        matchs.extend([i,i,i])
    resultats = []
    for i in range(len(df.index)):
        resultats.extend([0,1,2])  
    montants = []
    for i in range(len(df.index)):
        montants.extend(df.loc[i][['bet_H?', 'bet_D?', 'bet_A?']])
    gain_paris(matchs, resultats, montants)
    
elif dataset == "Saison 16-17":    
    seuil_pari = .6
    df['Prob_H'] = 0.0
    df['Prob_D'] = 0.0
    df['Prob_A'] = 0.0
    try:
        df.loc[304:]['Prob_H'] = y_pred_proba[:,0]
    except:
        pass
    df.loc[304:]['Prob_D'] = y_pred_proba[:,1]
    df.loc[304:]['Prob_A'] = y_pred_proba[:,2]
    
    df['bet_H?'] = np.where((df['Prob_H']>df['Prob_bk_H']) & (df['Prob_H']>seuil_pari), 1 , 0)
    df['bet_H?'] = np.where((df['Prob_H']-df['Prob_bk_H']>0.15) & (df['Prob_H']>seuil_pari), 3 , df['bet_H?'])
    df['bet_D?'] = np.where((df['Prob_D']>df['Prob_bk_D']) & (df['Prob_D']>seuil_pari), 1 , 0)
    df['bet_D?'] = np.where((df['Prob_D']-df['Prob_bk_D']>0.15) & (df['Prob_D']>seuil_pari), 3 , df['bet_D?'])
    df['bet_A?'] = np.where((df['Prob_A']>df['Prob_bk_A']) & (df['Prob_A']>seuil_pari), 1 , 0)
    df['bet_A?'] = np.where((df['Prob_A']-df['Prob_bk_A']>0.15) & (df['Prob_A']>seuil_pari), 3 , df['bet_A?'])
    df['bet_A?'] = np.where((df['Prob_A']>df['Prob_bk_A']) & (df['Prob_A']>seuil_pari), 1 , 0)
    df = df.fillna(0)
    
#Dictionnaire pour attribuer un chiffre à un resultat de pari
    dic_resultat = {0: 'bk_H', 1: 'bk_D', 2: 'bk_A'}
#Simulation de paris
    matchs = []
    for i in range(len(df.index)):
        matchs.extend([i,i,i])
    resultats = []
    for i in range(len(df.index)):
        resultats.extend([0,1,2])  
    montants = []
    for i in range(len(df.index)):
        montants.extend(df.loc[i][['bet_H?', 'bet_D?', 'bet_A?']])
    gain_paris(matchs, resultats, montants)
    
elif dataset == "Saison 17-18":            
    seuil_pari = .6
    df['Prob_H'] = 0.0
    df['Prob_D'] = 0.0
    df['Prob_A'] = 0.0
    try:
        df.loc[273:]['Prob_H'] = y_pred_proba[:,0]
    except:
        pass
    df.loc[273:]['Prob_D'] = y_pred_proba[:,1]
    df.loc[273:]['Prob_A'] = y_pred_proba[:,2]

    df['bet_H?'] = np.where((df['Prob_H']>df['Prob_bk_H']) & (df['Prob_H']>seuil_pari), 1 , 0)
    df['bet_H?'] = np.where((df['Prob_H']-df['Prob_bk_H']>0.15) & (df['Prob_H']>seuil_pari), 3 , df['bet_H?'])
    df['bet_D?'] = np.where((df['Prob_D']>df['Prob_bk_D']) & (df['Prob_D']>seuil_pari), 1 , 0)
    df['bet_D?'] = np.where((df['Prob_D']-df['Prob_bk_D']>0.15) & (df['Prob_D']>seuil_pari), 3 , df['bet_D?'])
    df['bet_A?'] = np.where((df['Prob_A']>df['Prob_bk_A']) & (df['Prob_A']>seuil_pari), 1 , 0)
    df['bet_A?'] = np.where((df['Prob_A']-df['Prob_bk_A']>0.15) & (df['Prob_A']>seuil_pari), 3 , df['bet_A?'])
    df['bet_A?'] = np.where((df['Prob_A']>df['Prob_bk_A']) & (df['Prob_A']>seuil_pari), 1 , 0)
    df = df.fillna(0)
    
#Dictionnaire pour attribuer un chiffre à un resultat de pari
    dic_resultat = {0: 'bk_H', 1: 'bk_D', 2: 'bk_A'}
#Simulation de paris
    matchs = []
    for i in range(len(df.index)):
        matchs.extend([i,i,i])
    resultats = []
    for i in range(len(df.index)):
        resultats.extend([0,1,2])  
    montants = []
    for i in range(len(df.index)):
        montants.extend(df.loc[i][['bet_H?', 'bet_D?', 'bet_A?']])
    gain_paris(matchs, resultats, montants)

st.write("Liste des matchs pour lesquels le modèle a parié :")	
df['montant'] = df['bet_H?'] + df['bet_D?'] + df['bet_A?']
#On garde les matchs parié
rslt_df = df[df['montant'] > 0]
rslt_df = rslt_df.reset_index(drop=True)

#Affichage en liste des matchs
#for i in range (len(rslt_df.index)):
#	st.write("Match du " + rslt_df.iloc[i]['date'] + " opposant " + rslt_df.iloc[i]['H'] + " à " + rslt_df.iloc[i]['A'] + ", Montant parié : " + str(rslt_df.iloc[i]['montant']) + " Bénéfice/Perte : " + str(rslt_df.iloc[i]['benefices']))

#Affichage dans un tableau
rslt_df = rslt_df.drop(['FTR','bk_H','bk_D','bk_A','Prob_bk_H','Prob_bk_D','Prob_bk_A','possession_h_moy','possession_a_moy','points_h','points_a','points_h_last_5','points_a_last_5','total_goals_h','total_goals_a','total_goals_taken_h','total_goals_taken_a','total_ontarget_h','total_ontarget_a','total_shot_off_target_h','total_shot_off_target_a','total_blocked_scoring_att_h','total_blocked_scoring_att_a','mean_rating_home','mean_rating_away','Prob_H','Prob_D','Prob_A','bet_H?','bet_D?','bet_A?'],axis=1)
rslt_df = undummify(rslt_df)
rslt_df = rslt_df.rename({'date': 'Date', 'H':'HomeTeam', 'A' : 'AwayTeam', 'benefices' : 'Bénéfice/Perte', 'montant' : 'Montant parié'}, axis = 1)
st.write(rslt_df)

