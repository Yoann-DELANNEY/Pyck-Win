
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

# Fonction de simulation d'un pari
def gain_paris(matchs, resultats, montants):
    benefices = 0
    for match, resultat, montant in zip(matchs, resultats, montants):
        if df.loc[match]['FTR'] == resultat:
            benefice = (df.loc[match][dic_resultat[resultat]] - 1)*montant
        else : benefice = -1*montant
        benefices += benefice
    pourcentage = benefices*100/montants.count(1)
    return st.write('Montant parié :', montants.count(1)), st.write('Gains/pertes :', benefices), st.write('Taux de rendement :', pourcentage, ' %')

st.title('Pyck & Win')

text = '''
---
'''
st.markdown(text)

dataset_container = st.sidebar.beta_expander("Configuration", True)
with dataset_container:
	dataset = st.selectbox("Choisir une saison", ("Saison 14-15", "Saison 15-16", "Saison 16-17","Saison 17-18"))
	
if dataset == "Saison 14-15":
	df = pd.read_csv(r"C:\Users\Olivier\Desktop\proj\Streamlit\bookmakers14-15.csv", index_col=0)	
elif dataset == "Saison 15-16":
	df = pd.read_csv(r"C:\Users\Olivier\Desktop\proj\Streamlit\bookmakers15-16.csv", index_col=0)
elif dataset == "Saison 16-17":
	df = pd.read_csv(r"C:\Users\Olivier\Desktop\proj\Streamlit\bookmakers16-17.csv", index_col=0)
elif dataset == "Saison 17-18":
	df = pd.read_csv(r"C:\Users\Olivier\Desktop\proj\Streamlit\bookmakers17-18.csv", index_col=0)
	
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

options = ["SVM","Random Forest Classifier","XGBoost"]
choix = st.radio("Choisir un modèle", options)

# SVM
if choix==options[0]: 
		clf=SVC(probability=True)
		parametres = {'C':[0.1,1,10], 'kernel':['rbf','linear', 'poly'], 'gamma':[0.001, 0.1, 0.5]}
		model = GridSearchCV(estimator=clf, param_grid=parametres)
		grille = model.fit(X_train_scaled,y_train)
		score = model.score(X_test_scaled,y_test)
		y_pred = model.predict(X_test_scaled)
		y_pred_proba = model.predict_proba(X_test_scaled)
    
# RF
if choix==options[1]:
    
		model = RandomForestClassifier(n_estimators = 1000,
								max_depth=4,
								n_jobs=-1)

		model.fit(X_train_scaled,y_train)
		score = model.score(X_test_scaled,y_test)
		y_pred = model.predict(X_test_scaled)
		y_pred_proba = model.predict_proba(X_test_scaled)
    
# XGBoost
if choix==options[2]:
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

st.write("Matrice de confusion")
#st.write(pd.crosstab(y_test, y_pred, rownames=['Classes réelles'], colnames=['Classes prédites']))
st.write(y_pred_proba)	
			
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
	df['bet_D?'] = np.where((df['Prob_D']>df['Prob_bk_D']) & (df['Prob_D']>seuil_pari), 1 , 0)
	df['bet_A?'] = np.where((df['Prob_A']>df['Prob_bk_A']) & (df['Prob_A']>seuil_pari), 1 , 0)
	df = df.fillna(0)
	st.write("ici1")

#dictionnaire pour attribuer un chiffre à un resultat de pari
	dic_resultat = {0: 'bk_H', 1: 'bk_D', 2: 'bk_A'}
# Simulation de paris
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
	df['bet_D?'] = np.where((df['Prob_D']>df['Prob_bk_D']) & (df['Prob_D']>seuil_pari), 1 , 0)
	df['bet_A?'] = np.where((df['Prob_A']>df['Prob_bk_A']) & (df['Prob_A']>seuil_pari), 1 , 0)
	df = df.fillna(0)
	st.write("ici2")
#dictionnaire pour attribuer un chiffre à un resultat de pari
	dic_resultat = {0: 'bk_H', 1: 'bk_D', 2: 'bk_A'}
# Simulation de paris
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
	df['bet_D?'] = np.where((df['Prob_D']>df['Prob_bk_D']) & (df['Prob_D']>seuil_pari), 1 , 0)
	df['bet_A?'] = np.where((df['Prob_A']>df['Prob_bk_A']) & (df['Prob_A']>seuil_pari), 1 , 0)
	df = df.fillna(0)
	st.write("ici3")
#dictionnaire pour attribuer un chiffre à un resultat de pari
	dic_resultat = {0: 'bk_H', 1: 'bk_D', 2: 'bk_A'}
# Simulation de paris
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
	df['bet_D?'] = np.where((df['Prob_D']>df['Prob_bk_D']) & (df['Prob_D']>seuil_pari), 1 , 0)
	df['bet_A?'] = np.where((df['Prob_A']>df['Prob_bk_A']) & (df['Prob_A']>seuil_pari), 1 , 0)
	df = df.fillna(0)
	st.write("ici4")
#dictionnaire pour attribuer un chiffre à un resultat de pari
	dic_resultat = {0: 'bk_H', 1: 'bk_D', 2: 'bk_A'}
# Simulation de paris
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

