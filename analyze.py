from pathlib import Path
import json, sqlite3, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
df=pd.read_csv(DATA/'dataset_part_2.csv')
raw=pd.read_csv(DATA/'dataset_part_1.csv')
assert len(df)==90 and not df.FlightNumber.duplicated().any()
bad={'False Ocean','False RTLS','False ASDS','None ASDS','None None'}
assert ((~df.Outcome.isin(bad)).astype(int)==df.Class).all()
df['Year']=pd.to_datetime(df.Date).dt.year
colors={0:'#d16b52',1:'#147d92'}
def save(name):
 plt.tight_layout();plt.savefig(FIG/(name+'.png'),bbox_inches='tight');plt.close()
for name,x,y in [('flight_site','FlightNumber','LaunchSite'),('payload_site','PayloadMass','LaunchSite'),('flight_orbit','FlightNumber','Orbit'),('payload_orbit','PayloadMass','Orbit')]:
 plt.figure(figsize=(9.3,4.7))
 for c,g in df.groupby('Class'):plt.scatter(g[x],g[y],label=f'Class {c}',c=colors[c],alpha=.8,s=45,edgecolors='white',linewidths=.4)
 plt.xlabel({'PayloadMass':'Payload mass (kg)','FlightNumber':'Flight number'}[x]);plt.ylabel(y);plt.legend(loc='best');save(name)
orbit=df.groupby('Orbit').Class.agg(['mean','size']).sort_values('mean')
plt.figure(figsize=(9.3,4.7));plt.bar(orbit.index,orbit['mean']*100,color='#147d92');plt.ylim(0,115);plt.ylabel('Class 1 rate (%)')
for i,r in enumerate(orbit.itertuples()):plt.text(i,r.mean*100+3,f'n={r.size}',ha='center',fontsize=10)
save('orbit_rate')
yr=df.groupby('Year').Class.agg(['mean','size'])
plt.figure(figsize=(9.3,4.7));plt.plot(yr.index,yr['mean']*100,'o-',color='#147d92');plt.ylim(0,110);plt.ylabel('Class 1 rate (%)');plt.xlabel('Launch year');plt.grid(alpha=.2)
for r in yr.itertuples():plt.annotate(f'n={r.size}',(r.Index,r.mean*100),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
save('yearly')

# SQL tasks use a separate 101-row historical course dataset.
sql_df=pd.read_csv(DATA/'Spacex.csv')
for c in sql_df.select_dtypes('object'):sql_df[c]=sql_df[c].str.strip()
conn=sqlite3.connect(ROOT/'launches.sqlite');sql_df.to_sql('launches',conn,if_exists='replace',index=False)
queries={
'Unique sites':'SELECT DISTINCT Launch_Site FROM launches ORDER BY Launch_Site',
'First five CCA records':"SELECT Date, Booster_Version, Launch_Site FROM launches WHERE Launch_Site LIKE 'CCA%' ORDER BY Date LIMIT 5",
'NASA CRS payload':"SELECT SUM(PAYLOAD_MASS__KG_) AS payload_kg FROM launches WHERE Customer = 'NASA (CRS)'",
'All NASA-related payload':"SELECT SUM(PAYLOAD_MASS__KG_) AS payload_kg FROM launches WHERE Customer LIKE '%NASA%'",
'F9 v1.1 average payload':"SELECT AVG(PAYLOAD_MASS__KG_) AS average_kg FROM launches WHERE Booster_Version LIKE 'F9 v1.1%'",
'First successful ground landing':"SELECT MIN(Date) AS first_date FROM launches WHERE Landing_Outcome='Success (ground pad)'",
'Drone landings with 4000-6000 kg':"SELECT Booster_Version, PAYLOAD_MASS__KG_ AS kg FROM launches WHERE Landing_Outcome='Success (drone ship)' AND PAYLOAD_MASS__KG_>4000 AND PAYLOAD_MASS__KG_<6000",
'Mission outcome counts':"SELECT Mission_Outcome, COUNT(*) AS count FROM launches GROUP BY Mission_Outcome",
'Maximum payload boosters':"SELECT Booster_Version, PAYLOAD_MASS__KG_ AS kg FROM launches WHERE PAYLOAD_MASS__KG_=(SELECT MAX(PAYLOAD_MASS__KG_) FROM launches)",
'2015 drone landing failures':"SELECT Date, Booster_Version, Launch_Site FROM launches WHERE Date LIKE '2015%' AND Landing_Outcome='Failure (drone ship)'",
'Landing outcome ranks':"SELECT Landing_Outcome, COUNT(*) AS count FROM launches WHERE Date BETWEEN '2010-06-04' AND '2017-03-20' GROUP BY Landing_Outcome ORDER BY count DESC, Landing_Outcome",
'Site success rates':"SELECT Launch_Site, COUNT(*) AS launches, SUM(CASE WHEN Landing_Outcome LIKE 'Success%' THEN 1 ELSE 0 END) AS successful_landings FROM launches GROUP BY Launch_Site"
}
sql_results={k:pd.read_sql_query(q,conn).to_dict('records') for k,q in queries.items()}
(ROOT/'queries.sql').write_text('\n\n'.join('-- '+k+'\n'+v+';' for k,v in queries.items()))

# Prelaunch predictors only. Outcome and final reuse counts are deliberately excluded.
num=['FlightNumber','PayloadMass','Flights','Block','GridFins','Reused','Legs']
cat=['Orbit','LaunchSite']
X=df[num+cat].copy();y=df.Class
for c in ['GridFins','Reused','Legs']:X[c]=X[c].astype(int)
pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
specs={
'Logistic regression':(LogisticRegression(max_iter=3000),{'model__C':[.01,.1,1,10]}),
'SVM':(SVC(),{'model__C':[.1,1,10],'model__kernel':['linear','rbf'],'model__gamma':['scale',.1]}),
'Decision tree':(DecisionTreeClassifier(random_state=42),{'model__max_depth':[2,3,4,None],'model__min_samples_leaf':[1,3,5]}),
'KNN':(KNeighborsClassifier(),{'model__n_neighbors':[3,5,7,9],'model__weights':['uniform','distance']})}
results=[];fitted={}
for name,(model,grid) in specs.items():
 gs=GridSearchCV(Pipeline([('pre',pre),('model',model)]),grid,cv=cv,scoring='accuracy',n_jobs=1)
 gs.fit(Xtr,ytr);pred=gs.predict(Xte);fitted[name]=gs
 results.append({'model':name,'cv_accuracy':gs.best_score_,'cv_std':float(gs.cv_results_['std_test_score'][gs.best_index_]),'test_accuracy':accuracy_score(yte,pred),'balanced_accuracy':balanced_accuracy_score(yte,pred),'f1':f1_score(yte,pred),'params':gs.best_params_})
best=max(results,key=lambda r:r['cv_accuracy']);bestmodel=fitted[best['model']];pred=bestmodel.predict(Xte)
cm=confusion_matrix(yte,pred)
plt.figure(figsize=(9.3,4.7));p=np.arange(4);plt.bar(p-.18,[r['cv_accuracy']*100 for r in results],.36,label='Training 5-fold CV',color='#147d92');plt.bar(p+.18,[r['test_accuracy']*100 for r in results],.36,label='Held-out test',color='#e0a448');plt.xticks(p,[r['model'] for r in results]);plt.ylim(0,110);plt.ylabel('Accuracy (%)');plt.legend();save('models')
fig,ax=plt.subplots(figsize=(6.4,4.8));ConfusionMatrixDisplay(cm,display_labels=['Class 0','Class 1']).plot(ax=ax,cmap='Blues',colorbar=False);save('confusion')
# Forward-time stress check. Refit selected hyperparameters on the first 72 rows.
chron=bestmodel.best_estimator_;chron.fit(X.iloc[:72],y.iloc[:72]);cp=chron.predict(X.iloc[72:])
chron_result={'accuracy':accuracy_score(y.iloc[72:],cp),'majority_baseline':float(y.iloc[72:].mean()),'confusion':confusion_matrix(y.iloc[72:],cp).tolist(),'note':'Secondary sensitivity check after model selection, not a second untouched test.'}
metrics={'n':len(df),'start':df.Date.min(),'end':df.Date.max(),'positives':int(y.sum()),'site':df.groupby('LaunchSite').Class.agg(['count','sum','mean']).reset_index().to_dict('records'),'orbit':orbit.reset_index().to_dict('records'),'year':yr.reset_index().to_dict('records'),'missing_raw':raw.isna().sum().to_dict(),'models':results,'best':best,'confusion':cm.tolist(),'chronological':chron_result,'sql':sql_results,'sql_rows':len(sql_df)}
(ROOT/'results.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))
