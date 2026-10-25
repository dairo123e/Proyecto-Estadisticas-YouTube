import pandas as pd, numpy as np, json, warnings, time
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
sns.set_theme(style="whitegrid")
R="/root/repo/"
d=pd.read_csv(R+"dataset_youtube_procesado.csv")
com=pd.read_csv(R+"comments.csv").drop(columns=["Unnamed: 0"],errors="ignore").dropna(subset=["Comment"])
com=com[com.Comment.astype(str).str.strip()!=""].drop_duplicates()
txt=com.groupby("Video ID")["Comment"].apply(lambda s:" ".join(s.astype(str))).rename("texto_comentarios")
d=d.merge(txt,on="Video ID",how="left"); d["texto_comentarios"]=d["texto_comentarios"].fillna("")
d["Title"]=d["Title"].astype(str)
# variables nuevas iteración 2
d["ratio_comentarios_likes"]=np.log1p(d["Comments"])-np.log1p(d["Likes"])
d["log_antiguedad"]=np.log1p(d["antiguedad_dias"])
d["mayusculas_titulo"]=d.Title.apply(lambda t: sum(c.isupper() for c in t)/max(len(t),1))
d["signos_titulo"]=d.Title.str.count(r"[!?]")
d["palabras_titulo"]=d.Title.str.split().str.len()
d["titulo_numero"]=d.Title.str.contains(r"\d").astype(int)
d["longitud_comentarios"]=d.texto_comentarios.str.len()/d.comentarios_analizados.clip(lower=1)
y=d["popularidad_alta"]
print("n",len(d),y.mean())
base_num=["log_likes_video","log_comentarios_video","log_promedio_likes_comentarios","comentarios_analizados",
 "proporcion_negativos","proporcion_neutrales","proporcion_positivos","longitud_titulo","antiguedad_dias"]
cat=["Keyword"]
idx_tr,idx_te=train_test_split(d.index,test_size=0.2,stratify=y,random_state=42)
Xtr,Xte,ytr,yte=d.loc[idx_tr],d.loc[idx_te],y.loc[idx_tr],y.loc[idx_te]
cv=StratifiedKFold(5,shuffle=True,random_state=42)
def prep(num,text=False,svd=True):
    t=[("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),
       ("cat",OneHotEncoder(handle_unknown="ignore"),cat)]
    if text:
        def tf(col,k,mf):
            steps=[("tfidf",TfidfVectorizer(max_features=mf,ngram_range=(1,2),min_df=3,sublinear_tf=True,stop_words="english"))]
            if svd: steps.append(("svd",TruncatedSVD(k,random_state=42)))
            return Pipeline(steps)
        t+= [("tit",tf("Title",20,2000),"Title"),("com",tf("texto_comentarios",30,5000),"texto_comentarios")]
    return ColumnTransformer(t)
def ev(name,model,Xa=Xtr,Xb=Xte,it=""):
    model.fit(Xa,ytr); p=model.predict(Xb); pr=model.predict_proba(Xb)[:,1]
    cvs=cross_val_score(model,Xa,ytr,cv=cv,scoring="f1",n_jobs=-1)
    r=dict(Iteracion=it,Modelo=name,Accuracy=accuracy_score(yte,p),Precision=precision_score(yte,p),Recall=recall_score(yte,p),
           F1=f1_score(yte,p),ROC_AUC=roc_auc_score(yte,pr),CV_F1=cvs.mean(),CV_std=cvs.std())
    print({k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items()}); return r,model,pr
res=[];store={}
# ---- Iteración 0: réplica Avance II
r,m,pr=ev("Gradient Boosting (Avance II)",Pipeline([("p",prep(base_num)),("m",GradientBoostingClassifier(random_state=42))]),it="0. Línea base");res.append(r);store["it0"]=(m,pr)
# ---- Iteración 1: nuevos algoritmos y ajuste de hiperparámetros
def tune(name,est,grid,num,text=False,it="",n=25):
    pipe=Pipeline([("p",prep(num,text)),("m",est)])
    s=RandomizedSearchCV(pipe,grid,n_iter=n,scoring="f1",cv=cv,random_state=42,n_jobs=-1).fit(Xtr,ytr)
    print(name,"best",s.best_params_,round(s.best_score_,4))
    r,m,pr=ev(name,s.best_estimator_,it=it); r["params"]=str({k.replace("m__",""):v for k,v in s.best_params_.items()}); return r,m,pr
g_gb={"m__n_estimators":[100,200,300,500],"m__learning_rate":[0.02,0.05,0.1],"m__max_depth":[2,3,4],"m__subsample":[0.7,0.85,1.0],"m__min_samples_leaf":[1,5,10]}
g_xgb={"m__n_estimators":[200,400,600],"m__learning_rate":[0.02,0.05,0.1],"m__max_depth":[2,3,4,5],"m__subsample":[0.7,0.85,1],"m__colsample_bytree":[0.6,0.8,1],"m__reg_lambda":[1,5,10]}
g_lgb={"m__n_estimators":[200,400,600],"m__learning_rate":[0.02,0.05,0.1],"m__num_leaves":[7,15,31],"m__min_child_samples":[10,20,40],"m__subsample":[0.7,0.85,1],"m__subsample_freq":[1],"m__colsample_bytree":[0.6,0.8,1]}
g_rf={"m__n_estimators":[300,500],"m__max_depth":[None,8,12,16],"m__min_samples_leaf":[1,2,4],"m__max_features":["sqrt",0.5]}
g_lr={"m__C":[0.01,0.1,0.5,1,5,10,50],"m__penalty":["l2"]}
XGB=XGBClassifier(random_state=42,eval_metric="logloss",n_jobs=1,verbosity=0)
LGB=LGBMClassifier(random_state=42,verbose=-1,n_jobs=1)
for nm,e,g in [("Gradient Boosting ajustado",GradientBoostingClassifier(random_state=42),g_gb),("XGBoost ajustado",XGB,g_xgb),
               ("LightGBM ajustado",LGB,g_lgb),("Random Forest ajustado",RandomForestClassifier(random_state=42,n_jobs=1),g_rf),
               ("Regresión logística ajustada",LogisticRegression(max_iter=3000),g_lr)]:
    r,m,pr=tune(nm,e,g,base_num,it="1. Ajuste de hiperparámetros",n=7 if "logística" in nm else 25); res.append(r); store["it1_"+nm]=(m,pr)
# ---- Iteración 2: ingeniería de variables + NLP (TF-IDF + SVD)
new_num=base_num+["ratio_comentarios_likes","mayusculas_titulo","signos_titulo","palabras_titulo","titulo_numero","longitud_comentarios"]
for nm,e,g,n in [("XGBoost + variables + TF-IDF",XGB,g_xgb,20),("Gradient Boosting + variables + TF-IDF",GradientBoostingClassifier(random_state=42),g_gb,15),
                 ("Regresión logística + variables + TF-IDF",LogisticRegression(max_iter=3000),g_lr,7)]:
    r,m,pr=tune(nm,e,g,new_num,text=True,it="2. Variables nuevas + NLP",n=n); res.append(r); store["it2_"+nm]=(m,pr)
# ---- Iteración 3: ensamble apilado
best_xgb=store["it1_XGBoost ajustado"][0]; best_gb=store["it1_Gradient Boosting ajustado"][0]; best_lr=store["it1_Regresión logística ajustada"][0]; best_lgb=store["it1_LightGBM ajustado"][0]
stack=StackingClassifier([("gb",best_gb),("xgb",best_xgb),("lgb",best_lgb),("lr",best_lr)],final_estimator=LogisticRegression(),cv=5,n_jobs=-1)
r,m,pr=ev("Stacking (GB+XGB+LGBM+LR)",stack,it="3. Ensamble");res.append(r);store["it3"]=(m,pr)
# ---- Iteración 4: escenario sin interacción del video (solo contenido + comentarios NLP)
num_nl=["log_promedio_likes_comentarios","comentarios_analizados","proporcion_negativos","proporcion_neutrales","proporcion_positivos",
        "longitud_titulo","antiguedad_dias","mayusculas_titulo","signos_titulo","palabras_titulo","titulo_numero","longitud_comentarios"]
for nm,e,g,n in [("XGBoost sin likes/comentarios + TF-IDF",XGB,g_xgb,20),("Regresión logística sin likes + TF-IDF",LogisticRegression(max_iter=3000),g_lr,7)]:
    r,m,pr=tune(nm,e,g,num_nl,text=True,it="4. Escenario sin likes ni comentarios del video",n=n); res.append(r); store["it4_"+nm]=(m,pr)
# solo texto
r,m,pr=tune("Regresión logística solo TF-IDF (texto)",LogisticRegression(max_iter=3000),g_lr,["comentarios_analizados"],text=True,it="4. Escenario sin likes ni comentarios del video",n=7); res.append(r); store["it4_txt"]=(m,pr)
df=pd.DataFrame(res); df.to_csv("resultados_iteraciones.csv",index=False); print(df.round(4).to_string())
import pickle; pickle.dump({k:v[1] for k,v in store.items()},open("probas.pkl","wb")); pickle.dump(store,open("store.pkl","wb"))
d.loc[idx_te,["Video ID"]].to_csv("test_ids.csv")
