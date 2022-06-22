import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

#------------------
# SIDEBAR
#------------------

with st.sidebar:
    SEED = int(st.number_input(label="Seed value",
                                min_value=1,
                                value=123))
    #umap
    st.write("---")
    st.write("UMAP parameters")
    umap_min_distance = int(st.number_input(label="Umap min distance",
                                min_value=0,
                                value=0))
    umap_n_neighbours = int(st.number_input(label="Umap n neighbours",
                                min_value=1,
                                value=10))
    #test ratio
    st.write("---")
    st.write("Test ratio")
    test_ratio = st.slider(label="test ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.3,key=1)

    #hyperparameters XGBRegressor
    st.write("---")
    st.write("Hyperparameters")
    hyp_gamma = st.slider(label="gamma", min_value=0.0, max_value=1.0, value=0.0, key=2)
    hyp_lr = st.slider(label="learning rate", min_value=0.001, max_value=1.0, value=0.05, key=3)
    hyp_depth = st.slider(label="max depth", min_value=2, max_value=10, value=6, key=4)
    hyp_lambda = st.slider(label="reg lambda", min_value=0.0, max_value=10.0, value=0.5, key=5)



#------------------
# BODY
#------------------

st.write("""

# Train your own model.
Import your own data and play with some of the hyperparameters here on the left. 
And look at how we went through everything.

## The data
""")

df = pd.read_csv('./advanced_tract_fmri_combined.csv')
df_flipped = pd.read_csv('./advanced_tract_fmri_combined_flip.csv')
st.dataframe(df,height=100)

st.write("""
Clean dataset:
""")

def preprocess_df(df):

    df = df.drop(columns=['Unnamed: 0','Unnamed: 0_x','Unnamed: 0_y'])
    left_lateralized = df.LI_fmri > 0
    df['left_lateralized'] = left_lateralized

    return df

df = preprocess_df(df)
df_flipped = preprocess_df(df_flipped)

st.dataframe(df_flipped,height=100)

#----------------
# UMAP BUTTON
if st.button(label="Run UMAP (may be slow)"):
    import umap

    # import umap-learn
    reducer1 = umap.UMAP(random_state=SEED,
                        n_neighbors =umap_n_neighbours,
                        min_dist=umap_min_distance
                        )

    reducer2 = umap.UMAP(random_state=SEED,
                        n_neighbors =umap_n_neighbours,
                        min_dist=umap_min_distance
                        )

    with st.spinner("Calculating UMAP"):
        reducer1.fit(df)
        reducer2.fit(df_flipped)

        # Uncomment to save
        # import joblib
        # joblib.dump(reducer1,'reducer_umap.sav')

        embedding1 = reducer1.transform(df)
        embedding2=  reducer2.transform(df_flipped)

        df_embedding = pd.DataFrame(embedding1)
        df_embedding.to_csv('population_embedding_dev.csv')

        df.to_csv('population_LI_values.csv')

    st.write("""
    Blue = right lateralized

    Red = left lateralized 
    """)

    fig, axs = plt.subplots(1,2,figsize=(13,6))

    scatter1 = axs[0].scatter(embedding1[:, 0], embedding1[:, 1], c=df.LI_fmri, cmap='seismic')
    scatter2 = axs[1].scatter(embedding2[:, 0], embedding2[:, 1], c=df_flipped.LI_fmri, cmap='seismic_r')

    fig.colorbar(scatter1, label='Laterality Index fMRI',ax=axs[0])
    fig.colorbar(scatter2, label='Laterality Index fMRI',ax=axs[1])

    axs[0].set_title('UMAP normal data')
    axs[1].set_title('UMAP flipped data')

    st.pyplot(fig)

# get patient indices of true right-lang patients
df_right = df[df.left_lateralized==False]
df_left = df[df.left_lateralized==True]
df_right_flip = df_flipped[df_flipped.left_lateralized==False]
df_left_flip = df_flipped[df_flipped.left_lateralized==True]

df_right = pd.concat([df_right,df_right_flip],axis=0)

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb


# TRAIN TEST SPLIT OF DATA
X_normal = df.iloc[:,1:29].drop(columns=['sum_L_fmri','sum_R_fmri','LI_fmri','left_lateralized'])
y_normal = df.iloc[:,22]
contras_normal = df.left_lateralized
X_train_normal, X_test_normal, y_train_normal, y_test_normal = train_test_split(X_normal, y_normal,
                                                                                random_state=SEED,
                                                                                test_size=test_ratio,
                                                                                stratify = contras_normal)


X_flipped = df_flipped.iloc[:,1:29].drop(columns=['sum_L_fmri','sum_R_fmri','LI_fmri','left_lateralized'])
y_flipped = df_flipped.iloc[:,22]
contras_flipped = df_flipped.left_lateralized
X_train_flipped, X_test_flipped, y_train_flipped, y_test_flipped = train_test_split(X_flipped, y_flipped,
                                                                                    random_state=SEED,
                                                                                    test_size=test_ratio,
                                                                                    stratify = contras_normal)

X_train = pd.concat([X_train_normal,X_train_flipped],axis=0)
y_train = pd.concat([y_train_normal,y_train_flipped],axis=0)
X_test = pd.concat([X_test_normal,X_test_flipped],axis=0)
y_test = pd.concat([y_test_normal,y_test_flipped],axis=0)

st.write("""
Train data:
""")

col1, col2 = st.columns(2)
with col1:
    st.write(f"X_train : {X_train.shape}")
    st.dataframe(X_train,height=100)

with col2:
    st.write(f"y_train : {y_train.shape}")
    st.dataframe(y_train,height=100)

st.write("""
Test data:
""")

col1, col2 = st.columns(2)
with col1:
    st.write(f"X_test : {X_test.shape}")
    st.dataframe(X_test,height=100)

with col2:
    st.write(f"y_test : {y_test.shape}")
    st.dataframe(y_test,height=100)

#--------------------
# MODEL TUNING
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
st.write("""
## Model
""")

#Round 1 of optimisation
param_grid_round1 = {
    'max_depth': [3,4,5],
    'learning_rate': [0.1,0.01,0.05],
    'gamma':[0,0.25,1.0],
    'reg_lambda':[0,1.0,10.0],
    'scale_pos_weight': [1,3,5]
}
#output:
#{'gamma': 0, 'learning_rate': 0.05, 'max_depth': 5, 'reg_lambda': 0, 'scale_pos_weight': 1}
st.write("Find the optimal hyperparameters to train the model")
st.write("Hyperparameter grid for gridsearch round 1")
st.json(param_grid_round1,expanded=False)
st.write("""
Best parameters after round1

{'gamma': 0.25, 'learning_rate': 0.05, 'max_depth': 5, 'reg_lambda': 0, 'scale_pos_weight': 1}

""")
# Round 2 of optimisation 
param_grid_round2 = {
    'max_depth': [5,6,7],
    'learning_rate': [0.05,0.01,0.005],
    'gamma':[0],
    'reg_lambda':[0,0.1,0.5,1],
    'scale_pos_weight': [1]
}   

st.write("Hyperparameter grid for gridsearch round 2")
st.json(param_grid_round2,expanded=False)
st.write("""
Best parameters after round 2

{'gamma': 0, 'learning_rate': 0.05, 'max_depth': 6, 'reg_lambda': 0.5, 'scale_pos_weight': 1}
""")

optimal_parameters = GridSearchCV(XGBRegressor(objective='reg:squarederror',
                                     seed = SEED,
                                     subsample=0.9,
                                     subsample_bytree=0.5),
                        #param grid here
                        param_grid_round2,
                        
                        scoring='neg_mean_absolute_error',
                        cv = 3,
                        n_jobs = 1,
                        verbose=10)

# RUN GRIDSEARCHCV TO FIND OPTIMAL PARAMATERS (uncomment to run again)
# optimal_parameters.fit(X_train, y_train)
# print(optimal_parameters.best_score_)
# print(optimal_parameters.best_params_)

#--------------------
# TRAIN MODEL

model = XGBRegressor(objective = 'reg:squarederror',
                    gamma = hyp_gamma,
                    learning_rate = hyp_lr,
                    max_depth = hyp_depth,
                    reg_lambda = hyp_lambda,
                    scale_pos_weight = 1)

eval_set = [(X_test,y_test)]

with st.spinner("Training model"):
    model.fit(X_train,y_train,
            early_stopping_rounds=20,
            # eval_metric = 'rmse',
            eval_set = eval_set,
            verbose=True)



    #UNCOMMENT TO SAVE MODEL
    # model.save_model("XGB_model_dev.json")

    st.write("training succesful")

    pred = model.predict(X_test)
    score = mean_squared_error(pred,y_test)

    st.write(f"""
        mean squared error : {score}
    """)

st.write("""
## What the model does

show explainability plots below (need to have trained model first though)
""")

with st.spinner("Loading all the plots"):
   
    from xgboost import plot_importance
    
    st.write('How much features cover data')
    ax1 = plot_importance(model, importance_type='cover')
   

    st.pyplot(ax1.figure)
   
    st.write("relative importance of each feature")
    ax2 = plot_importance(model)
    st.pyplot(ax2.figure)

    import shap
    st.write(    
    '''
    calculate shap values as described in this paper: explainable AI for tree based models
    https://www.nature.com/articles/s42256-019-0138-9

    ''')
   
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test_normal)

    # visualize the first prediction's explanation
    fig, axs = plt.subplots()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(shap.plots.waterfall(shap_values[0]))
    st.pyplot(shap.summary_plot(shap_values, X_test_normal))
    
    shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(X_test_normal)
    st.pyplot(shap.summary_plot(shap_interaction_values, X_test_normal))


    fig, axs = plt.subplots(figsize= (50,30))
    import graphviz

    st.write("Example of a tree")
    st.graphviz_chart(xgb.to_graphviz(model,num_trees=60),
                        use_container_width=True)

    st.write("""
    
    ## Performance Visualized 
    
    """)

test_df = pd.DataFrame()
test_df['true'] = y_test
test_df['pred'] = model.predict(X_test)
test_df['left_lateralized'] = y_test > 0
test_df.head()

X_real_life = df_right.iloc[:,1:29].drop(columns=['sum_L_fmri','sum_R_fmri','LI_fmri','left_lateralized'])
print(X_real_life)
y_real_life = df_right.iloc[:,22]

df_test_real_life = pd.DataFrame()
df_test_real_life['true'] = y_real_life
df_test_real_life['pred'] = model.predict(X_real_life)
# df_test_real_life['left_lateralized'] = str('actual')

df_combined_test = pd.concat([df_test_real_life,test_df],axis=0)

master_alpha = 0.5
fig, axs = plt.subplots(figsize=(10,10))

axs.scatter(df_combined_test.true,df_combined_test.pred,alpha=0.5)
axs.axvline(0)
axs.set_xlabel('LI_fMRI')
axs.set_ylabel('XGBoost Prediction')
st.pyplot(fig)

def get_posnegs(true_list,pred_list):

  true_bool = true_list
  pred_bool = pred_list

  eps = 10e-9 #value added below to avoid division by zero errors

  TP = 0
  TN = 0
  FP = 0
  FN = 0

  for ii in range(len(true_bool)):
    if true_bool[ii] == True and pred_bool[ii] == True:
      TP +=1
    elif true_bool[ii] == False and pred_bool[ii] == False:
      TN +=1
    elif true_bool[ii] == True and pred_bool[ii] == False:
      FN +=1
    elif true_bool[ii] == False and pred_bool[ii] == True:
      FP +=1
  
  TPR = TP / (TP + FN +eps) #sensitivity, recall 
  TNR = TN / (TN + FP +eps) #specificity
  FPR = FP / (FP + TN +eps) #fall-out
  FNR = FN / (FN + TP +eps) #miss-rate
  PPV = TP / (TP + FP +eps) #precision 
  NPV = TN / (TN + FN +eps) #NPV
  FDR = FP / (FP + TP +eps) #false discovery
  FOR = FN / (FN + TN +eps) #false omission
  F1 = 2*(PPV*TPR / (PPV+TPR +eps))

  return [TP, TN, FP, FN, TPR, TNR, FPR, FNR, PPV, NPV, FDR, FOR, F1] 

min_val = np.min(df_combined_test.pred)
max_val = np.max(df_combined_test.pred)

steps = 100

thresholds = np.linspace(min_val, max_val,steps)

true_vals = list(df_combined_test.true)
pred_vals = list(df_combined_test.pred)


true_bool = [x > 0 for x in true_vals]

TPR1 = []
TPR2 = []
FPR1 = []
FPR2 = []
PPV1 = []
PPV2 = []

for thr in thresholds:
  pred_bool1 = [x>thr for x in pred_vals]
  pred_bool2 = [x<thr for x in pred_vals]

  results1 = get_posnegs(true_bool,pred_bool1)
  TPR1.append(results1[4])
  FPR1.append(results1[6])
  PPV1.append(results1[8])
  results2 = get_posnegs(true_bool,pred_bool2)
  TPR2.append(results2[4])
  FPR2.append(results2[6])
  PPV2.append(results2[8])

fig, axs = plt.subplots(1,2)
AUC1 = np.sum(TPR1) / steps
AUC2 = np.sum(TPR2) / steps
axs[0].plot(FPR1, TPR1, label= f'AUC: {AUC1:0.3f}')
axs[0].set_title('ROC 1: value > thr')

axs[1].plot(FPR2, TPR2,label= f'AUC: {AUC2:0.3f}')
axs[1].set_title('ROC 2: value < thr')

for ax_ind in range(2):
  axs[ax_ind].set_xlim(-0.1,1.1)
  axs[ax_ind].set_ylim(-0.1,1.1)
  axs[ax_ind].plot([0,1],[0,1],'r--')
  axs[ax_ind].set_xlabel('FPR')
  axs[ax_ind].set_ylabel('TPR')
  axs[ax_ind].legend(loc='lower right')

fig.tight_layout()

st.pyplot(fig)

fig, axs = plt.subplots(1,2)
axs[0].plot(TPR1, PPV1)
axs[0].set_title('PR 1: value > thr')

axs[1].plot(TPR2, PPV2)
axs[1].set_title('PR 2: value < thr')

for ax_ind in range(2):
  axs[ax_ind].set_xlabel('Recall')
  axs[ax_ind].set_ylabel('Precision')
  axs[ax_ind].set_xlim(-0.1,1.1)
  axs[ax_ind].set_ylim(-0.1,1.1)
  axs[ax_ind].plot([0,1],[1,0],'k--')

fig.tight_layout()

st.pyplot(fig)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

TPR2 = np.array(TPR1)
FPR2 = np.array(FPR1)
distances = []
goal_coordinate = np.array([0,1])
for ii in range(TPR2.shape[0]):
  current_coordinate = np.array([FPR2[ii],TPR2[ii]])
  distance = np.linalg.norm(goal_coordinate - current_coordinate)
  distances.append(distance)

min_index = np.argmin(distances)
optimal_cutoff = thresholds[min_index]
print(optimal_cutoff)

fig, axs = plt.subplots(1,2,figsize=(10,5))
axs[0].plot(distances)
axs[0].plot(min_index,distances[min_index],'r*')
axs[0].set_xlabel('threshold index')
axs[0].set_ylabel('distance from upper left corner')

axs[1].plot(FPR2,TPR2,label= f'AUC: {AUC1:0.3f}')
axs[1].plot(FPR2[min_index],TPR2[min_index],'r*',label=f'XGB cutoff val: {optimal_cutoff:0.3f}')
axs[1].legend(loc='lower right')
axs[1].set_xlabel('FPR')
axs[1].set_ylabel('TPR')
axs[1].plot([0,1],[0,1],'r--')
axs[1].set_title('ROC curve')

df_combined_test = pd.concat([df_test_real_life,test_df],axis=0)
df_combined_test['true_bool'] = df_combined_test.true > 0
df_combined_test['pred_bool'] = df_combined_test.pred < optimal_cutoff

st.pyplot(fig)


colors = {True: 'b', #values fround in the dataframe
          False:'r',
          'actual':'g'}

# custom_color = [colors[r] for r in df_combined_test['left_lateralized']]
master_alpha = 0.5
fig, axs = plt.subplots(1,2,figsize=(10,5))

scatter = axs[0].scatter(df_combined_test.true,df_combined_test.pred,alpha=0.5)

axs[0].axvline(0)
axs[0].axhline(optimal_cutoff,color='r',label=f'XGB cutoff val: {optimal_cutoff:0.3f}')
legend1 = axs[0].legend(loc='upper right')
axs[0].add_artist(legend1)

axs[0].set_xlabel('LI fMRI')
axs[0].set_ylabel('XGBoost Prediction')

cm = confusion_matrix(df_combined_test.true_bool, df_combined_test.pred_bool)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(ax=axs[1])
axs[1].set_title('Language left? : 0=no, 1=yes')

st.pyplot(fig)

df_combined_test.to_csv("population_scores_dev.csv")