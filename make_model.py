import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import random
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import Normalizer
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Integer, Continuous
import matplotlib.pyplot as plt
import numpy as np
import umap

USE_SUBSET = True
SEED = 123
TEST_RATIO = 0.3
NORMALIZE = False
UMAP_AUGMENT = False
GENETIC_GRIDSEARCH = False
NORMAL_GRIDSEARCH = False
UMAP_SHOW = False
SUBSET_TEST = False
SHOW_PLOTS = False
SAVE_MODEL = False #if true specify a "count" in main
LOAD_MODEL = True

random.seed(SEED)

def main(df,test_subset,count=None,model_path=None):
  def augment_X(df):
    """add more features to X to add more info in the system"""
    def make_LI(L, R):
      return (L -R ) / (L + R)


    df['LI_length'] = make_LI(df.tract_length_L, df.tract_length_R)
    df['LI_span'] = make_LI(df.tract_span_L, df.tract_span_R)
    df['LI_curl'] = make_LI(df.tract_curl_L, df.tract_curl_R)
    df['LI_elongation'] = make_LI(df.tract_elongation_L, df.tract_elongation_R)
    df['LI_surface_area'] = make_LI(df.tract_surface_area_L, df.tract_surface_area_R)
    df['LI_irregularity'] = make_LI(df.tract_irregularity_L, df.tract_irregularity_R)

    return df



  #one hot encode the method
  onehot_method = pd.get_dummies(df.method, prefix='method')
  df = pd.concat([onehot_method, df],axis=1)
  df = df.drop(columns=['method','tract'])


  #Train test split (custom) 
  subjects_R = list(set(df[df.LI_fmri < 0].subject_id))
  subjects_L = list(set(df[df.LI_fmri > 0].subject_id))

  subjects = subjects_R + subjects_L

  test_num_R = round(TEST_RATIO*len(subjects_R))
  test_num_L = round(TEST_RATIO*len(subjects_L)) 

  print(f"choosing {test_num_R} righters and {test_num_L} lefters for test set")

  test_ids_R = random.sample(subjects_R,test_num_R)
  test_ids_L = random.sample(subjects_L, test_num_L)

  test_ids = test_ids_L + test_ids_R
  train_ids = [x for x in subjects if x not in test_ids]

  df_train = df[df.subject_id.isin(train_ids)]
  df_test = df[df.subject_id.isin(test_ids)]


  X_train = df_train.drop(columns = ['subject_id','flipped','LI_fmri','LI_fmri_flipped'])
  X_test = df_test.drop(columns = ['subject_id','flipped','LI_fmri','LI_fmri_flipped'])

  y_train = df_train.LI_fmri
  y_train[df_train.flipped==True] = df.LI_fmri_flipped

  y_test = df_test.LI_fmri

  y_test[df_test.flipped==True] = df.LI_fmri_flipped

  X_train = augment_X(X_train)
  X_test = augment_X(X_test)


  def umap_augment_X(X_train, X_test):
      X_umap = X_train

      #Add UMAP embeddings to X data
      for neighbours in [5,10,50,100,200]:
          for min_dist in [0,1,5,10,20,50]:
              print(neighbours,min_dist)

              reducer = umap.UMAP(random_state=SEED,n_neighbors=10,min_dist=0)
              reducer.fit(X_umap)
              embedding_train = reducer.transform(X_train)
              embedding_test = reducer.transform(X_test)

              emb_tr_1 = embedding_train[:,0]
              emb_tr_2 = embedding_train[:,1]
              
              emb_ts_1 = embedding_test[:,0]
              emb_ts_2 = embedding_test[:,1]

              X_train[f"emb_n{neighbours}_d{min_dist}_1"] = emb_tr_1
              X_train[f"emb_n{neighbours}_d{min_dist}_2"] = emb_tr_2
              X_test[f"emb_n{neighbours}_d{min_dist}_1"] = emb_ts_1
              X_test[f"emb_n{neighbours}_d{min_dist}_2"] = emb_ts_2

      return X_train, X_test

  if UMAP_SHOW:
    X_umap = df[df.flipped==False].drop(columns = ['subject_id','flipped','LI_fmri','LI_fmri_flipped'])
    reducer = umap.UMAP(random_state=SEED,n_neighbors=10,min_dist=0)
    reducer.fit(X_umap)
    embedding1 = reducer.transform(X_umap)
    fig, axs = plt.subplots()
    scatter1 = axs.scatter(embedding1[:, 0], embedding1[:, 1], c=df[df.flipped==False].LI_fmri, cmap='seismic')
    fig.colorbar(scatter1, label='Laterality Index fMRI',ax=axs)
    plt.show()


  if UMAP_AUGMENT:
      X_train, X_test = umap_augment_X(X_train, X_test)
  #Normalizing data?
  if NORMALIZE:
      norm_transformer = Normalizer().fit(X_train)

      X_train = norm_transformer.transform(X_train)
      X_test = norm_transformer.transform(X_test)

  if LOAD_MODEL:
    model = XGBRegressor()
    model.load_model(model_path)
    model_params = "loaded_model"

  else:
      
    # Hyperparameter search - GENETIC
    if GENETIC_GRIDSEARCH:
        param_grid_genetic = {
            'max_depth': Integer(3,15),
            'learning_rate': Continuous(0.0001,1,distribution='log-uniform'),
            'gamma':Continuous(0,5,distribution='uniform'),
            'reg_lambda':Continuous(0,10,distribution='uniform'),
            'scale_pos_weight': Continuous(0,2),
            'n_estimators':Integer(5,500),
            'grow_policy': Categorical(["depthwise","lossguide"]),
            'min_child_weight':Continuous(0,400),
            'subsample':Continuous(0.5,1),
            'colsample':Continuous(0.5,1),
            'reg_alpha':Continuous(0,10,distribution='uniform'),
            'base_score':Continuous(-1,1),
            'num_parallel_tree':Integer(1,20)
        }   

        evolved_optimal_parameters = GASearchCV(estimator=XGBRegressor(objective='reg:squarederror',
                                            seed = SEED),
                                            param_grid = param_grid_genetic,
                                            scoring='neg_mean_absolute_error',
                                            criteria='max',
                                            cv = 4,
                                            n_jobs = 20,
                                            verbose=True)

        evolved_optimal_parameters.fit(X_train, y_train)
        model_params = evolved_optimal_parameters.best_params_
        
        model = XGBRegressor(objective = 'reg:squarederror',
                            params = evolved_optimal_parameters.best_params_)
        print(evolved_optimal_parameters.best_params_)
        eval_set = [(X_test,y_test)]
        model.fit(X_train,y_train,
                    early_stopping_rounds=20,
                    # eval_metric = 'rmse',
                    eval_set = eval_set,
                    verbose=True)

    if NORMAL_GRIDSEARCH:
        param_grid_round = {
        'max_depth': [10],
        'learning_rate': [0.05],
        'gamma':[0],
        'reg_lambda':[1],
        'scale_pos_weight': [1]
        }

        optimal_parameters = GridSearchCV(XGBRegressor(objective='reg:squarederror',
                                        seed = SEED,
                                        subsample=0.9,
                                        subsample_bytree=0.5),
                            #param grid here
                            param_grid_round,
                            scoring='neg_mean_absolute_error',
                            cv = 3,
                            n_jobs = -1,
                            verbose=10)
        
        optimal_parameters.fit(X_train, y_train)
        print(optimal_parameters.best_score_)
        model_params = optimal_parameters.best_params_
        

        model = XGBRegressor(objective = 'reg:squarederror',
                            params = optimal_parameters.best_params_)
        print(optimal_parameters.best_params_)
        eval_set = [(X_test,y_test)]
        model.fit(X_train,y_train,
                    early_stopping_rounds=20,
                    # eval_metric = 'rmse',
                    eval_set = eval_set,
                    verbose=True)

    if (not NORMAL_GRIDSEARCH) & (not GENETIC_GRIDSEARCH):
        model = XGBRegressor(objective = 'reg:squarederror')
        eval_set = [(X_test,y_test)]
        model.fit(X_train,y_train,
                    early_stopping_rounds=20,
                    # eval_metric = 'rmse',
                    eval_set = eval_set,
                    verbose=True)
        model_params= {"test":"no params specifically"}

  # from sklearn.neural_network import MLPRegressor
  # model = MLPRegressor(hidden_layer_sizes=(1000,1000,1000),early_stopping=True,verbose=True,max_iter=10000)
  # model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  y_pred_train = model.predict(X_train)

  fig, axs = plt.subplots()
  axs.scatter(y_test[df_test.flipped==False],y_pred[df_test.flipped==False],alpha=0.5)
  axs.axvline(0)
  axs.set_xlabel('LI_fMRI')
  axs.set_ylabel('XGBoost Prediction')
  # plt.show()
  def make_ROC(y_test, y_pred):
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
  
    min_val = np.min(y_pred)
    max_val = np.max(y_pred)

    steps = 100

    thresholds = np.linspace(min_val, max_val,steps)

    true_vals = list(y_test)
    pred_vals = list(y_pred)


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
    import sklearn
    AUC1 = sklearn.metrics.auc(FPR1, TPR1)
    AUC2 = sklearn.metrics.auc(FPR2, TPR2)

    axs[0].plot(FPR1, TPR1, label= f'AUC: {AUC1:0.3f}')
    axs[0].set_title('ROC 1: value > thr')

    axs[1].plot(FPR2, TPR2,label= f'AUC: {AUC2:0.3f}')
    axs[1].set_title('ROC 2: value < thr')

    if AUC1 > AUC2:
      AUC_best = AUC1
      TPR_best = TPR1
      FPR_best = FPR1
    else:
      AUC_best = AUC2
      TPR_best = TPR2
      FPR_best = FPR2


    for ax_ind in range(2):
      axs[ax_ind].set_xlim(-0.1,1.1)
      axs[ax_ind].set_ylim(-0.1,1.1)
      axs[ax_ind].plot([0,1],[0,1],'r--')
      axs[ax_ind].set_xlabel('FPR')
      axs[ax_ind].set_ylabel('TPR')
      axs[ax_ind].legend(loc='lower right')

    fig.tight_layout()

    # plt.show()

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

    # plt.show()


    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


    distances = []
    goal_coordinate = np.array([0,1])
    for ii in range(len(TPR_best)):
      current_coordinate = np.array([FPR_best[ii],TPR_best[ii]])
      distance = np.linalg.norm(goal_coordinate - current_coordinate)
      distances.append(distance)

    min_index = np.argmin(distances)
    optimal_cutoff = thresholds[min_index]
    print(optimal_cutoff)

    fig, axs = plt.subplots(1,2)
    axs[0].plot(distances)
    axs[0].plot(min_index,distances[min_index],'r*')
    axs[0].set_xlabel('threshold index')
    axs[0].set_ylabel('distance from upper left corner')

    axs[1].plot(FPR_best,TPR_best,label= f'AUC: {AUC_best:0.3f}')
    axs[1].plot(FPR_best[min_index],TPR_best[min_index],'r*',label=f'XGB cutoff val: {optimal_cutoff:0.3f}')
    axs[1].legend(loc='lower right')
    axs[1].set_xlabel('FPR')
    axs[1].set_ylabel('TPR')
    axs[1].plot([0,1],[0,1],'r--')
    axs[1].set_title('ROC curve')

    # plt.show()

    fig, axs = plt.subplots(1,2)

    custom_colors = ["c" for x in range(df_test.shape[0])]
    alpha_list = [0.2 for x in range(df_test.shape[0])]
    for ii in range(len(custom_colors)):
        if (df_test.flipped.iloc[ii]==False) & (df_test.LI_fmri.iloc[ii]<0):
            custom_colors[ii] = "g"
            alpha_list[ii] = 0.8
        if (df_test.flipped.iloc[ii]==True) & (df_test.LI_fmri.iloc[ii]>0):
          custom_colors[ii] = "b"
          alpha_list[ii] = 0.8

    scatter = axs[0].scatter(y_test,y_pred,alpha=alpha_list,c=custom_colors)

    axs[0].axvline(0)
    axs[0].axhline(optimal_cutoff,color='r',label=f'XGB cutoff val: {optimal_cutoff:0.3f}')
    legend1 = axs[0].legend(loc='upper left',
                            framealpha=0.2)
    axs[0].add_artist(legend1)

    axs[0].set_xlabel('LI fMRI')
    axs[0].set_ylabel('XGBoost Predicted LI')
    axs[0].set_title('Regression Performance')
    y_pred_bool = y_pred > optimal_cutoff # if predicted value above cutoff -> left
    y_test_bool = y_test > 0 # if test values are above 0 -> left

    cm = confusion_matrix(y_test_bool,y_pred_bool)
    print(cm)

    y_true_right = y_pred[(df_test.flipped==False) & (df_test.LI_fmri < 0)]
    print(y_true_right)
    y_true_right_bool = y_true_right < optimal_cutoff
    right_correct = np.sum(y_true_right_bool)
    right_num = len(y_true_right_bool)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test_bool, y_pred_bool)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['right','left'])
    disp.plot(ax=axs[1])
    axs[1].set_title(f"Accuracy: {acc:0.3f}. AUC: {AUC_best:0.3f}")
    return {"acc":acc,"AUC":AUC_best,"right_correct":right_correct,"right_num": right_num, "optimal_cutoff":optimal_cutoff,
            "custom_colors":custom_colors,"alpha_list":alpha_list}

  ROC_results = make_ROC(y_test, y_pred)
  main.make_ROC = make_ROC

  if SUBSET_TEST:

    #Test on a different subset
    tensor_X = test_subset
    tensor_X = tensor_X.dropna()

    tensor_X = augment_X(tensor_X)
    onehot_method_tensor_X = pd.get_dummies(tensor_X.method, prefix='method')
    tensor_X = pd.concat([onehot_method_tensor_X, tensor_X],axis=1)
    tensor_X = tensor_X.drop(columns=['method','tract'])
    y_true = tensor_X.LI_fmri
    tensor_X = tensor_X.drop(columns = ['subject_id','flipped','LI_fmri','LI_fmri_flipped'])

    tensor_X = tensor_X.astype({'clean':'bool','ACT':'bool'})
    if NORMALIZE:
      tensor_X = norm_transformer.transform(tensor_X)

    y_pred = model.predict(tensor_X)

    custom_colors = ["c" for x in range(test_subset.shape[0])]
    alpha_list = [0.2 for x in range(test_subset.shape[0])]
    for ii in range(len(custom_colors)):
        if (test_subset.flipped.iloc[ii]==False) & (test_subset.LI_fmri.iloc[ii]<0):
            custom_colors[ii] = "g"
            alpha_list[ii] = 0.8
        if (test_subset.flipped.iloc[ii]==True) & (test_subset.LI_fmri.iloc[ii]<0):
          custom_colors[ii] = "b"
          alpha_list[ii] = 0.8

    fig, axs= plt.subplots()
    axs.scatter(y_true, y_pred,c=custom_colors, alpha=alpha_list)

    axs.axvline(0)
    axs.axhline(ROC_results["optimal_cutoff"],color='r',label=f'XGB cutoff val: {ROC_results["optimal_cutoff"],:0.3f}')
    legend1 = axs.legend(loc='upper left',
                            framealpha=0.2)
    axs.add_artist(legend1)

    axs.set_xlabel('LI fmri')
    axs.set_ylabel('Xgboost Prediction Tensor')
    axs.set_title("tested on subset")

    
  if SHOW_PLOTS:
    plt.show()
  print(ROC_results["acc"], ROC_results["AUC"])
  print(f'{ROC_results["right_correct"]} out of {ROC_results["right_num"]} found')


  if SAVE_MODEL:
    model.save_model(f'XGB_batch_{count}.json')

  return {"model_params":model_params,
          "accuracy":ROC_results["acc"],
          "AUC": ROC_results["AUC"],
          "right_correct": ROC_results["right_correct"],
          "right_num":ROC_results["right_num"],
          "y_pred":y_pred,
          "y_test":y_test,
          "custom_colors":ROC_results["custom_colors"],
          "alpha_list":ROC_results["alpha_list"],
          "optimal_cutoff":ROC_results["optimal_cutoff"],
          }

if __name__ == '__main__':
  # Load data
  df_original = pd.read_csv("AF_tract_fmri_data.csv")
  df_original = df_original.drop(columns = ["Unnamed: 0.1", "Unnamed: 0"] )

  test_subset = df_original[(df_original.ACT==False)]

  if USE_SUBSET:
      #filter out subset of data to test on
      df = df_original[df_original.ACT==True]

      df = df.dropna()
  else:

      df = df_original 

  df0 = df_original[(df_original.ACT==True) & (df_original.clean==True) & (df_original.method=="iFOD2")]
  df1 = df_original[(df_original.ACT==True) & (df_original.clean==True) & (df_original.method=="Tensor_Prob")]
  df2 = df_original[(df_original.ACT==True) & (df_original.clean==False) & (df_original.method=="iFOD2")]
  df3 = df_original[(df_original.ACT==True) & (df_original.clean==False) & (df_original.method=="Tensor_Prob")]
  df4 = df_original[(df_original.ACT==False) & (df_original.clean==True) & (df_original.method=="iFOD2")]
  df5 = df_original[(df_original.ACT==False) & (df_original.clean==True) & (df_original.method=="Tensor_Prob")]
  df6 = df_original[(df_original.ACT==False) & (df_original.clean==False) & (df_original.method=="iFOD2")]
  df7 = df_original[(df_original.ACT==False) & (df_original.clean==False) & (df_original.method=="Tensor_Prob")]

  df8 = df_original[(df_original.clean==True) & (df_original.method=="iFOD2")]
  df9 = df_original[(df_original.clean==True) & (df_original.method=="Tensor_Prob")]
  df10 = df_original[(df_original.clean==False) & (df_original.method=="iFOD2")]
  df11 = df_original[(df_original.clean==False) & (df_original.method=="Tensor_Prob")]

  df12 = df_original[(df_original.ACT==True) & (df_original.method=="iFOD2")]
  df13 = df_original[(df_original.ACT==True) & (df_original.method=="Tensor_Prob")]
  df14 = df_original[(df_original.ACT==False) & (df_original.method=="iFOD2")]
  df15 = df_original[(df_original.ACT==False) & (df_original.method=="Tensor_Prob")]

  df16 = df_original[(df_original.ACT==True) & (df_original.clean==False)]
  df17 = df_original[(df_original.ACT==True) & (df_original.clean==True)]
  df18 = df_original[(df_original.ACT==False) & (df_original.clean==False)]
  df19 = df_original[(df_original.ACT==False) & (df_original.clean==True)]

  df20 = df_original[(df_original.ACT==True)]
  df21 = df_original[(df_original.ACT==False)]
  df22 = df_original[(df_original.clean==True)]
  df23 = df_original[(df_original.clean==False)]
  df24 = df_original[(df_original.method=="iFOD2")]
  df25 = df_original[(df_original.method=="Tensor_Prob")]

  df26 = df_original
  
  df_list = [df0, df1, df2, df3, df4, df5, df6, df7, df8, df9,
             df10, df11, df12, df13, df14, df15, df16, df17, df18, df19,
             df20, df21, df22, df23, df24, df25, df26]

  # result = main(df0,test_subset,count=0)
  # results_list = []
  # count=0
  # for df in df_list:
  #   results = main(df,test_subset,count)
  #   results_list.append(results)
  #   count+=1

  # df_results = pd.DataFrame(results_list)
  # df_results.to_csv("results_nogridsearch.csv")
  # print(pd.DataFrame(results_list))


  result0 = main(df0,test_subset,model_path="XGB_batch_0.json")
  result8 = main(df0,test_subset,model_path="XGB_batch_8.json")
  result20 = main(df0,test_subset,model_path="XGB_batch_20.json")

  result_list = [result0, result8, result20]
  y_pred_list = [result0["y_pred"], result8["y_pred"], result20["y_pred"]]
  p = np.array(y_pred_list)
  pred = [x for xs in y_pred_list for x in xs]

  y_test_list = [result0["y_test"], result8["y_test"], result20["y_test"]]
  t = result0["y_test"]

  test = [x for xs in y_test_list for x in xs]
  custom_col_list = [result0["custom_colors"], result8["custom_colors"], result20["custom_colors"]]
  alpha_list_all = [result0["alpha_list"], result8["alpha_list"], result20["alpha_list"]]

  cutoff_list = [result0["optimal_cutoff"], result8["optimal_cutoff"], result20["optimal_cutoff"]]
  cut = np.array(cutoff_list)

  alphs = [x for xs in alpha_list_all for x in xs]
  cols = [x for xs in custom_col_list for x in xs]
  cs = result0["custom_colors"]

  p_mean = np.mean(p,axis=0)
  p_sum = np.sum(p,axis=0)

  cut_mean = np.mean(cut)
  cut_sum = np.sum(cut)


  fig, axs = plt.subplots(2,2)

  output = {"test":t,"pred":p_mean,"cols":result0["custom_colors"],"alphs":result0["alpha_list"]}
  df_output = pd.DataFrame(output)
  print(df_output)
  df_output.to_csv("population_performance_mean_model.csv")

  axs[0,0].scatter(y_test_list, y_pred_list,c=cols,alpha=alphs)
  axs[0,0].set_title("3 best models results")
  axs[0,0].set_xlabel("LI fmri")
  axs[0,0].set_ylabel("LI Xgboost")
  axs[0,0].axvline(0)

  axs[0,1].scatter(result0["y_test"],result0["y_pred"],c=result0["custom_colors"],alpha=result0["alpha_list"])
  axs[0,1].set_title("Best model so far")
  axs[0,1].set_xlabel("LI fmri")
  axs[0,1].set_ylabel("LI Xgboost")
  axs[0,1].axvline(0)
  axs[0,1].legend()

  axs[1,0].scatter(t,p_mean,c=result0["custom_colors"],alpha=result0["alpha_list"])
  axs[1,0].set_title("Mean of 3 best models")
  axs[1,0].set_xlabel("LI fmri")
  axs[1,0].set_ylabel("LI Xgboost")
  axs[1,0].axvline(0)
  axs[1,0].legend()

  axs[1,1].scatter(t,p_sum,c=result0["custom_colors"],alpha=result0["alpha_list"])
  axs[1,1].set_title("Sum of all results")
  axs[1,1].set_xlabel("LI fmri")
  axs[1,1].set_ylabel("LI Xgboost")
  axs[1,1].axvline(0)
  axs[1,1].legend()

  fig.tight_layout()

  ROC2_results = main.make_ROC(t,p_mean)

  plt.show()


  # for result in result_list:
  #   y_pred = result["y_pred"]
  #   y_test = result["y_test"]
  #   custom_colors = result["custom_colors"]
  #   alpha_list = result["alpha_list"]
  #   optimal_cutoff = result["optimal_cutoff"]

    