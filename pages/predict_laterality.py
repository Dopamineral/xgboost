import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
import numpy as np
import matplotlib.pyplot as plt

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


with st.sidebar:
    st.write("""Upload the csv file that was created earlier in **tck data app**""")
    data_file = st.sidebar.file_uploader(".csv file with tract metrics")

# TODO: Predict based on average of 3 models.
# TODO: add all the logic necessary to make the dashboard
# TODO: Reformat csv so it matches the df for the model
# TODO: Load the model and run prediction succesfully

st.write("""
# Laterality Prediction
Upload file in sidebar to get laterality prediction
""")

if data_file is not None:
    df = pd.read_csv(data_file)
    df_display = df.T.astype(str)
    df_display.columns={"features"}
    st.dataframe(df_display)
    

    optimal_cutoff = 0.015 #as found in make_model.py!

    df_ready = df.drop(columns=['id','sex','age','tract_name'])
    X_test = augment_X(df_ready)
    model1 = XGBRegressor()
    model2 = XGBRegressor()
    model3 = XGBRegressor()
    model1.load_model("XGB_batch_0.json")
    model2.load_model("XGB_batch_8.json")
    model3.load_model("XGB_batch_20.json")

    pred1 = model1.predict(X_test)
    pred2 = model2.predict(X_test)
    pred3 = model3.predict(X_test)

    mean_pred = np.mean([pred1,pred2,pred3])

    st.info(f"succesfully loaded models")
    st.info(f"succesfully made predictions")

    st.write(mean_pred)

    # reducer = joblib.load('reducer_umap.sav')
    # embedding = reducer.transform(df_ready)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Model prediction")
        st.metric(label="Predicted Laterality Index", value = f'{mean_pred:0.4f}')


    with col2:
        st.write("Possible Clinical conclusion")
        if mean_pred < optimal_cutoff:
            st.metric(label="Laterality", value = "Right Dominance")
        else:
            st.metric(label="Laterality", value = "Left Dominance")

    
    st.write("""

    ## Location compared to reference dataset

    To have a more nuanced view of the prediction, please see where this data value falls among the data the the model was tested on.

    """)
    df_pop = pd.read_csv('./population_performance_mean_model.csv')
    
    fig, axs = plt.subplots()
    axs.scatter(x=df_pop["test"],y=df_pop["pred"],
                c=df_pop["cols"], alpha=df_pop["alphs"])
    axs.axvline(0)
    axs.axhline(optimal_cutoff,color='r',label=f"XGBoost model cutoff: {optimal_cutoff}")
    axs.axhline(mean_pred,color="lime",label=f"predicted LI: {mean_pred:0.4f}")
    axs.set_title("Model test set")
    axs.set_xlabel("LI fMRI")
    axs.set_ylabel("Predicted LI XGBoost")
    axs.text(-0.3,0.2,"LEFT",color="r")
    axs.text(-0.3,-0.2,"RIGHT",color="r")
    axs.legend()

    st.pyplot(fig)
    

    