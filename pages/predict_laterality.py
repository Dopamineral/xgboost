import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt

optimal_cutoff = 0.003

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
    st.dataframe(df)

    df_ready = df.drop(columns=['id','sex','age','tract_name'])

    model = XGBRegressor()
    model.load_model("XGB_model_dev.json")
    st.info(f"succesfully loaded model: {model}")
    pred=-0.21
    # reducer = joblib.load('reducer_umap.sav')
    # embedding = reducer.transform(df_ready)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Model prediction")
        st.metric(label="Predicted Laterality Index", value = pred, delta="0.85 certainty")


    with col2:
        st.write("Possible Clinical conclusion")
        st.metric(label="Laterality", value = "Right Dominance")




    pred=-0.21


    st.write("""

    ## Location compared to training data

    To have a more nuanced view of the prediction, please see where this data value falls among the data the the model was trained on.

    """)
    df_combined_test = pd.read_csv('./population_scores_dev.csv')
    fig, axs = plt.subplots()

    master_alpha = 0.5
    fig, axs = plt.subplots()

    axs.scatter(df_combined_test.true,df_combined_test.pred,alpha=0.5)
    axs.axvline(0)
    axs.axhline(optimal_cutoff,color='r',label=f'XGB cutoff val: {optimal_cutoff:0.3f}')
    axs.axhline(pred,color='g',label=f'Predicted Laterality index: {pred}')
    legend1 = axs.legend(loc='upper right')
    axs.add_artist(legend1)

    axs.set_xlabel('LI fMRI')
    axs.set_ylabel('XGBoost Prediction')
    st.pyplot(fig)


    st.write("""

    ## Location in embedding space

    Again for more nuance, see where the current prediction falls in the UMAP mebedding space of the data that the model was trained on.

    """)
    embedding_df = pd.read_csv('population_embedding_dev.csv')
    LI_df = pd.read_csv('population_LI_values.csv')

    fig, axs = plt.subplots()
    scatter1 = axs.scatter(embedding_df.iloc[:, 1], embedding_df.iloc[:, 2], c=LI_df.LI_fmri, cmap='seismic')
    axs.scatter([6],[-0.5],color='g',label="PREDICTION",marker='X')
    legend1 = axs.legend(loc='upper left')
    axs.add_artist(legend1)

    fig.colorbar(scatter1, label='Laterality Index fMRI',ax=axs)
    st.pyplot(fig)