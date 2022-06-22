from re import L
from tractogram_data import Subject
import streamlit as st
from io import StringIO, BytesIO
import nibabel.streamlines as nibs
from dipy.tracking.utils import density_map
import numpy as np
import nibabel as nib

st.write(
f"""
# 🧮 Fiber tract data extractor.

This app extracts features from **.tck** files in accordance with this publication:
[Shape analysis of the human association pathways, Fang-Cheng  Yeh, Neuroimage, 2020](https://www.sciencedirect.com/science/article/pii/S1053811920308156)

"""
)

st.sidebar.write("""
You found the sidebar, well done. Input all the metadata you want below and upload your files when you're ready.

If you don't have any .tck files, try these out for size:
""")

with open("AF_left.tck", "rb") as tck_file_left:
    tck_L_byte = tck_file_left.read()

with open("AF_right.tck", "rb") as tck_file_right:
    tck_R_byte = tck_file_right.read()

st.sidebar.download_button(label="💾 download Test TCK left",
                            data = tck_L_byte,
                            file_name="AF_left.tck",
                             mime='application/octet-stream')

st.sidebar.download_button(label="💾 download Test TCK right",
                            data = tck_R_byte,
                            file_name="AF_right.tck",
                             mime='application/octet-stream')

st.sidebar.write("""
# Left tract

""")
left_file = st.sidebar.file_uploader("Left tract .tck file")

st.sidebar.write("# Right tract")
right_file = st.sidebar.file_uploader("Right tract .tck file")


with st.sidebar:
    st.write('## tract info')
    tract_info = st.text_input("what kind of tract? (will be type in dataframe)")
    st.write(f"tract info: '{tract_info}'")
    st.sidebar.write(" # Subject Metadata")


    sub_id = st.text_input('subject id')
    st.write(f"id: '{sub_id}'")
    sex = st.selectbox("Subject sex",("female","male","other"))
    st.write(f"sex: {sex}")
    age = st.slider('Subject Age', 0, 130, 50)
    st.write(f'age: {age}')
    
    #create some padding
    for i in range(5):
        st.write("")
    
    st.write("""
This app is hosted online for all to use.
To address some privacy concerns:
- No data is stored online. [find more info here](https://docs.streamlit.io/knowledge-base/using-streamlit/where-file-uploader-store-when-deleted)
- The app is open source for you to inspect. [github repo here](https://github.com/Dopamineral/xgboost)
- The "download csv" button below creates data based on everything you can see on the page. And only you have access to this data.
- Once you close or refresh the browser the data is gone forever. It's annoying but it's safe.

    """)

    for i in range(10):
        st.write("")
    st.write("RP, AR, SS -  2022")


sub_metadata = {"id":str(sub_id),
                "sex":str(sex),
                "age":str(age),
                "tract_name":str(tract_info)}


if (left_file is not None) & (right_file is not None):

    st.write("""
    ## Calculated data from the .tck files
    
    """)
    with st.spinner('Calculating...'):
                
        tract_L = nibs.load(left_file)
        tract_R = nibs.load(right_file)

        sub = Subject(tract_L,
                    tract_R,
                    sub_metadata)

    st.success('Done!')
    
    df_display = sub.df.T.astype(str)
    df_display.columns={"features"}
    st.table(df_display)

    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')
    
    csv = convert_df(sub.df)


    st.write("""
    
    Download this CSV to keep a copy of this data.
    You will **need this file** for the **laterality prediction** in the other page. 
    """)
    st.download_button(label=" 📥 Download data as CSV",
                        data=csv,
                        file_name='tract_data.csv',
                        mime='text/csv')

   
    
else:
    st.write("""
    ## <- check the sidebar to get started
    """)

