from re import L
from tractogram_data import Subject
import streamlit as st
from io import StringIO, BytesIO
import nibabel.streamlines as nibs


st.write(
f"""
# Fiber tract data extractor.
"""
)


st.sidebar.write("# Left tract")
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
    sex = st.selectbox("Subject sex",("male","female","Undefined"))
    st.write(f"sex: {sex}")
    age = st.slider('Subject Age', 0, 130, 50)
    st.write(f'age: {age}')
    
    #create some padding
    for i in range(10):
        st.write("")
    
    st.write("""
    RP, AR, SS -  2022
    """)


sub_metadata = {"id":str(sub_id),
                "sex":str(sex),
                "age":str(age),
                "tract_name":str(tract_info)}


if (left_file is not None) & (right_file is not None):

    sub_path_L = left_file
    test = nibs.load(sub_path_L)
    st.write(test)

    # sub = Subject(sub_path_L,
    #             sub_path_R,
    #             sub_metadata)

    
    # df_display = sub.df.T.astype(str)

    # st.table(df_display)

    # @st.cache
    # def convert_df(df):
    #     return df.to_csv().encode('utf-8')
    
    # csv = convert_df(sub.df)

    # st.download_button(label="Download data as CSV",
    #                     data=csv,
    #                     file_name='tract_data.csv',
    #                     mime='text/csv')
else:
    st.write("""
    ## <- check the sidebar to get started
    """)