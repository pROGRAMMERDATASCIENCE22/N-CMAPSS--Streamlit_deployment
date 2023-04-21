import streamlit as st
import pandas as pd
from PIL import Image

import numpy as np
from os.path import join as opj
import matplotlib.pyplot as plt


# Page setting
#st.set_page_config(layout="wide")
st.set_page_config(page_title="Model Deployment", page_icon=":sun:", layout="centered",
                           initial_sidebar_state="auto",
                           menu_items=None)
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


st.sidebar.header('***N-CMAPSS DEPLOYMENT***')
st.sidebar.subheader("Choose the chapter of the Data Story")

files = ['N-CMAPSS_DS01-005.h5','N-CMAPSS_DS04.h5',
         'N-CMAPSS_DS08a-009.h5','N-CMAPSS_DS05.h5',
         'N-CMAPSS_DS02-006.h5','N-CMAPSS_DS08c-008.h5',
         'N-CMAPSS_DS03-012.h5','N-CMAPSS_DS07.h5','N-CMAPSS_DS06.h5']
f_c = [1,2,3]
all_files=pd.read_csv('File_DevUnits_TestUnits.csv')
all_files = all_files.drop(columns=['Unnamed: 0'])
all_files = all_files.set_index('File')
all_flights=pd.read_csv('File_DevUnits_TestUnits_allflightclass.csv')
all_flights = all_flights.drop(columns=['Unnamed: 0'])
color = ['#b1db57','#dbce57', '#b1db57','#dbce57','#b1db57','#dbce57','#b1db57','#dbce57','#b1db57']
def to_color(x):
    return ["background-color: %s"%i for i in color]
all_flights.style.apply(to_color, axis=0)

def user_input_units():

    file_number = st.selectbox("**Choose the File ID to view all the Development and Test Units**", options=files)
    st.write("File  :", file_number)
    #st.checkbox("Use container width", value=False, key="use_container_width")
    #[lambda x: st.table(all_files.iloc[x]) for x in range(9)]
    for i in range(9):
        if file_number == files[i]:
            st.table(all_files.iloc[i])
    file_number = st.selectbox("**Choose the File ID to view Development and Test Units across the Flight classes**",
                                       options=files)
    st.write("File  :", file_number)


    for i in range(9):
        if file_number == files[i]:
            st.checkbox("Use container width", value=False, key="use_container_width")
            st.table(all_flights.iloc[i].T)

    return #features


genre = st.sidebar.radio(
    " ",
    ('Home', 'Model work flow', 'Development and test', 'EDA', 'ML models', 'ML Model and Loss plots', 'Inference'))
if genre == 'Home':
    import base64
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
            f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
            unsafe_allow_html=True
        )
    add_bg_from_local(r'12__Featured_Image_AeroShow_Bangalore.png')
if genre == 'Model work flow':
    st.subheader("Aircraft Engine RUL prediction ML model development work flow")
    image = Image.open(r'MLworkflow.png')
    st.image(image, caption="", use_column_width='always')
if genre == 'EDA':
    # opt = st.sidebar.radio("Plots for analysis",('Lineplot', 'Scatter', 'Boxplot', 'KDE', 'Waterfall'))
    # if opt == "Lineplot":
    t2, t3, t4, t5 = st.tabs(["Line plots", "Distribution of data", "Time Series", "HeatMaps"])
    with t2:
        t2.subheader("Altitude ranging across Flight classes 1, 2, 3 ")
        st.text(["Flight class vs Altitude"])
        image = Image.open(r'D:\IISc\CAPSTONE\Streamlit codes\EDA Images\\Flight_Altitude.png')
        st.image(image, caption="", use_column_width='always')
        st.write(" Higher the Flight class ( length of the flight ) higher the altitude ")

    with t3:
        t10, t12 = st.tabs(["Box Plot", "Distribution between development and Test"])
        with t10:
            t10.subheader("Distributions of Target values across Units")
            image = Image.open('D:\IISc\CAPSTONE\Streamlit codes\EDA Images\\box_unit_target.png')
            st.image(image, caption="", use_column_width='always')
        with t12:
            t12.text(
                """
                Data Distribution between development set and test set is shown for DS01 & DS08a.
                It is observed that the data in both the sets are not normally distributed and hence
                data normalization is required before fitting ML model.
                """
                )
            t13, t14 = st.tabs(["DS01_005--KDE", "DS008a-009--KDE"])
            with t13:
                image = Image.open(r'D:\IISc\CAPSTONE\Streamlit codes\EDA Images\kde1.png')
                st.image(image, caption="", use_column_width='always')
            with t14:
                image = Image.open(r'D:\IISc\CAPSTONE\Streamlit codes\EDA Images\kde2.png')
                st.image(image, caption="", use_column_width='always')
    with t4:
        image = Image.open(r'D:\IISc\CAPSTONE\Streamlit codes\EDA Images\time series final.png')
        st.image(image, caption='')
        st.text(
            """
            Above chart shows sensor data from A, W & Xs variables for all engines/units from DS01, DS04 & DS08a h5files. There is inverse 
            relation observed between Altitude, flight duration to the RUL (Target). Higher the altitude & duration – lower the RUL of engine
            and vice-versa. Inverse relationship trend between Alt, duration to RUL is reflecting in all files across all engines except 
            DS04 & DS08a h5 files. There is a possibility of incorrect data or data extraction issues which shows a detrend behaviour in 
            DS04 & DS08a files, by considering this data there is high probability that predictions might be away from +/- 5% error margin. 
            Hence DS04, DS08a  files are excluded for model training. A separate trained model built considering DS04 & DS08a  files as an 
            edge case to check the RUL predictions.
            """
            )
    with t5:
        t5.subheader("Heat maps")
        t2, t5 = st.tabs(["Xs_DS02-006", "Xs_DS05"])
        with t2:
            t2.subheader("Xs_DS02-006")
            image = Image.open('Xs_DS02-006.png')
            st.image(image, caption='')
        with t5:
            t5.subheader("Xs_DS05")
            image = Image.open('Xs_DS05.png')
            st.image(image, caption='')
elif genre == 'Development and test':
    df = user_input_units()


elif genre == 'ML models':
    st.subheader("NN MODELS ")
    t1, t2, t3, t4 = st.tabs(['GRU', 'DEEPGRU', 'LARGEST GRU', 'TRANSFORMER'])
    with t1:
         image = Image.open('D:\IISc\CAPSTONE\Streamlit codes\EDA Images\gru_cnn_dc.png')
         st.image(image, caption = "")
    with t2:
         image = Image.open('D:\IISc\CAPSTONE\Streamlit codes\EDA Images\deepgrucnnfc.png')
         st.image(image, caption = "")
    with t3:
         image = Image.open('D:\IISc\CAPSTONE\Streamlit codes\EDA Images\largestcudnngru.png')
         st.image(image, caption = "")
    with t4:
         image = Image.open('D:\IISc\CAPSTONE\Streamlit codes\EDA Images\\transformer.png')
         st.image(image, caption = "")
    
elif genre == 'ML Model and Loss plots':
    #st.sidebar.radio("Choose the model plot",('DEEP GRU CNN','GRU CNN DC', 'TRANSFORMER BASIC','TRANSFORMER ADVANCED', 'RANDOM PLOTS'))
    st.subheader("NN model training, testing and Loss plots")
    t1, t2, t3, t4, t10 =st.tabs(["GRU","Deep GRU","Basic Transformer","Advanced Transformer", "Random"])
    with t1:
        t5, t6, t7, t8, t9 = st.tabs(["03-012-Unit 15", "05-Unit9", "06-Unit10", "08c-Unit9", "Loss Plot"])
        with t5:
            t5.subheader("RMSE--15.38")
            image = Image.open(r'D:\IISc\CAPSTONE\Streamlit codes\EDA Images\GRU_N-CMAPSS_DS03-012_unit15_test_w50_s1_bs256_lr0.001_sub10_rmse-15.38.png')
            st.image(image, caption='RMSE--15.38')
        with t6:
            t6.subheader("RMSE--15.38")
            image = Image.open(r'D:\IISc\CAPSTONE\Streamlit codes\EDA Images\GRU_N-CMAPSS_DS05_unit9_test_w50_s1_bs256_lr0.001_sub10_rmse-15.38.png')
            st.image(image, caption='RMSE--15.38')
        with t7:
            t7.subheader("RMSE--15.38")
            image = Image.open(r'D:\IISc\CAPSTONE\Streamlit codes\EDA Images\GRU_N-CMAPSS_DS06_unit10_test_w50_s1_bs256_lr0.001_sub10_rmse-15.38.png')
            st.image(image, caption='RMSE--15.38')
        with t8:
            t8.subheader("RMSE--15.38")
            image = Image.open(r'D:\IISc\CAPSTONE\Streamlit codes\EDA Images\GRU_N-CMAPSS_DS08c-008_unit9_test_w50_s1_bs256_lr0.001_sub10_rmse-15.38.png')
            st.image(image,caption='RMSE--15.38')
        with t9:
            t9.subheader("RMSE--15.38")
            image = Image.open(r'GRU_training_w50_s1_bs256_sub10_lr0.001.png')
            st.image(image, caption='GRU ---Loss Plot')
    with t2:
        t5, t6, t7, t8, t9 = st.tabs(["03-012-Unit 15", "05-Unit9", "06-Unit10", "08c-Unit9", "Loss Plot"])
        with t5:
            t5.subheader("RMSE--17.45")
            image = Image.open(r'D:\IISc\CAPSTONE\Streamlit codes\EDA Images\Deep_GRU_N-CMAPSS_DS03-012_unit15_test_w50_s1_bs256_lr0.001_sub10_rmse-17.45.png')
            st.image(image, caption='RMSE--17.45')
        with t6:
            t6.subheader("RMSE--17.45")
            image = Image.open(r'D:\IISc\CAPSTONE\Streamlit codes\EDA Images\Deep_GRU_N-CMAPSS_DS05_unit9_test_w50_s1_bs256_lr0.001_sub10_rmse-17.45.png')
            st.image(image, caption='RMSE--17.45')
        with t7:
            t7.subheader("RMSE--17.45")
            image = Image.open(r'D:\IISc\CAPSTONE\Streamlit codes\EDA Images\Deep_GRU_N-CMAPSS_DS06_unit10_test_w50_s1_bs256_lr0.001_sub10_rmse-17.45.png')
            st.image(image, caption='RMSE--17.45')
        with t8:
            t8.subheader("RMSE--17.45")
            image = Image.open(r'D:\IISc\CAPSTONE\Streamlit codes\EDA Images\Deep_GRU_N-CMAPSS_DS08c-008_unit9_test_w50_s1_bs256_lr0.001_sub10_rmse-17.45.png')
            st.image(image, caption='RMSE--17.45')
        with t9:
            t9.subheader("RMSE--17.45")
            image = Image.open('Deep_GRU_training_w50_s1_bs256_sub10_lr0.001.png')
            st.image(image, caption='Deep GRU ---Loss Plot')
    with t3:
        t5, t6, t7, t8, t9 = st.tabs(["03-012-Unit 15", "05-Unit9", "06-Unit10", "08c-Unit9", "Loss Plot"])
        with t5:
            t5.subheader("RMSE--4.8")
            image = Image.open('TB__N-CMAPSS_DS03-012_unit15_test_w50_s1_bs256_lr0.001_sub10_rmse-4.8.png')
            st.image(image, caption='RMSE--4.8')
        with t6:
            t6.subheader("RMSE--4.8")
            image = Image.open('TB__N-CMAPSS_DS05_unit9_test_w50_s1_bs256_lr0.001_sub10_rmse-4.8.png')
            st.image(image, caption='RMSE--4.8')
        with t7:
            t7.subheader("RMSE--4.8")
            image = Image.open('TB__N-CMAPSS_DS06_unit10_test_w50_s1_bs256_lr0.001_sub10_rmse-4.8.png')
            st.image(image, caption='RMSE--4.8')
        with t8:
            t8.subheader("RMSE--4.8")
            image = Image.open('TB__N-CMAPSS_DS08c-008_unit9_test_w50_s1_bs256_lr0.001_sub10_rmse-4.8.png')
            st.image(image, caption='RMSE--4.8')
        with t9:
            t9.subheader("RMSE--4.8")
            image = Image.open('TB__training_w50_s1_bs256_sub10_lr0.001.png')
            st.image(image, caption='Basic Transformer--- Loss Plot')
    with t4:
        t5, t6, t7, t8, t9 = st.tabs(["03-012-Unit 15", "05-Unit9", "06-Unit10", "08c-Unit9", "Loss Plot"])
        with t5:
            t5.subheader("RMSE--4.68")
            image = Image.open('TA__N-CMAPSS_DS03-012_unit15_test_w50_s1_bs256_lr0.001_sub10_rmse-4.68.png')
            st.image(image, caption='RMSE--4.68')
        with t6:
            t6.subheader("RMSE--4.68")
            image = Image.open('TA__N-CMAPSS_DS05_unit9_test_w50_s1_bs256_lr0.001_sub10_rmse-4.68.png')
            st.image(image, caption='RMSE--4.68')
        with t7:
            t7.subheader("RMSE--4.68")
            image = Image.open('TA__N-CMAPSS_DS06_unit10_test_w50_s1_bs256_lr0.001_sub10_rmse-4.68.png')
            st.image(image, caption='RMSE--4.68')
        with t8:
            t8.subheader("RMSE--4.68")
            image = Image.open('TA__N-CMAPSS_DS08c-008_unit9_test_w50_s1_bs256_lr0.001_sub10_rmse-4.68.png')
            st.image(image, caption='RMSE--4.68')
        with t9:
            t9.subheader("RMSE--4.68")
            image = Image.open('TA__training_w50_s1_bs256_sub10_lr0.001.png')
            st.image(image, caption='Advanced Transformer--- Loss Plot')
    with t10:
        t11, t12, t13, t14, t15 = st.tabs(["01-005-Unit 7", "01-005-Unit 8", "01-005-Unit 8", "02-006-Unit 14", "08c-008-Unit 10"])
        with t11:
            t11.subheader("RMSE--86.89")
            image = Image.open('Random_N-CMAPSS_DS01-005_unit7_test_w50_s1_bs256_lr0.001_sub1_rmse-86.89.png')
            st.image(image, caption='RMSE--86.89')
        with t12:
            t12.subheader("RMSE--4.31")
            image = Image.open('Random_N-CMAPSS_DS01-005_unit8_test_w50_s1_bs256_lr0.001_sub10_rmse-4.31.png')
            st.image(image, caption='RMSE--4.31')
        with t13:
            t13.subheader("RMSE--15.58")
            image = Image.open('Random_N-CMAPSS_DS01-005_unit8_test_w50_s1_bs256_lr0.001_sub10_rmse-15.58.png')
            st.image(image, caption='RMSE--15.58')
        with t14:
            t14.subheader("RMSE--2.97")
            image = Image.open('Random_N-CMAPSS_DS02-006_unit14_test_w50_s1_bs256_lr0.001_sub10_rmse-2.97.png')
            st.image(image, caption='RMSE--2.97')
        with t15:
            t15.subheader("RMSE--86.89")
            image = Image.open('Random_N-CMAPSS_DS08c-008_unit10_test_w50_s1_bs256_lr0.001_sub1_rmse-86.89.png')
            st.image(image, caption='RMSE--86.89')
elif genre == 'Inference':
    opt = st.sidebar.radio("Final summary and prediction from the models",('Report', 'Model Deployment', 'Conclusion'))
    st.subheader("Inference of Model development, training and testing with RMSE")
    if opt == 'Report':
        st.text(
            """
            Different Loss Functions have been tried out. MSLE, NASAScore, Huber, MSE.
            NASAScore is a custom written LossFunction. Refer [2] for the formula.
            However MSE turned out to be the best Loss function. Though others are marginally worse.
            Model(GRU+CNN+FC) considered here was of 16.4M FLOPS ."""
        )
        image = Image.open('D:\IISc\CAPSTONE\Streamlit codes\EDA Images\Loss_RMSE_NASA_score.png')
        st.image(image, caption='Loss_RMSE_NASA_score')
        st.text(
            """
            Despite the fact that samples were randomly shuffled across engines, 
            across datasets, RUL predictions for the lower half (~50->0) were significantly 
            more accurate than the predictions for higher RULs (100->~50).
            Model(GRU+CNN+FC) considered here was of 49.6M FLOPS
            """
        )
        image = Image.open('D:\IISc\CAPSTONE\Streamlit codes\EDA Images\RUL_span_of_each_engine.png')
        st.image(image, caption = 'RUL span of each engine')
        st.text(
            """
            Across, different flight classes, Flight Class 1 was subsequently easy to predict, despite 
            having the least number of samples. Flight Class 2, Flight Class 3 were harder to predict accurately.
            Nevertheless, a single model was used for all the three classes. But it showcases that it can be a 
            good idea to develop flight class based models.
            Model(GRU+CNN+FC) considered here was of 49.6M FLOPS.
            """
        )
        image = Image.open('D:\IISc\CAPSTONE\Streamlit codes\EDA Images\Flight_class_Scores.png')
        st.image(image, caption='Flight class Scores')
        # image = Image.open('D:\IISc\CAPSTONE\Streamlit codes\EDA Images\Loss_RMSE_NASA_score.png')
        # st.image(image, caption='Loss_RMSE_NASA_score')
        # image = Image.open('Model-Flops-inference-time-per-sample.png')
        # st.image(image, caption='Flops inference time per sample')
        # image = Image.open('D:\IISc\CAPSTONE\Streamlit codes\EDA Images\Model-Flops-critical-full-score.png')
        # st.image(image, caption='Critical full scores')
        # image = Image.open('D:\IISc\CAPSTONE\Streamlit codes\EDA Images\Model-Flops-score.png')
        # st.image(image, caption='Best scores')
        st.text(
            """
            Overall, by just comparing the main models for inference speed, RMSE, NASA Score.
            Evaluation time per sample on GTX 1080 Ti.
            """
        )
        image = Image.open('D:\IISc\CAPSTONE\Streamlit codes\EDA Images\Model-Flops-inference-time-per-sample.png')
        st.image(image, caption='Model flops inference time per sample')

    if opt == 'Model Deployment':
        prediction_csv_dir = "prediction_csv_dir"
        filenames = ['N-CMAPSS_DS02-006', 'N-CMAPSS_DS07', 'N-CMAPSS_DS06', 'N-CMAPSS_DS01-005',
                     'N-CMAPSS_DS05', 'N-CMAPSS_DS03-012', 'N-CMAPSS_DS08c-008', 'N-CMAPSS_DS08a-009', 'N-CMAPSS_DS04']

        st.title("Model Deployment")
        filename = st.sidebar.selectbox(label="Choose any dataset to pick engines from:", options=filenames)
        file_devtest_df = pd.read_csv("ML_File_DevUnits_TestUnits.csv")

        units_index_test = np.fromstring(
            file_devtest_df[file_devtest_df.File == filename + '.h5']["Test Units"].values[0][1:-1],
            dtype=np.float, sep=" ").tolist()
        print(units_index_test)
        unit = st.sidebar.selectbox(label="Choose any Engine unit to predict RULs from:",
                                    options=[int(x) for x in units_index_test])
        rul_df = pd.read_csv(opj(prediction_csv_dir, "{}_Unit_{}.csv".format(filename, int(unit))))

        gt_rul = st.sidebar.slider(label="Choose a ground truth RUL",
                                   min_value=int(min(rul_df.RUL.values)), max_value=int(max(rul_df.RUL.values)))
        models = st.sidebar.multiselect(label="Choose Models:",
                                        options=["Transformer", "LargestCUDNN", "GRUCNNDC", "DeepGRU"])

        temp_df = rul_df[rul_df.RUL == gt_rul]
        index = temp_df.index.values[0]

        fig = plt.figure(figsize=(8, 8))
        plt.step([x * 10 + 10 for x in sorted(rul_df.RUL.values)], rul_df.RUL.values, label="RUL range", alpha=0.5)
        plt.scatter(index * 10 + 5, temp_df.RUL.values[0], label="Ground Truth", marker='*')
        if 'Transformer' in models:
            plt.scatter(index * 10 + 5, temp_df.Transformer.values[0], label="Transformer", marker='o')
        if 'LargestCUDNN' in models:
            plt.scatter(index * 10 + 5, temp_df.LargestCUDNN.values[0], label="LargestCUDNN", marker='x')
        if 'GRUCNNDC' in models:
            plt.scatter(index * 10 + 5, temp_df.GRUCNNDC.values[0], label="GRUCNNDC", marker='D')
        if 'DeepGRU' in models:
            plt.scatter(index * 10 + 5, temp_df.DeepGRU.values[0], label="DeepGRU", marker=',')
        plt.xlabel("Timestamp")
        plt.ylabel("RUL")
        #plt.grid(axis='y', color='0.95')
        plt.grid(which='both')
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.5)
        plt.legend(title='Model:')
        plt.title('Unit: {}'.format(unit))

        # plt.plot()
        st.pyplot(fig)

    if opt == 'Conclusion':
        t1, t2, t3 = st.tabs(["COMPARISON-TABLE", "NASA-SCORES", "RMSE-SCORES"])
        with t1:
            image = Image.open('D:\IISc\CAPSTONE\Streamlit codes\EDA Images\INFERENCE_Dataset-Scores.png')
            st.image(image, caption="Inference Dataset Scores")
            image = Image.open('D:\IISc\CAPSTONE\Streamlit codes\EDA Images\INFERENCE_Percent_Samples_Scores.png')
            st.image(image, caption="Inference Percent Samples Scores")

        with t2:
            image = Image.open('D:\IISc\CAPSTONE\Streamlit codes\EDA Images\FULL_NASA_dual_color.png')
            st.image(image, caption='NASA_SCORES')
        with t3:
            image = Image.open('D:\IISc\CAPSTONE\Streamlit codes\EDA Images\FULL_RMSE_dual_color.png')
            st.image(image, caption='RMSE-SCORES')

#st.write(df)
# uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
# for uploaded_file in uploaded_files:
#     bytes_data = uploaded_file.read()
#     st.write("filename:", uploaded_file.name)
#     st.write(bytes_data)


# options = st.multiselect(
#     'Choose the units',
#     [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

# st.write('You selected:', options)
#st.write(m.run(windows=15))
