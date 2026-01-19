import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import requests

from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix


#logger

def log(messsage):
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}]{messsage}")


#session state Intialization 
if "cleaned_data" not in st.session_state:
    st.session_state.cleaned_saved=False

#Folder Setup 
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
RAW_DIR=os.path.join(BASE_DIR,"data","raw")
CLEAN_DIR=os.path.join(BASE_DIR,"data","cleaned")

os.makedirs(RAW_DIR,exist_ok=True)
os.makedirs(CLEAN_DIR,exist_ok=True)

log("Application Started")
log(f"RAW_DIR={RAW_DIR}")
log(f"CLEAN_DIR={CLEAN_DIR}")

#PAGE CONFIGURATION
st.set_page_config(
    "End-to-End SVM",
    layout="wide",
)
st.title("End-to-End SVM Platform")

#sidebar

st.sidebar.title("SVM Settings")
kernel=st.sidebar.selectbox("kernel",["linear","rbf","poly","sigmoid"])
C=st.sidebar.slider("C(Regularization)",0.01,10.0,1.0)
gamma=st.sidebar.selectbox("Gamma",["scale","auto"])

log(f"SVM Settings:kernel={kernel},C={C},gamma={gamma}")

#Step 1:Data Ingestion
st.header("Step 1:Data Ingestion")
log("Step 1:Data Ingestion Started")

option=st.radio("Choose Data Source",["Download Dataset","Upload CSV"])

df=None
raw_path=None

if option=="Download Dataset":
    if st.button("Download Iris Dataset"):
        log("Downloading Iris Dataset")
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

        response=requests.get(url)
        raw_path=os.path.join(RAW_DIR,"iris.csv")
        with open(raw_path,"wb") as f:
            f.write(response.content)

        df=pd.read_csv(raw_path)
        st.success("Dataset Downloaded Successfully")
        log(f"Iris Dataset saved at {raw_path}")

if option=="Upload CSV":
    upload_file=st.file_uploader("upload your CSV file",type=["csv"])
    if upload_file:
        raw_path=os.path.join(RAW_DIR,upload_file.name)
        with open(raw_path,"wb") as f:
            f.write(upload_file.getbuffer())

        df=pd.read_csv(raw_path)
        st.success("File Uploaded Successfully")
        log(f"Uploaded data saved at {raw_path}")
        
#===========================================

#Step 2:EDA
if df is not None:
    st.header("Step 2:Exploratory Data Analysis (EDA)")
    log("Step 2:EDA Started")

    st.dataframe(df.head())
    st.write("Shape of the dataset:",df.shape)
    st.write("Missing values:",df.isnull().sum())

    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True),annot=True,cmap="coolwarm",ax=ax)
    st.pyplot(fig)
    

    log("Step 2:EDA Completed")

#=========================

#Step 3:Data Cleaning

if df is not None:
    st.header("Step 3:Data Cleaning ")
    Strategy=st.selectbox("Missing Value Strategy",["Mean","Median","Drop Rows"])

    df_clean= df.copy()

    if Strategy=="Drop Rows":
        df_clean=df_clean.dropna()

    else:
        for col in df_clean.select_dtypes(include=np.number):
            if Strategy=="Mean":
                df_clean[col]=df_clean[col].fillna(df_clean[col].mean())
            else:
                df_clean[col]=df_clean[col].fillna(df_clean[col].median())

    st.session_state.df_clean=df_clean
    st.success("Data Cleaning Completed")
else:
    st.info("Please Complete Step 1 (Data Ingestion ) first....")

#=========================

#Step 4:Save Cleaned Data

if st.button("Save Cleaned Data"):

    if st.session_state.df_clean is None:
        st.error("No cleaned data found.Please complete Step 3 First..")

    else:
        time_stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_filename=f"cleaned_data_{time_stamp}.csv"
        clean_path=os.path.join(CLEAN_DIR,clean_filename)

        st.session_state.df_clean.to_csv(clean_path,index=False)
        st.success(f"Cleaned Data saved ")
        st.info(f"Saved at : {clean_path}")
        log(f"Cleaned data saved at {clean_path}")

#============================

#Step 5:Load Cleaned Data


st.header("Step 5:Load Cleaned Data")
clean_files=os.listdir(CLEAN_DIR)

if not clean_files:
    st.warning("No cleaned dataset found. Please save one in step 4.")
    log("No cleaned dataset available ")
else:
    selected=st.selectbox("Select Cleaned Dataset",clean_files)
    df_model=pd.read_csv(os.path.join(CLEAN_DIR,selected))

    st.success(f"Loaded dataset: {selected}")
    log(f"Loaded cleaned dataset:{selected}")

    st.dataframe(df_model.head())

#=========================

#Step 6:Train Svm

st.header("Step 6:Train SVM Model")
log("Step 6 Started:SVM Training")
target=st.selectbox("Select Target Column",df_model.columns)
y=df_model[target]

if y.dtype=="object":
    y=LabelEncoder().fit_transform(y)
    log("Target Column Encoded")

#Select numeric features only

x=df_model.drop(columns=[target])
x=x.select_dtypes(include=np.number)

if x.empty:
    st.error("No Numeric Features available for training")
    st.stop()

#Scale features
scaler=StandardScaler()
x=scaler.fit_transform(x)

#Train-test split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

#svm 
model=SVC(kernel=kernel,C=C,gamma=gamma)
model.fit(X_train,y_train)

#Evaluate
y_pred=model.predict(X_test)
acc=accuracy_score(y_test,y_pred)

st.success(f" accuracy: {acc*100:.2f}%")
log(f"SVM Training Completed with accuracy: {acc*100:.2f}%")

cm=confusion_matrix(y_test,y_pred)
fig,ax=plt.subplots()
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax)
st.pyplot(fig)







