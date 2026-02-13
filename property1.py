import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("C:\\Users\\Ashfaque Raza\\Downloads\\Bengaluru_House_Data.csv")

# ── Pre-process: drop rows with nulls in key columns ──────────────────────────
df = df.dropna(subset=['area_type', 'bhk', 'balcony', 'price'])
df['bhk']     = pd.to_numeric(df['bhk'],     errors='coerce')
df['balcony'] = pd.to_numeric(df['balcony'], errors='coerce')
df['price']   = pd.to_numeric(df['price'],   errors='coerce')
df = df.dropna(subset=['bhk', 'balcony', 'price'])

# ── Encode & scale area_type ───────────────────────────────────────────────────
le = LabelEncoder()
df['area_type_encoded'] = le.fit_transform(df['area_type'])
scaler = StandardScaler()
df['area_type_scaled'] = scaler.fit_transform(df['area_type_encoded'].values.reshape(-1, 1))

# ── Features & target ─────────────────────────────────────────────────────────
x = df[['area_type_scaled', 'bhk', 'balcony']].values
y = df['price'].values

# ── Train KNN once at startup ─────────────────────────────────────────────────
KNN = KNeighborsRegressor(n_neighbors=5)
KNN.fit(x, y)

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.title("Bengaluru Property")

menu = st.sidebar.radio('Menu', ['Home', 'Price Summary'])

# ── Home ──────────────────────────────────────────────────────────────────────
if menu == 'Home':
    st.header('Property Chart')

    if st.checkbox('Show Property Data'):
        st.table(df.head(100))

    st.title('Graphs')
    graph = st.selectbox('Different Types of Graph', ['Bar plot', 'Histogram'])

    if graph == 'Bar plot':
        fig, ax = plt.subplots(figsize=(6, 4))
        sn.barplot(data=df, x='bhk', y='price', estimator=np.mean, ax=ax)
        ax.set_title('Average Price by BHK')
        st.pyplot(fig)

    if graph == 'Histogram':
        fig, ax = plt.subplots(figsize=(6, 4))
        sn.histplot(df['price'], kde=True, bins=30, ax=ax)
        ax.set_title('Price Distribution')
        st.pyplot(fig)

# ── Price Summary ─────────────────────────────────────────────────────────────
if menu == 'Price Summary':
    st.header("Price Summary according to your choice")

    area_type_options = list(le.classes_)

    value_  = st.radio("Choose the area type", area_type_options)
    value   = st.slider("How much BHK you want", 1, 50)
    value1  = st.number_input("Number of balconies", min_value=0, max_value=4, step=1)

    if st.button('Price Prediction ($)'):
        # Encode and scale user input exactly like training data
        area_encoded = le.transform([value_])[0]
        area_scaled  = scaler.transform([[area_encoded]])[0][0]

        input_data = np.array([[area_scaled, float(value), float(value1)]])

        prediction = KNN.predict(input_data)

        st.success(
            f"The predicted price for a **{value_}**, "
            f"**{value} BHK**, **{int(value1)} balcony** house is "
            f"**${prediction[0]:,.2f} **"
        )

        score = KNN.score(x, y)
        st.info(f"Model Training Accuracy: {score:.2%}")
