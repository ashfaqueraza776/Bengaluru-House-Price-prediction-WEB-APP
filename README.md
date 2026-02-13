# Bengaluru-House-Price-prediction-WEB-APP
The Bengaluru Property Price Predictor is an interactive web application built using Python and Streamlit that allows users to explore real estate data from Bengaluru, India, and predict property prices based on key inputs.
The application has two main sections. The Home section provides a visual overview of the dataset, including a bar chart showing average property prices by BHK configuration and a histogram displaying the overall price distribution. Users can also browse the raw property data directly within the app.
The Price Summary section is powered by a K-Nearest Neighbours (KNN) regression model trained on real Bengaluru housing data. Users select an area type (such as Super Built-up or Carpet Area), specify the number of BHKs and balconies, and the model instantly predicts an estimated property price in Lakhs (₹). The input features are preprocessed using Label Encoding and Standard Scaling to match the training pipeline, ensuring accurate and consistent predictions.
Tech Stack: Python, Streamlit, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.
Dataset: Bengaluru House Data — containing property listings with attributes such as area type, BHK count, balcony count, and price.
This project demonstrates the application of machine learning in the real estate domain, combining data visualisation and predictive modelling into a single, user-friendly interface.
<img width="1898" height="910" alt="image" src="https://github.com/user-attachments/assets/d1b275d4-7f9b-4535-96c7-d9ef29555847" />

<img width="1898" height="908" alt="image" src="https://github.com/user-attachments/assets/85066360-fad8-4a6f-9a9f-b9e016001686" />

