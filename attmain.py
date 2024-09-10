import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import io

# Function to handle negative value shifting
def neg_shift(x):
    x = x.copy()
    for i in range(x.shape[1]):
        if np.min(x[:, i]) < 0:
            x[:, i] = x[:, i] - np.min(x[:, i])
    return x

st.title("Identify Key Attributes in Your Data Using Machine Learning")

st.write("""
Upload your CSV data, specify the class column (the column you want to analyze), 
and the app will identify and display the top attributes based on their importance using a Random Forest model.
""")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload CSV", type="csv")

# Input fields for ID and class columns
id_column = st.text_input("Enter ID column name (optional)", "")
class_column = st.text_input("Enter class column name (required)")

if uploaded_file is not None and class_column:
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(uploaded_file)

    if class_column not in df.columns:
        st.error("Class column not found. Please check the column name.")
    else:
        # Remove ID column if specified
        if id_column and id_column in df.columns:
            df = df.drop(columns=[id_column])

        # Preprocessing pipelines
        num_pipeline = make_pipeline(
            SimpleImputer(strategy='median'),
            FunctionTransformer(neg_shift, feature_names_out='one-to-one'),
            FunctionTransformer(np.log1p, feature_names_out='one-to-one'),
            StandardScaler()
        )

        cat_pipeline = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OneHotEncoder()
        )

        preprocess = ColumnTransformer([
            ('num', num_pipeline, make_column_selector(dtype_include=[int, float])),
            ('cat', cat_pipeline, make_column_selector(dtype_include=object))
        ])

        X = preprocess.fit_transform(df.drop([class_column], axis=1))
        y = df[class_column]

        # Handle missing values in the target variable if needed
        if y.isnull().any():
            si = SimpleImputer(strategy='most_frequent')
            y = si.fit_transform(y.values.reshape(-1, 1)).ravel()

        # Encode the target variable if it's categorical
        if y.dtype == object:
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Random Forest model
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        accuracy = rf.score(X_test, y_test)
        feat_imp = rf.feature_importances_
        columns = preprocess.get_feature_names_out()
        importance = pd.DataFrame({'Features': columns, 'Importance': feat_imp}).sort_values('Importance', ascending=False).reset_index(drop=True).round(3).head(10)

        # Display model accuracy
        st.write(f"Model Accuracy: {accuracy:.2f}")

        # Display feature importances as a table
        st.write("Top Attributes by Importance:")
        st.dataframe(importance)

        # Plot feature importances using Plotly
        fig = px.bar(importance, x='Importance', y='Features', title='Top Attributes', labels={'Importance': 'Importance Value', 'Features': 'Attribute'}, orientation='h')
        fig.update_layout(yaxis_title='Attribute', xaxis_title='Importance Value', yaxis=dict(categoryorder='total ascending'))
        st.plotly_chart(fig)

        # Option to download the feature importance data
        csv_data = importance.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="feature_importance.csv",
            mime="text/csv"
        )
else:
    st.warning("Please upload a CSV file and provide the class column name.")
