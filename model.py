import sys

import numpy as np
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import io
from contextlib import redirect_stdout


def make_preprocessor():
    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ], verbose=True)

    numerical_transfomer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ], verbose=True)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transfomer, ['points']),
            ('cat', categorical_transformer, ['country', 'variety', 'province', 'winery']),
            ('drop', 'drop', ['region_1', 'region_2', 'designation', 'description']),
        ], remainder='passthrough', verbose=True)

    return preprocessor


def model(data):
    st.header('Preprocessing')
    preprocessor = make_preprocessor()

    processed_data = data.dropna(axis=0, subset=['price'])
    X = processed_data.drop('price', axis=1)
    y = processed_data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    np.random.seed(10)

    with io.StringIO() as buf, redirect_stdout(buf):
        result = preprocessor.fit_transform(X_train, y_train)
        output = buf.getvalue()
        st.code(output)
        st.write('Shape of output:')
        st.write(result.shape)
