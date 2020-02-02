import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache
def load_data():
    data = pd.read_csv('./datasets/wine-reviews/winemag-data-130k-v2.csv')
    data.drop('Unnamed: 0', inplace=True, axis=1)
    print(data)
    return data


def main():
    data = load_data().copy(deep=True)
    data.dropna(subset=['country', 'variety'], axis=0, inplace=True)

    max_price = 300.0
    price = st.slider('Price range', 0.0, max_price, 20.0)
    country = st.selectbox('Country', ['All'] + data['country'].unique().tolist())
    variety = st.selectbox('Variety', ['All'] + data['variety'].unique().tolist())
    keywords = st.text_area('Keywords', '')

    if st.button('Search'):
        vectorizer = CountVectorizer()
        matrix = vectorizer.fit_transform(data['description'].append(pd.Series(keywords)))
        similarities = cosine_similarity(matrix[-1], matrix).flatten()
        data['similarity'] = similarities[:-1]

        data = data[data['price'].between(price, max_price)]
        if country != 'All':
            data = data[data['country'] == country]

        if variety != 'All':
            data = data[data['variety'] == variety]

        data.sort_values(by='similarity', ascending=False, inplace=True)

        results = data[['country', 'description', 'points', 'price', 'title', 'variety', 'similarity']].head(
            10).reset_index(drop=True)
        st.table(results)


if __name__ == '__main__':
    main()
