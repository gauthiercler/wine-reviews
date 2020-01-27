import streamlit as st

from eda import eda
from load import load_data
from model import model

pages = {
    'Exploratory data analysis': eda,
    'Model Training': model
}


def main():

    data = load_data()
    menu = st.sidebar.radio('Menu', list(pages.keys()))
    pages[menu](data)


if __name__ == '__main__':
    main()
