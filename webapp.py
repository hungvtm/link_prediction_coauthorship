import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import get_network_map

# Load the data
df = pd.read_csv('data/Author_Institute_Lat_Long_Topic.csv', index_col=0)
author_name = df['Name']
author_name1 = author_name.copy().to_list()


# =================================================================================================


# Home page
def home():
    # app attributes
    st.title('Home')
    options = st.selectbox('Select Author', author_name1)
    st.write('You selected:', options)

    # get author id 
    author_id = df[df['Name'] == options]['Author'].values.tolist()[0]

    # process about 50 seconds
    with st.spinner('Processing...'):
        directory = get_network_map.predict(author_id)
 
    # show html file
    components.html(open(directory, 'r').read(), height=800, width=800)

    # show map

    # show word cloud


# =================================================================================================
    

# About page
def about():
    st.title('Add a new author')
    st.write('Welcome to the Add page')


# Web has two pages
PAGES = {
    "Home": home,
    "About": about
}

def main():
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page()

    


if __name__ == "__main__":
    main()

