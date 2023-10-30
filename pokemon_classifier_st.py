import streamlit as st
import random
import pandas as pd
import numpy as np
import os
import time
import requests
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title='Pokedex',
    page_icon='https://i.pinimg.com/736x/51/07/75/510775920002ed607ff0a5582932214a.jpg'
)

st.image('cover_project_st2.jpeg',use_column_width=True)
st.title('Pokedex')

class_names = ['Charmander', 'Mewtwo', 'Pikachu', 'Slowpoke', 'Squirtle']

st.cache_data()
def predict_class(image, model):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (256, 256))
    image = np.expand_dims(image / 255, 0)
    prediction = model.predict(image)
    return prediction

def path_to_image_html(path):
    return '<img src="' + path + '" width="60" >'

st.cache_data()
def print_data(pokelist):
    url = 'https://pokeapi.co/api/v2/pokemon/'
    df = pd.DataFrame(data=np.zeros((5, 4)),
                      columns=['Name', 'Type', 'Description', 'Image'],
                      index=range(1, 6)
                      )
    sprites_path = 'https://github.com/gaiatravaglini/Pokemon_image_classifier/blob/master/sprites/'
    sprites = []
    for i, poke in enumerate(pokelist):
        response = requests.get(url + poke.lower())
        if response.status_code != 200:
            df.iloc[i, 0] = poke
            df.iloc[i, 1] = 'Error fetching data from API'
            df.iloc[i, 2] = 'Error fetching data from API'
            df.iloc[i, 3] = 'Error fetching data from API'
            sprites.append(sprites_path+'0.png?raw=true')
        else:
            jresponse = response.json()
            type = jresponse['types'][0]['type']['name']
            id = jresponse['id']
            species_url = jresponse['species']['url']
            species_response = requests.get(species_url)
            species_response = species_response.json()
            description = ''
            for d in species_response['flavor_text_entries']:
                if d['language']['name'] == 'en':
                    description = d['flavor_text']
                    break
            df.iloc[i, 0] = poke.capitalize()
            df.iloc[i, 1] = type.capitalize()
            description = description.replace('\n', ' ')
            df.iloc[i, 2] = description
            sprites.append(sprites_path + str(id) + '.png?raw=true')

    df['Image'] = sprites
    return df, sprites

pokelist = ['Charmander', 'Mewtwo', 'Pikachu', 'Slowpoke', 'Squirtle']
df, sprites = print_data(pokelist)

file= st.file_uploader('Choose a file',type=['jpeg', 'jpg', 'bmp', 'png'])
if file is None:
    st.write("Please, upload a file")
else:
    test_image = Image.open(file)
    option = st.selectbox(
    "Which model would you like to use?",
    ("VGG16", "MobileNet"),
    index=None,
    placeholder="Select model...",
    )

    if option == "VGG16":
        st.write("You selected the VGG16 model.")
        vgg16_model = load_model('pokemon__classifier_vgg16.h5')
        st.write('Uploading an image:')
        st.image(test_image, caption="Pokemon Image", width=400)
        pred = predict_class(np.asarray(test_image), vgg16_model)
        result = class_names[np.argmax(pred)]
        formatted_pred = [(label, prob) for label, prob in zip(class_names, pred[0])]
        with st.spinner('Model working....'):
            time.sleep(3)
        with st.success('Classified'):
            output = 'The pokemon is most likely a ' + result
            time.sleep(1)
        st.markdown(output)
        if st.checkbox('Class Labels and Probabilities:'):
            for label, prob in formatted_pred:
                st.write(f"{label}: {prob:.6f}")
        if st.checkbox("Pokemon Information:"):
            matching_pokemon = df[df['Name'].str.lower() == result.lower()]
            st.write(matching_pokemon.to_html(escape=False, formatters={'Image': path_to_image_html}), unsafe_allow_html=True)
        
        
    elif option == "MobileNet":
        st.write("You selected the MobileNet model.")
        mob_model = load_model('pokemon__classifier_Mobilenet.h5')
        st.write('Uploading an image:')
        st.image(test_image, caption="Pokemon Image", width=400)
        pred = predict_class(np.asarray(test_image), mob_model)
        formatted_pred = [(label, prob) for label, prob in zip(class_names, pred[0])]
        result = class_names[np.argmax(pred)]
        with st.spinner('Model working....'):
            time.sleep(3)
        with st.success('Classified'):
            output = 'The pokemon is most likely a ' + result
            time.sleep(1)
        st.markdown(output)
        if st.checkbox('Class Labels and Probabilities:'):
            for label, prob in formatted_pred:
                st.write(f"{label}: {prob:.6f}")
        if st.checkbox("Pokemon Information:"):
            matching_pokemon = df[df['Name'].str.lower() == result.lower()]
            st.write(matching_pokemon.to_html(escape=False, formatters={'Image': path_to_image_html}), unsafe_allow_html=True)
        