import streamlit as st
import os
import logging
import sys
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.llms import Replicate
from llama_index import ServiceContext, set_global_service_context
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from st_audiorec import st_audiorec
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from llama_index import ServiceContext
from llama_index import StorageContext, load_index_from_storage
from PIL import Image
import streamlit as st
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
import pyttsx3
import speech_recognition as sr
import requests
from io import StringIO
import googlemaps
import re
import pandas as pd
import folium
from streamlit_folium import st_folium, folium_static
import time


os.environ["REPLICATE_API_TOKEN"] = <ENTER REPLICATE_API_TOKEN>

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# The replicate endpoint
LLAMA_13B_V2_CHAT = "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"

log_meal_api = <ENTER LOG_MEAL_API>
logmeal_headers = {'Authorization': 'Bearer ' + log_meal_api}

CLIP_DROP_API = <ENTER CLIP_DROP_API>

# Set your Google Maps API key
GOOGLE_MAPS_API_KEY = <ENTER GOOGLE_MAPS_API_KEY>

# Initialize the Google Maps client
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

# inject custom system prompt into llama-2
def custom_completion_to_prompt(completion: str) -> str:
    return completion_to_prompt(
        completion,
        system_prompt=(
            "You are a Q&A assistant. Your goal is to answer questions as "
            "accurately as possible is the instructions and context provided."
        ),
    )


llm = Replicate(
    model=LLAMA_13B_V2_CHAT,
    temperature=0.01,
    # override max tokens since it's interpreted
    # as context window instead of max tokens
    context_window=4096,
    # override completion representation for llama 2
    completion_to_prompt=custom_completion_to_prompt,
    # if using llama 2 for data agents, also override the message representation
    messages_to_prompt=messages_to_prompt,
)

# set a global service context
ctx = ServiceContext.from_defaults(llm=llm)
set_global_service_context(ctx)

embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")
service_context = ServiceContext.from_defaults(embed_model=embed_model)

storage_context = StorageContext.from_defaults(persist_dir='Data_Indexes')
index1 = load_index_from_storage(storage_context, service_context=service_context)
query_engine = index1.as_query_engine()


# App title
logo = Image.open('RestoBot logo.png')
about = "Hungry and looking for the perfect dish? Our chatbot is ready to delight your taste buds. Ask me anything about restaurants, recipes and nutrition; and I'll provide you with mouthwatering answers.\nExamples:\n1. Suggest me a budget friendly italian restaurant.\n2. I have chicken thighs, what can I make?\n3. What is a simple recipe with 30gm of protein?"
st.set_page_config(page_title="RestoBot",page_icon=logo,layout="wide",menu_items={'About': about})


if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def generate_response(prompt_input):
    response = query_engine.query(prompt_input)
    return response

def speechtotext():
    r = sr.Recognizer()
    while(1):    
        try:
             
            # use the microphone as source for input.
            with sr.Microphone() as source2:
                 
                r.adjust_for_ambient_noise(source2, duration=0.2)
                 
                #listens for the user's input 
                audio2 = r.listen(source2)
                 
                # Using google to recognize audio
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()

                st.session_state.messages.append({"role": "user", "content": MyText})
                with st.chat_message("user"):
                    st.write(MyText)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = generate_response(MyText)
                        st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)

            break

                 
        except sr.RequestError as e:
            st.write("Could not request results; {0}".format(e))
             
        except sr.UnknownValueError:
            print("unknown error occurred")

def texttospeech():
    text = st.session_state.messages[-1]["content"]
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def extract_restaurant_names(text):
    print("entered extract")
    pattern = r'\d+\.\s*([A-Za-z\'\s]+)[,-]'
    matches = re.findall(pattern, text)
    restaurant_names = [match.strip() for match in matches]
    return restaurant_names

def display_restaurant_locations(restaurant_names):
    df = pd.DataFrame(columns=['name','lat','lng'])
    for name in restaurant_names:
        places_result = gmaps.places(name + " Los Angeles")
        if places_result['status'] == 'OK' and places_result['results']:
            location = places_result['results'][0]['geometry']['location']
            new_row = {'name': name, **location}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = df.rename(columns={'lng': 'lon'})
    print(df)
    return df

def foliummaps(df):
    map_center = [df['lat'].mean(), df['lon'].mean()]
    mymap = folium.Map(location=map_center, zoom_start=10)

    # Add markers and text labels to the map
    for index, row in df.iterrows():
        folium.Marker([row['lat'], row['lon']], popup=row['name']).add_to(mymap)
    # Display the map
    return(mymap)

def display_map(resta_list):
    coords = display_restaurant_locations(resta_list)
    mymap = foliummaps(coords)
    return mymap

def get_recipe_name(paragraph):
    recipe_name_pattern = re.compile(r'Recipe: (\b\w+\b\s+){4}')
    match = recipe_name_pattern.search(paragraph)
    recipe_name = match.group(1) if match else None
    return recipe_name

def embedimage(response):
    response = get_recipe_name(response)
    r = requests.post('https://clipdrop-api.co/text-to-image/v1',
      files = {
          'prompt': (None, 'response', 'text/plain')
      },
      headers = { 'x-api-key': CLIP_DROP_API}
    )
    if (r.ok):
      return r.content
    else:
      r.raise_for_status()

def recoimage():
    uploaded_image = st.session_state.uploaded_image1
    if uploaded_image is not None:
        imgread = uploaded_image.read()
        with open('upload_img.jpeg', 'wb') as f: 
            f.write(imgread)
        img = 'upload_img.jpeg'
        url = 'https://api.logmeal.es/v2/image/segmentation/complete'
        resp = requests.post(url,files={'image': open(img, 'rb')}, headers=logmeal_headers)
        result = resp.json()["segmentation_results"]
        result_list = [json_obj['recognition_results'] for json_obj in result]
        result_names = [name['name'] for recognition_results in result_list for name in recognition_results]
        result_names = str(result_names)
        st.session_state.messages.append({"role": "user", "content": "Image - "+result_names})
        #with st.chat_message("user"):
        #    st.write("Image - "+result_names)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(result_names+"For these words, give me a recipe and a restaurant where I can find it.")
                #st.write(response.response)
        message = {"role": "assistant", "content": response.response}
        st.session_state.messages.append(message)

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

words_to_find = ["image", "Image", "images","Images","Picture","Pictures","picture","pictures","pic","pics"]
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            present_words = [word for word in words_to_find if word in prompt]
            resta_list = extract_restaurant_names(response.response)
            print(resta_list)
            if present_words:
                st.write(response.response)
                image = embedimage(response.response)
                st.image(image)
                message = {"role": "assistant", "content": image}
                st.session_state.messages.append(message)
            elif resta_list:
                st.write(response.response)
                mymap = display_map(resta_list)
                folium_static(mymap)
            else:
                print("In else")
                st.write(response.response)
            
    message = {"role": "assistant", "content": response.response}
    st.session_state.messages.append(message)

if st.button("Talk :speaking_head_in_silhouette:"):
    speechtotext()

if st.button("Hear :loudspeaker:"):
    texttospeech()

st.button('or Write below :writing_hand:',disabled=True)

uploaded_file = st.file_uploader("Choose an image",on_change=recoimage,key='uploaded_image1')

with st.sidebar:
    logo = Image.open('RestoBot logo.png')
    st.image(logo)

    st.write('<div style="text-align: justify;">Hungry and looking for the perfect dish? Our chatbot is ready to delight your taste buds. Ask me anything about restaurants, recipes and nutrition; and I will provide you with mouthwatering answers.</div>',unsafe_allow_html=True)
    st.write('<br>',unsafe_allow_html=True)
    st.caption("Developed by Team RestoBot<br>Shardul Nazirkar<br>Nachiket Dunbray<br>Niharika Abhange",unsafe_allow_html=True)