import streamlit as st
from PIL import Image
import requests
import json
from io import BytesIO
import base64
import os

st.set_page_config(page_title="Your Twitter Story Creatror App", page_icon='📱')
st.header("Turn your Photos into Amazing Twitter Stories")

def tweetgenerated(userprompt):  
  import requests
  import json
  import requests
  import json
  #prompt = 'a visionary Singapore city'
  prompt = userprompt
  url = "https://apps.beam.cloud/mandd"
  payload = {'myprompt': prompt}
  headers = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "Authorization": "Basic xxxxxxxxxxxxxxxxxxxxxx=",
    "Connection": "keep-alive",
    "Content-Type": "application/json"
  }

  response = requests.request("POST", url, 
    headers=headers,
    data=json.dumps(payload)
  )
  res = response.content
  #print(res)
  import textwrap
  json_object = json.loads(res)
  blogpost  = json_object["blogpost"]
  print("Suggested tweet content:\n")
  print(textwrap.fill(blogpost,60))
  print("-------")
  return blogpost


def callAPI(i,p,stoi,gs):
  """
  i = image path, string
  p = prompt, string
  stoi = strenght of the original image, float
  gs = guidance scale of the prompt, float
  return PILLOW IMAGE
  """
  # SETUP INIT_IMAGE AND PROMPT
  prompt = p # "singapore landscape in Fantasy Art style"
  # encoded_image is the stream for  the API POST
  input_image = i # '/content/singapore.jpeg'
  with open(input_image, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
  # API CALL
  url = "https://apps.beam.cloud/h7qs7"
  payload = {'origimage': encoded_image,
            'myprompt': prompt,
            'stoi' : stoi,
            'gs' : gs}
  headers = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "Authorization": "Basic xxxxxxxxxxxxxxxxxxxxxxxx=",
    "Connection": "keep-alive",
    "Content-Type": "application/json"
  }

  response = requests.request("POST", url,
    headers=headers,
    data=json.dumps(payload))
  # GET THE RESPONSE
  # return the byte stream into a string
  res = response.content.decode('utf-8')
  # convert the string into a json dict
  json_object = json.loads(res)
  rec_image = json_object["gen_image"]
  png_recovered = base64.b64decode(rec_image)
  return png_recovered

def main():
    #st.image('banner.png', use_column_width=True)
    st.markdown("1. Select a photo from your pc\n 2. Give AI  a topic\n3. AI generate an amazing picture\n4. AI generate the twitter post and hashtags\n5. Pick the best of AI generation")


    uploaded_file = st.file_uploader('Upload an image file')
    if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, width=500, caption='Your original image')
            image.save("input.png")

    st.write("Insert the twitter topic:")
    tweet = st.text_area('Insert here your twitter topic. Example: a visionary Singapore city', "", height = 50, key = 'tweet')
    if st.session_state.tweet == "":
        st.warning('You need to have a topic...', icon="⚠️")
    else:
       btnTweet = st.button("Generate Tweet and Tags", type='primary')  
       if btnTweet:
            with st.spinner('Generating Tweet...'):
                b = tweetgenerated(st.session_state.tweet)
            st.markdown("## Your tweet content")
            st.write(b)
            with st.spinner('Generating your AI image...'):
                theprompt = f"landscape of {st.session_state.tweet}"
                imgprompt = f"{theprompt} in Fantasy Art style."
                imgpath = os.path.abspath(os.getcwd())+'/input.png'
                AIimg = callAPI(imgpath,imgprompt,0.82,16)
                genimage = Image.open(BytesIO(AIimg)).convert('RGB')
                genimage.save("ouput.png")
                st.image(genimage, width=500, caption='Your generated image')
                            

if __name__ == '__main__':
   main()
