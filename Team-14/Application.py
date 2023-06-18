# # import streamlit as st
# # from PIL import Image
# # import requests
# # from io import BytesIO
# # import cv2
# # import numpy as np
# # import tempfile

# # from roboflow import Roboflow

# # from roboflow import Roboflow
# # rf = Roboflow(api_key="dO5XvoTKKqU8ynw36Fbj")
# # project = rf.workspace().project("anamoly-detection")
# # model = project.version(4).model

# # st.set_page_config(page_title="Anamoly Detection with YOLOv5(Fights)", page_icon="🔍")

# # st.sidebar.title("About")
# # st.sidebar.info("This app is a demonstration of the Amrita School of Engineering ID Card Detector. The model is trained using a custom YOLOv5 model.")

# # st.title("Anamoly Detection with YOLOv5(Fights)")

# # # Use the file uploader for both images and videos
# # uploaded_file = st.file_uploader("Choose an image or video file...", type=['jpeg','jpg', 'mp4'])

# # if uploaded_file is not None:

# #     file_bytes = uploaded_file.read()

# #     if uploaded_file.type == 'image/jpeg':
# #         image = Image.open(BytesIO(file_bytes))
# #         st.image(image, caption='Uploaded Image.', use_column_width=True)
# #         st.write("")

# #         st.write("Detecting...")
# #         with open("temp.jpg", "wb") as f:
# #             f.write(file_bytes)

# #         with st.spinner('Processing...'):
# #             results = model.predict("temp.jpg", confidence=28, overlap=30).save("prediction.jpg")

# #         st.success("Detection complete!")
# #         st.image('prediction.jpg', caption='Processed Image.', use_column_width=True)

# #     elif uploaded_file.type == 'video/mp4':
# #         video_file = tempfile.NamedTemporaryFile(delete=False) 
# #         video_file.write(file_bytes)

# #         vidcap = cv2.VideoCapture(video_file.name)

# #         stframe = st.empty()

# #         # skip frames
# #         skip_frames = 4  # example: process every 20th frame (change this to a number that suits your needs)
# #         frame_count = 0
        
# #         success, image = vidcap.read()
# #         while success:
# #             # Save the image to a temporary file
# #             if frame_count % skip_frames == 0:
# #                 cv2.imwrite("frame.jpg", image)

# #                 # Call the predict method on your model
# #                 results = model.predict("frame.jpg", confidence=28, overlap=30).save("prediction.jpg")

# #                 image = cv2.imread('prediction.jpg')
# #                 stframe.image(image, caption='Processed frame.', use_column_width=True)
            
# #             success, image = vidcap.read()
# #             frame_count += 1

# #         st.success("Detection complete!")

# # else:
# #     st.error("Please upload a file.")

# # st.markdown("---")

# import streamlit as st
# from PIL import Image
# import requests
# from io import BytesIO
# import cv2
# import numpy as np
# import tempfile
# from roboflow import Roboflow

# # Setup
# rf = Roboflow(api_key="dO5XvoTKKqU8ynw36Fbj")
# project = rf.workspace().project("anamoly-detection")
# model = project.version(4).model

# st.set_page_config(page_title="Anamoly Detection with YOLOv5(Fights)", page_icon="🔍")

# # Sidebar
# st.sidebar.title("About")
# st.sidebar.info("This app is a demonstration of the Amrita School of Engineering ID Card Detector. The model is trained using a custom YOLOv5 model.")
# st.sidebar.title("Settings")
# confidence_level = st.sidebar.slider("Confidence level for detection", min_value=0.0, max_value=1.0, value=0.5)

# # Main
# st.title("Anamoly Detection with YOLOv5(Fights)")

# # Use the file uploader for both images and videos
# uploaded_file = st.file_uploader("Choose an image or video file...", type=['jpeg','jpg', 'mp4'])

# if uploaded_file is not None:
#     file_bytes = uploaded_file.read()

#     if uploaded_file.type == 'image/jpeg':
#         image = Image.open(BytesIO(file_bytes))
#         st.image(image, caption='Uploaded Image.', use_column_width=True)

#         st.write("Detecting...")
#         with open("temp.jpg", "wb") as f:
#             f.write(file_bytes)

#         with st.spinner('Processing...'):
#             results = model.predict("temp.jpg", confidence=confidence_level, overlap=30).save("prediction.jpg")

#         st.success("Detection complete!")
#         st.image('prediction.jpg', caption='Processed Image.', use_column_width=True)

#     elif uploaded_file.type == 'video/mp4':
#         video_file = tempfile.NamedTemporaryFile(delete=False) 
#         video_file.write(file_bytes)

#         vidcap = cv2.VideoCapture(video_file.name)

#         stframe = st.empty()

#         # skip frames
#         skip_frames = 4  # example: process every 20th frame (change this to a number that suits your needs)
#         frame_count = 0
        
#         success, image = vidcap.read()
#         while success:
#             # Save the image to a temporary file
#             if frame_count % skip_frames == 0:
#                 cv2.imwrite("frame.jpg", image)

#                 # Call the predict method on your model
#                 results = model.predict("frame.jpg", confidence=confidence_level, overlap=30).save("prediction.jpg")

#                 image = cv2.imread('prediction.jpg')
#                 stframe.image(image, caption='Processed frame.', use_column_width=True)
            
#             success, image = vidcap.read()
#             frame_count += 1

#         st.success("Detection complete!")

# else:
#     st.error("Please upload a file.")

# st.markdown("---")

# # About the Project
# st.markdown("""
# ## About the Project
# This app is a demonstration of Anamoly Detection with YOLOv5(Fights) developed by Amrita School of Engineering.

# ## The Team
# - Vimal, Data Scientist.
# - Likhith, ML Engineer.
# - Gurunadh, Front-end Developer.
# """)
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import cv2
import numpy as np
import tempfile
from roboflow import Roboflow

# Setup
rf = Roboflow(api_key="dO5XvoTKKqU8ynw36Fbj")
project = rf.workspace().project("anamoly-detection")
model = project.version(4).model
st.set_page_config(page_title="Anamoly Detection with YOLOv5(Fights)", page_icon="🔍")
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}
.big-font {
    font-size:30px !important;
    color: #5a9;
}
.red-text {
    color:#ff0000 !important;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("About")
st.sidebar.info("Fight detection . The model is trained using a custom YOLOv5 model.")
st.sidebar.title("Settings")
confidence_level = st.sidebar.slider("Confidence level for detection", min_value=0.0, max_value=1.0, value=0.5)

# Main
st.markdown("""
<p class="big-font">Anamoly Detection with YOLOv5(Fights)</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image or video file...", type=['jpeg','jpg', 'mp4'])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    if uploaded_file.type == 'image/jpeg':
        image = Image.open(BytesIO(file_bytes))
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        st.write("Detecting...")
        with open("temp.jpg", "wb") as f:
            f.write(file_bytes)

        with st.spinner('Processing...'):
            results = model.predict("temp.jpg", confidence=confidence_level, overlap=30).save("prediction.jpg")

        st.success("Detection complete!")
        st.image('prediction.jpg', caption='Processed Image.', use_column_width=True)

    elif uploaded_file.type == 'video/mp4':
        video_file = tempfile.NamedTemporaryFile(delete=False) 
        video_file.write(file_bytes)

        vidcap = cv2.VideoCapture(video_file.name)

        stframe = st.empty()

        skip_frames = 4  
        frame_count = 0
        
        success, image = vidcap.read()
        while success:
            if frame_count % skip_frames == 0:
                cv2.imwrite("frame.jpg", image)

                results = model.predict("frame.jpg", confidence=confidence_level, overlap=30).save("prediction.jpg")

                image = cv2.imread('prediction.jpg')
                stframe.image(image, caption='Processed frame.', use_column_width=True)
            
            success, image = vidcap.read()
            frame_count += 1

        st.success("Detection complete!")

else:
    st.error("Please upload a file.")

st.markdown("---")

st.markdown("""
## About the Project
This app is a demonstration of Anamoly Detection with YOLOv5(Fights) developed by Amrita School of Engineering.

## The Team
- Vimal, 
- Likhith,
- Gurunadh,
""", unsafe_allow_html=True)
