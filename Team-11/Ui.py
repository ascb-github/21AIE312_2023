import streamlit as st
import cv2
import torch
import numpy as np
import base64

def main():

    st.set_page_config(layout="wide")

    # CSS for setting the background image
    main_bg = "/Users/vishalraghav/Downloads/fotor-ai-20230608145532.png"
    main_bg_ext = "png"

    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Pothole Detection System")

    # Load the model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/vishalraghav/Downloads/best.pt')

    file_up = st.sidebar.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if file_up is not None:

        for i, file in enumerate(file_up):

            image = cv2.imdecode(np.fromstring(file.read(), np.uint8), 1)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display the uploaded image on the left-hand side
            col1, col2 = st.columns(2)

            with col1:

                st.image(image, caption=f'Uploaded Image {i+1}', use_column_width=True)

            if st.button(f'Predict {i+1}', key=f'predict_button_{i}'):

                # Inference
                results = model(image)

                # Display the predicted image and details on the right-hand side
                with col2:

                    # Draw bounding boxes on the predicted image
                    for j, x in enumerate(results.xyxy[0]):

                        x1, y1, x2, y2, conf, cls = x
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        cv2.putText(image, f'{results.names[int(cls)]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    st.image(image, caption=f'Processed Image {i+1}', use_column_width=True)

                    # Display the box coordinates, area, and estimated distance
                    for j, x in enumerate(results.xyxy[0]):

                        x1, y1, x2, y2, conf, cls = x
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                        area = abs((x2 - x1) * (y2 - y1))  # Area in pixels
                        distance = image.shape[0] - y2  # Distance from the bottom of the image
                        # Assuming 1 pixel = 0.01 meter (you need to adjust this based on your actual scale)
                        scale = 0.01
                        area_meters = area * (scale ** 2)  # Area in meters
                        distance_meters = distance * scale  # Distance in meters

                        st.write(f"Detection {j+1}:")
                        st.write(f"- Class: {results.names[int(cls)]}")
                        st.write(f"- Confidence: {conf:.2f}")
                        st.write(f"- Bounding Box Coordinates: (x1, y1) = ({x1}, {y1}), (x2, y2) = ({x2}, {y2})")
                        st.write(f"- Area: {area_meters:.2f} square meters")
                        st.write(f"- Estimated Distance: {distance_meters:.2f} meters")

                        st.markdown("---")

if __name__ == "__main__":
    main()
    
