import streamlit as st
#from streamlit_option_menu import option_menu
    
st.write("# Automated Crop Health Analysis System! 👋")


st.markdown(
    """
    Our platform utilizes cutting-edge machine learning technology to diagnose crop health issues swiftly and accurately. With a diverse range of over 50 diseases covered, you can trust our system to identify issues from Apple Scab to Tomato Late Blight.
    ### How Does it Works?
    Simply upload an image of your affected plant, and our AI-powered algorithm will analyze it in seconds, providing you with a diagnosis and recommended actions to take.
    ### Technology we have used
    - NumPy and Pandas: For data manipulation and preprocessing.
    - TensorFlow and Keras: To build and train deep learning models for image classification.
    - OpenCV (cv2): For image processing and manipulation tasks.
    - Matplotlib and Seaborn: For data visualization and result analysis.
    - Scikit-learn: For evaluating model performance and generating classification reports.
    - Streamlit: For creating an interactive and user-friendly web interface.
    """
)   


import streamlit as st
import pandas as pd

# Define the data
data = {
    'CLASS': ['Apple Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Bacterial leaf blight in rice leaf', 'Blight in corn Leaf', 'Blueberry healthy', 'Brown spot in rice leaf', 'Cercospora leaf spot', 'Cherry (including sour) Powdery mildew', 'Cherry (including_sour) healthy', 'Common Rust in corn Leaf', 'Corn (maize) healthy', 'Garlic', 'Grape Black rot', 'Grape Esca Black Measles', 'Grape Leaf blight Isariopsis Leaf Spot', 'Grape healthy', 'Gray Leaf Spot in corn Leaf', 'Leaf smut in rice leaf', 'Orange Haunglongbing Citrus greening', 'Peach healthy', 'Pepper bell Bacterial spot', 'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Raspberry healthy', 'Sogatella rice', 'Soybean healthy', 'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites Two spotted spider mite', 'Tomato Target Spot', 'Tomato Tomato mosaic virus', 'Tomato healthy', 'algal leaf in tea', 'anthracnose in tea', 'bird eye spot in tea', 'brown blight in tea', 'cabbage looper', 'corn crop', 'ginger', 'healthy tea leaf', 'lemon canker', 'onion', 'potassium deficiency in plant', 'potato crop', 'potato hollow heart', 'red leaf spot in tea', 'tomato canker'],
    'IMAGE COUNT': [5443, 5368, 2376, 3553, 108, 3094, 3245, 108, 170, 2273, 1847, 3526, 2511, 132, 10195, 11956, 9299, 915, 1550, 108, 17585, 778, 2689, 3194, 2700, 2700, 329, 802, 70, 10994, 2398, 985, 5743, 2700, 5154, 2570, 4782, 4525, 3791, 1007, 3437, 305, 270, 270, 305, 211, 281, 122, 200, 165, 54, 49, 108, 162, 386, 51]
}

# Create a DataFrame
df = pd.DataFrame(data)
st.write('### Disease class on Dataset')

# Create a bar chart
st.bar_chart(df.set_index('CLASS'))


# Define the diseases
diseases = ['Apple Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Bacterial leaf blight in rice leaf', 'Blight in corn Leaf', 'Brown spot in rice leaf', 'Cercospora leaf spot', 'Cherry (including sour) Powdery mildew', 'Cherry (including_sour) healthy', 'Common Rust in corn Leaf', 'Grape Black rot', 'Grape Esca Black Measles', 'Grape Leaf blight Isariopsis Leaf Spot', 'Gray Leaf Spot in corn Leaf', 'Leaf smut in rice leaf', 'Orange Haunglongbing Citrus greening', 'Pepper bell Bacterial spot', 'Potato Early blight', 'Potato Late blight', 'Sogatella rice', 'Strawberry Leaf scorch', 'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites Two spotted spider mite', 'Tomato Target Spot', 'Tomato Tomato mosaic virus', 'algal leaf in tea', 'anthracnose in tea', 'bird eye spot in tea', 'brown blight in tea', 'potassium deficiency in plant', 'potato hollow heart', 'red leaf spot in tea', 'tomato canker']


