import pickle
import streamlit as st
import pandas as pd

# Load the RandomForestClassifier model
try:
    rf_classifier = pickle.load(open('rf_classifier.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file not found. Please make sure the pickle file 'rf_classifier.pkl' exists.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model file: {e}")
    st.stop()

# Streamlit app
def main():
    st.title("Crop Prediction ")
    st.write("Enter environmental factors to predict suitable crop")

    # User inputs
    humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=0)
    moisture = st.slider("Moisture (%)", min_value=0, max_value=100, value=0)
    light_intensity = st.slider("Light Intensity", min_value=0, max_value=1000, value=0)

    if st.button("Predict"):
        # Create input DataFrame
        input_data = {
            'Humidity': [humidity],
            'Moisture': [moisture],
            'LightIntensity': [light_intensity]
        }
        input_df = pd.DataFrame(input_data)

        # Predict suitable crop
        predicted_crop = rf_classifier.predict(input_df)
        st.write("Predicted Crop:", predicted_crop[0])

if __name__ == "__main__":
    main()
