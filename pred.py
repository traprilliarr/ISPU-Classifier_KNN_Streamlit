import streamlit as st
import pandas as pd
import numpy as np

def knn_predict(train_set, test_point, k):
    distances = []
    for label in train_set:
        for train_point in train_set[label]:
            distance = np.linalg.norm(np.array(train_point) - np.array(test_point))
            distances.append((distance, label))
    
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [distances[i][1] for i in range(k)]
    predicted_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
    
    return predicted_label

def app():
    st.title("ISPU Prediction")

    st.write('Enter your data for prediction:')
    pm10 = st.number_input('pm10')
    pm25 = st.number_input('pm25')
    so2 = st.number_input('so2')
    co = st.number_input('co')
    o3 = st.number_input('o3')
    no2 = st.number_input('no2')
    hc = st.number_input('hc')

    test_point = [pm10, pm25, so2, co, o3, no2, hc]

    train_set = {
        0: [[71.71, 84.65, 11.47, 5.73, 8.48, 9.91, 8.60], [71.57, 84.62, 11.36, 5.74, 8.46, 9.91, 8.73]],  
        1: [[73.21, 85.15, 11.29, 5.76, 8.38, 9.94, 9.10], [73.44, 85.62, 11.29, 5.76, 8.36, 10.09, 9.23]],
        2: [[39.50, 38.71, 13.94, 3.20, 8.75, 11.25, 6.67], [39.00, 32.26, 13.94, 2.31, 9.58, 11.25, 6.67]],
        3: [[27.50, 25.81, 9.62, 2.61, 8.75, 10.00, 5.56], [27.00, 26.00, 9.50, 2.60, 8.70, 10.10, 5.60]],
        4: [[46.00, 52.84, 22.60, 5.64, 7.92, 6.88, 6.67], [45.50, 52.00, 22.50, 5.60, 7.90, 6.80, 6.60]]
    }

    k = 3

    if st.button('Predict'):
        predicted_label = knn_predict(train_set, test_point, k)
        status_mapping = {0: 'baik', 1: 'sedang', 2: 'tidak sehat', 3: 'sangat tidak sehat', 4: 'berbahaya'}
        st.write(f'Predicted status: {status_mapping[predicted_label]}')

if __name__ == '__main__':
    app()
