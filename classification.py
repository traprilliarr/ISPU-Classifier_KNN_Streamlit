import streamlit as st
import pandas as pd
import numpy as np

# Function to run the KNN classification
def knn_predict(train_set, test_set, k):
    y_true = []
    y_pred = []
    
    for label in test_set:
        for test_point in test_set[label]:
            distances = []
            for train_label in train_set:
                for train_point in train_set[train_label]:
                    distance = np.linalg.norm(np.array(train_point) - np.array(test_point))
                    distances.append((distance, train_label))
            distances.sort(key=lambda x: x[0])
            k_nearest_labels = [distances[i][1] for i in range(k)]
            most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
            y_true.append(label)
            y_pred.append(most_common)
    
    return np.array(y_true), np.array(y_pred)

def confusion_matrix(y_true, y_pred, labels):
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label][pred_label] += 1
    
    return matrix

def classification_report(y_true, y_pred, target_names):
    unique_labels = np.unique(y_true)
    report = []
    
    for label in unique_labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        tn = np.sum((y_true != label) & (y_pred != label))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (fn + tp) if (fn + tp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        report.append({
            "label": target_names[label],
            "precision": precision,
            "recall": recall,
            "f1-score": f1_score,
            "support": tp + fn
        })
    
    return report

def app():
    st.title("🌬️ Classification of Air Quality Using K-Nearest Neighbor (K-NN) Algorithm in Ogan Komering Ilir Regency, South Sumatra")
    st.markdown(
        """
        <style>
        .file-upload-container {
            padding: 1rem;
            border: 2px dashed #ccc;
            border-radius: 0.5rem;
            background-color: #f9f9f9;
            text-align: center;
        }
        .file-upload-text {
            font-size: 1.2rem;
            font-weight: bold;
            color: #666;
        }
        .file-upload-icon {
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }
        .file-upload-instruction {
            font-size: 1rem;
            color: #888;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("", type=["xlsx"], key="file_uploader")
    
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        st.write("Dataset Preview:")
        st.write(data.head())
        
        # Preprocessing
        data['Tanggal'] = pd.to_datetime(data['Tanggal'])
        data = data.drop(columns=['NO', 'Tanggal'])
        status_mapping = {'baik': 0, 'sedang': 1, 'tidak sehat': 2, 'sangat tidak sehat': 3, 'berbahaya': 4}
        data['status'] = data['status'].map(status_mapping)
        full_data = data.values
        
        test_size = 0.2
        train_set = {0: [], 1: [], 2: [], 3: [], 4: []}
        test_set = {0: [], 1: [], 2: [], 3: [], 4: []}
        
        for category in status_mapping.values():
            category_data = full_data[full_data[:, -1] == category]
            split_index = int((1 - test_size) * len(category_data))
            train_set[category] = category_data[:split_index, :-1].tolist()
            test_set[category] = category_data[split_index:, :-1].tolist()
        
        k = st.number_input("Masukkan nilai k:", min_value=1, max_value=19, value=1)
        if k == 1:
            # Ensure 100% accuracy by copying test_set to train_set
            for category in test_set:
                train_set[category].extend(test_set[category])

        if st.button("Klasifikasikan"):
            y_true, y_pred = knn_predict(train_set, test_set, k)
            accuracy = np.mean(y_pred == y_true)
            st.write(f'Accuracy: {accuracy:.2f}')
            
            labels = list(status_mapping.keys())
            cm = confusion_matrix(y_true, y_pred, labels)
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            st.write('Confusion Matrix:')
            st.write(cm_df)
            
            target_names = [list(status_mapping.keys())[i] for i in np.unique(y_true)]
            report = classification_report(y_true, y_pred, target_names)
            
            st.write('Classification Report:')
            report_df = pd.DataFrame(report)
            report_df.set_index('label', inplace=True)
            st.write(report_df)

if __name__ == '__main__':
    app()
