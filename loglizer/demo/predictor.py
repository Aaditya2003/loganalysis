from joblib import load
from loglizer.loglizer import dataloader, preprocessing
import pandas as pd
import numpy as np

# Load your model and feature extractor from the files
def predictor(struct_log = r'C:\Users\DeLL\PycharmProjects\py 11.6\logparser\demo\AEL_result\HDFS.log_structured.csv',label_file = r'C:\Users\DeLL\PycharmProjects\py 11.6\loglizer\data\HDFS\anomaly_label.csv'):
    model = load(r'C:\Users\DeLL\PycharmProjects\py 11.6\loglizer\data\savedmodels\model_2k.joblib')

    feature_extractor = load(r'C:\Users\DeLL\PycharmProjects\py 11.6\loglizer\data\savedmodels\feature_extractor_2k.joblib')

    # Load and parse the Source Logs log file using the dataloader module
    # struct_log = r'C:\Users\DeLL\PycharmProjects\pythonProject1\logparser\demo\AEL_result\HDFS.log_structured.csv' # The structured log file
    # struct_log = r'C:\Users\DeLL\PycharmProjects\py 11.6\loglizer\data\HDFS\HDFS_100k.log_structured.csv'
    #label_file = '../data/HDFS/anomaly_label.csv'  # The anomaly label file

    (x_train, y_train, train_bid), (x_test, y_test, test_bid), x_train_id, x_test_id = dataloader.load_HDFS(struct_log,
                                                                                                            label_file=label_file,
                                                                                                            window='session',
                                                                                                            train_ratio=0.99,
                                                                                                            split_type='uniform',
                                                                                                            save_csv=True)

    # Extract features from the Source Logs log file using the feature extractor
    x_test = feature_extractor.transform(x_test)

    # Apply the model on the Source Logs features and get the predictions
    y_pred = model.predict(x_test)

    print(y_pred)

    x_test_id['Prediction'] = y_pred

    # x_test_id.to_csv('x_test_id_predictions.csv', index=True)
    df_anomaly = x_test_id[x_test_id['Prediction'] == 1].copy()
    #
    print("Number of rows in df_anomaly:", df_anomaly.shape[0])

    # df_anomaly = x_test_id[x_test_id['Prediction'] == 1].copy()
    #
    # # Set the index of df_anomaly to be the BlockId
    df_anomaly.set_index('block_id', inplace=True)

    df_anomaly.to_csv(r'C:\Users\DeLL\PycharmProjects\py 11.6\loglizer\data\otherdata\x_test_id_predictionshd.csv',
                      index=True)

    data = pd.read_csv(struct_log)
    block_ids = pd.read_csv(r'C:\Users\DeLL\PycharmProjects\py 11.6\loglizer\data\otherdata\x_test_id_predictionshd.csv')

    # Extract the block IDs from the block_ids DataFrame as a list
    block_id_list = block_ids['block_id'].tolist()

    # Filter data based on block IDs
    filtered_data = data[data['Content'].str.contains('|'.join(block_id_list))]
    filtered_data.to_csv(r"C:\Users\DeLL\PycharmProjects\py 11.6\privateGPT\input\anomal data\filtered_data.csv",
                         index=False)

    # Display the filtered data
    # Assuming filtered_data is a list of dictionaries where each dictionary represents a row
    # Columns to be selected
    selected_columns = ["LineId", "Date", "Time", "Pid", "Level", "Component", "EventId"]

    # Rename columns for the output
    column_rename_mapping = {
        "LineId": "line no.",
        "Date": "Date",
        "Time": "Time",
        "Pid": "user_id",
        "EventId": "error code"
    }

    # Create an explicit copy of the filtered_data DataFrame
    selected_data = filtered_data.copy()

    # Select desired columns
    selected_data = selected_data[selected_columns]

    # Rename columns
    selected_data.rename(columns=column_rename_mapping, inplace=True)

    # Loop through each row and print as a formatted string
    for index, row in selected_data.iterrows():
        formatted_output = " ".join(f"{column}: {value}" for column, value in row.items())
        print(formatted_output)
    # Evaluate the model performance using the evaluate method
    #precision, recall, f1 = model.evaluate(x_test, y_test)
#predictor()