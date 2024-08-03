import pandas as pd
from loglizer.loglizer.models import DecisionTree
from loglizer.loglizer import dataloader, preprocessing
from joblib import dump, load


# Load and preprocess the dataset
def trainer(struct_log = r'C:\Users\DeLL\PycharmProjects\py 11.6\loglizer\data\HDFS\HDFS_100k.log_structured.csv',
    label_file = r'C:\Users\DeLL\PycharmProjects\py 11.6\loglizer\data\HDFS\anomaly_label.csv'):
    # struct_log = r'C:\Users\DeLL\PycharmProjects\pythonProject1\logparser\demo\AEL_result\HDFS.log_structured.csv' # The structured log file
    # struct_log = r'C:\Users\DeLL\PycharmProjects\py 11.6\loglizer\data\HDFS\HDFS_100k.log_structured.csv'
    # label_file = r'C:\Users\DeLL\PycharmProjects\py 11.6\loglizer\data\HDFS\anomaly_label.csv'  # The anomaly label file
    (x_train, y_train, train_bid), (x_test, y_test, test_bid), x_train_id, x_test_id = dataloader.load_HDFS(
        struct_log,
        label_file=label_file,
        window='session',
        train_ratio=1,
        split_type='uniform')

    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')

    # x_test = feature_extractor.transform(x_test)

    model = DecisionTree()
    model.fit(x_train, y_train)
    dump(model, r'C:\Users\DeLL\PycharmProjects\py 11.6\loglizer\data\savedmodels\model_2k.joblib')
    dump(feature_extractor,
         r'C:\Users\DeLL\PycharmProjects\py 11.6\loglizer\data\savedmodels\feature_extractor_2k.joblib')

    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)

    # print('Test validation:')
    # print(x_test)
    #
    # # Finally make predictions and alter on anomaly cases
    # y_test = model.predict(x_test)
    # print(y_test)
    # precision, recall, f1 = model.evaluate(x_test, y_test)
    # print('Test validation:')

    # precision, recall, f1 = model.evaluate(x_test, y_test)
    # y_test = model.predict(x_test)
    # precision, recall, f1 = model.evaluate(x_test, y_test)
    #
    # # Load the original log data
    # original_df = pd.read_csv(struct_log)
    #
    # # Get the indices of the samples used in x_test
    # filtered_original_df = original_df.iloc[x_test.index]
    # test_indices = x_test.index
    #
    # # Filter the original log data to get rows corresponding to x_test samples
    # filtered_original_df = original_df.iloc[test_indices]
    #
    # # Add the corresponding y_test and y_pred_test columns
    # # filtered_original_df['True_Label'] = y_test
    # filtered_original_df['Predicted_Label'] = y_test
    #
    # # Save the result to a CSV file
    # filtered_original_df.to_csv('result.csv', index=False)
    #
    # print('Results saved to result.csv')
trainer()
