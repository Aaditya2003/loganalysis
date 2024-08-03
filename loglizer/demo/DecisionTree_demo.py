#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd
sys.path.append('../')
from loglizer.loglizer.models import DecisionTree
from loglizer.loglizer import dataloader, preprocessing

struct_log = r'C:\Users\DeLL\PycharmProjects\loglizer\loglizer\data\HDFS\HDFS_100k.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log,
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=0.5,
                                                                split_type='uniform')

    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    results_df = pd.DataFrame({'log_entry': x_test})

    # Save the DataFrame to a CSV file
    results_df.to_csv('test_results.csv', index=False)

    x_test = feature_extractor.transform(x_test)

    model = DecisionTree()
    model.fit(x_train, y_train)

    # print('Train validation:')
    # precision, recall, f1 = model.evaluate(x_train, y_train)

    print('Test validation:')
    # print(x_test)
    #
    # # Finally make predictions and alter on anomaly cases
    # y_test = model.predict(x_test)
    # print(y_test)
    # precision, recall, f1 = model.evaluate(x_test, y_test)
    print('Test validation:')

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



