import streamlit as st
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, List, Union


def encode_categorical(dataframe, columns):
    encoder = LabelEncoder()
    for col in columns:
        dataframe[col] = encoder.fit_transform(dataframe[col])
    return dataframe

def find_high_correlation_pairs(corr_matrix, threshold=0.5):
    corr_pairs = []
    columns = corr_matrix.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            value = corr_matrix.iloc[i, j]
            if abs(value) >= threshold:
                corr_pairs.append((columns[i], columns[j], value))
    return corr_pairs

def drop_highly_correlated_features(dataframe, corr_matrix, threshold=0.5):
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            if abs(corr_matrix.iloc[i, j]) > threshold:
                to_drop.add(col2)
    dataframe = dataframe.drop(columns=list(to_drop))
    return dataframe, list(to_drop)

def time_based_split(X, Y, split_year=2005):
    print(f"\nPerforming Time-based Split (train <= {split_year}, test > {split_year})...")
    train_mask = X['Year'] <= split_year
    test_mask = X['Year'] > split_year
    
    if not train_mask.any():
        raise ValueError(f"No data found for training (Year <= {split_year}). Adjust split_year or check data.")
    if not test_mask.any():
        raise ValueError(f"No data found for testing (Year > {split_year}). Adjust split_year or check data.")
    
    X_train = X[train_mask]
    y_train = Y[train_mask]
    X_test = X[test_mask]
    y_test = Y[test_mask]
    
    print(f"Train set size: {len(X_train)}, years: {X_train['Year'].min()}-{X_train['Year'].max()}")
    print(f"Test set size: {len(X_test)}, years: {X_test['Year'].min()}-{X_test['Year'].max()}")
    
    return X_train, X_test, y_train, y_test

def create_splits(X_train_time, y_train_time, X_test_time, y_test_time):
    splits = {}
    
    splits['seen'] = {
        'X_train': X_train_time.copy(),
        'y_train': y_train_time.copy(),
        'X_test': X_test_time.copy(),
        'y_test': y_test_time.copy()}
    
    unique_countries = sorted(X_train_time['Area'].unique())
    if len(unique_countries) < 2:
        raise ValueError("Not enough unique countries for unseen_countries split. Need at least 2.")
    num_train_countries = int(len(unique_countries) * 0.8)
    train_countries = unique_countries[:num_train_countries]
    test_countries = unique_countries[num_train_countries:]
    
    splits['unseen_countries'] = {
        'X_train': X_train_time[X_train_time['Area'].isin(train_countries)].copy(),
        'y_train': y_train_time[X_train_time['Area'].isin(train_countries)].copy(),
        'X_test': X_test_time[X_test_time['Area'].isin(test_countries)].copy(),
        'y_test': y_test_time[X_test_time['Area'].isin(test_countries)].copy()}
    
    unique_crops = sorted(X_train_time['Crop'].unique())
    if len(unique_crops) < 2:
        raise ValueError("Not enough unique crops for unseen_crops split. Need at least 2.")
    num_train_crops = int(len(unique_crops) * 0.7)
    train_crops = unique_crops[:num_train_crops]
    test_crops = unique_crops[num_train_crops:]
    
    splits['unseen_crops'] = {
        'X_train': X_train_time[X_train_time['Crop'].isin(train_crops)].copy(),
        'y_train': y_train_time[X_train_time['Crop'].isin(train_crops)].copy(),
        'X_test': X_test_time[X_test_time['Crop'].isin(test_crops)].copy(),
        'y_test': y_test_time[X_test_time['Crop'].isin(test_crops)].copy()}
    
    splits['unseen_both'] = {
        'X_train': X_train_time[X_train_time['Area'].isin(train_countries) &
                               X_train_time['Crop'].isin(train_crops)].copy(),
        'y_train': y_train_time[X_train_time['Area'].isin(train_countries) &
                               X_train_time['Crop'].isin(train_crops)].copy(),
        'X_test': X_test_time[X_test_time['Area'].isin(test_countries) &
                             X_test_time['Crop'].isin(test_crops)].copy(),
        'y_test': y_test_time[X_test_time['Area'].isin(test_countries) &
                             X_test_time['Crop'].isin(test_crops)].copy()}
    
    for split_name, data in splits.items():
        if not len(data['X_train']) or not len(data['X_test']):
            raise ValueError(f"Split '{split_name}' has empty train or test set. Check data distribution.")
    
    return splits

def scale_features_and_create_weights(splits, numerical_columns):
    scaler = MinMaxScaler()
    
    for split_name, data in splits.items():
        print(f"\nProcessing {split_name} split...")
        X_train_scaled = data['X_train'].copy()
        X_train_scaled[numerical_columns] = scaler.fit_transform(X_train_scaled[numerical_columns])
        X_test_scaled = data['X_test'].copy()
        X_test_scaled[numerical_columns] = scaler.transform(X_test_scaled[numerical_columns])
        X_train_temp = X_train_scaled.copy()
        X_train_temp['Country_Crop'] = X_train_temp['Area'].astype(str) + '_' + X_train_temp['Crop'].astype(str)
        group_freq = X_train_temp['Country_Crop'].value_counts()
        X_train_temp['SampleWeight'] = 1 / X_train_temp['Country_Crop'].map(group_freq)
        splits[split_name]['X_train'] = X_train_temp.drop(['Country_Crop', 'SampleWeight'], axis=1)
        splits[split_name]['X_test'] = X_test_scaled
        splits[split_name]['y_train'] = data['y_train']
        splits[split_name]['y_test'] = data['y_test']
        splits[split_name]['sample_weights'] = X_train_temp['SampleWeight'].values
        splits[split_name]['scaler'] = scaler
        print(f"Train size: {len(splits[split_name]['X_train'])}")
        print(f"Test size: {len(splits[split_name]['X_test'])}")
    
    return splits

def prepare_transfer_learning_data(X_test_unseen, y_test_unseen, test_size=0.7, random_state=42):
    print("Preparing data for transfer learning...")
    if len(X_test_unseen) == 0:
        raise ValueError("Unseen countries test set is empty. Cannot create transfer learning data.")
    X_adapt, X_test_unseen_final, y_adapt, y_test_unseen_final = train_test_split(
        X_test_unseen, y_test_unseen, test_size=test_size, random_state=random_state)
    if len(X_adapt) == 0 or len(X_test_unseen_final) == 0:
        raise ValueError("Transfer learning split resulted in empty adaptation or test set.")
    print(f"Adaptation set size: {len(X_adapt)}")
    print(f"Final test set size: {len(X_test_unseen_final)}")
    return X_adapt, X_test_unseen_final, y_adapt, y_test_unseen_final

def process_data(X, Y, numerical_columns):
    try:
        X_train_time, X_test_time, y_train_time, y_test_time = time_based_split(X, Y)
        splits = create_splits(X_train_time, y_train_time, X_test_time, y_test_time)
        splits = scale_features_and_create_weights(splits, numerical_columns)
        X_adapt, X_test_unseen_final, y_adapt, y_test_unseen_final = prepare_transfer_learning_data(
            splits['unseen_countries']['X_test'], splits['unseen_countries']['y_test'])
        splits['transfer_learning'] = {
            'X_adapt': X_adapt,
            'y_adapt': y_adapt,
            'X_test_unseen_final': X_test_unseen_final,
            'y_test_unseen_final': y_test_unseen_final}
        return splits
    except Exception as e:
        raise ValueError(f"Error in process_data: {e}")