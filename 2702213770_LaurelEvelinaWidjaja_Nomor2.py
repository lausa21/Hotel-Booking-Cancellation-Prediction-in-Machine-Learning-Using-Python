# # 2702213770 - Laurel Evelina Widjaja - Nomor 2
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import statistics as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import joblib

class handleData: 
    def __init__(self, file_path):
        self.file_path = file_path
        self.data, self.input_df, self.output_df = [None] * 3

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        print(self.data.info())
        print(self.data['booking_status'].value_counts())
        
    def create_input_output(self, target_column):
        self.input_df = self.data.drop(target_column, axis=1)
        self.output_df = self.data[target_column]

class handleModel: 
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5

    def del_id(self, id_column):
        self.input_data = self.input_data.drop(id_column, axis=1)

    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
        self.input_data, self.output_data, test_size=test_size, random_state=random_state)

    def cek_missing_values(self):
        print("Missing value di x_train")
        print(self.x_train.isna().sum())
        print("Missing value di x_test")
        print(self.x_test.isna().sum())

    def checkOutlier(self, col):
        boxplot = self.x_train.boxplot(column=[col]) 
        plt.show()

    def impute_mode(self, column):
        return st.mode(self.x_train[column])
    
    def impute_median(self, column):
        return self.x_train[column].median()

    def impute_missing_values(self, column, imputation):
        self.x_train[column] = self.x_train[column].fillna(imputation)
        self.x_test[column] = self.x_test[column].fillna(imputation)

    def drop_duplicate(self, target_name):
        self.y_train = pd.Series(self.y_train, name=target_name)
        self.y_test = pd.Series(self.y_test, name=target_name)

        train_combined = pd.concat([self.x_train, self.y_train], axis=1)
        test_combined = pd.concat([self.x_test, self.y_test], axis=1)

        train_combined = train_combined.drop_duplicates().reset_index(drop=True)
        test_combined = test_combined.drop_duplicates().reset_index(drop=True)

        self.x_train = train_combined.drop(columns=target_name)
        self.y_train = train_combined[target_name]

        self.x_test = test_combined.drop(columns=target_name)
        self.y_test = test_combined[target_name]

    def label_encoding(self, column):
        self.label_enc = LabelEncoder()
        self.x_train[column] = self.label_enc.fit_transform(self.x_train[column])
        self.x_test[column] = self.label_enc.transform(self.x_test[column])

    def target_encoding(self):
        self.target_enc = LabelEncoder()
        self.y_train = self.target_enc.fit_transform(self.y_train)

    def oneHot_encoding(self):
        self.mst_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        mst_train = pd.DataFrame(self.mst_encoder.fit_transform(self.x_train[['market_segment_type']]),
                                columns=self.mst_encoder.get_feature_names_out(['market_segment_type']))
        mst_test = pd.DataFrame(self.mst_encoder.transform(self.x_test[['market_segment_type']]),
                                columns=self.mst_encoder.get_feature_names_out(['market_segment_type']))

        self.tmp_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        tmp_train = pd.DataFrame(self.tmp_encoder.fit_transform(self.x_train[['type_of_meal_plan']]),
                                columns=self.tmp_encoder.get_feature_names_out(['type_of_meal_plan']))
        tmp_test = pd.DataFrame(self.tmp_encoder.transform(self.x_test[['type_of_meal_plan']]),
                                columns=self.tmp_encoder.get_feature_names_out(['type_of_meal_plan']))

        self.x_train = pd.concat([self.x_train.reset_index(drop=True), mst_train, tmp_train], axis=1)
        self.x_test = pd.concat([self.x_test.reset_index(drop=True), mst_test, tmp_test], axis=1)

        self.x_train.drop(['market_segment_type', 'type_of_meal_plan'], axis=1, inplace=True)
        self.x_test.drop(['market_segment_type', 'type_of_meal_plan'], axis=1, inplace=True)

    def create_model_rf(self):
        self.rf_model = RandomForestClassifier()

    def train_model_rf(self):
        self.rf_model.fit(self.x_train, self.y_train)

    def make_prediction_rf(self):
        self.y_predict_rf = self.rf_model.predict(self.x_test) 

    def inv_transform_rf(self):
        self.y_predict2_rf = self.target_enc.inverse_transform(self.y_predict_rf)
        
    def evaluate_model_rf(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict2_rf))

    def save_model_to_file(self, model_filename, labelEnc_filename, targetEnc_filename, mstEnc_filename, tmpEnc_filename):
        joblib.dump(self.rf_model, model_filename, compress=3)
        with open(labelEnc_filename, 'wb') as file:
            pickle.dump(self.label_enc, file) 
        with open(targetEnc_filename, 'wb') as file:
            pickle.dump(self.target_enc, file)
        with open(mstEnc_filename, 'wb') as mst_file:
            pickle.dump(self.mst_encoder, mst_file)
        with open(tmpEnc_filename, 'wb') as tmp_file:
            pickle.dump(self.tmp_encoder, tmp_file)


file_path = 'Dataset_B_hotel.csv'  
handle_data = handleData(file_path)
handle_data.load_data()
handle_data.create_input_output('booking_status')
input_df = handle_data.input_df
output_df = handle_data.output_df

handle_model = handleModel(input_df, output_df)
handle_model.del_id('Booking_ID')
handle_model.split_data()

tmp_impute = handle_model.impute_mode('type_of_meal_plan')
handle_model.impute_missing_values('type_of_meal_plan', tmp_impute)
rcps_impute = handle_model.impute_mode('required_car_parking_space')
handle_model.impute_missing_values('required_car_parking_space', rcps_impute)
handle_model.checkOutlier('avg_price_per_room')
appr_impute = handle_model.impute_median('avg_price_per_room')
handle_model.impute_missing_values('avg_price_per_room', appr_impute)
handle_model.drop_duplicate('booking_status')

handle_model.label_encoding('room_type_reserved')
handle_model.oneHot_encoding()
handle_model.target_encoding()
handle_model.create_model_rf()
handle_model.train_model_rf()
handle_model.make_prediction_rf()
handle_model.inv_transform_rf()
handle_model.evaluate_model_rf()
handle_model.save_model_to_file('rf_model_oop.pkl', 
                                'room_type_reserved_encode_oop.pkl',
                                'booking_status_encode_oop.pkl', 
                                'market_segment_type_encode_oop.pkl',
                                'type_of_meal_plan_encode_oop.pkl')