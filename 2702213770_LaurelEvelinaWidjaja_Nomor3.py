# 2702213770 - Laurel Evelina Widjaja - Nomor 3
import pickle
import pandas as pd
import warnings
import joblib
warnings.filterwarnings('ignore')

def load_model(model_filename):
    """Load the best model from a pickle file"""
    with open(model_filename, 'rb') as file:
        model = joblib.load(file)
    return model

def load_encoder(encoder_filename):
    """Load the label encoder from a pickle file"""
    with open(encoder_filename, 'rb') as file:
        enc = pickle.load(file)
    return enc

def preprocess_user_input(user_input, label_enc, mst_enc, tmp_enc):
    """Encode the categorical user input and return the encoded input as a DataFrame."""
    try:
        # Encode the 'room_type_reserved' using label encoding
        user_input['room_type_reserved'] = label_enc.transform([user_input['room_type_reserved']])[0]
        
        # One Hot Encoding for 'type_of_meal_plan' using tmp_enc
        tmp_encoded = tmp_enc.transform([[user_input['type_of_meal_plan']]])
        tmp_encoded_df = pd.DataFrame(tmp_encoded, columns=tmp_enc.get_feature_names_out())

        # One Hot Encoding for 'market_segment_type' using mst_enc
        mst_encoded = mst_enc.transform([[user_input['market_segment_type']]])
        mst_encoded_df = pd.DataFrame(mst_encoded, columns=mst_enc.get_feature_names_out())

        # Combine all encoded features with the original input (numeric fields)
        user_input_encoded = pd.DataFrame([user_input]).drop(['type_of_meal_plan', 'market_segment_type'], axis=1)
        
        # Combine the original data with the one-hot encoded DataFrames
        user_input_encoded = pd.concat([user_input_encoded, tmp_encoded_df, mst_encoded_df], axis=1)

        return user_input_encoded
    except Exception as e:
        print(f"Error in preprocessing user input: {e}")
        raise

def predict_with_model(model, label_enc, target_enc, mst_enc, tmp_enc, user_input):
    """Make a prediction and return the booking status."""
    # Preprocess and encode user input
    preprocessed_input = preprocess_user_input(user_input, label_enc, mst_enc, tmp_enc)
    # Convert the DataFrame to a list of numeric values for the model
    user_input_list = preprocessed_input.values.tolist()
    # Predict the outcome
    prediction = model.predict(user_input_list)
    # Convert prediction back to original category
    booking_status = target_enc.inverse_transform(prediction)  
    
    return booking_status[0]

def main():
    model = load_model('rf_model_oop.pkl')
    label_enc = load_encoder('room_type_reserved_encode_oop.pkl')
    target_enc = load_encoder('booking_status_encode_oop.pkl')
    mst_enc = load_encoder('market_segment_type_encode_oop.pkl')
    tmp_enc = load_encoder('type_of_meal_plan_encode_oop.pkl')

    user_input = {
        'no_of_adults': 2,
        'no_of_children': 1,
        'no_of_weekend_nights': 1,
        'no_of_week_nights': 2,
        'type_of_meal_plan': 'Meal Plan 1',
        'required_car_parking_space': 0.0,
        'room_type_reserved': 'Room_Type 1',
        'lead_time': 50,
        'arrival_year': 2017,
        'arrival_month': 9,
        'arrival_date': 5,
        'market_segment_type': 'Online',
        'repeated_guest': 0,
        'no_of_previous_cancellations': 0,
        'no_of_previous_bookings_not_canceled': 0,
        'avg_price_per_room': 80.0,
        'no_of_special_requests': 1
    }
    prediction = predict_with_model(model, label_enc, target_enc, mst_enc, tmp_enc, user_input)
    print(f"The predicted output is: {prediction}")

if __name__ == "__main__":
    main()
