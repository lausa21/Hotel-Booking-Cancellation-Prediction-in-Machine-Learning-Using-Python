# 2702213770 - Laurel Evelina Widjaja - Nomor 4
import streamlit as st
import joblib
import pandas as pd

# Load the machine learning model
model = joblib.load('rf_model_oop.pkl')
mst_encoder = joblib.load('market_segment_type_encode_oop.pkl') 
tmp_encoder = joblib.load('type_of_meal_plan_encode_oop.pkl') 
rtr_encoder = joblib.load('room_type_reserved_encode_oop.pkl')
bs_encoder = joblib.load('booking_status_encode_oop.pkl')

def main():
    st.title('2702213770 - Laurel Evelina Widjaja - UTS Model Deployment')

    # Input dari user
    no_of_adults = st.sidebar.number_input('Jumlah Orang Dewasa', min_value=1, value=1)
    no_of_children = st.sidebar.number_input('Jumlah Anak Kecil', min_value=0, value=0)
    no_of_weekend_nights = st.sidebar.number_input('Jumlah Malam Akhir Pekan', min_value=0, value=0)
    no_of_week_nights = st.sidebar.number_input('Jumlah Malam dalam Seminggu', min_value=0, value=0)
    type_of_meal_plan = st.sidebar.selectbox('Jenis Paket Makanan', ['Not Selected', 'Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3'])
    required_car_parking_space = st.sidebar.radio('Apakah Membutuhkan Tempat Parkir?', ['Tidak', 'Iya'])
    room_type_reserved = st.sidebar.selectbox('Jenis Kamar', ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
    lead_time = st.sidebar.number_input('Jumlah Hari Antara Tanggal Pemesanan dan Tanggal Kedatangan', min_value=0, max_value=500, value=0)
    arrival_year = st.sidebar.number_input('Tahun Kedatangan', min_value=2017)
    arrival_month = st.sidebar.number_input('Bulan Kedatangan', min_value=1, max_value=12)
    arrival_date = st.sidebar.number_input('Tanggal Kedatangan', min_value=1, max_value=31)
    market_segment_type = st.sidebar.selectbox('Segmen Pasar', ['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation'])
    repeated_guest = st.sidebar.radio('Apakah Sebelumnya Pernah Menginap di Sini?', ['Tidak', 'Iya'])
    no_of_previous_cancellations = st.sidebar.number_input('Jumlah Pembatalan Sebelumnya', min_value=0, value=0)
    no_of_previous_bookings_not_canceled = st.sidebar.number_input('Jumlah Pesanan Tidak Dibatalkan Sebelumnya', min_value=0, value=0)
    avg_price_per_room = st.sidebar.number_input('Harga Rata-Rata per Hari Pemesanan', min_value=0.00, value=99.45)
    no_of_special_requests = st.sidebar.slider('Jumlah Permintaan Khusus', min_value=0, max_value=10, value=0)

    # Perhitungan dari input user
    required_car_parking_space = 1 if required_car_parking_space == 'Iya' else 0
    repeated_guest = 1 if repeated_guest == 'Iya' else 0

    if st.button('Make Prediction'):
        features = pd.DataFrame([{
            'no_of_adults': no_of_adults,
            'no_of_children': no_of_children,
            'no_of_weekend_nights': no_of_weekend_nights,
            'no_of_week_nights': no_of_week_nights,
            'type_of_meal_plan': type_of_meal_plan,
            'required_car_parking_space': required_car_parking_space,
            'room_type_reserved': room_type_reserved,
            'lead_time': lead_time,
            'arrival_year': arrival_year,
            'arrival_month': arrival_month,
            'arrival_date': arrival_date,
            'market_segment_type': market_segment_type,
            'repeated_guest': repeated_guest,
            'no_of_previous_cancellations': no_of_previous_cancellations,
            'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
            'avg_price_per_room': avg_price_per_room,
            'no_of_special_requests': no_of_special_requests
        }])

        # Encoding fitur kategorikal
        mst_encoded = pd.DataFrame(mst_encoder.transform(features[['market_segment_type']]), columns=mst_encoder.get_feature_names_out(['market_segment_type']))
        tmp_encoded = pd.DataFrame(tmp_encoder.transform(features[['type_of_meal_plan']]), columns=tmp_encoder.get_feature_names_out(['type_of_meal_plan']))
        rtr_encoded = rtr_encoder.transform([room_type_reserved])  # Label encoding menghasilkan array satu dimensi
        rtr_encoded_df = pd.DataFrame(rtr_encoded, columns=['room_type_reserved'])

        # Gabungkan fitur numerik dan hasil encoding
        numeric_features = features.drop(columns=['market_segment_type', 'type_of_meal_plan', 'room_type_reserved'])
        final_features = pd.concat([numeric_features, mst_encoded, tmp_encoded, rtr_encoded_df], axis=1)

        final_features = final_features[model.feature_names_in_]

        result = make_prediction(final_features)
        decoded_result = bs_encoder.inverse_transform([result])
        st.success(f'The prediction is: {decoded_result[0]}')

def make_prediction(features):
    prediction = model.predict(features)
    return prediction[0]

if __name__ == '__main__':
    main()

