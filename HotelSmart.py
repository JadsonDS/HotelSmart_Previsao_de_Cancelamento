import streamlit as st
import numpy as np
import pickle
from boruta import BorutaPy
from PIL import Image

# Configuração da página
st.set_page_config(page_title="Hotéis - Previsão de Cancelamento", layout="wide")
st.title("🏨 HotelSmart - Previsão de Cancelamento de Reservas")
st.markdown("### Insira os dados da reserva para prever se será cancelada")

# Carregar e mostrar a logo
logo = Image.open("logo.png")
st.image(logo, width=200)  # Ajuste a largura conforme necessário

# Carregar modelo e transformadores
with open('models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('models/boruta_selector.pkl', 'rb') as file:
    boruta_selector = pickle.load(file)

with open('models/best_xgb.pkl', 'rb') as file:
    model = pickle.load(file)

# --- Organização do Formulário em 4 colunas por linha ---
# Linha 1
col1, col2, col3, col4 = st.columns(4)

with col1:
    no_of_adults = st.number_input("Adultos", min_value=0)
with col2:
    no_of_children = st.number_input("Crianças", min_value=0)
with col3:
    no_of_weekend_nights = st.number_input("Noites Fim Semana", min_value=0)
with col4:
    no_of_week_nights = st.number_input("Noites de Semana", min_value=0)

# Linha 2
col5, col6, col7, col8 = st.columns(4)

with col5:
    type_of_meal_plan = st.selectbox("Plano de Refeição", options=[0, 1, 2, 3])
with col6:
    required_car_parking_space = st.selectbox("Estacionamento", options=[0, 1])
with col7:
    room_type_reserved = st.selectbox("Tipo de Quarto", options=[0, 1, 2, 3, 4, 5, 6])
with col8:
    lead_time = st.number_input("Lead Time", min_value=0)

# Linha 3
col9, col10, col11, col12 = st.columns(4)

with col9:
    arrival_month = st.selectbox("Mês de Chegada", options=list(range(1, 13)))
with col10:
    market_segment_type = st.selectbox("Segmento de Mercado", options=[0, 1, 2, 3, 4])
with col11:
    repeated_guest = st.selectbox("Hóspede Repetido", options=[0, 1])
with col12:
    no_of_previous_cancellations = st.number_input("Cancelamentos Anteriores", min_value=0)

# Linha 4
col13, col14, col15, col16 = st.columns(4)

with col13:
    no_of_previous_bookings_not_canceled = st.number_input("Reservas Não Canceladas", min_value=0)
with col14:
    avg_price_per_room = st.number_input("Preço Médio por Quarto", min_value=0.0)
with col15:
    no_of_special_requests = st.number_input("Solicitações Especiais", min_value=0)
with col16:
    total_nights = st.number_input("Total de Noites", min_value=0)

# Linha 5 (opcional, se quiser incluir total_people)
col17, _, _, _ = st.columns(4)
with col17:
    total_people = st.number_input("Total de Pessoas", min_value=0)

# Botão de previsão
if st.button("Prever Cancelamento"):
    # Criar array com as 17 features na mesma ordem do treinamento
    input_data = np.array([[
        no_of_adults,
        no_of_children,
        no_of_weekend_nights,
        no_of_week_nights,
        type_of_meal_plan,
        required_car_parking_space,
        room_type_reserved,
        lead_time,
        arrival_month,
        market_segment_type,
        repeated_guest,
        no_of_previous_cancellations,
        no_of_previous_bookings_not_canceled,
        avg_price_per_room,
        no_of_special_requests,
        total_nights,
        total_people
    ]])

    # Aplicar escalonamento e seleção de features
    input_data_scaled = scaler.transform(input_data)
    input_data_selected = input_data_scaled[:, boruta_selector.support_]

    # Fazer previsão
    prediction = model.predict(input_data_selected)
    result = "Cancelamento previsto" if prediction[0] == 1 else "Reserva segura"
    st.success(result)