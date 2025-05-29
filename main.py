from pickletools import unicodestringnl
import streamlit as st
import pickle
import requests
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import squarify
from dotenv import load_dotenv
from groq import Groq
load_dotenv()

def load_model(filename):
    with open(os.path.join("models", filename), "rb") as file:
        model = pickle.load(file)
    return model

st.set_page_config(page_title="CATASTRONET", layout="wide",page_icon="🌋")


with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    st.markdown("""
    <div class="tit">
        CATASTRONET
    </div>
    """, unsafe_allow_html=True)

    with open("ouragan.jpg", "rb") as file:
        img_data = file.read()
    img_base64 = base64.b64encode(img_data).decode()

    st.sidebar.markdown(
        f"""
        <img class="sidebar-img" src="data:image/jpeg;base64,{img_base64}">
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown("<div style='text-align:center;'>CatastroNet🌫</div>",unsafe_allow_html=True)
    
    menu = ["Home🏠","Dashboard📈","Prediction💹","Assistance🤖"]
    choice = st.sidebar.selectbox("Board",menu)
    
    if choice =='Home🏠':

        st.title("🌍 CatastroNet - Système intelligent de gestion des catastrophes naturelles")

        st.markdown("""
        Bienvenue sur *CatastroNet*, votre plateforme intelligente dédiée à l’analyse, la prévention et la gestion des catastrophes naturelles.

        Grâce à l’intelligence artificielle, notre application vous permet de :
        - 📊 Analyser les risques liés aux catastrophes naturelles (séismes, inondations, sécheresses, etc.)
        - 🌐 Visualiser les zones à risque selon différents paramètres
        - 🛡 Mieux anticiper pour protéger les populations et les infrastructures

        ---

        💡 Sélectionnez une option dans le menu latéral pour commencer l’analyse ou explorer les données.

        """)
            
    elif choice == 'Dashboard📈':
        @st.cache_data
        def load_data():
            df = pd.read_csv("full_dataset.csv")
            return df
        df = load_data()
        
        st.subheader('📊Tableau de Surveillance')
        ville_filter = st.selectbox("Choisir une ville", df['ville'].unique())
        df = df[df['ville'] == ville_filter]
        
        avg = df['avg_air_temperature'].mean()
        count_married = np.mean(df['max_air_temperature'])
        balance = np.mean(df['min_air_temperature'])
        #creation d'indicateur
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric(label=" Avg air T° ", value=round(avg), delta=round(avg))
        kpi2.metric(label=" Max air T° ", value=int(count_married), delta=round(count_married))
        kpi3.metric(label=" Min air T° ", value=round(balance), delta=round(balance))
        
        col1, col2 = st.columns(2)
        a = col1.button("page1")
        b = col2.button("page2")

        if a:
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("🎯 Barplot")
                fig1, ax = plt.subplots()
                sns.barplot(x='humidity', y='wind_speed', data=df, palette='Oranges', ax=ax)
                st.pyplot(fig1)

            with col2:
                st.subheader("🧭 Donut Chart")
                counts = df['avg_air_temperature'].value_counts()
                labels = counts.index
                sizes = counts.values
                colors = sns.color_palette('pastel')[0:len(labels)]
                fig2, ax = plt.subplots()
                ax.pie(sizes, labels=labels, colors=colors, startangle=90,
                    wedgeprops=dict(width=0.4), autopct='%1.1f%%')
                ax.axis('equal')
                st.pyplot(fig2)

            with col3:
                st.subheader("📈 Courbe")
                fig3, ax = plt.subplots()
                ax.plot(df['historical_index'], df['snowfall'], color='teal')
                ax.set_xlabel('historical_index')
                ax.set_ylabel('snowfall')
                st.pyplot(fig3)
        elif b:
            # Deuxième ligne
            col4, col5, col6 = st.columns(3)
            with col4:
                st.subheader("📊 Histogramme")
                fig4, ax = plt.subplots()
                sns.histplot(x='event_duration_days', data=df, color='green', ax=ax)
                st.pyplot(fig4)

            with col5:
                st.subheader("🎻 Violinplot")
                fig6, ax = plt.subplots()
                sns.violinplot(x="ground_deformation", y="seismic_activity", data=df, ax=ax, palette='muted')
                st.pyplot(fig6)
                
            with col6:
                st.subheader("📉 Lineplot")
                fig9, ax = plt.subplots()
                sns.lineplot(x="uv_index", y="rainfall", data=df, ax=ax, color='purple')
                st.pyplot(fig9)
        
        st.dataframe(df)

    elif choice == 'Prediction💹':
        st.subheader('💹Tableau de Prediction')
        tab1,tab2,tab3,tab4 = st.tabs(["Menu","🌍Terrestre","🌊Aquatique","🪐Atmosphérique"])
        with tab1:
            st.markdown("Passer à Terrestre, Aquatique ou Atmosphérique pour effectuer une prediction")
            st.markdown("<div class='fleche'>➡</div>",unsafe_allow_html=True)
            
        with tab2:
            st.markdown("Choisir Le Type de Prediction Que Vous Souhaitez Faire et Remplir Les Données")
            
            
            with st.form(key='Details'):  
                st.markdown("<div class='tite'>🌋Eruption Volcanique</div>",unsafe_allow_html=True)
                model_volcan = load_model("volcan.pkl")  
                col1,col2 = st.columns(2)
                with col1:
                    seismic_activity = st.number_input("Activité sysmique",step=0.1,key='seismic_activity')
                    gas_emission = st.number_input("Emission de Gaz",step=0.1,key='gas_emission')
                    ground_deformation = st.number_input("Deformation du Sol",step=0.1,key='ground_deformation')
                    crater_temperature = st.number_input("Temperature du Cratère",step=0.1,key='crater_temperature')
                    
                with col2:
                    eruption_history_score = st.number_input("Score Historique des Eruptions",step=0.1,key='eruption_history_score1')
                    magma_chamber_pressure = st.number_input("Pression de la Chambre Magmatique",step=0.1,key='magma_chamber_pressure')
                    ash_emission_rate = st.number_input("Taux d'émission des cendres ",step=0.1,key='ash_emission_rate')
                    crust_thickness = st.number_input("Epaisseur de la croute",step=0.1,key='crust_thickness')
                    
                data = [[seismic_activity,gas_emission,ground_deformation,crater_temperature,
                    eruption_history_score,magma_chamber_pressure,ash_emission_rate,crust_thickness]] 
                
                volcano = st.form_submit_button('**Predict Eruption Volcanique**')
                if volcano:
                    with st.spinner('Calcul en cours... ⏳'):
                        prediction = model_volcan.predict(data)[0]
                        time.sleep(1)  # simule un délai
                    
                    st.success('Prédiction terminée 🎉')
                    
                    st.markdown(
                        f"""
                        <div style="font-size:18px; margin-top:20px; margin-bottom:20px;">
                            La prédiction est : <strong>{prediction}</strong>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    if prediction == 1:
                        st.error(
                            "⚠️ Risque d'Éruption Volcanique"
                        )
                    else:
                        st.success(
                            "✅ Pas de Risque d'Éruption Volcanique"
                        )
            
            with st.form(key='Detail'):    
                st.markdown("<div class='tite'>🌪Séisme</div>",unsafe_allow_html=True)
                model_seisme = load_model("seisme.pkl")
                    
                col3,col4 = st.columns(2)
                with col3:
                    tectonic_plate_motion = st.number_input("Mouvements des Plaques",step=0.1,key='tectonic_plate_motion')
                    seismic_history_score = st.number_input("Score Historique Sismique",step=0.1,key='seismic_history_score')
                    ground_stress_level = st.number_input("Niveau de Stress au Sol",step=0.1,key='ground_stress_level')
                    microquake_frequency = st.number_input("Fréquence du Micro-séquisme",step=0.1,key='microquake_frequency1')
                    
                with col4:
                    gas_emission_level = st.number_input("Niveau des Gaz d'Emission",step=0.1,key='gas_emission_level')
                    depth_to_bedrock = st.number_input("Profondeur j'usqu'au substrat Rocheux",step=0.1,key='depth_to_bedrock')
                    water_table_level = st.number_input("Niveau de la Nappe Phréatique",step=0.1,key='water_table_level')
                    rock_density = st.number_input("Densité Rocheuse",step=0.1,key='rock_density')
                    
                seisme = [[tectonic_plate_motion,seismic_history_score,ground_stress_level,microquake_frequency,
                    gas_emission_level,depth_to_bedrock,water_table_level,rock_density]] 
        
                seismo = st.form_submit_button('**Predict seisme**')
                if seismo:
                    with st.spinner('calcul en cours..'):
                        prediction = model_seisme.predict(seisme)[0]
                        time.sleep(1)
                    st.success('Prediction terminée')
                    st.markdown(f"La Prediction est égal à:**{prediction}**")
                    if prediction == 1:
                        st.error("⚠️ Risque de Séisme")
                    else:
                        st.success("✅ Pas de Risque de Séisme")
            
            with st.form(key='Detailse'):    
                st.markdown("<div class='tite'>☀Sècheresse</div>",unsafe_allow_html=True)
                model_secheresse = load_model("secheresse.pkl")
                    
                col5,col6 = st.columns(2)
                with col5:
                    precipitation = st.number_input("Precipitation",step=0.,key='precipitation1')
                    soil_moisture = st.number_input("Humidité du sol",step=0.1,key='soil_moisture1')
                    temperature = st.number_input("Temperature du Sol",step=0.1,key='temperature2')
                    evapotranspiration = st.number_input("Evapo-transpiration",step=0.1,key='evapotranspiration1')
                    
                with col6:
                    vegetation_index = st.number_input("Index de la Vegetation",step=0.1,key='vegetation_index1')
                    water_reservoir_level = st.number_input("Niveau du reservoir d'eau",step=0.1,key='water_reservoir_level1')
                    wind_speed = st.number_input("Vitesse du Vent",step=0.1,key='wind_speed1')
                    drought_history_score = st.number_input("Historique de la Secheresse",step=0.1,key='drought_history_score1')
                    
                secheresse = [[precipitation,soil_moisture,temperature,evapotranspiration,
                    vegetation_index,water_reservoir_level,wind_speed,drought_history_score]] 
        
                secheresso = st.form_submit_button('**Predict Secheresse**')
                if secheresso:
                    with st.spinner('calcul en cours..'):
                        prediction = model_secheresse.predict(secheresse)[0]
                        time.sleep(1)
                    st.success('Prediction terminée')
                    st.markdown(f"La Prediction est égal à:**{prediction}**")
                    if prediction == 1:
                        st.error("⚠️ Risque de Sècheresse")
                    else:
                        st.success("✅ Pas de Risque de Sècheresse")
            
            with st.form(key='Detailx'):    
                st.markdown("<div class='tite'>❄Tempête de Neige</div>",unsafe_allow_html=True)
                model_neige = load_model("neige.pkl")
                
                col7,col8 = st.columns(2)
                with col7:
                    temperature = st.number_input("Temperature°",step=0.1,key='temperature3')
                    humidity = st.number_input("Humidité de l'Air",step=0.1,key='humidity1')
                    wind_speed = st.number_input("Vitesse du Vent",step=0.1,key='wind_speed2')
                    atmospheric_pressure = st.number_input("Pression Atmosphérique",step=0.1,key='atmospheric_pressure1')
                    
                with col8:
                    precipitation_probability = st.number_input("Probabilité de Precipitation",step=0.1,key='precipitation_probability1')
                    snow_depth = st.number_input("Profondeur de Neige",step=0.1,key='snow_depth1')
                    cloud_cover = st.number_input("Couverture Nuageuse",step=0.1,key='cloud_cover1')
                    storm_history_score = st.number_input("Score Historique des Tempêtes",step=0.1,key='storm_history_score1')
                    
                neige = [[temperature,humidity,wind_speed,atmospheric_pressure,
                    precipitation_probability,snow_depth,cloud_cover,storm_history_score]] 
        
                neigo = st.form_submit_button('**Predict Tempête de Neige**')
                if neigo:
                    with st.spinner('calcul en cours..'):
                        prediction = model_neige.predict(neige)[0]
                        time.sleep(1)
                    st.success('Prediction terminée')
                    st.markdown(f"La Prediction est égal à:**{prediction}**")
                    if prediction == 1:
                        st.error("⚠️ Risque d'une Tempête de Neige")
                    else:
                        st.success("✅ Pas de Risque d'une Tempête de Neige")
            
        with tab3:
            st.markdown("Choisir Le Type de Prediction Que Vous Souhaitez Faire et Remplir Les Données")
            
            with st.form(key='Detour1'):
                st.markdown("<div class='tite'>☔Inondation</div>",unsafe_allow_html=True)
                model_inondation = load_model("inondation.pkl")
                    
                col1,col2 = st.columns(2)
                with col1:
                    rainfall = st.number_input("Niveau de Precipitation",step=0.1,key='rainfall1')
                    river_level = st.number_input("Niveau de Rivière",step=0.1,key='river_level1')
                    soil_saturation = st.number_input("Saturation au Sol",step=0.1,key='soil_saturation1')
                    urban_coverage = st.number_input("Couverture Urbaine",step=0.1,key='urbain_coverage')
                    drainage_capacity = st.number_input("Capacité de Drainage",step=0.1,key='drainage_capacity')
                    
                with col2:
                    humidity = st.number_input("Humidité",step=0.1,key='humidite2')
                    wind_speed = st.number_input("Vitesse vent ",step=0.1,key='wind_speed3')
                    flood_history_score = st.number_input("Score Historique des inondations ",step=0.1,key='flood_history_score')
                    local_topography = st.number_input("Topographie Locale",step=0.1,key='local_topography')
                    runoff_speed = st.number_input("Vitesse de ruissellement",step=0.1,key='runoff_speed')
                    
                inondation = [[rainfall,river_level,soil_saturation,urban_coverage,drainage_capacity,
                    humidity,wind_speed,flood_history_score,local_topography,runoff_speed]] 
        
                inondo = st.form_submit_button('**Predict Inondation**')
                if inondo:
                    with st.spinner('calcul en cours..'):
                        prediction = model_inondation.predict(inondation)[0]
                        time.sleep(1)
                    st.success('Prediction terminée')
                    st.markdown(f"La Prediction est égal à:**{prediction}**")
                    if prediction == 1:
                        st.error("⚠️ Risque d'Inondation")
                    else:
                        st.success("✅ Pas de Risque d'Inondation")
            
            with st.form(key='Detour2'):  
                st.markdown("<div class='tite'>🌀Ouragan</div>",unsafe_allow_html=True)
                model_ouragan = load_model("ouragan.pkl")
                
                col3,col4 = st.columns(2)
                with col3:
                    sea_surface_temp = st.number_input("T° de la Surface de la Mer",step=0.1,key='sea_surface_temp')
                    air_pressure = st.number_input("Pression Atmosphérique",step=0.1,key='air_pressure_')
                    humidity = st.number_input("Humidité du sol",step=0.1,key='himidity3')
                    wind_shear = st.number_input("Coupe Vent",step=0.1,key='wind_shear')
                    cloud_cover = st.number_input("Couverture Nuageuse",step=0.1,key='cloud_cover2')
                    
                    
                with col4:
                    precipitation = st.number_input("Precipitation",step=0.1,key='precipita')
                    sst_anomaly = st.number_input("Anomalie sst",step=0.1,key='sst_anomaly')
                    atmospheric_instability = st.number_input("Instabilité Atmosphérique",step=0.1,key='atmospheric')
                    storm_history_score = st.number_input("Score Historique des Tempêtes",step=0.1,key='storm')
                    land_temperature = st.number_input("T° Terrestre",step=0.1,key='land_temperature')
                    
                ouragan = [[sea_surface_temp,air_pressure,humidity,wind_shear,cloud_cover,
                    precipitation,sst_anomaly,atmospheric_instability,storm_history_score,land_temperature]] 
        
                ouragano = st.form_submit_button('**Predict Ouragan**')
                if ouragano:
                    with st.spinner('calcul en cours..'):
                        prediction = model_ouragan.predict(ouragan)[0]
                        time.sleep(1)
                    st.success('Prediction terminée')
                    st.markdown(f"La Prediction est égal à:**{prediction}**")
                    if prediction == 1:
                        st.error("⚠️ Risque d'Ouragan")
                    else:
                        st.success("✅ Pas de Risque d'Ouragan")
            
            with st.form(key='Detour'):
                st.markdown("<div class='tite'>🌊Tsunami</div>",unsafe_allow_html=True)
                model_tsunami = load_model("tsunami.pkl")
                    
                col5,col6 = st.columns(2)
                with col5:
                    earthquake_magnitude = st.number_input("Magnitude du Tremblement de Terre",step=0.1,key='magnitude')
                    earthquake_depth = st.number_input("Profondeur du Tremblement de Terre",step=0.1,key='depth')
                    distance_to_coast = st.number_input("Distance de la Côte",step=0.1,key='distance_to_cote')
                    seafloor_displacement = st.number_input("Déplacement des Fonds Marins",step=0.1,key='seafloor_displacement')
                    fault_type_score = st.number_input("Scrore des Defauts",step=0.1,key='fault_type_score')
                    
                    
                with col6:
                    aftershocks_count = st.number_input("Repliques",step=0.1,key='aftershocks_count')
                    tectonic_zone_score = st.number_input("Score Zone Tectonique",step=0.1,key='tectonic')
                    ocean_depth_at_epicenter = st.number_input("Profondeur de l'Océan à l'Epicentre",step=0.1,key='ocean_depth')
                    wave_amplification_potential = st.number_input("Potentiel d'Amplification des Ondes",step=0.1,key='amplification')
                    historical_tsunami_frequency = st.number_input("Frequence Historique du Tsunami",step=0.1,key='historical_tsunami')
                
                tsunami = [[earthquake_magnitude,earthquake_depth,distance_to_coast,
                            seafloor_displacement,fault_type_score,aftershocks_count,
                            tectonic_zone_score,ocean_depth_at_epicenter,wave_amplification_potential,
                            historical_tsunami_frequency]] 
        
                tsunamo = st.form_submit_button('**Predict Tsunami**')
                if tsunamo:
                    with st.spinner('calcul en cours..'):
                        prediction = model_tsunami.predict(tsunami)[0]
                        time.sleep(1)
                    st.success('Prediction terminée')
                    st.markdown(f"La Prediction est égal à:**{prediction}**")
                    if prediction == 1:
                        st.error("⚠️ Risque de Tsunami")
                    else:
                        st.success("✅ Pas de Risque de Tsunami")
                
        with tab4:
            st.markdown("Choisir Le Type de Prediction Que Vous Souhaitez Faire et Remplir Les Données")
            
            with st.form(key='Orage'):
                st.markdown("<div class='tite'>⛈Orage</div>",unsafe_allow_html=True)
                model_orage = load_model("orage.pkl")
                    
                col1,col2 = st.columns(2)
                with col1:
                    air_temperature = st.number_input("T° de l'air",step=0.1,key='air_temperature3')
                    humidity = st.number_input("Humidité",step=0.1,key='humidity4')
                    pressure = st.number_input("Pression",step=0.1,key='pressure4')
                    dew_point = st.number_input("Point de Rosée",step=0.1,key='dew_point')
                    wind_speed = st.number_input("Vitesse du Vent",step=0.1,key='wind_speed4')
                    
                    
                with col2:
                    wind_direction_variability = st.number_input("Variabilité de la vitesse du vent",step=0.1,key='wind_direction')
                    cloud_cover = st.number_input("Couverture Nuageuse",step=0.1,key='cloud_cover3')
                    cape = st.number_input("CAPE",step=0.1,key='cape')
                    lifted_index = st.number_input("Index Surelevé",step=0.1,key='lifted_index')
                    historical_thunderstorm_score = st.number_input("Score d'Orage Historique",step=0.1,key='historical_thunderstorm_score')
                    
                orage = [[air_temperature,humidity,pressure,dew_point,wind_speed,
                            wind_direction_variability,cloud_cover,cape,lifted_index,
                            historical_thunderstorm_score]] 
        
                orago = st.form_submit_button('**Predict Orage/Tornade**')
                if orago:
                    with st.spinner('calcul en cours..'):
                        prediction = model_orage.predict(orage)[0]
                        time.sleep(1)
                    st.success('Prediction terminée')
                    st.markdown(f"La Prediction est égal à:**{prediction}**")
                    if prediction == 1:
                        st.error("⚠️ Risque d'un Orage")
                    else:
                        st.success("✅ Pas de Risque d'un Orage")

            with st.form(key='Froid'):
                st.markdown("<div class='tite'>🥶Vague de Froid</div>",unsafe_allow_html=True)
                model_froid = load_model("froid.pkl")
                    
                col3,col4 = st.columns(2)
                with col3:
                    avg_air_temperature = st.number_input("T° Moyenne de l'Air",step=0.1,key='aat')
                    min_air_temperature = st.number_input("T° minimale de l'air",step=0.1,key='mat')
                    wind_speed = st.number_input("Vitesse du Vent",step=0.1,key='ws')
                    humidity = st.number_input("Humidité",step=0.1,key='h')
                    pressure = st.number_input("Pression",step=0.1,key='p')
                    
                    
                with col4:
                    snow_cover_days = st.number_input("Jours Couvert de Neige",step=0.1,key='scd')
                    intrusion_polaire = st.number_input("Intrusion Polaire",step=0.1,key='ip')
                    cold_wave_duration = st.number_input("Durée de la vague Froide",step=0.1,key='cwd')
                    cloud_coverage = st.number_input("Couverture Cloud",step=0.1,key='cc')
                    historical_cold_wave_index = st.number_input("Indice Historique des VF",step=0.1,key='hcwi')
                    
                froid = [[avg_air_temperature,min_air_temperature,wind_speed,humidity,pressure,
                            snow_cover_days,intrusion_polaire,cold_wave_duration,cloud_coverage,
                            historical_cold_wave_index]] 
        
                froido = st.form_submit_button('**Predict Vague de Froid**')
                if froido:
                    with st.spinner('calcul en cours..'):
                        prediction = model_froid.predict(froid)[0]
                        time.sleep(1)
                    st.success('Prediction terminée')
                    st.markdown(f"La Prediction est égal à:**{prediction}**")
                    if prediction == 1:
                        st.error("⚠️ Risque d'une Vague de Froid")
                    else:
                        st.success("✅ Pas de Risque d'une Vague de Froid")
            
            with st.form(key='Canicule'):
                st.markdown("<div class='tite'>🥵Canicule</div>",unsafe_allow_html=True)
                model_canicule = load_model("canicule.pkl")
                    
                col5,col6 = st.columns(2)
                with col5:
                    avg_air_temperature = st.number_input("T° Moyenne de l'Air",step=0.1,key='aat1')
                    max_air_temperature = st.number_input("T° Minimale de l'Air",step=0.1,key='mat1')
                    humidity = st.number_input("Humidité",step=0.1,key='hu')
                    uv_index = st.number_input("Index UV",step=0.1,key='uv_index')
                    wind_speed = st.number_input("Vitesse du Vent",step=0.1,key='wp1')
                    
                    
                with col6:
                    night_temp = st.number_input("T° de Nuit",step=0.1,key='nigth_temp')
                    heat_duration_days = st.number_input("Jours de Durée de Chaleur",step=0.1,key='hdd')
                    drought_index = st.number_input("Indice de Sècheresse",step=0.1,key='drought_index')
                    cloud_coverage = st.number_input("Couverture Cloud",step=0.1,key='cloud_c')
                    urban_heat_island_index = st.number_input("Indice de l'île de Chaleur Urbaine",step=0.1,key='uhii')
                    
                canicule = [[avg_air_temperature,max_air_temperature,humidity,uv_index,wind_speed,
                            night_temp,heat_duration_days,drought_index,cloud_coverage,
                            urban_heat_island_index]] 
        
                caniculo = st.form_submit_button('**Predict Canicule**')
                if caniculo:
                    with st.spinner('calcul en cours..'):
                        prediction = model_canicule.predict(canicule)[0]
                        time.sleep(1)
                    st.success('Prediction terminée')
                    st.markdown(f"La Prediction est égal à:**{prediction}**")
                    if prediction == 1:
                        st.error("⚠️ Risque d'une Canicule")
                    else:
                        st.success("✅ Pas de Risque d'une Canicule")
                    
                    
    elif choice == 'Assistance🤖':

        def img_to_base64(img_path):
            with open(img_path, "rb") as img_file:
                b64_str = base64.b64encode(img_file.read()).decode()
            return b64_str

        img_b64 = img_to_base64("chatbot.png")
        html_img = f'<img src="data:image/png;base64,{img_b64}" class="chatbot-image" width="150">'

        st.markdown(html_img, unsafe_allow_html=True)

        # Zone de texte et bouton côte à côte
        message = st.text_input('En quoi puis-je vous aider 💬', key='chat_input')
        
        send = st.button("Envoyer", key='send_btn', help="Cliquez pour envoyer votre message")

        client = Groq(api_key=os.environ.get("GROQ_API_KEYS"))
        print("Clé chargée :", os.environ.get("GROQ_API_KEYS"))

        if message:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": message,
                    }
                ],
                model="llama-3.3-70b-versatile",
            )

            st.markdown(f'<div class="chat-response">{chat_completion.choices[0].message.content}</div>', unsafe_allow_html=True)
        
        
        
if __name__ == '__main__':
    main()