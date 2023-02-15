import streamlit as st
import pandas as pd
import plotly.express as px
import zipfile
import lightgbm as lgb
from PIL import Image

#st.set_page_config(layout="wide")
MODEL_PATH = 'data/'
MODEL_NAME = '2023-02-08_lgbm_model.zip'
UIC_PATH = 'data/unique_uic.csv'
TIMETABLE_PATH = 'data/tagesfahrplan.csv'
SELECTED_LINES = ['RB 71', 'RB 72', 'RB 73', 'RB 74', 'RB 76']

# COLUMN NAMES
COLUMN_DAY_OF_SERVICE = "betriebstag"
COLUMN_TIME = "planzeit_an"
COLUMN_TRIP_NUMBER = "zugnummer"
COLUMN_STATION = "ibnr"
COLUMN_PREVIOUS_STATION = "ibnr_vorher"
COLUMN_STOP_NUMBER_IN_TRIP = "lfd"
COLUMN_ROUTE_ID = "linie_railml"  # Meint die Linie, nicht die Fahrtnummer
COLUMN_TIME_SEC = 'time_sec'
COLUMN_TIME_HOURS = 'time_hours'
COLUMN_YEAR = 'year'
COLUMN_MONTH = 'month'
COLUMN_DAY = 'day'
COLUMN_DAY_OF_WEEK = 'day_of_week'
COLUMN_OCCUPANCY_VALUE = 'occupancy'
COLUMN_PREVIOUS_OCCUPANCY_VALUE = 'previous_occupancy_value'
COLUMN_UIC = 'uic'

XList=[
    COLUMN_TIME_SEC,
    COLUMN_YEAR,
    COLUMN_MONTH,
    COLUMN_DAY,
    COLUMN_DAY_OF_WEEK,
    COLUMN_UIC,
    COLUMN_TRIP_NUMBER,
    COLUMN_STATION,
    COLUMN_PREVIOUS_OCCUPANCY_VALUE,
    COLUMN_PREVIOUS_STATION,
    COLUMN_STOP_NUMBER_IN_TRIP,
]
y = COLUMN_OCCUPANCY_VALUE

def derived_columns_time(df):
    """
    Create new columns derived from existing date an time columns.
    The following columns need to be datetime objects:
    time,time_actual,day_of_service
    """
    df[COLUMN_TIME_SEC] = df[COLUMN_TIME].dt.hour * 3600 + df[COLUMN_TIME].dt.minute * 60 + df[COLUMN_TIME].dt.second
    #df[COLUMN_DELTA] = (df[COLUMN_TIME_ACTUAL] - df[COLUMN_TIME]).dt.total_seconds()
    df[COLUMN_YEAR] = df[COLUMN_DAY_OF_SERVICE].dt.year
    df[COLUMN_MONTH] = df[COLUMN_DAY_OF_SERVICE].dt.month
    df[COLUMN_DAY] = df[COLUMN_DAY_OF_SERVICE].dt.day
    df[COLUMN_DAY_OF_WEEK] = df[COLUMN_DAY_OF_SERVICE].dt.dayofweek
    df[COLUMN_TIME_HOURS] = df[COLUMN_TIME].dt.hour
    
    return df


def number_in_trip(df: pd.DataFrame):
    """
    number rows for each trip (fahrt_halt_lauf)
    supposes that rows are already in correct order for Arrival/Departure and sorted by time
    it also needs the column COLUMN_PREVIOUS_STATION

    the two rows corresponding to arrival/departure at the same station get the same number

    :param df:
    :return: df with COLUMN_STOP_NUMBER_IN_TRIP filled
    """
    df["stop_counter_helper"] = (df[COLUMN_STATION] != df[COLUMN_PREVIOUS_STATION]).astype(int)
    df[COLUMN_STOP_NUMBER_IN_TRIP] = df.groupby(COLUMN_TRIP_NUMBER)["stop_counter_helper"].cumsum()
    df.drop("stop_counter_helper", axis=1, inplace=True)
    return df


def data_preparation_one_trip(one_trip, occupancy_first_stop=0, day= pd.Timestamp.today(), uicnr=0):
    df = one_trip.copy()
    df[COLUMN_DAY_OF_SERVICE] = day
    # df[COLUMN_DAY_OF_SERVICE]= pd.to_datetime(df[COLUMN_DAY_OF_SERVICE])
    df[COLUMN_TIME]= pd.to_datetime(df[COLUMN_TIME])
    df = derived_columns_time(df)
    df[COLUMN_TIME] = df[COLUMN_TIME].dt.time.astype(str)
    df = number_in_trip(df)
    # unknown cols
    df[COLUMN_UIC] = uicnr
    df[COLUMN_PREVIOUS_OCCUPANCY_VALUE] = occupancy_first_stop

    return df


def iterative_prediction(beispiel1_it):
    """
    Predict occupancy for each stop. Iterative because each prediction 
    is based on the prediction of the previous stop.
    :param df:
    :return: series with predictions
    """
    beispiel1_it = beispiel1_it.copy()
    first_iteration = True

    for index, row in beispiel1_it.iterrows():
        if first_iteration:
            beispiel1_it.loc[index,'prediction'] = beispiel1_it.loc[index,'previous_occupancy_value']
            prediction = regressor.predict(row[XList].astype(float))[0]
            first_iteration = False
        else:
            # alte vorhersage
            beispiel1_it.loc[index,'previous_occupancy_value'] = prediction
            row['previous_occupancy_value'] = prediction

            
            # aktuelle vorhersage
            prediction = regressor.predict(row[XList].astype(float))[0]
            beispiel1_it.loc[index,'prediction'] = prediction
        
        
    return beispiel1_it['prediction']


def prediction_one_trip(df):
    df2 =  df.copy()
    df2['Belegungsprognose'] = iterative_prediction(df[XList])
    df2['Haltestelle'] =  df2[COLUMN_TIME] +" "+ df2['bhflang'] 
    return df2


def barplot_one_trip(df2):

    #tripnr = df2[COLUMN_TRIP_NUMBER].astype(int).iloc[0]
    return px.bar(df2, x='Haltestelle', y='Belegungsprognose', title='Belegungsprognose je Haltestelle auf der Fahrt')


def load_data():
    """
    Load relevant data from files
    """
    # load model
    archive = zipfile.ZipFile(MODEL_PATH+MODEL_NAME, 'r')
    model_name_ext = MODEL_NAME.split('.')[0]+'.mdl'
    regressor = lgb.Booster(model_str=archive.read(model_name_ext).decode())
    # regressor = lgb.Booster(model_file=model_path+model_name)

    # load timetable
    fpl = pd.read_csv(TIMETABLE_PATH, sep=';', )
    fpl = fpl[fpl[COLUMN_ROUTE_ID].isin(SELECTED_LINES)].copy()
    # easy timetable string
    fpl_string = fpl.groupby([COLUMN_ROUTE_ID,COLUMN_TRIP_NUMBER]).agg(
        abfahrt = ('planzeit_ab', 'first'),
        von = ('bhflang', 'first'),
        nach = ('bhflang', 'last'),
        ankunft = ('planzeit_an','last')
    ).reset_index()
    fpl_string['Fahrtinfo'] = fpl_string[COLUMN_ROUTE_ID]+' um '+fpl_string['abfahrt'] + ' von '+  fpl_string['von'] + ' nach ' + fpl_string['nach'] + ' um ' + fpl_string['ankunft'] + ' ('+fpl_string[COLUMN_TRIP_NUMBER].astype(int).astype(str) +')'
    fpl_string = fpl_string.sort_values('Fahrtinfo')

    #load uic list
    uic = pd.read_csv(UIC_PATH, sep=';')

    return regressor, fpl, fpl_string, uic 



################### App
image = Image.open('data/etc_logo.png')
st.image(image)

st.title('Belegungsprognosen für vlexx')

data_load_state = st.text('Lade Modell...')
regressor, fpl, fpl_string, uic = load_data()
data_load_state.text("Modell geladen")

st.subheader('Wählen Sie eine Fahrt:')

dropdown_train_number = st.selectbox('Wähle Zug',fpl_string['Fahrtinfo'])

c1, c2, c3 = st.columns(3)
with c1:
    first_occupancy_value = st.number_input('Belegung an erster Haltestelle', step=1)
with c2:
    d = st.date_input("Betriebstag",pd.Timestamp.today() )
with c3:
    uicnr = st.selectbox('Wagennummer',uic[COLUMN_UIC].dropna().astype(str))


st.markdown('Die Prognosen im Fahrverlauf bauen auf einander auf. Wählen Sie eine Belegung am ersten Halt um die Prognosen für den weiteren Fahrtverlauf zu beeinflussen.')

selected_trip = fpl_string.loc[fpl_string['Fahrtinfo']==dropdown_train_number,COLUMN_TRIP_NUMBER].iloc[0]


one_trip = fpl[(fpl[COLUMN_DAY_OF_SERVICE]==fpl[COLUMN_DAY_OF_SERVICE].min())&(fpl[COLUMN_TRIP_NUMBER]==selected_trip)]
one_trip = data_preparation_one_trip(one_trip, first_occupancy_value, pd.to_datetime(d),uicnr)
one_trip = prediction_one_trip(one_trip)
st.subheader('Belegungsprognose zur Fahrt')
st.plotly_chart(barplot_one_trip(one_trip))
st.dataframe(one_trip[[COLUMN_ROUTE_ID,'planzeit_an','planzeit_ab','bhflang','Belegungsprognose']])
