"""
File: src/app.py

Streamlit app for this project. Includes data visualization, processing,
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import os
from io import StringIO

@st.cache_data
def load_data():
    """Loads the dataset for visualization purposes"""
    try:
        # Try to load the 50-bin aggregated data first (smaller file)
        data_path = "data/merged.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df
        else:
            st.error(f"Data file not found: {data_path}")
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def load_weather_data():
    """Load weather data for visualization"""
    try:
        weather_path = "data/weather_data.csv"
        if os.path.exists(weather_path):
            df = pd.read_csv(weather_path)
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Error loading weather data: {str(e)}")
        return None

def load_model():
    """Load the trained model results"""
    try:
        model_path = "results_50bins.pkl"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_results = pickle.load(f)
            return model_results
        else:
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def title():
    """Introductory text for app"""
    st.title("NYC Traffic Risk Forecaster")
    st.subheader("A spatial Poisson regression model that maps location and weather conditions to predicted accident risk")

    st.markdown("""
    This application visualizes traffic accident patterns in NYC and provides risk forecasting based on:
    - **Location**: Spatial binning of NYC areas
    - **Weather conditions**: Temperature, precipitation, wind, etc.
    - **Temporal patterns**: Time-based trends
    """)

def data_overview_tab(df: pd.DataFrame):
    """Display data overview and basic statistics"""
    if df is not None:
        st.header("Data Overview")
        st.text("The dataset consists of accidents that occurred from January 2013 to August 2025.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records (Number of accidents)", f"{len(df):,}")
        with col2:
            if 'accident_count' in df.columns:
                st.metric("Total Accidents", f"{df['accident_count'].sum():,}")
        with col3:
            if 'date' in df.columns:
                date_range = pd.to_datetime(df['date']).max() - pd.to_datetime(df['date']).min()
                st.metric("Date Range", f"{date_range.days} days")

        st.subheader("Dataset Sample")
        st.dataframe(df.head(10))

        if st.checkbox("Show dataset info"):
            st.text("Dataset Information:")
            info_data = {
                "Column": df.columns,
                "Dtype": [str(df[col].dtype) for col in df.columns]
            }
            info_df = pd.DataFrame(info_data).set_index("Column")
            st.dataframe(info_df)

def visualizations_tab(df, weather_df):
    """Create various visualizations"""
    st.header("Visualizations")

    if df is not None:
        # Time series of accidents
        if 'date' in df.columns and 'accident_count' in df.columns:
            st.subheader("Accident Trends Over Time")
            df['date'] = pd.to_datetime(df['date'])
            daily_accidents = df.groupby('date')['accident_count'].sum().reset_index()

            fig = px.line(daily_accidents, x='date', y='accident_count',
                         title='Daily Accident Counts')
            fig.update_layout(xaxis_title="Date", yaxis_title="Number of Accidents")
            st.plotly_chart(fig, use_container_width=True)

        # Spatial distribution if lat/lon available
        if all(col in df.columns for col in ['latitude', 'longitude', 'accident_count']):
            st.subheader("Spatial Distribution of Accidents")
            sample_size = min(1000, len(df))
            sample_df = df.sample(n=sample_size)

            fig = px.scatter_mapbox(
                sample_df,
                lat='latitude',
                lon='longitude',
                size='accident_count',
                color='accident_count',
                hover_data=['accident_count'],
                color_continuous_scale='Reds',
                mapbox_style='open-street-map',
                zoom=10,
                title=f'Accident Distribution (Sample of {sample_size} locations)'
            )
            st.plotly_chart(fig, use_container_width=True)

        # Weather correlation if weather data is available
        if weather_df is not None:
            st.subheader("Weather Impact Analysis")
            weather_cols = [col for col in weather_df.columns if col in ['rain_1h', 'snow_1h', 'wind_speed', 'visibility']]
            if weather_cols:
                selected_weather = st.selectbox("Select weather variable:", weather_cols)
                if selected_weather in weather_df.columns:
                    fig = px.histogram(weather_df, x=selected_weather,
                                     title=f'Distribution of {selected_weather}')
                    st.plotly_chart(fig, use_container_width=True)

def model_results_tab(model_results):
    """Display model results and predictions"""
    st.header("Model Results")

    if model_results is not None:
        st.write("Model results loaded successfully!")
        st.write("Available keys:", list(model_results.keys()) if isinstance(model_results, dict) else "Not a dictionary")

        # Display basic model info
        st.json(str(model_results)[:500] + "..." if len(str(model_results)) > 500 else str(model_results))
    else:
        st.warning("No model results found. Train a model first to see results here.")

def fetch_weather():
    """Fetches the current weather conditions (hourly) from the OpenWeatherData API"""
    st.subheader("Current Weather")
    st.info("Weather API integration coming soon!")

def app():
    """Main app function"""
    title()

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Visualizations", "Model Results", "Weather"])

    # Load data
    df = load_data()
    weather_df = load_weather_data()
    model_results = load_model()

    with tab1:
        data_overview_tab(df)

    with tab2:
        visualizations_tab(df, weather_df)

    with tab3:
        model_results_tab(model_results)

    with tab4:
        fetch_weather()

if __name__ == "__main__":
    app()

