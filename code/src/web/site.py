import streamlit as st
import requests
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import ast

st.set_page_config(layout="wide")

@st.cache_data
def process(server_url, method, sensitivity_score, max_fraction_anomalies, debug, input_data_set):
    full_server_url = f"{server_url}/{method}?sensitivity_score={sensitivity_score}&max_fraction_anomalies={max_fraction_anomalies}&debug={debug}"
    r = requests.post(
        full_server_url,
        data=input_data_set,
        headers={"Content-Type": "application/json"}
    )
    return r

# Used as a helper method for creating lists from JSON.
@st.cache_data
def convert_univariate_list_to_json(univariate_str):
    # Remove brackets if they exist.
    univariate_str = univariate_str.replace('[', '').replace(']', '')
    univariate_list = univariate_str.replace(' ', '').split(',')
    df = pd.DataFrame([float(u) for u in univariate_list], columns=["value"])
    df["key"] = ""
    return df.to_json(orient="records")

@st.cache_data
def convert_multivariate_list_to_json(multivariate_str):
    mv_ast = ast.literal_eval(multivariate_str)
    return json.dumps([{"key": k, "vals": v} for idx,[k,v] in enumerate(mv_ast)])

def main():
    st.write(
    """
    # Finding Ghosts in Your Data

    This is an outlier detection application based on the book Finding Ghosts in Your Data (Apress, 2022).  The purpose of this site is to provide a simple interface for interacting with the outlier detection API we build over the course of the book.

    ## Instructions
    First, select the method you wish to use for outlier detection.  Then, enter the dataset you wish to process.  This dataset should be posted as a JSON array with the appropriate attributes.  The specific attributes you need to enter will depend on the method you chose above.

    If you switch between methods, you will see a sample dataset corresponding to the expected structure of the data.  Follow that pattern for your data.
    """
    )

    server_url = "http://localhost/detect"
    method = st.selectbox(label="Choose the method you wish to use.", options = ("univariate", "multivariate", "timeseries/single", "timeseries/multiple"))
    sensitivity_score = st.slider(label = "Choose a sensitivity score.", min_value=1, max_value=100, value=50)
    max_fraction_anomalies = st.slider(label = "Choose a max fraction of anomalies.", min_value=0.01, max_value=1.0, value=0.1)
    debug = st.checkbox(label="Run in Debug mode?")
    convert_to_json = st.checkbox(label="Convert data in list to JSON format?  If you check this box, enter data as a comma-separated list of values.")
    
    if method == "univariate":
        starting_data_set = """[
        {"key": "1","value": 1},
        {"key": "2", "value": 2},
        {"key": "3", "value": 3},
        {"key": "4", "value": 4},
        {"key": "5", "value": 5},
        {"key": "6", "value": 6},
        {"key": "8", "value": 95}
    ]"""
    elif method == "multivariate":
        starting_data_set = """[
        {"key":"1","vals":[22.46, 17.69, 8.04, 14.11]},
        {"key":"2","vals":[22.56, 17.69, 8.04, 14.11]},
        {"key":"3","vals":[22.66, 17.69, 8.04, 14.11]},
        {"key":"4","vals":[22.76, 17.69, 8.04, 14.11]},
        {"key":"5","vals":[22.896, 17.69, 8.04, 14.11]},
        {"key":"6","vals":[22.9, 22.69, 8.04, 14.11]},
        {"key":"7","vals":[22.06, 17.69, 8.04, 14.11]},
        {"key":"8","vals":[22.16, 17.69, 9.15, 14.11]},
        {"key":"9","vals":[22.26, 17.69, 8.04, 14.11]},
        {"key":"10","vals":[22.36, 178.69, 8.04, 14.11]},
        {"key":"11","vals":[22.46, 17.69, 8.04, 14.11]},
        {"key":"12","vals":[22.56, 17.69, 8.04, 14.11]},
        {"key":"13","vals":[22.66, 17.69, 8.04, 14.11]},
        {"key":"14","vals":[22.76, 17.69, 8.04, 14.11]},
        {"key":"15","vals":[22.86, 17.69, 8.04, 14.11]},
        {"key":"16","vals":[22.76, 17.69, 8.04, 14.11]},
        {"key":"17","vals":[22.66, 17.69, 8.04, 14.11]},
        {"key":"18","vals":[22.56, 17.69, 8.04, 14.11]},
        {"key":"19","vals":[22.46, 17.69, 8.04, 14.11]},
        {"key":"20","vals":[22.36, 17.69, 8.04, 14.11]},
        {"key":"21","vals":[22.26, 17.69, 8.04, 14.11]}
    ]"""
    else:
        starting_data_set = "Select a method."
    input_data = st.text_area(label = "Data to process (in JSON format):", value=starting_data_set, height=300)

    if st.button(label="Detect!"):
        if method=="univariate" and convert_to_json:
            input_data = convert_univariate_list_to_json(input_data)
        if method=="multivariate" and convert_to_json:
            input_data = convert_multivariate_list_to_json(input_data)
        resp = process(server_url, method, sensitivity_score, max_fraction_anomalies, debug, input_data)
        res = json.loads(resp.content)
        df = pd.DataFrame(res['anomalies'])

        if method=="univariate":
            st.header('Anomaly score per data point')
            colors = {True: '#481567', False: '#3CBB75'}
            g = px.scatter(df, x=df["value"], y=df["anomaly_score"], color=df["is_anomaly"], color_discrete_map=colors,
                        symbol=df["is_anomaly"], symbol_sequence=['square', 'circle'],
                        hover_data=["sds", "mads", "iqrs", "grubbs", "gesd", "dixon", "gaussian_mixture"])
            st.plotly_chart(g, use_container_width=True)


            tbl = df[['key', 'value', 'anomaly_score', 'is_anomaly', 'sds', 'mads', 'iqrs', 'grubbs', 'gesd', 'dixon', 'gaussian_mixture']]
            st.write(tbl)

            if debug:
                col11, col12 = st.columns(2)

                with col11:                
                    st.header('Debug weights')
                    st.write(res['debug_weights'])

                with col12:
                    st.header("Tests Run")
                    st.write(res['debug_details']['Test diagnostics']['Tests Run'])
                    if "Extended tests" in res['debug_details']['Test diagnostics']:
                        st.write(res['debug_details']['Test diagnostics']['Extended tests'])
                    if "Gaussian mixture test" in res['debug_details']['Test diagnostics']:
                        st.write(res['debug_details']['Test diagnostics']['Gaussian mixture test'])

                col21, col22 = st.columns(2)

                with col21:
                    st.header("Base Calculations")
                    st.write(res['debug_details']['Test diagnostics']['Base calculations'])

                with col22:
                    st.header("Fitted Calculations")
                    if "Fitted calculations" in res['debug_details']['Test diagnostics']:
                        st.write(res['debug_details']['Test diagnostics']['Fitted calculations'])

                col31, col32 = st.columns(2)

                with col31:
                    st.header("Initial Normality Checks")
                    if "Initial normality checks" in res['debug_details']['Test diagnostics']:
                        st.write(res['debug_details']['Test diagnostics']['Initial normality checks'])

                with col32:
                    st.header("Fitted Normality Checks")
                    if "Fitted Lambda" in res['debug_details']['Test diagnostics']:
                        st.write(f"Fitted Lambda = {res['debug_details']['Test diagnostics']['Fitted Lambda']}")
                    if "Fitted normality checks" in res['debug_details']['Test diagnostics']:
                        st.write(res['debug_details']['Test diagnostics']['Fitted normality checks'])
                    if "Fitting Status" in res['debug_details']['Test diagnostics']:
                        st.write(res['debug_details']['Test diagnostics']["Fitting Status"])

                st.header("Full Debug Details")
                st.json(res['debug_details'])
        elif method=="multivariate":
            st.header('Anomaly score per data point')
            colors = {True: '#481567', False: '#3CBB75'}
            df = df.sort_values(by=['anomaly_score'], ascending=False)
            g = px.bar(df, x=df["key"], y=df["anomaly_score"], color=df["is_anomaly"], color_discrete_map=colors,
                        hover_data=["vals", "anomaly_score_cof", "anomaly_score_loci"], log_y=True)
            st.plotly_chart(g, use_container_width=True)


            tbl = df[['key', 'vals', 'anomaly_score', 'is_anomaly', 'anomaly_score_cof', 'anomaly_score_loci']]
            st.write(tbl)

            if debug:
                col11, col12 = st.columns(2)

                with col11:                
                    st.header("Tests Run")
                    st.write(res['debug_details']['Tests run'])
                    st.write(res['debug_details']['Test diagnostics'])

                with col12:
                    st.header("Outlier Determinants")
                    st.write(res['debug_details']['Outlier determination'])

                st.header("Full Debug Details")
                st.json(res['debug_details'])


if __name__ == "__main__":
    main()