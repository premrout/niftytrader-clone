# One-click setup script for NiftyTrader Streamlit MVP
# This script creates the folder structure and files as defined in the MVP scaffold.
# Run in PowerShell:  ./setup_nifty.ps1

$base = "C:\Users\premr\nifty_app"
New-Item -ItemType Directory -Force -Path $base

# Helper to write files
function Write-File($path, $content) {
    $dir = Split-Path $path
    if (!(Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    Set-Content -Path $path -Value $content
}

# requirements.txt
Write-File "$base\requirements.txt" @"
streamlit
pandas
numpy
yfinance
plotly
requests
nsepython
cachetools
"@

# app.py (launcher)
Write-File "$base\app.py" @"
import streamlit as st
st.set_page_config(page_title="NiftyTrader Clone", layout="wide")

st.title("NiftyTrader.in Clone — Streamlit MVP")
st.write("Use the sidebar to navigate through pages.")
"@

# Homepage
Write-File "$base\pages\0_Homepage.py" @"
import streamlit as st

st.title("Homepage — NiftyTrader Clone")
st.write("Live market overview and quick links coming here.")
"@

# Option Chain
Write-File "$base\pages\1_Option_Chain.py" @"
import streamlit as st
st.title("Option Chain")
st.write("Option chain table and charts will be rendered here.")
"@

# PCR page
Write-File "$base\pages\2_PCR.py" @"
import streamlit as st
st.title("PCR (Put Call Ratio)")
st.write("PCR charts will go here.")
"@

# Max Pain page
Write-File "$base\pages\3_Max_Pain.py" @"
import streamlit as st
st.title("Max Pain")
st.write("Max Pain calculations and heatmap.")
"@

# Live Charts
Write-File "$base\pages\4_Live_Charts.py" @"
import streamlit as st
st.title("Live Charts")
st.write("Candlestick charts with indicators.")
"@

# Utilities folder
Write-File "$base\utils\__init__.py" ""
Write-File "$base\utils\nse_client.py" @"
# placeholder for NSE data fetch utilities
"@

# Run script
Write-File "$base\run_local.bat" @"
@echo off
cd %~dp0
python -m streamlit run app.py
"@

Write-Output "Setup complete. Navigate to $base and run run_local.bat"
