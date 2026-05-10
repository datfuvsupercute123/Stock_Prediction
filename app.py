
import streamlit as st
import torch, joblib, torch.nn as nn
import numpy as np
import pandas as pd

st.set_page_config(page_title="VN Stock Historical AI", layout="wide")
st.title("CNN-BiGRU Stock Prediction")

class CNN_BiGRU(nn.Module):
    def __init__(self, in_sz, out_sz):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_sz, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.rnn = nn.GRU(128, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, out_sz)
        )

    def forward(self, x):
        c_out = self.cnn(x.permute(0, 2, 1))
        _, h_n = self.rnn(c_out.permute(0, 2, 1))
        return self.fc(torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1))

@st.cache_resource
def load_assets():
    try:
        cfg = joblib.load('config_kdays_vn.joblib')
        scl = joblib.load('target_scalers_kdays_vn.joblib')
        payload = joblib.load('real_data_payload.joblib')
        
        net = CNN_BiGRU(cfg['in_sz'], cfg['out_sz'])
        net.load_state_dict(torch.load('vn_kdays.pth', map_location='cpu'))
        net.eval()
        return net, scl, cfg, payload
    except Exception as e:
        st.error(f"Missing files: {e}")
        return None, None, None, None

model, scalers, config, payload = load_assets()

if model is not None:
    # Sidebar selection
    tickers = sorted(payload['df_subset']['Ticker'].unique())
    ticker = st.sidebar.selectbox("Select Ticker:", tickers)
    
    # Filter dates for the selected ticker
    ticker_df = payload['df_subset'][payload['df_subset']['Ticker'] == ticker]
    available_dates = ticker_df['Date'].dt.strftime('%Y-%m-%d').tolist()
    
    # User selects a specific end date for the window
    selected_date_str = st.sidebar.selectbox("Select Calculation Date (End of 60-day window):", available_dates[::-1])

    if st.button("Predict from Selected Date"):
        # Find index of the selected date
        target_idx = ticker_df[ticker_df['Date'].dt.strftime('%Y-%m-%d') == selected_date_str].index[0]
        
        # Get start index (target_idx - 59)
        all_ticker_indices = ticker_df.index.tolist()
        pos = all_ticker_indices.index(target_idx)
        
        if pos >= config['WINDOW_SIZE'] - 1:
            window_indices = all_ticker_indices[pos - config['WINDOW_SIZE'] + 1 : pos + 1]
            input_data = payload['data'][window_indices]
            X = torch.tensor(input_data).float().unsqueeze(0)

            with torch.no_grad():
                preds = model(X).numpy()

            real_diffs = scalers[ticker].inverse_transform(preds)[0]
            
            st.success(f"Forecasting for {ticker} starting after {selected_date_str}")
            
            # Metrics
            met_cols = st.columns(5)
            for i, d in enumerate(real_diffs):
                met_cols[i].metric(f"T+{i+1} Forecast", f"{d:+,.0f} VND")

            # Chart
            chart_df = pd.DataFrame({
                "Day": [f"T+{i+1}" for i in range(5)], 
                "Price Change": real_diffs
            })
            st.line_chart(chart_df.set_index("Day"))
        else:
            st.error(f"Not enough historical data before {selected_date_str} to form a {config['WINDOW_SIZE']} day window.")
