#app.py
import streamlit as st
import requests
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Optional
import base64
from io import BytesIO
import numpy as np
from io import StringIO
import streamlit.components.v1 as components

# Configure Streamlit page
st.set_page_config(
    page_title="Forest Fire Risk Monitor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #ff6b6b, #ffa726);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #007bff;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s ease-in-out;
    }

    .metric-card h2, .metric-card h3 {
        color: #212529 !important; 
    }

    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .risk-high {
        background: #ffebee;
        border-left-color: #f44336;
    }
    
    .risk-moderate {
        background: #fff3e0;
        border-left-color: #ff9800;
    }
    
    .risk-low {
        background: #e8f5e8;
        border-left-color: #4caf50;
    }
    
    .sidebar-section {
        background: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .alert-banner {
        background: #d32f2f;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .info-banner {
        background: #1976d2;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Fix for Streamlit dataframe styling performance */
    .stDataFrame {
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "https://forest-fire-monitor.onrender.com"
RISK_COLORS = {
    0: '#4CAF50',  # Green for Low
    1: '#FF9800',  # Orange for Moderate
    2: '#F44336'   # Red for High
}
RISK_LABELS = {0: 'Low', 1: 'Moderate', 2: 'High'}

class FireMonitorApp:
    def __init__(self):
        self.api_base_url = API_BASE_URL
    
    def make_api_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with proper JSON handling"""
        try:
            url = f"{self.api_base_url}/{endpoint}"
            
            if params is None:
                params = {}
            
            # Convert date object to string if needed
            if 'date' in params and hasattr(params['date'], 'strftime'):
                params['date'] = params['date'].strftime('%Y-%m-%d')
            
            if endpoint == "predict-fires":
                response = requests.post(
                    url, 
                    json=params,
                    headers={'Content-Type': 'application/json'},
                    timeout=60
                )
            else:
                response = requests.get(url, params=params, timeout=60)
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    return result
                except ValueError as e:
                    st.error(f"❌ Invalid JSON response: {e}")
                    return {}
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
                return {}
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 Cannot connect to API. Please ensure your FastAPI server is running on http://localhost:8000")
            return {}
        except requests.exceptions.Timeout:
            st.error("🚨 API request timed out. Please try again.")
            return {}
        except Exception as e:
            st.error(f"🚨 Request failed: {e}")
            return {}
        
    def create_map(self, fire_data: List[Dict], view_type: str = "markers") -> folium.Map:
        """Create Folium map with fire data"""
        # Center map on India
        india_center = [20.5937, 78.9629]
        m = folium.Map(
            location=india_center,
            zoom_start=5,
            tiles='OpenStreetMap'
        )
        
        if not fire_data:
            return m
        
        if view_type == "markers":
            # Create marker clusters for each risk level
            low_risk_cluster = MarkerCluster(name="Low Risk", show=True)
            moderate_risk_cluster = MarkerCluster(name="Moderate Risk", show=True)
            high_risk_cluster = MarkerCluster(name="High Risk", show=True)
            
            clusters = {
                0: low_risk_cluster,
                1: moderate_risk_cluster,
                2: high_risk_cluster
            }
            
            for point in fire_data:
                lat, lon = point['latitude'], point['longitude']
                risk_level = point['risk_level']
                
                popup_content = f"""
                <div style="width: 200px;">
                    <h4>Fire Detection</h4>
                    <p><strong>Date:</strong> {point['acq_date']}</p>
                    <p><strong>Time:</strong> {point['acq_time']}</p>
                    <p><strong>Risk Level:</strong> <span style="color: {RISK_COLORS[risk_level]};">{RISK_LABELS[risk_level]}</span></p>
                    <p><strong>Brightness:</strong> {point['bright_ti4']:.2f}</p>
                    <p><strong>FRP:</strong> {point['frp']:.2f}</p>
                    <p><strong>Confidence:</strong> {point['confidence'].upper()}</p>
                    <p><strong>Satellite:</strong> {point['satellite']}</p>
                    <p><strong>Day/Night:</strong> {'Day' if point['daynight'] == 'D' else 'Night'}</p>
                </div>
                """
                
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_content, max_width=250),
                    icon=folium.Icon(
                        color='darkred' if risk_level == 2 else 'orange' if risk_level == 1 else 'green',
                        icon='fire'
                    )
                ).add_to(clusters[risk_level])
            
            for cluster in clusters.values():
                cluster.add_to(m)
                
        elif view_type == "heatmap":
            heat_data = []
            for point in fire_data:
                weight = (point['risk_level'] + 1) * max(point['frp'], 1)
                heat_data.append([point['latitude'], point['longitude'], weight])
            
            HeatMap(
                heat_data,
                min_opacity=0.2,
                max_zoom=18,
                radius=15,
                blur=10,
                gradient={
                    0.2: 'blue',
                    0.4: 'lime',
                    0.6: 'orange',
                    0.8: 'red',
                    1.0: 'darkred'
                }
            ).add_to(m)
        
        folium.LayerControl().add_to(m)
        return m
    
    def display_map(self, fire_map: folium.Map):
        """Display folium map using streamlit-folium with error handling"""
        try:
            # Convert map to HTML
            map_html = fire_map._repr_html_()
            
            # Display using components
            components.html(map_html, height=600, scrolling=True)
            
        except Exception as e:
            st.error(f"Error displaying map: {e}")
            st.info("💡 Try refreshing the page or switching to a different view type.")
    
    def create_risk_distribution_chart(self, statistics: Dict) -> go.Figure:
        """Create risk distribution pie chart"""
        labels = ['Low Risk', 'Moderate Risk', 'High Risk']
        values = [
            statistics.get('low_risk', 0),
            statistics.get('moderate_risk', 0),
            statistics.get('high_risk', 0)
        ]
        colors = ['#4CAF50', '#FF9800', '#F44336']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            textinfo='label+percent+value',
            textposition='inside'
        )])
        
        fig.update_layout(
            title="Fire Risk Distribution",
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_temporal_analysis(self, fire_data: List[Dict]) -> go.Figure:
        """Create temporal analysis chart"""
        if not fire_data:
            return go.Figure()
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(fire_data)
        df['datetime'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time'].str.zfill(4).str[:2] + ':' + df['acq_time'].str.zfill(4).str[2:])
        df['hour'] = df['datetime'].dt.hour
        
        # Group by hour and risk level
        hourly_risk = df.groupby(['hour', 'risk_level']).size().reset_index(name='count')
        
        fig = go.Figure()
        
        for risk_level in [0, 1, 2]:
            risk_data = hourly_risk[hourly_risk['risk_level'] == risk_level]
            fig.add_trace(go.Bar(
                x=risk_data['hour'],
                y=risk_data['count'],
                name=RISK_LABELS[risk_level],
                marker_color=RISK_COLORS[risk_level]
            ))
        
        fig.update_layout(
            title="Fire Detection by Hour of Day",
            xaxis_title="Hour of Day",
            yaxis_title="Number of Detections",
            barmode='stack',
            height=400
        )
        
        return fig
    
    def export_data_to_csv(self, fire_data: List[Dict]) -> str:
        """Export fire data to CSV"""
        if not fire_data:
            return ""
        
        df = pd.DataFrame(fire_data)
        return df.to_csv(index=False)
    
    def run(self):
        """Run the Streamlit application"""
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🔥 Early Detection and Risk Mapping of Forest Fires Using Remote Sensing and AI</h1>
            <p>Real-time monitoring of forest fire risk levels across India using NASA FIRMS data</p>
        </div>
        """, unsafe_allow_html=True)

        # Sidebar
        st.sidebar.markdown("## 🎛️ Control Panel")
        
        # Date picker
        st.sidebar.markdown("### 📅 Date")
        date = st.sidebar.date_input(
            "Date",
            value=datetime.now() - timedelta(days=1),
            max_value=datetime.now()
        )
        
        # Day range
        st.sidebar.markdown("### 📊 Day Range")
        day_range = st.sidebar.slider(
            "Select day range",
            min_value=1,
            max_value=10,
            value=3
        )

        # Region selector
        st.sidebar.markdown("### 🗺️ Region")
        region = st.sidebar.selectbox(
            "Select region",
            options=['complete', 'north', 'south', 'east', 'west'],
            format_func=lambda x: {
                'complete': 'Complete India',
                'north': 'North India',
                'south': 'South India', 
                'east': 'East India',
                'west': 'West India'
            }.get(x, x)
        )

        # View type toggle
        st.sidebar.markdown("### 🗺️ Map View")
        view_type = st.sidebar.radio(
            "Choose visualization type:",
            ["markers", "heatmap"],
            format_func=lambda x: "🔵 Marker Clusters" if x == "markers" else "🔥 Heatmap"
        )
        
        # Filters
        st.sidebar.markdown("### 🔍 Filters")
        
        # Risk level filter
        risk_filter = st.sidebar.selectbox(
            "Risk Level",
            options=[None, 0, 1, 2],
            format_func=lambda x: "All Levels" if x is None else RISK_LABELS[x]
        )
        
        # Day/Night filter
        daynight_filter = st.sidebar.selectbox(
            "Day/Night",
            options=[None, "D", "N"],
            format_func=lambda x: "All Times" if x is None else ("Day" if x == "D" else "Night")
        )
        
        # Auto-refresh toggle
        st.sidebar.markdown("### 🔄 Auto-Refresh")
        auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh (5 min)")
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()
        
        # Main content - Prepare API parameters
        params = {
            "date": date.strftime("%Y-%m-%d"),
            "day_range": day_range,
            "region": region
        }
        
        if risk_filter is not None:
            params["risk_level"] = risk_filter
        if daynight_filter:
            params["daynight"] = daynight_filter
        
        # Show loading message
        with st.spinner("🔍 Fetching fire data from NASA FIRMS API..."):
            # Fetch data from API
            response = self.make_api_request("predict-fires", params)
        
        # Response validation
        if not response:
            st.error("❌ No response received from API. Please check if the backend server is running.")
            st.info("💡 Make sure your FastAPI server is running on http://localhost:8000")
            return
        
        if "error" in response:
            st.error(f"❌ API Error: {response['error']}")
            return
        
        fire_data = response.get("data", [])
        statistics = response.get("statistics", {})
        
        # Validate data
        total_records = len(fire_data)
        if total_records == 0:
            st.warning("⚠️ No fire data found for the selected criteria.")
            st.info("💡 Try adjusting your date range, region, or removing some filters.")
            return
        
        
        # Display alert banner for high-risk areas
        high_risk_count = statistics.get("high_risk", 0)
        if high_risk_count > 0:
            st.markdown(f"""
            <div class="alert-banner">
                🚨 HIGH RISK ALERT: {high_risk_count} high-risk fire areas detected!
            </div>
            """, unsafe_allow_html=True)
        
        # Display statistics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Total Detections</h3>
                <h2>{statistics.get('total_points', 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card risk-low">
                <h3>🟢 Low Risk</h3>
                <h2>{statistics.get('low_risk', 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card risk-moderate">
                <h3>🟡 Moderate Risk</h3>
                <h2>{statistics.get('moderate_risk', 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card risk-high">
                <h3>🔴 High Risk</h3>
                <h2>{statistics.get('high_risk', 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Map visualization
        st.markdown("## 🗺️ Interactive Fire Risk Map")
        
        if fire_data:
            # Create map
            fire_map = self.create_map(fire_data, view_type)
            
            # Display map with improved method
            self.display_map(fire_map)
        else:
            st.info("📍 No fire data available for the selected filters and date.")
        
        # Analytics section
        if fire_data:
            st.markdown("## 📈 Analytics Dashboard")
            
            # Create two columns for charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk distribution chart
                risk_chart = self.create_risk_distribution_chart(statistics)
                st.plotly_chart(risk_chart, use_container_width=True)
            
            with col2:
                # Temporal analysis
                temporal_chart = self.create_temporal_analysis(fire_data)
                st.plotly_chart(temporal_chart, use_container_width=True)
        
        # Data export section
        if fire_data:
            st.markdown("## 📥 Data Export")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV export
                csv_data = self.export_data_to_csv(fire_data)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name=f"fire_data_{date}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # JSON export
                json_data = json.dumps(fire_data, indent=2)
                st.download_button(
                    label="📋 Download JSON",
                    data=json_data,
                    file_name=f"fire_data_{date}.json",
                    mime="application/json"
                )
        
        # Detailed data table with performance optimization
        if fire_data:
            st.markdown("## 📋 Detailed Fire Data")

            # Convert to DataFrame for better display
            df = pd.DataFrame(fire_data)

            # Add a 'risk_label' column for easier styling
            if 'risk_level' in df.columns:
                df['risk_label'] = df['risk_level'].map(RISK_LABELS)

            # Select columns to display
            display_columns = [
                'acq_date', 'acq_time', 'latitude', 'longitude', 
                'risk_label', 'bright_ti4', 'frp', 'confidence', 
                'satellite', 'daynight'
            ]

            # Filter columns that exist in the DataFrame
            available_columns = [col for col in display_columns if col in df.columns]
            
            # Performance optimization: limit rows for styling
            MAX_ROWS_TO_STYLE = 1000
            
            if len(df) > MAX_ROWS_TO_STYLE:
                st.warning(f"⚠️ Large dataset ({len(df):,} rows). Showing first {MAX_ROWS_TO_STYLE:,} rows for performance.")
                # Show paginated view
                page_size = 100
                num_pages = min((len(df) - 1) // page_size + 1, 10)  # Limit to 10 pages max
                
                page = st.selectbox("Select page", range(1, num_pages + 1), index=0)
                start_idx = (page - 1) * page_size
                end_idx = min(start_idx + page_size, len(df))
                
                display_df = df.iloc[start_idx:end_idx]
                
                if available_columns:
                    st.dataframe(display_df[available_columns], use_container_width=True)
                else:
                    st.dataframe(display_df, use_container_width=True)
                    
                st.info(f"Showing rows {start_idx + 1}-{end_idx} of {len(df):,} total records")
                
            else:
                # Apply styling only for smaller dataframes
                if available_columns and 'risk_label' in df.columns:
                    def highlight_risk(row):
                        if row['risk_label'] == 'High':
                            return ['background-color: #ffebee; color: #212529;'] * len(row)
                        elif row['risk_label'] == 'Moderate':
                            return ['background-color: #fff3e0; color: #212529;'] * len(row)
                        else:
                            return ['background-color: #e8f5e8; color: #212529;'] * len(row)

                    try:
                        styled_df = df[available_columns].style.apply(highlight_risk, axis=1)
                        st.dataframe(styled_df, use_container_width=True)
                    except Exception:
                        # Fallback to unstyled if styling fails
                        st.dataframe(df[available_columns], use_container_width=True)
                else:
                    st.dataframe(df, use_container_width=True)
        
        # Auto-refresh logic
        if auto_refresh:
            time.sleep(300)  # 5 minutes
            st.rerun()
        
        # Footer with information
        st.markdown("---")
        st.markdown(f"""
        <div class="info-banner">
            <p>📡 Data Source: NASA FIRMS (Fire Information for Resource Management System)</p>
            <p>🔄 Data is updated in near real-time. Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>🗺️ Current region: {region.title()} India</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar information
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ℹ️ Information")
        st.sidebar.markdown("""
        **Risk Levels:**
        - 🟢 **Low**: Minimal fire risk
        - 🟡 **Moderate**: Moderate fire risk
        - 🔴 **High**: High fire risk - immediate attention required
        
        **Regions:**
        - **Complete**: All of India
        - **North**: Punjab, UP, etc.
        - **South**: Karnataka, Tamil Nadu, etc.
        - **East**: West Bengal, Bihar, etc.
        - **West**: Maharashtra, Gujarat, etc.
        """)
        
        # Performance metrics
        if fire_data:
            st.sidebar.markdown("### 📊 Performance Stats")
            processing_time = response.get("processing_time", "N/A")
            if processing_time != "N/A":
                st.sidebar.metric("Processing Time", f"{processing_time}s")
            st.sidebar.metric("Data Points", len(fire_data))
            st.sidebar.metric("Coverage Period", f"{day_range} days")
            st.sidebar.metric("Region", region.title())

# Main execution
if __name__ == "__main__":
    app = FireMonitorApp()
    app.run()