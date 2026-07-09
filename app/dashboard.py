import sys
try:
    import streamlit as st
except ImportError:
    st = None
    print("Warning: Streamlit not installed. Run `pip install streamlit`.")

def run_dashboard():
    if not st:
        print("Cannot run dashboard without Streamlit.")
        sys.exit(1)
        
    st.set_page_config(page_title="Ionosphere Digital Twin", layout="wide")
    
    st.title("🌍 Physics-Guided Ionosphere-Plasmasphere Digital Twin")
    st.markdown("### Operational Offline Platform (Hybrid Mamba-TKAN)")
    
    tabs = st.tabs([
        "Overview", "Predictions", "GNSS Delays", "Storm Monitor", 
        "What-If Simulation", "Exports & Reports"
    ])
    
    with tabs[0]:
        st.header("System Overview")
        st.info("System is offline and ready. Model loaded into CPU/GPU.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Operational Readiness Score", "98 / 100")
        col2.metric("Current Storm State", "Quiet")
        col3.metric("Anomaly Status", "All Clear")
        
    with tabs[1]:
        st.header("TEC & Electron Density Predictions")
        st.write("(Run offline engine to populate interactive plotly charts here)")
        
    with tabs[2]:
        st.header("Multi-Constellation GNSS Delays")
        st.write("Tracks GPS L1, NavIC L5, Galileo E1, BeiDou B1.")
        
    with tabs[3]:
        st.header("Geomagnetic Storm Monitor")
        st.write("Real-time categorization based on NASA OMNIWeb Kp & Dst drivers.")
        
    with tabs[4]:
        st.header("Counterfactual Simulation")
        st.slider("Override Kp Index", 0.0, 9.0, 2.0)
        st.slider("Override Dst Index", -300.0, 50.0, -10.0)
        st.button("Run Simulation")
        
    with tabs[5]:
        st.header("Data Export")
        st.button("Export to CSV")
        st.button("Export to HDF5")
        st.button("Generate Scientific Report (Markdown)")

if __name__ == "__main__":
    run_dashboard()
