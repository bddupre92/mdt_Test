"""
Test Dashboard

A simple test dashboard to verify Streamlit functionality.
"""

import streamlit as st

# Set page configuration first
st.set_page_config(
    page_title="Test Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Test Dashboard")
    
    st.write("This is a test dashboard to verify Streamlit functionality.")
    
    # Add some test widgets
    st.sidebar.title("Test Controls")
    test_option = st.sidebar.selectbox(
        "Select an option",
        ["Option 1", "Option 2", "Option 3"]
    )
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Test Section 1")
        st.write(f"Selected option: {test_option}")
        
        if st.button("Click me!"):
            st.success("Button clicked!")
    
    with col2:
        st.header("Test Section 2")
        test_slider = st.slider("Test slider", 0, 100, 50)
        st.write(f"Slider value: {test_slider}")

if __name__ == "__main__":
    main() 