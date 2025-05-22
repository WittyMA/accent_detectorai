"""
Streamlit app for English Accent Detection
"""

import os
import sys
import tempfile
import logging
from pathlib import Path
import streamlit as st

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import custom modules
from src.utils.video_processor import VideoProcessor
from src.models.accent_detector import AccentDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory for audio files
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize processors
video_processor = VideoProcessor(output_dir=UPLOAD_FOLDER)
accent_detector = AccentDetector()

def main():
    # Set page config
    st.set_page_config(
        page_title="English Accent Detector",
        page_icon="🎤",
        layout="centered"
    )
    
    # Header
    st.title("English Accent Detector")
    st.markdown("""
    This tool analyzes spoken English from videos to detect accent types and provide confidence scores.
    
    Enter a public video URL (YouTube, Loom, direct MP4 link, etc.) containing English speech to analyze the speaker's accent.
    """)
    
    # Input form
    with st.form("accent_form"):
        video_url = st.text_input("Video URL", placeholder="https://www.youtube.com/watch?v=...")
        submit_button = st.form_submit_button("Analyze Accent")
    
    # Process video URL when form is submitted
    if submit_button and video_url:
        try:
            with st.spinner("Processing video and analyzing accent..."):
                # Extract audio from video
                audio_path = video_processor.extract_audio_from_url(video_url)
                
                # Detect accent
                result = accent_detector.detect_accent(audio_path)
                
                # Display results
                st.success("Analysis complete!")
                
                # Display accent type
                st.subheader("Detected Accent")
                st.markdown(f"<h2 style='text-align: center;'>{result['accent']}</h2>", unsafe_allow_html=True)
                
                # Display confidence score
                st.subheader("Confidence Score")
                st.progress(result['confidence_score'] / 100)
                st.text(f"{result['confidence_score']}% confidence")
                
                # Display explanation
                st.subheader("Analysis Explanation")
                st.info(result['explanation'])
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.caption("REM Waste Accent Detection Tool | Built for hiring evaluation purposes")

if __name__ == "__main__":
    main()
