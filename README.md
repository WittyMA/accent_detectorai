# English Accent Detector - Streamlit Cloud Version

A simplified application that analyzes speaker accents from video URLs, providing accent classification and confidence scores for English speakers. This version is specifically designed for Streamlit Cloud deployment.

## Overview

This application allows users to:

1. Input a public video URL (YouTube, Loom, direct MP4 links, etc.)
2. Extract audio from the video
3. Analyze the speaker's accent to detect English language speaking candidates
4. Receive results including:
   - Classification of the accent (e.g., British, American, Australian, etc.)
   - Confidence score for English accent (0-100%)
   - A brief explanation of the analysis

## Features

- **Video URL Input**: Accepts various video sources including YouTube, Vimeo, Loom, and direct video file links
- **Audio Extraction**: Automatically extracts audio from videos for analysis
- **Accent Detection**: Identifies various English accents using simplified analysis
- **Confidence Scoring**: Provides a confidence percentage for the accent classification
- **Streamlit Cloud Compatible**: Designed specifically to work on Streamlit Cloud
- **Docker Support**: Ready-to-use Dockerfile for containerized deployment

## Deployment Options

### Streamlit Cloud Deployment (Recommended)

1. Fork or clone this repository to your GitHub account
2. Sign up for a free account at [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app in Streamlit Cloud
4. Connect to your GitHub repository
5. Set the main file path to `streamlit_app.py`
6. Deploy with a single click

### Local Deployment

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/accent-detector-streamlit.git
   cd accent-detector-streamlit
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install ffmpeg (if not already installed):
   - Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

5. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

### Docker Deployment

1. Build the Docker image:
   ```
   docker build -t accent-detector-streamlit .
   ```

2. Run the container:
   ```
   docker run -p 8501:8501 accent-detector-streamlit
   ```

3. Access the application at `http://localhost:8501`

## Troubleshooting

### Common Issues

1. **Error: "Command 'yt-dlp' not found"**
   - Solution: Ensure yt-dlp is installed: `pip install yt-dlp`

2. **Error: "ffmpeg not found"**
   - Solution: Install ffmpeg and ensure it's in your system PATH

3. **Error: Video URL not accessible**
   - Solution: Verify the video URL is public and accessible

4. **Streamlit Cloud deployment fails**
   - Solution: Check the logs in Streamlit Cloud for specific error messages
   - Ensure all dependencies are in requirements.txt
   - Verify the main file path is set correctly

## Technical Details

This version uses:
- Direct yt-dlp commands for audio extraction (avoiding pydub dependency)
- Simplified file structure with a single entry point
- Minimal dependencies for maximum compatibility

## Limitations

- The system is designed for English accent detection only
- The simplified version uses a rule-based approach rather than ML models
- Performance may vary based on audio quality and background noise

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for video downloading
- [Streamlit](https://streamlit.io/) for the interactive UI framework
