# ⚡ YouTube Summarizer

A high-performance YouTube video analysis platform with dual applications: a comprehensive main app and an ultra-fast version optimized for maximum speed. Features intelligent dual-mode operation with AI-powered analysis and lightweight fallback methods.

## 🚀 Available Applications

### 📱 Main Application (`streamlit_app.py`)
- **🎯 Comprehensive Analysis**: Full-featured video summarization
- **🌍 Multilingual Support**: Hindi translation and language detection
- **🤖 Dual-Mode Architecture**: AI features with intelligent fallbacks
- **📊 Rich Visualizations**: Charts, sentiment analysis, and content insights

### ⚡ Ultra-Fast Version (`app_ultra_fast.py`)
- **🚀 Speed Optimized**: Lightning-fast processing with parallel execution
- **📈 Performance Tracking**: Real-time speed metrics and processing history
- **🔄 Parallel Processing**: Multi-threaded analysis for maximum efficiency
- **⚡ Quick Analysis Mode**: Ultra-fast basic summaries when speed is critical

## ✨ Core Features

- **🎯 Smart Video Processing**: Automatically extracts YouTube transcripts and metadata
- **🌍 Multilingual Support**: Hindi translation and intelligent language detection
- **🤖 Dual-Mode Operation**: AI transcription for local development, fallback methods for cloud deployment
- **📊 Intelligent Summarization**: AI-powered summaries with extractive fallback
- **📈 Advanced Analytics**: Sentiment analysis, word clouds, and content categorization
- **⚡ Performance Options**: Choose between comprehensive analysis or ultra-fast processing
- **🎨 Modern Interface**: Clean, responsive Streamlit web applications

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ (recommended for best compatibility)
- Internet connection for YouTube access

### Easy Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/youtube-summarizer.git
   cd youtube-summarizer
   ```

2. **Run the setup script**
   ```bash
   python setup_local.py
   ```
   This will install dependencies and optionally configure AI features.

3. **Start the application**
   
   **Option A: Main Application (Comprehensive)**
   ```bash
   streamlit run streamlit_app.py
   ```
   
   **Option B: Ultra-Fast Version (Speed Optimized)**
   ```bash
   streamlit run app_ultra_fast.py
   ```
   
   **Option C: Use Windows launcher**
   ```bash
   start_app.bat
   ```

4. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Paste any YouTube URL and start analyzing!

### Manual Installation

If you prefer manual setup:

1. **Install basic dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Optional: Enable AI features**
   Uncomment the AI packages in `requirements.txt` and install:
   ```bash
   pip install transformers torch pytube pydub SpeechRecognition faster-whisper
   ```

## 🔧 How It Works

### Application Modes

**📱 Main Application (`streamlit_app.py`):**
- **🔄 Dual-Mode Operation**: AI mode with fallback capability
- **📊 Comprehensive Analysis**: Full feature set with rich visualizations
- **🌍 Translation Features**: Hindi translation integration
- **🎨 Enhanced UI**: Rich dashboard with multiple analysis sections

**⚡ Ultra-Fast Application (`app_ultra_fast.py`):**
- **🚀 Performance First**: Optimized for maximum processing speed
- **🔄 Parallel Processing**: Multi-threaded analysis pipeline
- **📈 Speed Tracking**: Real-time performance metrics
- **⚡ Quick Mode**: Ultra-fast basic summary option

### Processing Pipeline

**🤖 AI Mode (Local Development):**
1. **📥 Video Input**: Paste any YouTube URL
2. **🔍 Content Analysis**: Extracts video metadata and transcripts
3. **🎤 AI Transcription**: Uses Whisper for videos without subtitles
4. **🧠 Advanced Summarization**: Transformer-based summary generation
5. **📊 Enhanced Analysis**: AI-powered sentiment analysis and insights

**☁️ Fallback Mode (Cloud/Lightweight):**
1. **📥 Video Input**: Same URL processing
2. **📝 Transcript Extraction**: YouTube Transcript API only
3. **📊 Extractive Summarization**: TF-IDF based sentence scoring
4. **🎯 Basic Analysis**: Keyword-based sentiment and content analysis
5. **⚡ Fast Performance**: Lightweight processing for cloud deployment

## 📁 Project Structure

```
youtube-summarizer/
├── streamlit_app.py      # Main comprehensive application
├── app_ultra_fast.py     # Ultra-fast performance-optimized version
├── requirements.txt      # Dependencies (AI packages commented out)
├── setup_local.py        # Easy setup script with guided installation
├── start_app.bat        # Windows launcher for main app
├── README.md            # Project documentation
├── CLEANUP_COMPLETE.md  # Repository cleanup summary
├── .streamlit/          # Streamlit configuration
│   └── config.toml      # UI theme and settings
└── .gitignore           # Git ignore rules
```

## 🌟 Key Features

### Dual Application Architecture
- **📱 Main App**: Full-featured comprehensive analysis
- **⚡ Ultra-Fast**: Speed-optimized with parallel processing
- **🔄 Same Core**: Shared functionality with different optimizations
- **📊 Consistent UI**: Similar interface design across both apps
### Smart Content Processing
- **⚡ Performance Optimization**: Ultra-fast app with parallel processing
- **📈 Speed Tracking**: Real-time performance metrics and history
- **🌍 Hindi Translation**: Google Translate integration for multilingual support
- **📝 Quality Enhancement**: Automatic transcript cleanup and formatting
- **🎯 Content Categorization**: Technology, Business, Education, etc.
- **📈 Timeline Analysis**: Segment-based content breakdown

### Performance Features (Ultra-Fast App)
- **🔄 Parallel Workers**: Configurable multi-threading (1-6 workers)
- **🚀 Quick Analysis Mode**: Ultra-fast basic summaries
- **📊 Processing Speed**: Real-time words/second metrics
- **🎯 Optimized Models**: Smaller, faster AI models (distilbart-cnn-6-6)
- **⚡ GPU Support**: Optional GPU acceleration
- **📈 Performance History**: Track speed improvements over time

## 🛠️ Configuration

### Application Selection
- **Main App**: Use `streamlit_app.py` for comprehensive analysis
- **Ultra-Fast**: Use `app_ultra_fast.py` for maximum speed
- **Launcher**: `start_app.bat` runs the main application by default

### Environment Variables
- **AI Features**: Set `AI_TRANSCRIPTION_AVAILABLE=False` to disable AI features
- **Translation**: Set `TRANSLATION_AVAILABLE=False` to disable translation features

### Ultra-Fast App Settings
- **Parallel Workers**: Configure in sidebar (1-6 workers)
- **GPU Acceleration**: Enable/disable GPU usage
- **Quick Analysis**: Toggle ultra-fast mode for basic summaries
- **Performance Tracking**: Automatic speed history logging

## 🔍 Troubleshooting

### Common Solutions

**AI Features Not Available**
```bash
# Enable AI features by uncommenting packages in requirements.txt
pip install transformers torch pytube pydub SpeechRecognition faster-whisper
```

**Installation Issues**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Port Configuration**
```bash
# Main app on custom port
streamlit run streamlit_app.py --server.port 8502

# Ultra-fast app on custom port  
streamlit run app_ultra_fast.py --server.port 8503
```

**Performance Issues (Ultra-Fast App)**
- Check parallel worker settings in sidebar
- Enable GPU acceleration if available
- Use Quick Analysis mode for maximum speed
- Monitor performance history for optimization

**Cloud Mode Warning Messages**
- "Transformers not available" - Expected in cloud mode, fallback methods will be used
- "AI Transcription Not Available" - Normal for lightweight deployment

### Python Compatibility
- **Recommended**: Python 3.9 or 3.10 for best compatibility
- **Supported**: Python 3.8+ including Python 3.13
- **Cloud Deployment**: Use Python 3.9/3.10 for Streamlit Cloud

## 🚀 Deployment Options

### Local Development (Full Features)
1. Run `python setup_local.py`
2. Enable AI features when prompted
3. Choose your application:
   - **Main**: `streamlit run streamlit_app.py`
   - **Ultra-Fast**: `streamlit run app_ultra_fast.py`

### Performance Testing
1. Use the ultra-fast app for speed benchmarks
2. Monitor processing speed in words/second
3. Optimize parallel worker settings
4. Compare performance across different video types

### Cloud Deployment (Lightweight Mode)
1. Keep AI packages commented in `requirements.txt`
2. Deploy main app to any cloud platform
3. Fallback methods provide excellent compatibility
4. Consider ultra-fast app for high-volume scenarios

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit** - Modern web app framework for both applications
- **Hugging Face Transformers** - AI processing capabilities
- **yt-dlp & PyTube** - YouTube video processing
- **OpenAI Whisper** - Speech recognition technology
- **Concurrent.futures** - Parallel processing for ultra-fast performance
- **distilBART** - Lightweight summarization model for speed optimization

---

**⭐ If you find this project helpful, please give it a star on GitHub!**

**Made with ❤️ and ⚡ for the open source community**
