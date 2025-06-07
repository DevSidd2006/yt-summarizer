#!/usr/bin/env python3
"""
Enhanced YouTube Summarizer with AI Transcription Support
Complete single-file application for easy deployment
"""

# Configure Streamlit page FIRST before any other imports
import streamlit as st

st.set_page_config(
    page_title="Enhanced YouTube Summarizer",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CRITICAL: Python 3.13 compatibility fix - Must be FIRST before any imports
# Create cgi module for Python 3.13 compatibility (required by httpx -> googletrans)
import sys
if 'cgi' not in sys.modules:
    try:
        import cgi
    except ImportError:
        import types
        import re
        cgi = types.ModuleType('cgi')
        
        def escape(s, quote=False):
            """Escape HTML characters"""
            s = s.replace("&", "&amp;")
            s = s.replace("<", "&lt;")
            s = s.replace(">", "&gt;")
            if quote:
                s = s.replace('"', "&quot;")
                s = s.replace("'", "&#x27;")
            return s
        
        def parse_header(line):
            """Parse a Content-type like header (needed by googletrans)"""
            parts = line.split(';')
            main_type = parts[0].strip()
            pdict = {}
            for p in parts[1:]:
                i = p.find('=')
                if i >= 0:
                    name = p[:i].strip().lower()
                    value = p[i+1:].strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                        value = value[1:-1]
                    pdict[name] = value
            return main_type, pdict
        
        cgi.escape = escape
        cgi.parse_header = parse_header
        sys.modules['cgi'] = cgi

# Now safe to import everything else
from youtube_transcript_api import YouTubeTranscriptApi
import nltk
from nltk.tokenize import sent_tokenize
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import tempfile
import re
import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import seaborn as sns
from collections import Counter
import threading
from concurrent.futures import ThreadPoolExecutor
import time

# Enhanced imports for AI transcription and translation (optional)
import os
import warnings
warnings.filterwarnings("ignore")

# Check if running on Streamlit Cloud
IS_STREAMLIT_CLOUD = os.getenv('STREAMLIT_SHARING') or os.getenv('STREAMLIT_CLOUD_MODE') or 'streamlit.io' in os.getenv('HOSTNAME', '')

# AI/ML Dependencies - Make optional for cloud deployment
AI_TRANSCRIPTION_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
WHISPER_AVAILABLE = False
WHISPER_TYPE = None

# Try to import transformers (with cloud deployment safety)
try:
    if not IS_STREAMLIT_CLOUD:  # Only load on local development
        from transformers import pipeline
        TRANSFORMERS_AVAILABLE = True
        print("✅ transformers loaded successfully")
    else:
        print("⚠️ Skipping transformers on Streamlit Cloud to prevent PyTorch conflicts")
        TRANSFORMERS_AVAILABLE = False
except ImportError as transform_error:
    print(f"⚠️ transformers import failed: {transform_error}")
    TRANSFORMERS_AVAILABLE = False
except Exception as e:
    print(f"⚠️ transformers loading error: {e}")
    TRANSFORMERS_AVAILABLE = False

# Try to import AI transcription dependencies (optional for cloud)
try:
    if not IS_STREAMLIT_CLOUD:  # Only load on local development
        import speech_recognition as sr
        from pytube import YouTube
        from pydub import AudioSegment
        AI_TRANSCRIPTION_AVAILABLE = True
        
        # Try to import faster-whisper (more stable alternative)
        try:
            from faster_whisper import WhisperModel
            WHISPER_AVAILABLE = True
            WHISPER_TYPE = "faster"
            print("✅ faster-whisper loaded successfully")
        except ImportError as whisper_error:
            print(f"⚠️ faster-whisper import failed: {whisper_error}")
            WHISPER_AVAILABLE = False
            WHISPER_TYPE = None
    else:
        print("⚠️ Skipping AI transcription imports on Streamlit Cloud")
        AI_TRANSCRIPTION_AVAILABLE = False
        print("To enable AI transcription, install with: pip install faster-whisper")
        
except ImportError as e:
    AI_TRANSCRIPTION_AVAILABLE = False
    WHISPER_AVAILABLE = False
    WHISPER_TYPE = None

# Translation support (cgi module already created above for Python 3.13 compatibility)
try:
    from googletrans import Translator
    TRANSLATION_AVAILABLE = True
    TRANSLATOR_TYPE = 'googletrans'
    print("✅ Google Translate loaded successfully (using googletrans with compatibility fix)")
except ImportError:
    try:
        from deep_translator import GoogleTranslator
        TRANSLATION_AVAILABLE = True
        TRANSLATOR_TYPE = 'deep'
        print("✅ Google Translate loaded successfully (using deep-translator)")
    except ImportError:
        TRANSLATION_AVAILABLE = False
        TRANSLATOR_TYPE = None
        print("⚠️ Google Translate not available. Install with: pip install googletrans==4.0.0-rc1")
except ImportError:
    try:
        from deep_translator import GoogleTranslator
        TRANSLATION_AVAILABLE = True
        TRANSLATOR_TYPE = 'deep'
        print("✅ Google Translate loaded successfully (using deep-translator)")
    except ImportError:
        TRANSLATION_AVAILABLE = False
        TRANSLATOR_TYPE = None
        print("⚠️ Google Translate not available. Install with: pip install googletrans==4.0.0-rc1")
    print(f"⚠️ AI transcription packages not available: {e}")
    print("To enable AI transcription, install with: pip install pytube pydub speechrecognition faster-whisper")

# Download required NLTK data
try:
    nltk.download('punkt_tab')
except:
    nltk.download('punkt')

# Global cache for models to improve performance - OPTIMIZED
@st.cache_resource
def load_summarizer():
    """Load and cache the OPTIMIZED summarization model for better speed"""
    if not TRANSFORMERS_AVAILABLE:
        st.warning("⚠️ Transformers not available - summarization disabled for cloud deployment")
        return None
    
    # Use faster, smaller model instead of bart-large-cnn for better performance
    try:
        from transformers import pipeline
        return pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", device=-1)
    except Exception as e:
        st.warning(f"⚠️ Failed to load primary summarization model: {e}")
        try:
            # Fallback to even smaller model
            return pipeline("summarization", model="t5-small", device=-1)
        except Exception as e2:
            st.error(f"❌ All summarization models failed to load: {e2}")
            return None

@st.cache_resource  
def load_sentiment_analyzer():
    """Load and cache OPTIMIZED sentiment analysis model"""
    if not TRANSFORMERS_AVAILABLE:
        st.warning("⚠️ Transformers not available - sentiment analysis disabled for cloud deployment")
        return None
    
    try:
        from transformers import pipeline
        # Use a more efficient model
        return pipeline("sentiment-analysis", 
                       model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                       device=-1)
    except Exception as e:
        st.warning(f"⚠️ Failed to load sentiment analysis model: {e}")
        return None

@st.cache_resource
def load_whisper_model():
    """Load and cache Whisper model for audio transcription"""
    if not AI_TRANSCRIPTION_AVAILABLE or not WHISPER_AVAILABLE:
        return None
    try:
        if WHISPER_TYPE == "faster":
            # Use faster-whisper (more stable)
            return WhisperModel("base", device="cpu", compute_type="int8")
        elif WHISPER_TYPE == "openai":
            # Use openai-whisper
            import whisper
            return whisper.load_model("base")
        else:
            return None
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

@st.cache_resource
def load_translator():
    """Load and cache Google Translator with optimized initialization"""
    if not TRANSLATION_AVAILABLE:
        return None
    try:
        global TRANSLATOR_TYPE
        if TRANSLATOR_TYPE == 'deep':
            from deep_translator import GoogleTranslator
            return GoogleTranslator(source='auto', target='en')
        elif TRANSLATOR_TYPE == 'googletrans':
            from googletrans import Translator
            return Translator()
        else:
            return None
    except Exception as e:
        st.warning(f"⚠️ Error loading translator: {e}")
        return None

# Session state for translator status to avoid repeated testing
def get_translator_with_status():
    """Get translator and test it only once per session with enhanced diagnostics"""
    if 'translator_tested' not in st.session_state:
        st.session_state.translator_tested = False
        st.session_state.translator_working = False
        st.session_state.translator_warning_shown = False
        st.session_state.translator_error_details = None
    
    translator = load_translator()
    if not translator:
        st.session_state.translator_error_details = "Failed to load translator object"
        return None
      # Test translator only once per session
    if not st.session_state.translator_tested:
        try:
            # Use threading for timeout on Windows
            import threading
            import time
            
            test_result = [None]
            test_error = [None]
            
            def test_translation():
                try:
                    global TRANSLATOR_TYPE
                    if TRANSLATOR_TYPE == 'deep':
                        # deep-translator API - returns string directly
                        result = translator.translate("test")
                        # Create a result object with text attribute for compatibility
                        test_result[0] = type('Result', (), {'text': result})()
                    elif TRANSLATOR_TYPE == 'googletrans':
                        # googletrans API - returns object with text attribute
                        result = translator.translate("test", dest='en')
                        test_result[0] = result
                    else:
                        raise Exception("Unknown translator type")
                except Exception as e:
                    test_error[0] = e
            
            # Run translation test in a separate thread with timeout
            test_thread = threading.Thread(target=test_translation)
            test_thread.daemon = True
            test_thread.start()
            test_thread.join(timeout=10)  # 10-second timeout
            
            if test_thread.is_alive():
                # Test timed out
                st.session_state.translator_working = False
                st.session_state.translator_error_details = "Translator test timed out (network issues)"
            elif test_error[0]:
                # Test failed with error
                raise test_error[0]
            elif test_result[0] and hasattr(test_result[0], 'text'):
                # Test successful
                st.session_state.translator_working = True
                st.session_state.translator_error_details = None
            else:
                # Test returned invalid result
                st.session_state.translator_working = False
                st.session_state.translator_error_details = "Translator returned invalid result"
                    
        except Exception as e:
            st.session_state.translator_working = False
            error_msg = str(e).lower()
            if "timeout" in error_msg or "connection" in error_msg or "network" in error_msg:
                st.session_state.translator_error_details = f"Network connectivity issue: {str(e)[:100]}"
            elif "403" in error_msg or "forbidden" in error_msg:
                st.session_state.translator_error_details = "Google Translate access blocked (403 error)"
            elif "429" in error_msg or "rate" in error_msg or "limit" in error_msg:
                st.session_state.translator_error_details = "Rate limit exceeded - try again later"
            elif "ssl" in error_msg or "certificate" in error_msg:
                st.session_state.translator_error_details = "SSL/Certificate issue - check network security"
            else:
                st.session_state.translator_error_details = f"Translation service error: {str(e)[:100]}"
        finally:
            st.session_state.translator_tested = True
    
    return translator if st.session_state.translator_working else None

# Separate function for testing translator (called only when explicitly needed)
def test_translator_connection():
    """Test translator connection - only call when explicitly needed"""
    try:
        translator = load_translator()
        if not translator:
            return False
        
        test_result = translator.translate("test", dest='en')
        return test_result and hasattr(test_result, 'text')
    except:
        return False

# ----------------------
# Enhanced Subtitle Functions
# ----------------------

def enhance_transcript_quality(transcript_data):
    """Enhance transcript quality by fixing common issues"""
    try:
        enhanced_transcript = []
        
        for entry in transcript_data:
            # Handle both FetchedTranscriptSnippet objects and dictionaries
            if hasattr(entry, 'text'):
                text = entry.text
                start = entry.start
                duration = getattr(entry, 'duration', 0)
            elif isinstance(entry, dict):
                text = entry.get('text', '')
                start = entry.get('start', 0)
                duration = entry.get('duration', 0)
            else:
                continue
            
            # Common auto-caption fixes
            text = fix_common_transcription_errors(text)
            
            # Skip very short or meaningless segments
            if len(text.strip()) < 3 or text.strip() in ['[Music]', '[Applause]', '[Laughter]', '...']:
                continue
            
            enhanced_transcript.append({
                'start': start,
                'duration': duration,
                'text': text
            })
        
        return enhanced_transcript
        
    except Exception as e:
        st.warning(f"Could not enhance transcript quality: {e}")
        return transcript_data

def fix_common_transcription_errors(text):
    """Fix common transcription and auto-caption errors"""
    # Common word corrections
    corrections = {
        # Technical terms
        ' ai ': ' AI ',
        ' api ': ' API ',
        ' ui ': ' UI ',
        ' ux ': ' UX ',
        ' ceo ': ' CEO ',
        ' cto ': ' CTO ',
        ' seo ': ' SEO ',
        ' pov ': ' POV ',
        ' roi ': ' ROI ',
        ' kpi ': ' KPI ',
        ' html ': ' HTML ',
        ' css ': ' CSS ',
        ' js ': ' JavaScript ',
        ' sql ': ' SQL ',
        
        # Common misheard words
        'definately': 'definitely',
        'seperate': 'separate',
        'occured': 'occurred',
        'recieve': 'receive',
        'theyre': "they're",
        'youre': "you're",
        'its ': "it's ",
        'dont': "don't",
        'cant': "can't",
        'wont': "won't",
        'didnt': "didn't",
        'wasnt': "wasn't",
        'isnt': "isn't",
        'arent': "aren't",
        'werent': "weren't",
        'wouldnt': "wouldn't",
        'shouldnt': "shouldn't",
        'couldnt': "couldn't",
        
        # YouTube-specific corrections
        'youtube': 'YouTube',
        'google': 'Google',
        'facebook': 'Facebook',
        'instagram': 'Instagram',
        'twitter': 'Twitter',
        'linkedin': 'LinkedIn',
        'tiktok': 'TikTok',
        'snapchat': 'Snapchat',
        
        # Remove filler words and noise
        ' um ': ' ',
        ' uh ': ' ',
        ' uhm ': ' ',
        ' er ': ' ',
        ' ah ': ' ',
        'you know ': '',
    }
    
    # Apply corrections
    text = ' ' + text.lower() + ' '  # Add spaces for word boundary matching
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    # Remove multiple spaces
    text = ' '.join(text.split())
    
    # Capitalize first letter of sentences
    sentences = text.split('. ')
    sentences = [s.strip().capitalize() for s in sentences if s.strip()]
    text = '. '.join(sentences)
    
    # Fix common punctuation issues
    text = re.sub(r'\s+([,.!?])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([,.!?])\s*([a-zA-Z])', r'\1 \2', text)  # Add space after punctuation
    
    return text.strip()

def get_enhanced_transcript(video_id, video_url, preferred_languages=None, enable_auto_translate=True, show_original_language=True):
    """Get transcript with multiple fallback methods and enhanced Hindi detection"""
    transcript_methods = []
    
    # Initialize default preferred languages if not provided
    if preferred_languages is None:
        preferred_languages = ["en"]
      # Method 1: Check for official English subtitles first (highest priority)
    try:
        st.info("🔍 Checking for official English subtitles...")
          # Enhanced error handling for YouTube API issues
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        except Exception as api_error:
            error_msg = str(api_error).lower()
            if "no element found" in error_msg or "xml" in error_msg:
                st.error("❌ **YouTube Transcript Access Restricted**")
                st.warning("⚠️ This video has restricted transcript access. Common causes:")
                st.info("   • **Age-restricted content** - requires login to access")
                st.info("   • **Private or unlisted video** - limited access permissions")
                st.info("   • **Regional restrictions** - video not available in your location")
                st.info("   • **Channel restrictions** - creator has disabled transcript access")
                st.info("   • **Copyright content** - automated systems may block transcript access")
                
                # Try direct fallback methods
                st.info("🔄 **Attempting alternative extraction methods...**")
                return try_alternative_transcript_methods(video_id, video_url)
                
            elif "could not retrieve" in error_msg or "transcript" in error_msg:
                st.error("❌ **No Transcripts Available**")
                st.info("💡 This video may not have subtitles or captions enabled.")
                st.info("🎤 **Suggested Solutions:**")
                st.info("   • Try using **AI Transcription** (enable in sidebar)")
                st.info("   • Check if video has manual captions")
                st.info("   • Try a different video from the same channel")
                
                # Try AI transcription if available
                if AI_TRANSCRIPTION_AVAILABLE:
                    st.info("🤖 **Attempting AI transcription as fallback...**")
                    return try_ai_transcription_fallback(video_id, video_url)
                else:
                    raise Exception("No transcripts available and AI transcription not installed")
                    
            elif "400" in error_msg or "bad request" in error_msg:
                st.error("❌ **HTTP 400 Error - Bad Request**")
                st.warning("⚠️ YouTube is blocking the transcript request. This can happen with:")
                st.info("   • **Restricted videos** - age-restricted or private content")
                st.info("   • **Rate limiting** - too many requests in short time")
                st.info("   • **Video access issues** - temporary YouTube restrictions")
                st.info("🔄 **Try Again:** Wait a few minutes and retry, or try a different video")
                raise Exception("YouTube transcript access blocked (HTTP 400)")
                
            else:
                st.warning(f"⚠️ YouTube API error: {str(api_error)[:100]}...")
                st.info("🔄 Attempting alternative transcript extraction methods...")
                return try_alternative_transcript_methods(video_id, video_url)
        
        # Show available languages if requested
        if show_original_language:
            available_languages = get_available_transcript_languages(video_id)
            if available_languages:
                lang_info = []
                for lang in available_languages[:5]:  # Show first 5 languages
                    status = "🤖" if lang['is_generated'] else "📝"
                    lang_info.append(f"{status} {lang['name']} ({lang['code']})")
                st.info(f"📋 Available languages: {', '.join(lang_info)}")
        
        # First, look for official English subtitles
        for transcript_info in transcript_list:
            if transcript_info.language_code.startswith('en') and not transcript_info.is_generated:
                try:
                    transcript = transcript_info.fetch()
                    if transcript and len(transcript) > 0:
                        enhanced = enhance_transcript_quality(transcript)
                        transcript_methods.append((f"Official English Subtitles (Enhanced)", enhanced))
                        st.success(f"✅ Found official English subtitles - using highest quality source")
                        return enhanced
                except Exception as e:
                    st.warning(f"Failed to fetch official English subtitles: {e}")
                    continue
        
        # If no official English, check for auto-generated English
        for transcript_info in transcript_list:
            if transcript_info.language_code.startswith('en') and transcript_info.is_generated:
                try:
                    transcript = transcript_info.fetch()
                    if transcript and len(transcript) > 0:
                        enhanced = enhance_transcript_quality(transcript)
                        transcript_methods.append((f"Auto-Generated English (Enhanced)", enhanced))
                        st.success(f"✅ Found auto-generated English captions")
                        return enhanced
                except Exception as e:
                    st.warning(f"Failed to fetch auto-generated English: {e}")
                    continue
          # If no English found, check if any available transcript is actually English before proceeding with Hindi
        st.info("🔍 Checking for English content in available transcripts...")
        
        # Check if any transcript contains English content (to avoid translating English content as Hindi)
        for transcript_info in transcript_list:
            try:
                # Skip if we already processed this language
                if transcript_info.language_code.startswith('en'):
                    continue
                    
                transcript = transcript_info.fetch()
                if transcript and len(transcript) > 0:
                    # Detect language of this transcript
                    detected_lang = detect_transcript_language(transcript)
                    
                    # If detected as English, use it directly
                    if detected_lang == 'en':
                        st.info(f"🇺🇸 Detected English content in {getattr(transcript_info, 'language', transcript_info.language_code)} transcript!")
                        enhanced = enhance_transcript_quality(transcript)
                        transcript_methods.append((f"English Content (from {getattr(transcript_info, 'language', transcript_info.language_code)})", enhanced))
                        st.success(f"✅ Using English content without translation")
                        return enhanced
                        
            except Exception as e:
                continue
        
        # If no English found, check for Hindi subtitles (both official and auto-generated)
        st.info("🇮🇳 No English subtitles found. Checking for Hindi content...")
        hindi_transcript = None
        hindi_source_type = None
        
        # Try all Hindi language variations and also check for auto-translate capability
        hindi_codes = ['hi', 'hi-IN', 'hi-Latn']  # Include transliterated Hindi
        
        # First try official Hindi subtitles
        for transcript_info in transcript_list:
            if transcript_info.language_code in hindi_codes and not transcript_info.is_generated:
                try:
                    transcript = transcript_info.fetch()
                    if transcript and len(transcript) > 0:
                        hindi_transcript = transcript
                        hindi_source_type = f"Official Hindi Subtitles"
                        st.success(f"✅ Found official Hindi subtitles")
                        break
                except Exception as e:
                    st.warning(f"Failed to fetch official Hindi subtitles: {e}")
                    continue        # If no official Hindi, try auto-generated Hindi
        if not hindi_transcript:
            for transcript_info in transcript_list:
                if transcript_info.language_code in hindi_codes and transcript_info.is_generated:
                    try:
                        # Try to get manually created transcript first, then auto-generated
                        transcript = transcript_info.fetch()
                        if transcript and len(transcript) > 0:
                            hindi_transcript = transcript
                            hindi_source_type = f"Auto-Generated Hindi Captions"
                            st.success(f"✅ Found auto-generated Hindi captions")
                            break
                    except Exception as e:
                        st.warning(f"Failed to fetch auto-generated Hindi: {e}")
                        # Try to get translated version if available
                        try:
                            if hasattr(transcript_info, 'translate'):
                                english_transcript = transcript_info.translate('en').fetch()
                                if english_transcript and len(english_transcript) > 0:
                                    enhanced = enhance_transcript_quality(english_transcript)
                                    transcript_methods.append((f"Hindi → English (Auto-translated)", enhanced))
                                    st.success(f"✅ Found auto-translated Hindi to English captions")
                                    return enhanced
                        except Exception as translate_error:
                            st.warning(f"Auto-translation also failed: {translate_error}")
                        continue
          # If still no Hindi found, try to use YouTube's auto-translate feature
        if not hindi_transcript:
            st.info("🔄 Trying YouTube's auto-translate feature...")
            for transcript_info in transcript_list:
                try:
                    # Try to translate any available transcript to English
                    if hasattr(transcript_info, 'translate'):
                        english_transcript = transcript_info.translate('en').fetch()
                        if english_transcript and len(english_transcript) > 0:
                            enhanced = enhance_transcript_quality(english_transcript)
                            source_lang = getattr(transcript_info, 'language', transcript_info.language_code)
                            transcript_methods.append((f"{source_lang} → English (YouTube Auto-translate)", enhanced))
                            st.success(f"✅ Used YouTube auto-translate from {source_lang} to English")
                            return enhanced
                except Exception as e:
                    continue
        
        # Try to get ANY transcript and detect if it's Hindi
        if not hindi_transcript:
            st.info("🔍 Checking all available transcripts for Hindi content...")
            for transcript_info in transcript_list:
                try:
                    transcript = transcript_info.fetch()
                    if transcript and len(transcript) > 0:
                        # First check if this transcript is actually English
                        detected_lang = detect_transcript_language(transcript)
                        
                        # If detected as English, use it directly
                        if detected_lang == 'en':
                            st.info(f"🇺🇸 Found English content in {getattr(transcript_info, 'language', transcript_info.language_code)} transcript!")
                            enhanced = enhance_transcript_quality(transcript)
                            transcript_methods.append((f"English Content (from {getattr(transcript_info, 'language', transcript_info.language_code)})", enhanced))
                            st.success(f"✅ Using English content without translation")
                            return enhanced
                        
                        # Only check for Hindi if not English
                        if detected_lang != 'en':
                            is_hindi, hindi_info = detect_hindi_content(transcript)
                            if is_hindi:
                                st.info(f"🇮🇳 Detected Hindi content in {getattr(transcript_info, 'language', transcript_info.language_code)} transcript")
                                hindi_transcript = transcript
                                hindi_source_type = f"Hindi Content in {getattr(transcript_info, 'language', transcript_info.language_code)} Transcript"
                                break
                except Exception as e:
                    continue
          # If Hindi transcript found, translate it to English
        if hindi_transcript and TRANSLATION_AVAILABLE and enable_auto_translate:
            st.info(f"🔄 Processing {hindi_source_type}...")
            
            # Enhance quality first
            enhanced_hindi = enhance_transcript_quality(hindi_transcript)
            
            # Show Hindi sample
            sample_text = ""
            for entry in enhanced_hindi[:3]:
                if hasattr(entry, 'text'):
                    sample_text += entry.text + " "
                elif isinstance(entry, dict):
                    sample_text += entry.get('text', '') + " "
            
            if sample_text.strip():
                with st.expander("🔍 Original Hindi Sample", expanded=False):
                    st.text_area("Original Hindi Text:", sample_text[:300], height=80, disabled=True, key="hindi_sample_1")
            
            # Translate Hindi to English
            translated_transcript = translate_hindi_to_english(enhanced_hindi, show_progress=True)
            
            if translated_transcript:
                transcript_methods.append((f"{hindi_source_type} → English Translation", translated_transcript))
                st.success(f"✅ Successfully translated Hindi content to English")
                return translated_transcript
            else:
                st.warning("⚠️ Hindi translation failed, trying other languages...")
        
        # If no Hindi or translation failed, try other preferred languages
        for preferred_lang in preferred_languages:
            if preferred_lang not in ['en', 'hi']:  # Skip already processed languages
                for transcript_info in transcript_list:
                    if transcript_info.language_code.startswith(preferred_lang):
                        try:
                            transcript = transcript_info.fetch()
                            if transcript and len(transcript) > 0:
                                if TRANSLATION_AVAILABLE and enable_auto_translate:
                                    st.info(f"🔄 Translating {transcript_info.language} to English...")
                                    translated_transcript = translate_transcript_to_english(transcript, transcript_info.language_code)
                                    if translated_transcript:
                                        enhanced = enhance_transcript_quality(translated_transcript)
                                        transcript_methods.append((f"Translated from {transcript_info.language} to English", enhanced))
                                        st.success(f"✅ Successfully translated {transcript_info.language} to English")
                                        return enhanced
                                else:
                                    enhanced = enhance_transcript_quality(transcript)
                                    transcript_methods.append((f"Original {transcript_info.language}", enhanced))
                                    st.warning(f"⚠️ Using original {transcript_info.language} (translation disabled)")
                                    return enhanced
                        except Exception as e:
                            st.warning(f"Failed to process {transcript_info.language}: {e}")
                            continue
          # Final fallback: try any available transcript with better error handling
        if not transcript_methods:
            st.info("🌍 Trying any available language...")
            for transcript_info in transcript_list:
                try:
                    # Try to fetch the transcript
                    transcript = transcript_info.fetch()
                    if transcript and len(transcript) > 0:
                        lang_code = transcript_info.language_code
                        lang_name = getattr(transcript_info, 'language', lang_code)
                        
                        # If it's not English, try to translate
                        if not lang_code.startswith('en') and TRANSLATION_AVAILABLE and enable_auto_translate:
                            st.info(f"🔄 Found {lang_name} subtitles, translating to English...")
                            translated_transcript = translate_transcript_to_english(transcript, lang_code)
                            if translated_transcript:
                                enhanced = enhance_transcript_quality(translated_transcript)
                                transcript_methods.append((f"Translated from {lang_name} to English", enhanced))
                                st.success(f"✅ Successfully translated {lang_name} to English")
                                return enhanced
                        
                        # If translation not available or failed, use original
                        enhanced = enhance_transcript_quality(transcript)
                        transcript_methods.append((f"Original {lang_name}", enhanced))
                        st.warning(f"⚠️ Using original {lang_name} (translation {'disabled' if not TRANSLATION_AVAILABLE else 'failed'})")
                        return enhanced
                except Exception as transcript_error:
                    st.warning(f"Failed to process {getattr(transcript_info, 'language', 'unknown')}: {transcript_error}")
                    continue                    
    except Exception as e:
        error_msg = str(e).lower()
        if "no element found" in error_msg or "xml" in error_msg:
            st.error("❌ YouTube transcript API error: Video may have restricted access or malformed subtitle data")
            st.info("💡 This often happens with:")
            st.info("   • Age-restricted videos")
            st.info("   • Recently uploaded videos")
            st.info("   • Videos with limited subtitle access")
        elif "no transcripts available" in error_msg:
            st.error("❌ No transcripts available for this video")
            st.info("💡 This video may not have subtitles or captions")
        else:
            st.warning(f"Could not list available transcripts: {str(e)[:100]}...")
        
        # If transcript listing failed, try direct methods as fallback
        st.info("🔄 Trying alternative transcript extraction methods...")
        
        # Try direct transcript fetch without listing first
        try:
            st.info("🔍 Attempting direct transcript fetch...")
            direct_transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            if direct_transcript:
                enhanced = enhance_transcript_quality(direct_transcript)
                transcript_methods.append(("Direct English Transcript", enhanced))
                st.success("✅ Successfully retrieved transcript via direct method")
                return enhanced
        except Exception as direct_error:
            st.warning(f"Direct transcript fetch also failed: {str(direct_error)[:50]}...")
        
        # Try auto-generated if direct failed
        try:
            st.info("🔍 Trying auto-generated transcripts...")
            auto_transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-US', 'en-GB', 'en'])
            if auto_transcript:
                enhanced = enhance_transcript_quality(auto_transcript)
                transcript_methods.append(("Auto-Generated English", enhanced))
                st.success("✅ Found auto-generated transcript")
                return enhanced
        except Exception as auto_error:
            st.warning(f"Auto-generated transcript fetch failed: {str(auto_error)[:50]}...")
    
    # Method 2: Use AI transcription (Whisper) as final fallback
    if not transcript_methods and AI_TRANSCRIPTION_AVAILABLE:
        st.warning("🎵 No subtitles found. Attempting AI transcription...")
        
        with st.spinner("Downloading audio for transcription..."):
            audio_path, error = download_audio_from_youtube(video_url, video_id)
            
            if audio_path and not error:
                whisper_transcript, whisper_error = transcribe_with_whisper(audio_path, video_id)
                
                if whisper_transcript and not whisper_error:
                    # Check language of AI transcript
                    if TRANSLATION_AVAILABLE and enable_auto_translate:
                        detected_lang = detect_transcript_language(whisper_transcript)
                        
                        # If already English, use directly without translation
                        if detected_lang == 'en':
                            st.info("🇺🇸 AI detected English audio - using directly without translation")
                            transcript_methods.append(("AI Transcription (English)", whisper_transcript))
                            st.success("✅ AI transcription completed (English)")
                            return whisper_transcript
                        
                        # Only translate if detected as Hindi
                        elif detected_lang == 'hi':
                            st.info("🇮🇳 AI detected Hindi audio, translating to English...")
                            translated_ai = translate_hindi_to_english(whisper_transcript, show_progress=True)
                            if translated_ai:
                                st.success("✅ AI transcription and Hindi translation completed")
                                return translated_ai
                        
                        # For other languages, show info but use original
                        else:
                            st.info(f"🌍 AI detected {detected_lang.upper()} audio - using original transcript")
                    
                    transcript_methods.append(("AI Transcription (Whisper)", whisper_transcript))
                    st.success("✅ AI transcription completed")
                    return whisper_transcript
                else:
                    st.error(f"AI transcription failed: {whisper_error}")
            else:
                st.error(f"Audio download failed: {error}")
    elif not transcript_methods:
        st.error("❌ No transcript methods available. Please install AI transcription packages or try a video with subtitles.")
        return "Error: No transcript could be obtained from any method"
    
    # Return the best available transcript
    if transcript_methods:
        method_name, transcript = transcript_methods[0]
        st.success(f"✅ Transcript obtained via: {method_name}")
        return transcript
    else:
        return "Error: No transcript could be obtained from any method"

def assess_transcript_quality(transcript_data):
    """Assess the quality of transcript and provide recommendations"""
    if isinstance(transcript_data, str):
        return {"quality": "unknown", "recommendations": ["Text-based input provided"]}
    
    if not transcript_data or len(transcript_data) == 0:
        return {"quality": "poor", "recommendations": ["No transcript data available"]}
    
    # Calculate quality metrics
    total_entries = len(transcript_data)
    
    # Handle both dictionary format and FetchedTranscriptSnippet objects
    text_parts = []
    for entry in transcript_data:
        try:
            if hasattr(entry, 'text'):
                text_parts.append(entry.text)
            elif isinstance(entry, dict) and 'text' in entry:
                text_parts.append(entry['text'])
            else:
                text_parts.append(str(entry))
        except:
            continue
    
    total_text = " ".join(text_parts)
    avg_segment_length = len(total_text.split()) / total_entries if total_entries > 0 else 0
    
    # Check for common quality indicators
    quality_score = 0
    recommendations = []
    
    # Length check
    if avg_segment_length > 3:
        quality_score += 25
    else:
        recommendations.append("Segments are very short - may indicate poor audio quality")
    
    # Coherence check (basic)
    if len(total_text.split()) > 100:
        quality_score += 25
    else:
        recommendations.append("Very short transcript - video may have limited speech")
    
    # Check for noise indicators
    noise_indicators = ['[Music]', '[Applause]', '[Laughter]', '...', 'um', 'uh']
    noise_count = sum(total_text.lower().count(indicator) for indicator in noise_indicators)
    
    if noise_count < len(total_text.split()) * 0.1:  # Less than 10% noise
        quality_score += 25
    else:
        recommendations.append("High noise level detected - consider manual review")
      # Timing consistency check
    if len(transcript_data) > 1:
        time_gaps = []
        for i in range(1, len(transcript_data)):
            try:
                # Handle both object attributes and dictionary access
                if hasattr(transcript_data[i], 'start'):
                    current_start = transcript_data[i].start
                    prev_start = transcript_data[i-1].start
                    prev_duration = getattr(transcript_data[i-1], 'duration', 0)
                else:
                    current_start = transcript_data[i]['start']
                    prev_start = transcript_data[i-1]['start']
                    prev_duration = transcript_data[i-1].get('duration', 0)
                
                gap = current_start - (prev_start + prev_duration)
                time_gaps.append(gap)
            except:
                continue
        
        avg_gap = sum(time_gaps) / len(time_gaps) if time_gaps else 0
        if avg_gap < 5:  # Less than 5-second gaps on average
            quality_score += 25
        else:
            recommendations.append("Large timing gaps detected - transcript may be incomplete")
    
    # Determine overall quality
    if quality_score >= 75:
        quality = "excellent"
    elif quality_score >= 50:
        quality = "good"
    elif quality_score >= 25:
        quality = "fair"
    else:
        quality = "poor"
        recommendations.append("Consider using AI transcription for better results")
    
    return {"quality": quality, "score": quality_score, "recommendations": recommendations}

# ----------------------
# AI Transcription Functions (Optional)
# ----------------------

def download_audio_from_youtube(video_url, video_id):
    """Download audio from YouTube video with enhanced error handling"""
    if not AI_TRANSCRIPTION_AVAILABLE:
        return None, "AI transcription packages not available"
    
    try:
        st.info("🎵 Initializing YouTube audio downloader...")
        
        # Enhanced YouTube object creation with better error handling
        try:
            yt = YouTube(video_url, use_oauth=False, allow_oauth_cache=False)
        except Exception as yt_error:
            error_msg = str(yt_error).lower()
            if "400" in error_msg or "bad request" in error_msg:
                return None, "HTTP 400 Error: YouTube blocked the request. This video may be age-restricted, private, or have download restrictions."
            elif "403" in error_msg or "forbidden" in error_msg:
                return None, "HTTP 403 Error: Access forbidden. Video may be private or geo-restricted."
            elif "404" in error_msg or "not found" in error_msg:
                return None, "HTTP 404 Error: Video not found. Check if the video ID is correct and the video exists."
            else:
                return None, f"YouTube connection error: {str(yt_error)[:100]}"
        
        # Check video availability
        try:
            title = yt.title
            st.info(f"📹 Found video: {title[:50]}...")
        except Exception as title_error:
            st.warning("⚠️ Could not verify video details, proceeding with caution...")
        
        # Get available audio streams with better filtering
        try:
            st.info("🔍 Searching for audio streams...")
            audio_streams = yt.streams.filter(only_audio=True, file_extension='mp4').order_by('abr').desc()
            
            if not audio_streams:
                # Try alternative audio formats
                audio_streams = yt.streams.filter(only_audio=True).order_by('abr').desc()
                
            if not audio_streams:
                # Last resort: try any stream with audio
                audio_streams = yt.streams.filter(adaptive=True, mime_type="audio/mp4")
                
            audio_stream = audio_streams.first() if audio_streams else None
            
            if not audio_stream:
                return None, "No audio streams available. Video may not have downloadable audio or may be restricted."
                
            st.info(f"🎵 Selected audio stream: {audio_stream.mime_type} at {audio_stream.abr or 'unknown'} quality")
            
        except Exception as stream_error:
            return None, f"Failed to get audio streams: {str(stream_error)}"
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, f"{video_id}.mp4")
        
        # Download audio with progress tracking
        try:
            st.info("⬇️ Downloading audio... (this may take a few minutes)")
            audio_stream.download(output_path=temp_dir, filename=f"{video_id}.mp4")
            
            # Verify download success
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:  # At least 1KB
                st.success("✅ Audio download completed successfully")
                return audio_path, None
            else:
                return None, "Download completed but file is empty or corrupted"
                
        except Exception as download_error:
            error_msg = str(download_error).lower()
            if "400" in error_msg:
                return None, "Download failed with HTTP 400: YouTube may have blocked the request due to rate limiting or video restrictions"
            elif "403" in error_msg:
                return None, "Download failed with HTTP 403: Access denied, video may be restricted"
            elif "timeout" in error_msg:
                return None, "Download timeout: Video may be too long or connection is slow"
            else:
                return None, f"Download failed: {str(download_error)}"
        
    except Exception as e:
        return None, f"Unexpected error in audio download: {str(e)}"

def convert_audio_for_whisper(audio_path):
    """Convert audio to format suitable for Whisper"""
    try:
        # Load audio file
        audio = AudioSegment.from_file(audio_path)
        
        # Convert to WAV format with 16kHz sample rate (Whisper's preferred format)
        wav_path = audio_path.replace('.mp4', '.wav')
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")
        
        return wav_path
        
    except Exception as e:
        return None

def transcribe_with_whisper(audio_path, video_id):
    """Transcribe audio using OpenAI Whisper or faster-whisper"""
    if not AI_TRANSCRIPTION_AVAILABLE or not WHISPER_AVAILABLE:
        return None, "AI transcription packages not available"
        
    try:
        whisper_model = load_whisper_model()
        if not whisper_model:
            return None, "Whisper model not available"
        
        st.info("🎤 Transcribing audio with AI (this may take a few minutes)...")
        
        # Convert audio format
        wav_path = convert_audio_for_whisper(audio_path)
        if not wav_path:
            return None, "Audio conversion failed"
        
        # Transcribe with appropriate Whisper implementation
        if WHISPER_TYPE == "faster":
            # Use faster-whisper
            segments, info = whisper_model.transcribe(wav_path, beam_size=5)
            
            # Convert faster-whisper output to YouTube transcript format
            transcript_data = []
            for segment in segments:
                transcript_data.append({
                    'start': segment.start,
                    'duration': segment.end - segment.start,
                    'text': segment.text.strip()
                })
        else:
            # Use openai-whisper
            result = whisper_model.transcribe(wav_path, verbose=False)
            
            # Convert Whisper output to YouTube transcript format
            transcript_data = []
            for segment in result.get('segments', []):
                transcript_data.append({
                    'start': segment['start'],
                    'duration': segment['end'] - segment['start'],
                    'text': segment['text'].strip()
                })
          # Clean up temporary files
        try:
            os.remove(audio_path)
            os.remove(wav_path)
            os.rmdir(os.path.dirname(audio_path))
        except:
            pass
        
        return transcript_data, None
        
    except Exception as e:
        return None, f"Whisper transcription failed: {str(e)}"

# ----------------------
# Translation Functions
# ----------------------

def detect_transcript_language(transcript_data):
    """Detect the language of the transcript"""
    if not TRANSLATION_AVAILABLE:
        return "unknown"
    
    try:
        translator = get_translator_with_status()
        if not translator:
            return "unknown"
        
        # Get a sample of text for language detection
        sample_text = ""
        count = 0
        for entry in transcript_data:
            if hasattr(entry, 'text'):
                sample_text += entry.text + " "
            elif isinstance(entry, dict):
                sample_text += entry.get('text', '') + " "
            
            count += 1
            if count >= 5 or len(sample_text) > 200:  # Use first 5 entries or 200 chars
                break
        
        if sample_text.strip():
            detected = translator.detect(sample_text)
            return detected.lang if detected else "unknown"
        else:
            return "unknown"
            
    except Exception as e:
        print(f"Language detection failed: {e}")
        return "unknown"

def translate_text_batch(texts, target_lang='en', source_lang=None):
    """Translate a batch of texts efficiently"""
    if not TRANSLATION_AVAILABLE:
        return texts
    
    try:
        translator = get_translator_with_status()
        if not translator:
            return texts
        
        translated_texts = []
          # Process in batches to avoid API limits
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_translated = []
            
            for text in batch:
                if len(text.strip()) == 0:
                    batch_translated.append(text)
                    continue
                
                try:
                    if source_lang:
                        result = translator.translate(text, src=source_lang, dest=target_lang)
                    else:
                        result = translator.translate(text, dest=target_lang)
                    
                    # Handle both regular string results and coroutine objects
                    if result:
                        translated_text = result.text if hasattr(result, 'text') else str(result)
                        # Ensure we have actual text, not a coroutine
                        if hasattr(translated_text, '__await__') or 'coroutine' in str(type(translated_text)):
                            translated_text = text  # Fallback to original
                        batch_translated.append(translated_text)
                    else:
                        batch_translated.append(text)
                    
                    # Small delay to be respectful to the API
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Translation failed for text: {e}")
                    batch_translated.append(text)  # Keep original if translation fails
            
            translated_texts.extend(batch_translated)
            
            # Progress update for large batches
            if len(texts) > 20:
                progress = min(100, int((i + batch_size) / len(texts) * 100))
                st.progress(progress / 100)
        
        return translated_texts
        
    except Exception as e:
        print(f"Batch translation failed: {e}")
        return texts

def translate_transcript_to_english(transcript_data, source_language_code=None):
    """Translate entire transcript to English"""
    if not TRANSLATION_AVAILABLE:
        st.error("❌ Translation not available. Install googletrans: pip install googletrans==4.0.0-rc1")
        return None
    
    try:
        st.info(f"🔄 Translating transcript from {source_language_code or 'detected language'} to English...")
        
        # Extract all text entries
        texts = []
        metadata = []
        
        for entry in transcript_data:
            if hasattr(entry, 'text'):
                texts.append(entry.text)
                metadata.append({
                    'start': entry.start,
                    'duration': getattr(entry, 'duration', 0)
                })
            elif isinstance(entry, dict):
                texts.append(entry.get('text', ''))
                metadata.append({
                    'start': entry.get('start', 0),
                    'duration': entry.get('duration', 0)
                })
        
        if not texts:
            st.error("❌ No text found in transcript to translate")
            return None
        
        # Show translation progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("🔄 Translating transcript...")
        
        # Translate texts in batches
        translated_texts = translate_text_batch(texts, target_lang='en', source_lang=source_language_code)
        
        progress_bar.progress(100)
        status_text.text("✅ Translation completed!")
        
        # Reconstruct transcript with translated text
        translated_transcript = []
        for i, translated_text in enumerate(translated_texts):
            if i < len(metadata):
                translated_transcript.append({
                    'start': metadata[i]['start'],
                    'duration': metadata[i]['duration'],
                    'text': translated_text
                })
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Successfully translated {len(translated_transcript)} transcript segments to English")
        
        return translated_transcript
        
    except Exception as e:
        st.error(f"❌ Translation failed: {str(e)}")
        return None

# ----------------------
# Hindi Detection and Translation Functions
# ----------------------

def detect_hindi_content(transcript_data):
    """Specifically detect if content is in Hindi with better encoding handling"""
    if not TRANSLATION_AVAILABLE:
        return False, "Translation not available"
    
    try:
        # Get a sample of text for Hindi detection
        sample_text = ""
        count = 0
        for entry in transcript_data:
            text = ""
            if hasattr(entry, 'text'):
                text = entry.text
            elif isinstance(entry, dict):
                text = entry.get('text', '')
              # Clean up encoding issues
            if text:
                try:
                    text = text.encode('utf-8', 'ignore').decode('utf-8')
                    # Filter out completely garbled text (mostly symbols)
                    if len(re.sub(r'[^\w\s]', '', text)) > len(text) * 0.3:  # At least 30% should be words
                        sample_text += text + " "
                except:
                    continue
            
            count += 1
            if count >= 10 or len(sample_text) > 500:  # Larger sample for Hindi detection
                break
        
        if sample_text.strip():
            translator = get_translator_with_status()
            if translator:
                # Try to detect language with better error handling
                try:
                    detected = translator.detect(sample_text)
                    if detected and hasattr(detected, 'lang'):
                        language_code = detected.lang
                        confidence = getattr(detected, 'confidence', 0.0)
                    else:
                        language_code = "unknown"
                        confidence = 0.0
                except Exception as detect_error:
                    st.warning(f"⚠️ Language detection failed: {str(detect_error)[:100]}")
                    language_code = "unknown"
                    confidence = 0.0
                
                # Check for Hindi language codes
                is_hindi = language_code in ['hi', 'hi-IN', 'hindi']
                
                # Additional check for Hindi text patterns (Devanagari script)
                hindi_chars = len(re.findall(r'[\u0900-\u097F]', sample_text))
                total_chars = len(sample_text.replace(' ', ''))
                hindi_percentage = (hindi_chars / total_chars * 100) if total_chars > 0 else 0
                
                # Also check for common Hindi words in Latin script
                hindi_words_latin = ['hai', 'hum', 'aap', 'kya', 'kaise', 'kahan', 'kaun', 'main', 'yeh', 'woh']
                hindi_word_count = sum(1 for word in hindi_words_latin if word in sample_text.lower())
                
                # Consider it Hindi if any of these conditions are met:
                # 1. Detected as Hindi by Google Translate
                # 2. Contains significant Devanagari script (>30%)
                # 3. Contains multiple common Hindi words in Latin script
                is_hindi = is_hindi or (hindi_percentage > 30) or (hindi_word_count >= 2)
                
                return is_hindi, {
                    'detected_language': language_code,
                    'confidence': confidence,
                    'hindi_script_percentage': hindi_percentage,
                    'hindi_words_found': hindi_word_count,
                    'sample_text': sample_text[:200] + "..." if len(sample_text) > 200 else sample_text
                }
        
        return False, "No valid text found for detection"
        
    except Exception as e:
        return False, f"Hindi detection failed: {str(e)}"

def translate_hindi_to_english(transcript_data, show_progress=True):
    """Specifically translate Hindi transcript to English with enhanced UI and better encoding"""
    if not TRANSLATION_AVAILABLE:
        st.error("❌ Translation not available. Install googletrans: pip install googletrans==4.0.0-rc1")
        return None
    
    try:
        st.info("🇮🇳 Hindi content detected! Translating to English...")
        
        # Show Hindi sample before translation
        sample_text = ""
        for entry in transcript_data[:3]:  # Show first 3 entries as sample
            if hasattr(entry, 'text'):
                text = entry.text.strip()
                # Clean up any encoding issues
                text = text.encode('utf-8', 'ignore').decode('utf-8')
                sample_text += text + " "
            elif isinstance(entry, dict):
                text = entry.get('text', '').strip()
                # Clean up any encoding issues                text = text.encode('utf-8', 'ignore').decode('utf-8')
                sample_text += text + " "
        
        if sample_text.strip():
            with st.expander("🔍 Original Hindi Sample", expanded=False):
                st.text_area("Original Hindi Text:", sample_text[:500], height=100, disabled=True, key="hindi_sample_3")
        
        # Extract all text entries with better encoding handling
        texts = []
        metadata = []
        
        for entry in transcript_data:
            if hasattr(entry, 'text'):
                text = entry.text.strip()
                # Clean up encoding and filter out empty/garbage text
                text = text.encode('utf-8', 'ignore').decode('utf-8')
                if len(text) > 0 and not all(ord(char) < 32 or ord(char) > 126 for char in text if char.isascii()):
                    texts.append(text)
                    metadata.append({
                        'start': entry.start,
                        'duration': getattr(entry, 'duration', 0)
                    })
            elif isinstance(entry, dict):
                text = entry.get('text', '').strip()
                # Clean up encoding and filter out empty/garbage text
                text = text.encode('utf-8', 'ignore').decode('utf-8')
                if len(text) > 0 and not all(ord(char) < 32 or ord(char) > 126 for char in text if char.isascii()):
                    texts.append(text)
                    metadata.append({
                        'start': entry.get('start', 0),
                        'duration': entry.get('duration', 0)
                    })
        
        if not texts:
            st.error("❌ No valid text found in transcript to translate")
            return None
        
        st.info(f"📝 Processing {len(texts)} text segments for translation...")
        
        # Show translation progress with Hindi-specific messaging
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("🔄 Translating Hindi to English...")
          # Translate texts in smaller batches for Hindi (better accuracy)
        translator = get_translator_with_status()
        if not translator:
            error_details = getattr(st.session_state, 'translator_error_details', 'Unknown error')
            st.error(f"❌ Could not initialize translator: {error_details}")
              # Provide specific troubleshooting based on error type
            if "timeout" in error_details.lower() or "network" in error_details.lower():
                st.info("🌐 **Network Issue Detected**")
                st.info("   • Check your internet connection")
                st.info("   • Try using a different network")
                st.info("   • Disable VPN if active")
                st.info("   • Wait a few minutes and try again")
            elif "403" in error_details or "forbidden" in error_details.lower():
                st.info("🚫 **Access Blocked**")
                st.info("   • Google Translate may be blocked in your region")
                st.info("   • Try using a VPN to a different location")
                st.info("   • Check if corporate firewall is blocking access")
            elif "429" in error_details or "rate" in error_details.lower():
                st.info("⏰ **Rate Limit Reached**")
                st.info("   • Too many translation requests")
                st.info("   • Wait 10-15 minutes before trying again")
                st.info("   • Try with a shorter video or text")
            else:
                st.info("🔧 **General Troubleshooting**")
                st.info("   • Restart the application")
                st.info("   • Check if googletrans is properly installed:")
                st.code("pip install googletrans==4.0.0-rc1")
                st.info("   • Try again in a few minutes")
            
            # Offer to continue with original Hindi text
            st.info("💡 **Alternative Options:**")
            st.info("   • You can manually copy and translate the Hindi text using other tools")
            st.info("   • Try again later when translation services are available")
            st.info("   • Some English subtitles may still be available for this video")
            
            return None
        translated_texts = []
        batch_size = 3  # Smaller batches for better Hindi translation
        successful_translations = 0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_translated = []
            
            for text in batch:
                if len(text.strip()) == 0:
                    batch_translated.append(text)
                    continue
                  # Skip if text contains mostly symbols or punctuation
                if len(re.sub(r'[^\w\s]', '', text)) < 3:
                    batch_translated.append("")
                    continue
                
                try:
                    # Multiple attempts with different source language specifications
                    result = None
                    translated_text = None
                    
                    for src_lang in ['hi', 'auto', None]:
                        try:
                            if src_lang:
                                result = translator.translate(text, src=src_lang, dest='en')
                            else:
                                result = translator.translate(text, dest='en')
                            
                            if result:
                                # Handle both regular text results and potential coroutine objects
                                if hasattr(result, 'text'):
                                    potential_text = result.text
                                    # Check if it's a coroutine object
                                    if hasattr(potential_text, '__await__') or 'coroutine' in str(type(potential_text)):
                                        continue  # Skip this result, try next method
                                    translated_text = str(potential_text).strip()
                                else:
                                    translated_text = str(result).strip()
                                
                                # Verify the translation is meaningful and not empty
                                if translated_text and len(translated_text) > 0:
                                    # Check it's not just symbols or garbled text
                                    if not all(ord(char) < 32 or ord(char) > 126 for char in translated_text if char.isascii()):
                                        batch_translated.append(translated_text)
                                        successful_translations += 1
                                        break
                        except Exception as inner_e:
                            continue
                    
                    # If no successful translation after all attempts
                    if not translated_text:
                        batch_translated.append("")
                    
                    # Longer delay for Hindi translation (better quality)
                    time.sleep(0.3)
                    
                except Exception as e:
                    st.warning(f"Hindi translation failed for segment: {str(e)[:100]}")
                    batch_translated.append("")  # Use empty string instead of original
            
            translated_texts.extend(batch_translated)
            
            # Update progress
            if show_progress:
                progress = min(100, int((i + batch_size) / len(texts) * 100))
                progress_bar.progress(progress / 100)
                status_text.text(f"🔄 Translating Hindi... {progress}% complete ({successful_translations} successful)")
        
        if show_progress:
            progress_bar.progress(100)
            status_text.text(f"✅ Hindi translation completed! ({successful_translations}/{len(texts)} successful)")
        
        # Filter out empty translations and reconstruct transcript
        translated_transcript = []
        for i, translated_text in enumerate(translated_texts):
            if i < len(metadata) and translated_text and len(translated_text.strip()) > 0:
                translated_transcript.append({                    'start': metadata[i]['start'],
                    'duration': metadata[i]['duration'],
                    'text': translated_text
                })
        
        if not translated_transcript:
            st.error("❌ No successful translations were produced")
            return None
        
        # Show translation sample
        if translated_transcript and show_progress:
            sample_translated = " ".join([entry['text'] for entry in translated_transcript[:3]])
            with st.expander("✅ Translation Sample", expanded=True):
                st.text_area("English Translation:", sample_translated[:500], height=100, disabled=True, key="translation_sample_1")
        
        # Clean up progress indicators
        if show_progress:
            progress_bar.empty()
            status_text.empty()
        
        st.success(f"✅ Successfully translated {len(translated_transcript)} Hindi transcript segments to English")
        return translated_transcript
        
    except Exception as e:
        st.error(f"❌ Hindi translation failed: {str(e)}")
        return None

def get_available_transcript_languages(video_id):
    """Get list of available transcript languages for a video with enhanced error handling"""
    try:
        # Try to get transcript list with better error handling
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        except Exception as api_error:
            error_msg = str(api_error).lower()
            if "no element found" in error_msg or "xml" in error_msg:
                print(f"XML parsing error for video {video_id}: YouTube returned malformed data")
                return []
            elif "could not retrieve" in error_msg:
                print(f"No transcripts available for video {video_id}")
                return []
            else:
                print(f"Error getting transcript list for {video_id}: {api_error}")
                return []
        
        languages = []
        
        for transcript_info in transcript_list:
            try:
                languages.append({
                    'code': transcript_info.language_code,
                    'name': getattr(transcript_info, 'language', transcript_info.language_code),
                    'is_generated': transcript_info.is_generated,
                    'is_translatable': getattr(transcript_info, 'is_translatable', False)
                })
            except Exception as lang_error:
                print(f"Error processing language info: {lang_error}")
                continue
        
        return languages
        
    except Exception as e:
        print(f"Error getting available languages: {e}")
        return []

def show_language_detection_results(video_id, transcript_data):
    """Display detailed language detection results with special Hindi handling"""
    try:
        # General language detection
        detected_lang = detect_transcript_language(transcript_data)
        
        # Specific Hindi detection
        is_hindi, hindi_info = detect_hindi_content(transcript_data)
        
        # Get available languages
        available_languages = get_available_transcript_languages(video_id)
        
        st.subheader("🌍 Language Detection Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🔍 Detected Language", detected_lang.upper() if detected_lang != "unknown" else "Unknown")
            
        with col2:
            if is_hindi:
                st.metric("🇮🇳 Hindi Content", "✅ YES")
                if isinstance(hindi_info, dict):
                    st.metric("📊 Confidence", f"{hindi_info.get('confidence', 0)*100:.1f}%")
            else:
                st.metric("🇮🇳 Hindi Content", "❌ NO")
                
        with col3:
            st.metric("📝 Available Languages", len(available_languages))
        
        # Show detailed Hindi information if detected
        if is_hindi and isinstance(hindi_info, dict):
            with st.expander("🇮🇳 Hindi Detection Details", expanded=True):
                st.write(f"**Detected Language Code:** `{hindi_info.get('detected_language', 'unknown')}`")
                st.write(f"**Script Analysis:** {hindi_info.get('hindi_script_percentage', 0):.1f}% Devanagari characters")
                
                if TRANSLATION_AVAILABLE:
                    st.success("✅ Hindi translation is available!")
                    if st.button("🔄 Translate Hindi to English", key="hindi_translate_btn"):
                        return True  # Signal to trigger translation                else:
                    st.warning("⚠️ Install translation support to translate Hindi content")
        
        return False
        
    except Exception as e:
        st.warning(f"Could not perform language detection: {e}")
        return False

def display_language_info(video_id, transcript_method="Unknown"):
    """Display detailed language information for the video"""
    try:
        available_languages = get_available_transcript_languages(video_id)
        
        if available_languages:
            st.subheader("🌍 Language Information")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Current Transcript:**")
                st.info(f"📝 {transcript_method}")
                
            with col2:
                st.write("**Available Languages:**")
                lang_display = []
                for lang in available_languages:
                    status = "🤖 Auto" if lang['is_generated'] else "📝 Manual"
                    translate = "🔄" if lang['is_translatable'] else "❌"
                    lang_display.append(f"{status} {translate} {lang['name']} ({lang['code']})")
                
                for lang_info in lang_display[:5]:  # Show first 5
                    st.text(lang_info)
                
                if len(lang_display) > 5:
                    with st.expander(f"➕ Show {len(lang_display) - 5} more languages"):
                        for lang_info in lang_display[5:]:
                            st.text(lang_info)
                            
    except Exception as e:
        st.warning(f"Could not retrieve language information: {e}")

# ----------------------
# YouTube API Functions
# ----------------------

def get_video_metadata(video_id):
    """Get video metadata using YouTube Data API (if available) or web scraping"""
    try:
        # Try to get basic info from YouTube page
        url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            content = response.text
            
            # Extract title
            title_match = re.search(r'<title>([^<]+)</title>', content)
            title = title_match.group(1).replace(' - YouTube', '') if title_match else "Unknown Title"
            
            # Extract channel name
            channel_match = re.search(r'"ownerChannelName":"([^"]+)"', content)
            channel = channel_match.group(1) if channel_match else "Unknown Channel"
            
            # Extract view count
            view_match = re.search(r'"viewCount":"(\d+)"', content)
            views = int(view_match.group(1)) if view_match else 0
            
            # Extract duration
            duration_match = re.search(r'"lengthSeconds":"(\d+)"', content)
            duration = int(duration_match.group(1)) if duration_match else 0
            
            return {
                'title': title,
                'channel': channel,
                'views': views,
                'duration': duration,
                'url': url
            }
    except Exception as e:
        print(f"Error getting metadata: {e}")
    
    return {
        'title': 'Unknown Title',
        'channel': 'Unknown Channel', 
        'views': 0,
        'duration': 0,
        'url': f"https://www.youtube.com/watch?v={video_id}"
    }

def analyze_sentiment_timeline(transcript_data):
    """Analyze sentiment changes throughout the video"""
    sentiment_analyzer = load_sentiment_analyzer()
    
    # If no sentiment analyzer available (cloud deployment), use fallback
    if sentiment_analyzer is None:
        return generate_fallback_sentiment(transcript_data)
    
    if isinstance(transcript_data, str):
        return [{'timestamp': '00:00', 'sentiment': 'NEUTRAL', 'score': 0.0}]
    
    sentiment_timeline = []
    segment_duration = 60  # 1-minute segments for sentiment analysis
    
    segments = create_timeline_segments(transcript_data, segment_duration)
    
    for segment in segments:
        text = segment['text'][:500]  # Limit text for faster processing
        if len(text.strip()) > 20:
            try:
                result = sentiment_analyzer(text)[0]
                sentiment_timeline.append({
                    'timestamp': segment['timestamp'],
                    'sentiment': result['label'],
                    'score': result['score'],
                    'text_sample': text[:100] + "..."
                })
            except:
                sentiment_timeline.append({
                    'timestamp': segment['timestamp'],
                    'sentiment': 'NEUTRAL',
                    'score': 0.0,
                    'text_sample': text[:100] + "..."
                })
    
    return sentiment_timeline

def generate_fallback_sentiment(transcript_data):
    """Fallback sentiment analysis using simple keyword-based approach"""
    if isinstance(transcript_data, str):
        return [{'timestamp': '00:00', 'sentiment': 'NEUTRAL', 'score': 0.0}]
    
    # Simple sentiment keywords
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'enjoy', 'happy', 'excited', 'awesome', 'perfect', 'best', 'brilliant']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'sad', 'angry', 'disappointed', 'worst', 'problem', 'issue', 'difficult', 'hard', 'struggle']
    
    sentiment_timeline = []
    segment_duration = 60  # 1-minute segments
    
    segments = create_timeline_segments(transcript_data, segment_duration)
    
    for segment in segments:
        text = segment['text'].lower()
        if len(text.strip()) > 20:
            positive_count = sum(1 for word in positive_words if word in text)
            negative_count = sum(1 for word in negative_words if word in text)
            
            if positive_count > negative_count:
                sentiment = 'POSITIVE'
                score = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
            elif negative_count > positive_count:
                sentiment = 'NEGATIVE'
                score = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
            else:
                sentiment = 'NEUTRAL'
                score = 0.5
            
            sentiment_timeline.append({
                'timestamp': segment['timestamp'],
                'sentiment': sentiment,
                'score': score,
                'text_sample': text[:100] + "..."
            })
    
    return sentiment_timeline

def extract_key_phrases(text):
    """Extract key phrases and topics from the text"""
    # Simple keyword extraction based on frequency and length
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    word_freq = Counter(words)
    
    # Filter out common words
    stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 
                  'were', 'said', 'what', 'make', 'like', 'time', 'very', 'when', 
                  'come', 'here', 'just', 'know', 'take', 'people', 'into', 'year', 
                  'your', 'good', 'some', 'could', 'them', 'think', 'would', 'should', 'may', 'might'}
    
    key_phrases = [word for word, freq in word_freq.most_common(20) 
                   if word not in stop_words and freq > 2]
    
    return key_phrases[:15]

def create_word_cloud(text):
    """Generate a word cloud from the text"""
    try:
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white',
                             colormap='viridis').generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    except Exception as e:
        print(f"Error creating word cloud: {e}")
        return None

def analyze_content_categories(text):
    """Categorize content into topics"""
    categories = {
        'Technology': ['technology', 'software', 'computer', 'digital', 'AI', 'machine learning', 'coding', 'programming', 'algorithm', 'data'],
        'Business': ['business', 'marketing', 'finance', 'money', 'profit', 'company', 'strategy', 'management', 'entrepreneur', 'startup'],
        'Education': ['learn', 'education', 'study', 'school', 'university', 'course', 'tutorial', 'knowledge', 'teaching', 'research'],
        'Entertainment': ['movie', 'music', 'game', 'fun', 'entertainment', 'show', 'comedy', 'drama', 'film', 'series'],
        'Health': ['health', 'fitness', 'medical', 'doctor', 'exercise', 'nutrition', 'wellness', 'medicine', 'diet', 'mental'],
        'Science': ['science', 'research', 'experiment', 'discovery', 'theory', 'physics', 'chemistry', 'biology', 'study', 'analysis'],
        'Lifestyle': ['lifestyle', 'travel', 'food', 'fashion', 'home', 'family', 'relationship', 'culture', 'cooking', 'beauty']
    }
    
    text_lower = text.lower()
    category_scores = {}
    
    for category, keywords in categories.items():
        score = sum(text_lower.count(keyword) for keyword in keywords)
        if score > 0:
            category_scores[category] = score
    
    return sorted(category_scores.items(), key=lambda x: x[1], reverse=True)[:3]

def generate_chapter_breakdown(transcript_data, num_chapters=5):
    """Generate chapter-like breakdown of the video"""
    if isinstance(transcript_data, str):
        return [{'title': 'Full Content', 'timestamp': '00:00', 'summary': transcript_data[:200]}]
    
    total_duration = transcript_data[-1]['start'] if transcript_data else 300
    chapter_duration = total_duration / num_chapters
    
    chapters = []
    summarizer = load_summarizer()
    
    for i in range(num_chapters):
        start_time = i * chapter_duration
        end_time = (i + 1) * chapter_duration
        
        # Get text for this chapter
        chapter_text = ""
        for entry in transcript_data:
            if start_time <= entry['start'] < end_time:
                chapter_text += entry['text'] + " "
        
        if chapter_text.strip():
            try:
                # Generate chapter summary
                if len(chapter_text.split()) > 50:
                    summary_result = summarizer(chapter_text[:1000], 
                                              max_length=80, min_length=30, do_sample=False)
                    summary = summary_result[0]['summary_text'] if summary_result else chapter_text[:100]
                else:
                    summary = chapter_text[:100]
                
                # Generate chapter title from summary
                words = summary.split()[:5]
                title = ' '.join(words).replace('.', '').title()
                
                chapters.append({
                    'title': title or f"Chapter {i+1}",
                    'timestamp': format_timestamp(start_time),
                    'summary': summary,
                    'duration': format_timestamp(end_time - start_time)
                })
            except Exception as e:
                print(f"Error generating chapter {i+1}: {e}")
                continue
    
    return chapters

# ----------------------
# Helper Functions
# ----------------------

def get_transcript(video_id, preferred_languages=None, enable_auto_translate=True, show_original_language=True):
    """Enhanced transcript function with multiple fallback methods"""
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    return get_enhanced_transcript(video_id, video_url, preferred_languages, enable_auto_translate, show_original_language)

def get_transcript_text(transcript_data):
    """Extract just the text from transcript data"""
    if isinstance(transcript_data, str):
        return transcript_data
    
    text_parts = []
    for t in transcript_data:
        try:
            # Handle FetchedTranscriptSnippet objects
            if hasattr(t, 'text'):
                text_parts.append(t.text)
            # Handle dictionary format
            elif isinstance(t, dict) and 'text' in t:
                text_parts.append(t['text'])
            else:
                text_parts.append(str(t))
        except:
            continue
    
    return " ".join(text_parts)

def format_timestamp(seconds):
    """Convert seconds to MM:SS or HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"

def create_timeline_segments(transcript_data, segment_duration=300):
    """Create timeline segments (default 5 minutes each)"""
    if isinstance(transcript_data, str):
        return [{"start": 0, "text": transcript_data, "timestamp": "00:00"}]
    
    segments = []
    current_segment = {"start": 0, "text": "", "entries": []}
    
    for entry in transcript_data:
        try:
            # Handle FetchedTranscriptSnippet objects
            if hasattr(entry, 'start') and hasattr(entry, 'text'):
                start_time = entry.start
                text = entry.text
            # Handle dictionary format
            elif isinstance(entry, dict):
                start_time = entry.get('start', 0)
                text = entry.get('text', '')
            else:
                continue
            
            # If we've exceeded the segment duration, start a new segment
            if start_time >= current_segment["start"] + segment_duration and current_segment["text"]:
                current_segment["timestamp"] = format_timestamp(current_segment["start"])
                segments.append(current_segment)
                current_segment = {"start": start_time, "text": "", "entries": []}
            
            current_segment["text"] += text + " "
            current_segment["entries"].append(entry)
        except Exception as e:
            print(f"Error processing transcript entry: {e}")
            continue
    
    # Add the last segment
    if current_segment["text"]:
        current_segment["timestamp"] = format_timestamp(current_segment["start"])
        segments.append(current_segment)
    
    return segments

def generate_detailed_summary(transcript_data, segment_duration=300):
    """Generate detailed timeline-based summary with parallel processing"""
    try:
        summarizer = load_summarizer()
        
        # Create timeline segments
        segments = create_timeline_segments(transcript_data, segment_duration)
        
        detailed_summary = "## 📋 **DETAILED VIDEO SUMMARY WITH TIMELINE**\n\n"
        
        # Process segments in parallel for speed
        def process_segment(segment_info):
            i, segment = segment_info
            timestamp = segment["timestamp"]
            text = segment["text"].strip()
            
            if len(text) < 100:  # Skip very short segments
                return None
            
            try:
                # Create detailed summary for this segment
                words = len(text.split())
                max_len = min(250, max(100, words // 2))
                min_len = min(80, max_len // 3)
                
                if words > 400:
                    chunks = chunk_text(text, 400)
                    segment_summary = ""
                    for chunk in chunks:
                        if len(chunk.strip()) > 50:
                            out = summarizer(chunk, max_length=150, min_length=60, do_sample=False)
                            if out and len(out) > 0 and 'summary_text' in out[0]:
                                segment_summary += out[0]['summary_text'] + " "
                else:
                    out = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
                    segment_summary = out[0]['summary_text'] if out and len(out) > 0 else "No summary available."
                
                return f"### 🕐 **[{timestamp}]** - Segment {i+1}\n{segment_summary.strip()}\n\n"
                
            except Exception as e:
                print(f"Error summarizing segment {i+1}: {e}")
                return None
        
        # Process up to 3 segments in parallel for better performance
       
        with ThreadPoolExecutor(max_workers=3) as executor:
            segment_results = list(executor.map(process_segment, enumerate(segments)))
        
        # Combine results
        for result in segment_results:
            if result:
                detailed_summary += result
        
        return detailed_summary
        
    except Exception as e:
        return f"Error generating detailed summary: {str(e)}"

def chunk_text(text, max_words):
    """Split text into chunks of specified word count"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    
    return chunks

def generate_comprehensive_summary(transcript_data):
    """Generate both overview and detailed summaries with advanced features"""
    try:
        # Get text from transcript
        if isinstance(transcript_data, str):
            full_text = transcript_data
        else:
            full_text = get_transcript_text(transcript_data)
        
        # Generate overview summary
        overview = generate_summary(full_text)
        
        # Generate detailed timeline summary
        detailed = generate_detailed_summary(transcript_data, segment_duration=180)
        
        # Generate key insights
        insights = generate_key_insights(full_text)
        
        # Get content categories
        categories = analyze_content_categories(full_text)
        category_text = ", ".join([f"{cat} ({score} mentions)" for cat, score in categories]) if categories else "General Content"
        
        # Extract key phrases
        key_phrases = extract_key_phrases(full_text)
        phrases_text = ", ".join(key_phrases[:10]) if key_phrases else "No key phrases detected"
        
        # Combine everything
        comprehensive = f"""
# 🎯 **EXECUTIVE SUMMARY**

{overview}

## 📊 **CONTENT ANALYSIS**
- **Primary Topics:** {category_text}
- **Key Phrases:** {phrases_text}
- **Content Type:** {"Educational/Tutorial" if any(word in full_text.lower() for word in ['tutorial', 'learn', 'how to', 'guide']) else "General Discussion"}

---

{detailed}

---

## 📈 **KEY INSIGHTS & TAKEAWAYS**

{insights}

## 🎬 **VIDEO STRUCTURE**

{generate_video_structure_analysis(transcript_data)}
"""
        
        return comprehensive
        
    except Exception as e:
        return f"Error generating comprehensive summary: {str(e)}"

def generate_video_structure_analysis(transcript_data):
    """Analyze video structure and pacing"""
    try:
        if isinstance(transcript_data, str):
            return "Structure analysis not available for text-only content."
        
        # Analyze speaking pace and content density
        segments = create_timeline_segments(transcript_data, 60)  # 1-minute segments
        
        structure_analysis = ""
        
        # Calculate words per minute for each segment
        pacing_data = []
        for segment in segments:
            word_count = len(segment['text'].split())
            pacing_data.append({
                'timestamp': segment['timestamp'],
                'words_per_minute': word_count,
                'content_density': 'High' if word_count > 200 else 'Medium' if word_count > 100 else 'Low'
            })
        
        # Find the most content-dense segments
        high_density_segments = [p for p in pacing_data if p['content_density'] == 'High']
        
        structure_analysis += f"- **Total Segments:** {len(segments)}\n"
        structure_analysis += f"- **High-Density Segments:** {len(high_density_segments)} segments with rapid information delivery\n"
        
        if high_density_segments:
            timestamps = [s['timestamp'] for s in high_density_segments[:3]]
            structure_analysis += f"- **Key Information Peaks:** {', '.join(timestamps)}\n"
        
        # Analyze intro/outro
        if len(segments) > 2:
            intro_words = len(segments[0]['text'].split())
            outro_words = len(segments[-1]['text'].split())
            structure_analysis += f"- **Introduction Density:** {'Detailed' if intro_words > 150 else 'Brief'}\n"
            structure_analysis += f"- **Conclusion Density:** {'Detailed' if outro_words > 150 else 'Brief'}\n"
        
        return structure_analysis
        
    except Exception as e:
        return f"Error analyzing video structure: {str(e)}"

def generate_key_insights(text):
    """Extract key insights and important points"""
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Split text into meaningful chunks
        sentences = sent_tokenize(text)
        
        # Find important sentences (longer sentences often contain more information)
        important_sentences = [s for s in sentences if len(s.split()) > 15]
        
        if not important_sentences:
            return "No key insights could be extracted."
        
        # Take every 10th sentence to get distributed insights
        key_sentences = important_sentences[::max(1, len(important_sentences)//10)][:15]
        
        insights_text = " ".join(key_sentences)
        
        if len(insights_text.split()) > 500:
            # Summarize the key insights
            out = summarizer(insights_text[:2000], max_length=200, min_length=100, do_sample=False)
            if out and len(out) > 0:
                return out[0]['summary_text']
        
        return insights_text[:1000] + "..." if len(insights_text) > 1000 else insights_text
        
    except Exception as e:
        return f"Error generating insights: {str(e)}"

def generate_summary(text, is_long_form=False):
    try:
        summarizer = load_summarizer()  # Use cached model
        
        # If no summarizer available (cloud deployment), use fallback method
        if summarizer is None:
            return generate_fallback_summary(text)
        
        # Clean and preprocess the text
        text = text.strip()
        if not text:
            return "No text to summarize."
        
        # Adjust chunk size based on content length
        word_count = len(text.split())
        if word_count > 10000:  # Long-form content (2+ hour podcasts)
            max_chunk = 400
            is_long_form = True
        elif word_count > 5000:  # Medium content
            max_chunk = 450
        else:  # Short content
            max_chunk = 512
            
        sentences = sent_tokenize(text)
        
        if not sentences:
            return "No sentences found to summarize."
        
        current_chunk = []
        chunks = []
        total_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_tokens = len(sentence.split())
            
            # Skip very long sentences that might cause issues
            if sentence_tokens > max_chunk:
                # Truncate very long sentences
                words = sentence.split()[:max_chunk]
                sentence = " ".join(words)
                sentence_tokens = len(words)
            
            if total_tokens + sentence_tokens <= max_chunk:
                current_chunk.append(sentence)
                total_tokens += sentence_tokens
            else:
                if current_chunk:  # Only add non-empty chunks
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                total_tokens = sentence_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        if not chunks:
            return "No content available for summarization."

        # For long-form content, create hierarchical summaries
        if is_long_form and len(chunks) > 20:
            return generate_hierarchical_summary(chunks, summarizer)
        else:
            return generate_standard_summary(chunks, summarizer)
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def generate_fallback_summary(text):
    """Fallback summarization method when AI models are not available (cloud deployment)"""
    try:
        # Clean and preprocess text
        text = text.strip()
        if not text:
            return "No text to summarize."
        
        sentences = sent_tokenize(text)
        if len(sentences) <= 3:
            return text
        
        # Extract key sentences using simple heuristics
        sentence_scores = []
        word_freq = {}
        
        # Calculate word frequencies
        words = text.lower().split()
        for word in words:
            word = re.sub(r'[^\w\s]', '', word)
            if len(word) > 3 and word not in ['this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said']:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score sentences based on word frequency
        for i, sentence in enumerate(sentences):
            sentence_words = sentence.lower().split()
            score = 0
            word_count = 0
            
            for word in sentence_words:
                word = re.sub(r'[^\w\s]', '', word)
                if word in word_freq:
                    score += word_freq[word]
                    word_count += 1
            
            if word_count > 0:
                # Position bonus for sentences at beginning and end
                position_bonus = 1.0
                if i < len(sentences) * 0.1:  # First 10%
                    position_bonus = 1.2
                elif i > len(sentences) * 0.9:  # Last 10%
                    position_bonus = 1.1
                
                avg_score = (score / word_count) * position_bonus
                sentence_scores.append((i, sentence, avg_score))
        
        # Sort by score and select top sentences
        sentence_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Select top 30% of sentences or minimum 3 sentences
        num_sentences = max(3, min(len(sentences) // 3, 10))
        selected_sentences = sentence_scores[:num_sentences]
        
        # Sort selected sentences by original order
        selected_sentences.sort(key=lambda x: x[0])
        
        # Create summary
        summary = " ".join([s[1] for s in selected_sentences])
        
        # Add a note about the summarization method
        summary = "📝 **Cloud Summary** (Extractive): " + summary
        
        return summary if summary else "Unable to generate summary."
    
    except Exception as e:
        return f"Error in fallback summarization: {str(e)}"

def generate_standard_summary(chunks, summarizer):
    """Generate summary for regular length content"""
    summary = ""
    for i, chunk in enumerate(chunks):
        try:
            if len(chunk.strip()) < 50:
                continue
                
            chunk_words = len(chunk.split())
            max_len = min(150, max(60, chunk_words // 3))
            min_len = min(30, max_len // 3)
            
            out = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)
            if out and len(out) > 0 and 'summary_text' in out[0]:
                summary += out[0]['summary_text'] + " "
        except Exception as e:
            print(f"Error summarizing chunk {i+1}: {e}")
            continue

    return summary.strip() if summary.strip() else "Unable to generate summary."

def generate_hierarchical_summary(chunks, summarizer):
    """Generate hierarchical summary for long-form content (podcasts)"""
    try:
        # Step 1: Summarize chunks in groups
        group_size = 5  # Process 5 chunks at a time
        intermediate_summaries = []
        
        for i in range(0, len(chunks), group_size):
            group = chunks[i:i+group_size]
            group_text = " ".join(group)
            
            if len(group_text.strip()) < 100:
                continue
                
            try:
                # Create intermediate summary for this group
                words = len(group_text.split())
                max_len = min(200, max(80, words // 4))
                min_len = min(50, max_len // 3)
                
                out = summarizer(group_text[:2000], max_length=max_len, min_length=min_len, do_sample=False)
                if out and len(out) > 0 and 'summary_text' in out[0]:
                    intermediate_summaries.append(out[0]['summary_text'])
            except Exception as e:
                print(f"Error summarizing group {i//group_size + 1}: {e}")
                continue
        
        if not intermediate_summaries:
            return "Unable to generate summary from long-form content."
        
        # Step 2: Create final summary from intermediate summaries
        final_text = " ".join(intermediate_summaries)
        if len(final_text.split()) > 1000:
            # If still too long, summarize again
            words = len(final_text.split())
            max_len = min(300, max(150, words // 3))
            min_len = min(100, max_len // 3)
            
            final_out = summarizer(final_text, max_length=max_len, min_length=min_len, do_sample=False)
            if final_out and len(final_out) > 0 and 'summary_text' in final_out[0]:
                return final_out[0]['summary_text']
        
        return final_text
        
    except Exception as e:
        return f"Error in hierarchical summarization: {str(e)}"

def generate_flashcards(text):
    sentences = sent_tokenize(text)
    flashcards = []
    for sent in sentences:
        words = sent.split()
        if len(words) > 6:
            question = f"What is the meaning or importance of: '{' '.join(words[:6])}...'"
            answer = sent
            flashcards.append((question, answer))
    return flashcards[:5]

def generate_mindmap(text):
    keywords = [word for word in text.split() if len(word) > 5]
    graph = nx.Graph()

    # Create a basic structure with a central node
    central_node = "Main Idea"
    graph.add_node(central_node)

    for kw in keywords[:10]:  # limit to 10 for clarity
        graph.add_edge(central_node, kw)

    return graph

def draw_mindmap(graph):
    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw(graph, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, ax=ax)
    return fig

def extract_video_id(url):
    """Extract video ID from various YouTube URL formats"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([^&\n?#]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)',
        r'youtu\.be\/([^&\n?#]+)',
        r'youtube\.com\/embed\/([^&\n?#]+)',
        r'youtube\.com\/v\/([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

# ----------------------
# Streamlit App
# ----------------------

# Page config is handled by streamlit_app.py entry point

def main():
    """Main Streamlit application function"""
    st.title("🚀 Enhanced YouTube Video Analyzer & Summarizer")
    st.markdown("*AI-powered analysis with advanced subtitle enhancement and AI transcription fallback*")

    # Cloud deployment notification
    if IS_STREAMLIT_CLOUD:
        st.info("☁️ **Running on Streamlit Cloud** - AI transcription and advanced ML features are disabled to ensure stable deployment. The app will focus on subtitle-based analysis.")
        st.info("💡 **Available Features**: YouTube transcript extraction, Hindi translation, basic summarization, and content analysis.")
    
    # Display AI transcription availability
    if AI_TRANSCRIPTION_AVAILABLE:
        st.success("✅ AI Transcription Available")
    else:
        if not IS_STREAMLIT_CLOUD:
            st.warning("⚠️ AI Transcription Not Available - Install with: `pip install openai-whisper pytube pydub SpeechRecognition`")
        else:
            st.info("🔍 **Cloud Mode**: Using subtitle-based processing for optimal performance")

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Transcript options
        st.subheader("📝 Transcript Options")
        force_ai_transcription = st.checkbox("🤖 Force AI Transcription", 
                                            value=False,
                                            help="Use AI transcription even if subtitles are available",
                                            disabled=not AI_TRANSCRIPTION_AVAILABLE)
        enhance_quality = st.checkbox("✨ Enhance Transcript Quality", 
                                    value=True,
                                    help="Apply automatic corrections to improve transcript quality")
        
        # Summary settings
        st.subheader("📝 Summary Settings")
        segment_duration = st.slider("Segment Duration (minutes)", 2, 10, 3)
        summary_detail = st.selectbox("Summary Detail Level", 
                                      ["Concise", "Detailed", "Comprehensive"], 
                                      index=1)
        # Analysis options
        st.subheader("📊 Analysis Options")
        enable_sentiment = st.checkbox("Sentiment Analysis", value=True)
        enable_wordcloud = st.checkbox("Word Cloud", value=True)
        enable_chapters = st.checkbox("Auto-Generate Chapters", value=True)
        
        # Translation settings
        st.subheader("🌍 Translation Options")
        if TRANSLATION_AVAILABLE:
            st.success("✅ Translation Available")
            enable_auto_translate = st.checkbox("🔄 Auto-translate foreign videos",
                                              value=True,
                                              help="Automatically translate non-English transcripts to English")
            enable_auto_hindi_translation = st.checkbox("🇮🇳 Auto-detect & translate Hindi",
                                                       value=True,
                                                       help="Automatically detect Hindi content and translate to English")
            show_original_language = st.checkbox("📝 Show original language info",
                                               value=True,
                                               help="Display detected language information")
            preferred_languages = st.multiselect(
                "🎯 Preferred subtitle languages (priority order)",
                ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "hi", "ar"],
                default=["en"],
                help="Languages to look for before falling back to translation"
            )
        else:
            st.warning("⚠️ Translation Not Available")
            st.info("Install with: `pip install googletrans==4.0.0rc1`")
            enable_auto_translate = False
            enable_auto_hindi_translation = False
            show_original_language = False
            preferred_languages = ["en"]
        
        # Advanced features
        st.subheader("🔬 Advanced Features")
        enable_key_phrases = st.checkbox("Key Phrase Extraction", value=True)
        enable_content_categorization = st.checkbox("Content Categorization", value=True)

        # Troubleshooting section
        st.subheader("🔧 Troubleshooting")
        
        with st.expander("❌ Common Issues & Solutions"):
            st.markdown("**Video Access Restricted:**")
            st.info("• Try enabling AI Transcription above")
            st.info("• Check if video is public/unlisted")
            st.info("• Wait and retry for rate limits")
            
            st.markdown("**No Transcripts Found:**")
            st.info("• Enable 'Force AI Transcription'")
            st.info("• Try videos with manual captions")
            st.info("• Install AI packages if needed")
            
            st.markdown("**Translation Failures:**")
            st.info("• Check internet connection")
            st.info("• Try disabling auto-translation")
            st.info("• Restart application")
            
            st.markdown("**Slow Processing:**")
            st.info("• Use 'Ultra Fast' version for speed")
            st.info("• Disable heavy features like word clouds")
            st.info("• Try shorter videos first")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        video_url = st.text_input("🔗 Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

    with col2:
        if st.button("🎯 Quick Analysis", help="Get a quick overview without full processing"):
            st.info("Quick analysis feature coming soon!")

    # Initialize transcript variable
    transcript = None

    if video_url:
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("❌ Invalid YouTube URL. Please enter a valid YouTube URL.")
            st.stop()

        # Get video metadata
        with st.spinner("🔍 Fetching video information..."):
            metadata = get_video_metadata(video_id)
        
        # Display video info
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📺 **Channel**", metadata['channel'])
        with col2:
            st.metric("👁️ **Views**", f"{metadata['views']:,}" if metadata['views'] else "N/A")
        with col3:
            duration_formatted = format_timestamp(metadata['duration']) if metadata['duration'] else "N/A"
            st.metric("⏱️ **Duration**", duration_formatted)
        with col4:
            st.metric("📝 **Title**", metadata['title'][:30] + "..." if len(metadata['title']) > 30 else metadata['title'])

        # Enhanced transcript extraction with quality assessment
        st.write("⏳ Extracting transcript with enhanced methods...")
        
        # Get transcript with enhanced methods
        if force_ai_transcription and AI_TRANSCRIPTION_AVAILABLE:
            st.info("🎤 Using AI transcription as requested...")
            video_url_full = f"https://www.youtube.com/watch?v={video_id}"
            audio_path, error = download_audio_from_youtube(video_url_full, video_id)
            if audio_path and not error:
                transcript, whisper_error = transcribe_with_whisper(audio_path, video_id)
                if whisper_error:
                    st.error(f"AI transcription failed: {whisper_error}")
                    st.stop()
            else:
                st.error(f"Audio download failed: {error}")
                st.stop()
        else:
            # Modify preferred languages to include Hindi if auto-Hindi detection is enabled
            transcript_preferred_languages = preferred_languages.copy() if preferred_languages else ["en"]
            
            # Add Hindi to preferred languages if auto-Hindi detection is enabled
            if enable_auto_hindi_translation and "hi" not in transcript_preferred_languages:
                transcript_preferred_languages.append("hi")
                st.info("🇮🇳 Hindi auto-detection enabled - including Hindi in transcript search...")
            
            translator_tested = False
            translator_working = False
              # Test translator only once per session
            translator = get_translator_with_status()
            
            # Only show warning if translation features are explicitly requested
            if not translator and (enable_auto_translate or enable_auto_hindi_translation):
                if 'translator_warning_shown' not in st.session_state or not st.session_state.translator_warning_shown:
                    st.info("💡 Translation features may be limited due to connectivity issues.")
                    if 'translator_warning_shown' not in st.session_state:
                        st.session_state.translator_warning_shown = True
            
            transcript = get_transcript(video_id, transcript_preferred_languages, enable_auto_translate, show_original_language)

        if isinstance(transcript, str) and "Error" in transcript:
            error_message = transcript.lower()
            
            # Provide specific guidance based on error type
            if "restricted access" in error_message or "xml parsing failed" in error_message:
                st.error("❌ **Video Access Restricted**")
                st.markdown("### 🚫 **This video has restricted transcript access**")
                st.info("**Possible reasons:**")
                st.info("• Age-restricted content requiring login")
                st.info("• Private or unlisted video")
                st.info("• Geographic restrictions")
                st.info("• Channel disabled transcript access")
                st.info("• Copyright protection measures")
                
                if AI_TRANSCRIPTION_AVAILABLE:
                    st.markdown("### 🤖 **Alternative Solution**")
                    st.info("Try AI transcription to bypass transcript restrictions:")
                    if st.button("🎤 **Try AI Transcription**", type="primary"):
                        st.rerun()
                else:
                    st.markdown("### 💡 **Suggested Solutions**")
                    st.info("1. Install AI transcription: `pip install openai-whisper pytube pydub`")
                    st.info("2. Try a different video from the same channel")
                    st.info("3. Check if the video has public access")
                    
            elif "no transcripts available" in error_message:
                st.error("❌ **No Transcripts Found**")
                st.markdown("### 📝 **This video doesn't have subtitles or captions**")
                
                if AI_TRANSCRIPTION_AVAILABLE:
                    st.markdown("### 🤖 **AI Transcription Available**")
                    st.success("Good news! We can generate transcripts using AI:")
                    if st.button("🎤 **Generate AI Transcript**", type="primary"):
                        st.rerun()
                else:
                    st.markdown("### 💡 **Solutions**")
                    st.info("1. **Install AI transcription** to generate transcripts:")
                    st.code("pip install openai-whisper pytube pydub SpeechRecognition")
                    st.info("2. **Try videos with captions** from channels that provide subtitles")
                    st.info("3. **Look for educational content** which often has better caption support")
                    
            elif "blocked" in error_message or "400" in error_message:
                st.error("❌ **YouTube Request Blocked**")
                st.markdown("### 🛡️ **YouTube is blocking the request**")
                st.warning("This can happen due to:")
                st.info("• Rate limiting (too many requests)")
                st.info("• IP restrictions")
                st.info("• Temporary YouTube API issues")
                st.info("• Video access restrictions")
                
                st.markdown("### 🔄 **What to try:**")
                st.info("1. **Wait 5-10 minutes** and try again")
                st.info("2. **Try a different video** to test the service")
                st.info("3. **Check if the video URL is correct**")
                if AI_TRANSCRIPTION_AVAILABLE:
                    st.info("4. **Use AI transcription** as an alternative")
                    if st.button("🤖 **Try AI Transcription**", type="primary"):
                        st.rerun()
            else:
                st.error(f"❌ {transcript}")
                
                # Generic fallback options
                if AI_TRANSCRIPTION_AVAILABLE:
                    st.markdown("### 🤖 **Try Alternative Method**")
                    if st.button("🎤 Try AI Transcription Instead", type="secondary"):
                        st.rerun()
            
            st.stop()

        # Check if transcript was successfully retrieved
        if not transcript or isinstance(transcript, str) and "Error" in transcript:
            st.error("❌ Transcript could not be retrieved.")
            st.stop()

        # Assess transcript quality
        quality_assessment = assess_transcript_quality(transcript)
        
        # Display quality assessment
        quality_color = {
            "excellent": "🟢",
            "good": "🟡", 
            "fair": "🟠",
            "poor": "🔴",
            "unknown": "⚪"
        }
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.success("✅ Transcript extracted successfully!")
        with col2:
            st.metric("📊 Quality", f"{quality_color.get(quality_assessment['quality'], '⚪')} {quality_assessment['quality'].title()}")
        with col3:
            if st.button("🔄 Try Different Method"):
                st.rerun()
        
        # Show quality recommendations if any
        if quality_assessment.get('recommendations'):
            with st.expander("💡 Quality Recommendations", expanded=False):
                for rec in quality_assessment['recommendations']:
                    st.info(f"• {rec}")
        # Apply quality enhancement if enabled
        if enhance_quality and not isinstance(transcript, str):
            transcript = enhance_transcript_quality(transcript)
            st.info("✨ Transcript quality enhancement applied")

        # ========== AUTOMATIC HINDI DETECTION AND TRANSLATION ==========
        # Check if the transcript contains Hindi content and auto-translate if needed
        if not isinstance(transcript, str) and enable_auto_hindi_translation:
            st.write("🔍 **Checking language content...**")
            
            # First detect the general language
            detected_lang = detect_transcript_language(transcript)
            
            # If already English, skip Hindi processing
            if detected_lang == 'en':
                st.success("🇺🇸 **English content detected** - no translation needed!")
                st.info("✅ Transcript is already in English, proceeding with analysis...")
            else:
                # Only check for Hindi if not English
                is_hindi, hindi_info = detect_hindi_content(transcript)
                
                if is_hindi:
                    st.info("🇮🇳 **Hindi content detected!** Automatically translating to English...")
                    
                    # Show Hindi detection details
                    if isinstance(hindi_info, dict):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🔍 Language", hindi_info.get('detected_language', 'hi').upper())
                        with col2:
                            st.metric("📊 Confidence", f"{hindi_info.get('confidence', 0)*100:.1f}%")
                    with col3:
                        st.metric("📝 Script", f"{hindi_info.get('hindi_script_percentage', 0):.1f}% Devanagari")
                    
                    # Automatically translate Hindi to English with comprehensive error handling
                    try:
                        translated_transcript = translate_hindi_to_english(transcript, show_progress=True)
                        
                        if translated_transcript and len(translated_transcript) > 0:
                            # Replace the original transcript with the translated one
                            transcript = translated_transcript
                            st.success("✅ **Hindi content successfully translated to English!** Proceeding with analysis...")
                            
                            # Show language info
                            with st.expander("🌍 Language Processing Details", expanded=False):
                                st.write("**Original Language:** Hindi (हिंदी)")
                                st.write("**Target Language:** English")
                                st.write("**Translation Method:** Google Translate API")
                                st.write("**Processing Status:** ✅ Completed")
                        else:
                            st.warning("⚠️ Hindi translation produced no results. Proceeding with original Hindi text...")
                            st.info("💡 **Tip:** You can still use the analysis features with Hindi text, though some features may work better with English.")
                            
                    except Exception as translation_error:
                        st.error(f"❌ Hindi translation failed: {str(translation_error)}")
                        st.warning("⚠️ Proceeding with original Hindi text...")
                        st.info("💡 **Tip:** You can still use the analysis features with Hindi text, though some features may work better with English.")
                else:
                    # Not Hindi content, show detection results for non-English languages
                    if detected_lang and detected_lang != "unknown":
                        st.info(f"✅ Content language detected: **{detected_lang.upper()}** (No translation needed)")
                    else:
                        st.info("✅ Language detection completed (Proceeding with original content)")

        # Continue with existing processing only if transcript is available
        if transcript and not isinstance(transcript, str):
            full_text = get_transcript_text(transcript)
            word_count = len(full_text.split())
            estimated_duration = word_count / 150  # Average speaking rate: 150 words/minute
            # Enhanced stats display
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Words", f"{word_count:,}")
            with col2:
                st.metric("⏰ Est. Duration", f"{estimated_duration:.1f} min")
            with col3:
                reading_time = word_count / 250  # Average reading speed
                st.metric("📖 Reading Time", f"{reading_time:.1f} min")
            with col4:
                complexity = "High" if word_count > 10000 else "Medium" if word_count > 3000 else "Low"
                st.metric("🧠 Complexity", complexity)
        
            if word_count > 10000:
                st.warning("🎧 **Long-form content detected!** Using advanced hierarchical processing...")
            
            # ========== LANGUAGE DETECTION STATUS ==========
            # Show detailed language detection results if enabled
            if show_original_language and not isinstance(transcript, str):
                with st.expander("🌍 Language Detection Results", expanded=False):
                    show_language_detection_results(video_id, transcript)
        
            # Progress tracking
            progress_container = st.container()
            with progress_container:
                if word_count > 3000:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text("🚀 Initializing advanced analysis...")
                    
            # Generate comprehensive analysis
            start_time = time.time()
            
            if word_count > 3000:
                status_text.text("📝 Generating comprehensive summary...")
                progress_bar.progress(25)
                
            comprehensive_summary = generate_comprehensive_summary(transcript)
            
            if word_count > 3000:
                progress_bar.progress(60)
                status_text.text("🔍 Analyzing content structure...")
            
            # Main summary display
            st.markdown("---")
            st.markdown(comprehensive_summary)
        
            # Advanced analysis tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Visualizations", "🎭 Sentiment Analysis", "🏷️ Chapters", "🔍 Deep Analysis", "💾 Export"])
            
            with tab1:
                st.subheader("📊 Visual Content Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if enable_wordcloud:
                        st.subheader("☁️ Word Cloud")
                        wordcloud_fig = create_word_cloud(full_text)
                        if wordcloud_fig:
                            st.pyplot(wordcloud_fig)
                            
                with col2:
                    st.subheader("🧠 Enhanced Mind Map")
                    mindmap_graph = generate_mindmap(full_text)
                    fig = draw_mindmap(mindmap_graph)
                    st.pyplot(fig)
                
                # Content categories visualization
                if enable_content_categorization:
                    categories = analyze_content_categories(full_text)
                    if categories:
                        st.subheader("📈 Content Categories")
                        cat_df = pd.DataFrame(categories, columns=['Category', 'Mentions'])
                        fig = px.bar(cat_df, x='Category', y='Mentions', 
                                   title="Content Category Distribution",
                                   color='Mentions', color_continuous_scale='viridis')
                        st.plotly_chart(fig, use_container_width=True)
        
            with tab2:
                if enable_sentiment:
                    st.subheader("🎭 Sentiment Timeline Analysis")
                
                if word_count > 3000:
                    status_text.text("🎭 Analyzing sentiment timeline...")
                    progress_bar.progress(80)
                
                sentiment_data = analyze_sentiment_timeline(transcript)
                
                if sentiment_data and len(sentiment_data) > 1:
                    # Create sentiment timeline chart
                    sentiment_df = pd.DataFrame(sentiment_data)
                    
                    # Map sentiment labels to colors
                    color_map = {'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'gray'}
                    sentiment_df['color'] = sentiment_df['sentiment'].map(color_map)
                    
                    fig = px.scatter(sentiment_df, x='timestamp', y='score', 
                                   color='sentiment', size='score',
                                   title="Sentiment Timeline Throughout Video",
                                   labels={'score': 'Sentiment Score', 'timestamp': 'Time'},
                                   hover_data=['text_sample'])
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Sentiment summary
                    positive_count = len([s for s in sentiment_data if s['sentiment'] == 'POSITIVE'])
                    negative_count = len([s for s in sentiment_data if s['sentiment'] == 'NEGATIVE'])
                    neutral_count = len([s for s in sentiment_data if s['sentiment'] == 'NEUTRAL'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("😊 Positive Segments", positive_count)
                    with col2:
                        st.metric("😞 Negative Segments", negative_count)
                    with col3:
                        st.metric("😐 Neutral Segments", neutral_count)
                else:
                    st.info("Sentiment analysis not available for this content.")
        
            with tab3:
                if enable_chapters:
                    st.subheader("🏷️ Auto-Generated Chapters")
                    
                    chapters = generate_chapter_breakdown(transcript, num_chapters=6)
                    
                    if chapters:
                        for i, chapter in enumerate(chapters):
                            with st.expander(f"📖 Chapter {i+1}: {chapter['title']} [{chapter['timestamp']}]"):
                                st.write(f"**Duration:** {chapter.get('duration', 'N/A')}")
                                st.write(f"**Summary:** {chapter['summary']}")
                                
                                # Create clickable timestamp (YouTube link)
                                youtube_url = f"https://www.youtube.com/watch?v={video_id}&t={chapter['timestamp'].replace(':', 'm').replace('m', 's')}s"
                                st.markdown(f"[🔗 Jump to this chapter]({youtube_url})")
                    else:
                        st.info("Chapters could not be generated for this content.")
        
            with tab4:
                st.subheader("🔍 Deep Content Analysis")
                
                # Display transcript quality details
                st.subheader("📊 Transcript Quality Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Quality Score", f"{quality_assessment.get('score', 0)}/100")
                    st.metric("Total Segments", len(transcript) if not isinstance(transcript, str) else 1)
                
                with col2:
                    avg_length = len(full_text.split()) / len(transcript) if transcript and not isinstance(transcript, str) else len(full_text.split())
                    st.metric("Avg Segment Length", f"{avg_length:.1f} words")
                    
                # Key phrases
                if enable_key_phrases:
                    key_phrases = extract_key_phrases(full_text)
                    if key_phrases:
                        st.subheader("🔑 Key Phrases & Topics")
                        
                        # Display as tags
                        phrase_html = ""
                        for phrase in key_phrases[:15]:
                            phrase_html += f'<span style="background-color: #e1f5fe; padding: 4px 8px; margin: 2px; border-radius: 12px; display: inline-block; font-size: 0.9em;">{phrase}</span> '
                        
                        st.markdown(phrase_html, unsafe_allow_html=True)
                
                # Enhanced flashcards
                st.subheader("📚 Interactive Flashcards")
                overview_summary = generate_summary(full_text)
                flashcards = generate_flashcards(overview_summary)
                
                if flashcards:
                    for i, (q, a) in enumerate(flashcards):
                        with st.expander(f"💡 Flashcard {i+1}"):
                            st.markdown(f"**❓ Question:** {q}")
                            if st.button(f"Show Answer {i+1}", key=f"answer_{i}"):
                                st.markdown(f"**✅ Answer:** {a}")
                
                # Timeline segments
                st.subheader("🕐 Detailed Timeline Segments")
                segments = create_timeline_segments(transcript, segment_duration=segment_duration*60)
                
                segment_df = pd.DataFrame([
                    {
                        'Timestamp': segment['timestamp'],
                        'Word Count': len(segment['text'].split()),
                        'Content Density': 'High' if len(segment['text'].split()) > 200 else 'Medium' if len(segment['text'].split()) > 100 else 'Low',
                        'Preview': segment['text'][:100] + "..."
                    }
                    for segment in segments
                ])
                
                st.dataframe(segment_df, use_container_width=True)
        
            with tab5:
                st.subheader("💾 Export & Download Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Summary export
                    summary_export = {
                        'video_title': metadata['title'],
                        'channel': metadata['channel'],
                        'url': metadata['url'],
                        'analysis_date': datetime.now().isoformat(),
                        'word_count': word_count,
                        'quality_assessment': quality_assessment,
                        'summary': comprehensive_summary,
                        'key_phrases': extract_key_phrases(full_text) if enable_key_phrases else [],
                        'sentiment_data': sentiment_data if enable_sentiment else []
                    }
                    
                    st.download_button(
                        label="📄 Download Summary (JSON)",
                        data=json.dumps(summary_export, indent=2),
                        file_name=f"youtube_summary_{video_id}.json",
                        mime="application/json"
                    )
                    
                    st.download_button(
                        label="📝 Download Summary (Text)",
                        data=comprehensive_summary,
                        file_name=f"youtube_summary_{video_id}.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    # Transcript export
                    st.download_button(
                        label="📜 Download Full Transcript",
                        data=full_text,
                        file_name=f"youtube_transcript_{video_id}.txt",
                        mime="text/plain"
                    )
                    
                    if chapters:
                        chapters_text = "\n\n".join([
                            f"Chapter {i+1}: {chapter['title']} [{chapter['timestamp']}]\n{chapter['summary']}"
                            for i, chapter in enumerate(chapters)
                        ])
                        
                        st.download_button(
                            label="📖 Download Chapters",
                            data=chapters_text,
                            file_name=f"youtube_chapters_{video_id}.txt",
                            mime="text/plain"
                        )
        
            # Final processing update
            if word_count > 3000:
                progress_bar.progress(100)
                processing_time = time.time() - start_time
                status_text.text(f"✅ Analysis complete! Processed in {processing_time:.1f} seconds")
                
                # Remove progress bar after completion
                time.sleep(2)
                progress_container.empty()
        
            # Footer with stats
            st.markdown("---")
            quality_badge = f"{quality_color.get(quality_assessment['quality'], '⚪')} {quality_assessment['quality'].title()}"
            st.markdown(f"**🔍 Analysis Summary:** Processed {word_count:,} words | Quality: {quality_badge} | Analysis time: {time.time() - start_time:.1f}s")
            
            # Display language information
            display_language_info(video_id, transcript_method="Enhanced Transcript")

def try_alternative_transcript_methods(video_id, video_url):
    """Try alternative methods to extract transcripts when main API fails"""
    st.info("🔄 **Alternative Method 1:** Direct transcript fetch...")
    
    # Try direct transcript fetch with different language codes
    direct_methods = [
        (['en'], "English"),
        (['en-US', 'en-GB'], "English variants"),
        (['auto'], "Auto-detection"),
        (['hi'], "Hindi"),
        (['es', 'fr', 'de', 'ja', 'ko'], "Other languages")
    ]
    
    for lang_codes, description in direct_methods:
        try:
            st.info(f"   → Trying {description}...")
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=lang_codes)
            if transcript and len(transcript) > 0:
                enhanced = enhance_transcript_quality(transcript)
                st.success(f"✅ **Success!** Found transcript using {description}")
                return enhanced
        except Exception as e:
            st.warning(f"   ✗ {description} failed: {str(e)[:50]}...")
            continue
    
    # If direct methods fail, try AI transcription
    if AI_TRANSCRIPTION_AVAILABLE:
        st.info("🔄 **Alternative Method 2:** AI transcription...")
        return try_ai_transcription_fallback(video_id, video_url)
    else:
        st.error("❌ **All alternative methods failed.**")
        st.info("💡 **Final Suggestions:**")
        st.info("   • Install AI transcription: `pip install openai-whisper pytube pydub`")
        st.info("   • Try a different video from the same channel")
        st.info("   • Check if the video has public access")
        st.info("   • Wait and try again later (temporary restrictions)")
        return "Error: All transcript extraction methods failed - video has restricted access"

def try_ai_transcription_fallback(video_id, video_url):
    """Try AI transcription as a fallback method"""
    try:
        st.info("🎤 **Starting AI transcription process...**")
        st.warning("⚠️ **Note:** AI transcription may take several minutes for longer videos")
        
        # Download audio
        with st.spinner("📥 Downloading audio from video..."):
            audio_path, error = download_audio_from_youtube(video_url, video_id)
            
            if error:
                if "400" in error:
                    st.error("❌ **Audio Download Failed (HTTP 400)**")
                    st.info("This video has audio download restrictions:")
                    st.info("   • **Copyright protection** - audio download blocked")
                    st.info("   • **Premium content** - requires subscription")
                    st.info("   • **Live stream** - cannot download live audio")
                    st.info("   • **Age restrictions** - requires authentication")
                elif "403" in error:
                    st.error("❌ **Audio Download Forbidden (HTTP 403)**")
                    st.info("   • **Geographic restrictions** - not available in your region")
                    st.info("   • **Channel restrictions** - creator disabled downloading")
                else:
                    st.error(f"❌ **Audio Download Error:** {error}")
                return f"Error: Audio download failed - {error}"
            
            if not audio_path:
                st.error("❌ **Audio Download Failed:** No audio file was created")
                return "Error: Audio download failed - no file created"
        
        # Transcribe with AI
        with st.spinner("🤖 Transcribing audio with AI (this may take a few minutes)..."):
            transcript, whisper_error = transcribe_with_whisper(audio_path, video_id)
            
            if whisper_error:
                st.error(f"❌ **AI Transcription Failed:** {whisper_error}")
                return f"Error: AI transcription failed - {whisper_error}"
            
            if not transcript:
                st.error("❌ **AI Transcription Failed:** No transcript was generated")
                return "Error: AI transcription failed - no content generated"
        
        st.success("✅ **AI transcription completed successfully!**")
        return transcript
        
    except Exception as e:
        st.error(f"❌ **AI Transcription Error:** {str(e)}")
        return f"Error: AI transcription failed - {str(e)}"

# ----------------------
# Application Entry Point
# ----------------------

if __name__ == "__main__":
    main()
