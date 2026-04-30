import streamlit as st
import pandas as pd
import re
import base64
from datetime import datetime
from collections import Counter

# Try importing NLP libraries with error handling
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob

    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer

    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# Try importing visualization libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


# Download NLTK data
@st.cache_resource
def download_nltk_data():
    if not NLP_AVAILABLE:
        return False
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('brown', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        return True
    except Exception:
        return False


nltk_success = download_nltk_data() if NLP_AVAILABLE else False


# Initialize VADER
@st.cache_resource
def get_vader_analyzer():
    if VADER_AVAILABLE:
        return VaderAnalyzer()
    elif NLP_AVAILABLE and nltk_success:
        return SentimentIntensityAnalyzer()
    return None


vader = get_vader_analyzer()

# Page configuration
st.set_page_config(
    page_title="SentimentIQ - Feedback Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Dark Blue Theme CSS
st.markdown("""
    <style>
    /* Main Theme Colors */
    :root {
        --primary: #1e3a5f;
        --primary-light: #2c5282;
        --primary-dark: #0f2440;
        --accent: #3182ce;
        --success: #2f855a;
        --warning: #b7791f;
        --danger: #9b2c2c;
        --bg-dark: #0a1628;
        --bg-card: #132440;
        --text-primary: #e2e8f0;
        --text-secondary: #a0aec0;
        --border: #2d3748;
    }

    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #132440 50%, #0f2440 100%);
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        border: 1px solid #3182ce;
        box-shadow: 0 8px 32px rgba(49, 130, 206, 0.2);
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(49, 130, 206, 0.1) 50%, transparent 70%);
        animation: shimmer 3s infinite;
    }

    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    .main-header h1 {
        color: #ffffff;
        font-size: 2.5em;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }

    .main-header p {
        color: #90cdf4;
        font-size: 1.1em;
        margin-top: 10px;
        position: relative;
        z-index: 1;
    }

    /* Cards */
    .feedback-card {
        background: linear-gradient(135deg, #132440 0%, #1a2d4a 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border: 1px solid #2d3748;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .feedback-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(49, 130, 206, 0.3);
    }

    .positive {
        border-left: 4px solid #48bb78;
        background: linear-gradient(135deg, #132440 0%, #1a3a2a 100%);
    }

    .neutral {
        border-left: 4px solid #ecc94b;
        background: linear-gradient(135deg, #132440 0%, #3a351a 100%);
    }

    .negative {
        border-left: 4px solid #fc8181;
        background: linear-gradient(135deg, #132440 0%, #3a1a1a 100%);
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1a2d4a 0%, #1e3a5f 100%);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #2d3748;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: #3182ce;
        box-shadow: 0 8px 25px rgba(49, 130, 206, 0.3);
        transform: translateY(-2px);
    }

    .metric-card h3 {
        color: #a0aec0;
        font-size: 1em;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-card h2 {
        color: #e2e8f0;
        font-size: 2em;
        margin: 10px 0;
        font-weight: 700;
    }

    .emoji-large {
        font-size: 52px;
        margin: 10px 0;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3182ce 0%, #2c5282 100%);
        color: white;
        border: 1px solid #4299e1;
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        font-size: 0.9em;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
        border-color: #63b3ed;
        box-shadow: 0 4px 15px rgba(49, 130, 206, 0.4);
        transform: translateY(-1px);
    }

    /* Text Areas */
    .stTextArea > div > div > textarea {
        background: #132440;
        border: 1px solid #2d3748;
        color: #e2e8f0;
        border-radius: 10px;
        padding: 15px;
        font-size: 16px;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: #3182ce;
        box-shadow: 0 0 0 2px rgba(49, 130, 206, 0.3);
    }

    /* Text Inputs */
    .stTextInput > div > div > input {
        background: #132440;
        border: 1px solid #2d3748;
        color: #e2e8f0;
        border-radius: 10px;
        padding: 10px 15px;
    }

    .stTextInput > div > div > input:focus {
        border-color: #3182ce;
        box-shadow: 0 0 0 2px rgba(49, 130, 206, 0.3);
    }

    /* Select Boxes */
    .stSelectbox > div > div > select {
        background: #132440;
        border: 1px solid #2d3748;
        color: #e2e8f0;
    }

    /* Multiselect */
    .stMultiSelect > div > div {
        background: #132440;
        border: 1px solid #2d3748;
        color: #e2e8f0;
    }

    /* Sidebar */
    .css-1d391kg, .css-1lcbmhc {
        background: linear-gradient(180deg, #0f2440 0%, #132440 100%);
        border-right: 1px solid #2d3748;
    }

    .sidebar .sidebar-content {
        background: transparent;
    }

    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3182ce 0%, #4299e1 100%);
    }

    /* Metrics */
    .stMetric {
        background: linear-gradient(135deg, #1a2d4a 0%, #1e3a5f 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2d3748;
    }

    .stMetric label {
        color: #a0aec0 !important;
    }

    .stMetric .css-1xarl3l {
        color: #e2e8f0 !important;
    }

    /* Info Boxes */
    .stAlert {
        background: #132440;
        border: 1px solid #2d3748;
        color: #e2e8f0;
        border-radius: 10px;
    }

    .stSuccess {
        background: #1a3a2a;
        border-color: #48bb78;
    }

    .stWarning {
        background: #3a351a;
        border-color: #ecc94b;
    }

    .stError {
        background: #3a1a1a;
        border-color: #fc8181;
    }

    /* DataFrames */
    .dataframe {
        background: #132440 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 10px !important;
    }

    .dataframe th {
        background: #1e3a5f !important;
        color: #e2e8f0 !important;
        border-bottom: 2px solid #3182ce !important;
    }

    .dataframe td {
        background: #132440 !important;
        color: #e2e8f0 !important;
        border-bottom: 1px solid #2d3748 !important;
    }

    /* Radio Buttons */
    .stRadio > div {
        background: #132440;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2d3748;
    }

    .stRadio label {
        color: #e2e8f0 !important;
    }

    /* Sliders */
    .stSlider > div > div > div {
        background: #3182ce;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #132440;
        border-radius: 10px;
        border: 1px solid #2d3748;
        padding: 5px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #a0aec0;
        border-radius: 8px;
    }

    .stTabs [aria-selected="true"] {
        background: #3182ce;
        color: white;
    }

    /* Divider */
    hr {
        border-color: #2d3748;
    }

    /* Caption */
    .caption {
        color: #a0aec0;
        font-size: 0.9em;
    }

    /* Link */
    a {
        color: #63b3ed;
        text-decoration: none;
    }

    a:hover {
        color: #90cdf4;
        text-decoration: underline;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #a0aec0;
        border-top: 1px solid #2d3748;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'feedback_history' not in st.session_state:
    st.session_state.feedback_history = []
if 'total_analyses' not in st.session_state:
    st.session_state.total_analyses = 0


# Enhanced keyword-based sentiment analysis
def analyze_sentiment_keywords(text):
    """Enhanced keyword-based sentiment analysis with context"""
    text_lower = text.lower()

    positive_words = {
        'excellent': 1.0, 'amazing': 1.0, 'outstanding': 1.0, 'perfect': 0.9,
        'brilliant': 0.9, 'fantastic': 0.9, 'wonderful': 0.9, 'superb': 0.9,
        'great': 0.7, 'good': 0.5, 'love': 0.8, 'happy': 0.7, 'pleased': 0.6,
        'satisfied': 0.6, 'awesome': 0.9, 'impressive': 0.8, 'recommend': 0.7,
        'enjoy': 0.6, 'delighted': 0.8, 'thank': 0.5, 'thanks': 0.5,
        'helpful': 0.6, 'best': 0.8, 'favorite': 0.7, 'incredible': 0.9,
        'exceptional': 0.9, 'remarkable': 0.8, 'phenomenal': 1.0,
        'nice': 0.4, 'decent': 0.3, 'fine': 0.2, 'okay': 0.1,
        'well': 0.3, 'better': 0.5, 'improved': 0.6
    }

    negative_words = {
        'terrible': -1.0, 'horrible': -1.0, 'awful': -1.0, 'disgusting': -1.0,
        'worst': -0.9, 'hate': -0.9, 'useless': -0.9, 'pathetic': -0.9,
        'bad': -0.5, 'poor': -0.6, 'disappointed': -0.7, 'disappointing': -0.7,
        'boring': -0.6, 'slow': -0.5, 'broken': -0.8, 'failed': -0.8,
        'failure': -0.8, 'problem': -0.6, 'issue': -0.5, 'complaint': -0.6,
        'unhappy': -0.7, 'frustrated': -0.7, 'angry': -0.8, 'rude': -0.8,
        'expensive': -0.4, 'overpriced': -0.6, 'waste': -0.7, 'never': -0.5,
        'annoying': -0.6, 'difficult': -0.5, 'confusing': -0.5
    }

    intensifiers = ['very', 'really', 'extremely', 'absolutely', 'completely',
                    'totally', 'highly', 'so', 'quite', 'incredibly']
    negations = ['not', 'no', "n't", 'never', 'neither', 'nor', 'hardly', 'barely']

    words = text_lower.split()
    total_score = 0
    word_count = 0

    for i, word in enumerate(words):
        negation_multiplier = 1
        if i > 0 and words[i - 1] in negations:
            negation_multiplier = -1

        intensifier_multiplier = 1
        if i > 0 and words[i - 1] in intensifiers:
            intensifier_multiplier = 1.5

        if word in positive_words:
            total_score += positive_words[word] * negation_multiplier * intensifier_multiplier
            word_count += 1
        elif word in negative_words:
            total_score += negative_words[word] * negation_multiplier * intensifier_multiplier
            word_count += 1

    if word_count > 0:
        avg_score = total_score / word_count
    else:
        avg_score = 0

    if avg_score > 0.15:
        sentiment = "Positive"
        emoji = "😊"
        confidence = min(abs(avg_score) * 100, 95)
        polarity = min(avg_score, 0.95)
    elif avg_score < -0.15:
        sentiment = "Negative"
        emoji = "😞"
        confidence = min(abs(avg_score) * 100, 95)
        polarity = max(avg_score, -0.95)
    else:
        sentiment = "Neutral"
        emoji = "😐"
        confidence = 50 + (abs(avg_score) * 30)
        polarity = avg_score

    return {
        'sentiment': sentiment,
        'emoji': emoji,
        'polarity': round(polarity, 3),
        'subjectivity': min(word_count / max(len(words), 1), 0.9),
        'confidence': round(confidence, 2)
    }


def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER"""
    if vader:
        scores = vader.polarity_scores(text)
        compound = scores['compound']

        if compound >= 0.05:
            sentiment = "Positive"
            emoji = "😊"
            confidence = min(abs(compound) * 100, 95)
        elif compound <= -0.05:
            sentiment = "Negative"
            emoji = "😞"
            confidence = min(abs(compound) * 100, 95)
        else:
            sentiment = "Neutral"
            emoji = "😐"
            confidence = 50 - (abs(compound) * 30)

        return {
            'sentiment': sentiment,
            'emoji': emoji,
            'polarity': round(compound, 3),
            'subjectivity': (scores['pos'] + scores['neg']) / max(scores['neu'], 0.1),
            'confidence': round(confidence, 2)
        }
    return analyze_sentiment_keywords(text)


def analyze_sentiment_textblob(text):
    """Analyze sentiment using TextBlob"""
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if polarity > 0.15:
            sentiment = "Positive"
            emoji = "😊"
            confidence = min(abs(polarity) * 100, 95)
        elif polarity < -0.15:
            sentiment = "Negative"
            emoji = "😞"
            confidence = min(abs(polarity) * 100, 95)
        else:
            sentiment = "Neutral"
            emoji = "😐"
            confidence = 50 + (abs(polarity) * 30)

        return {
            'sentiment': sentiment,
            'emoji': emoji,
            'polarity': round(polarity, 3),
            'subjectivity': round(subjectivity, 3),
            'confidence': round(confidence, 2)
        }
    except Exception:
        return analyze_sentiment_keywords(text)


def analyze_sentiment(text):
    """Main sentiment analysis using ensemble of methods"""
    vader_result = analyze_sentiment_vader(text)

    if NLP_AVAILABLE:
        try:
            textblob_result = analyze_sentiment_textblob(text)

            avg_polarity = (vader_result['polarity'] + textblob_result['polarity']) / 2
            avg_subjectivity = (vader_result['subjectivity'] + textblob_result['subjectivity']) / 2
            avg_confidence = (vader_result['confidence'] + textblob_result['confidence']) / 2

            if avg_polarity > 0.1:
                sentiment = "Positive"
                emoji = "😊"
                confidence = min(avg_confidence, 95)
            elif avg_polarity < -0.1:
                sentiment = "Negative"
                emoji = "😞"
                confidence = min(avg_confidence, 95)
            else:
                sentiment = "Neutral"
                emoji = "😐"
                confidence = 50 + (abs(avg_polarity) * 20)

            return {
                'sentiment': sentiment,
                'emoji': emoji,
                'polarity': round(avg_polarity, 3),
                'subjectivity': round(avg_subjectivity, 3),
                'confidence': round(confidence, 2),
                'vader_score': vader_result['polarity'],
                'textblob_score': textblob_result['polarity']
            }
        except Exception:
            return vader_result
    else:
        return vader_result


def extract_keywords(text):
    """Extract keywords with emotions"""
    emotions = {
        'happy': '😊', 'sad': '😢', 'angry': '😠', 'love': '❤️',
        'great': '👍', 'bad': '👎', 'amazing': '✨', 'terrible': '💔',
        'good': '✅', 'poor': '❌', 'excellent': '🌟', 'awful': '😱'
    }

    words = re.findall(r'\b\w+\b', text.lower())
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'can', 'i', 'you', 'he', 'she', 'it', 'we',
                  'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its',
                  'our', 'their', 'this', 'that', 'these', 'those', 'am'}

    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    word_counts = Counter(keywords).most_common(15)

    enhanced_keywords = []
    for word, count in word_counts:
        em = emotions.get(word, '')
        enhanced_keywords.append((f"{em} {word}" if em else word, count))

    return enhanced_keywords


def generate_wordcloud(feedback_list):
    """Generate word cloud from feedback"""
    if not WORDCLOUD_AVAILABLE or not feedback_list:
        return None

    try:
        all_text = ' '.join([fb['text'] for fb in feedback_list])
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='#0a1628',
            max_words=100,
            collocations=False,
            colormap='Blues'
        ).generate(all_text)

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0a1628')
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    except Exception:
        return None


# Main Header
st.markdown("""
    <div class="main-header">
        <h1>📊 SentimentIQ</h1>
        <p>Professional Feedback Analysis & Sentiment Intelligence Platform</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Analysis Controls")

    analysis_mode = st.radio(
        "Select Mode",
        ["🎯 Single Analysis", "📝 Batch Analysis", "📊 History Dashboard"],
        key="analysis_mode"
    )

    st.markdown("---")

    # Statistics
    st.markdown("### 📈 Platform Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Analyses", st.session_state.total_analyses)
    with col2:
        if st.session_state.feedback_history:
            positive_count = sum(1 for fb in st.session_state.feedback_history if fb['sentiment'] == 'Positive')
            st.metric("Satisfaction", f"{(positive_count / len(st.session_state.feedback_history) * 100):.1f}%")
        else:
            st.metric("Satisfaction", "0%")

    st.markdown("---")

    if st.session_state.feedback_history:
        st.markdown("### 💾 Data Export")
        df_export = pd.DataFrame(st.session_state.feedback_history)
        csv = df_export.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="feedback_history.csv">📥 Download CSV Report</a>'
        st.markdown(href, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔧 System Status")
    if vader:
        st.success("✅ VADER Engine Active")
    elif NLP_AVAILABLE and nltk_success:
        st.success("✅ NLP Engine Active")
    else:
        st.warning("⚡ Basic Analysis Mode")

# Main Content Area
if analysis_mode == "🎯 Single Analysis":
    st.header("Single Feedback Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        feedback_text = st.text_area(
            "Enter Customer Feedback:",
            placeholder="Type or paste customer feedback here for analysis...",
            height=150,
            key="single_feedback"
        )

        customer_name = st.text_input("Customer Name (Optional):", placeholder="Enter customer name...")

        if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
            if feedback_text.strip():
                with st.spinner("🔄 Processing with AI models..."):
                    result = analyze_sentiment(feedback_text)
                    keywords = extract_keywords(feedback_text)

                    feedback_record = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'customer': customer_name if customer_name else "Anonymous",
                        'text': feedback_text,
                        'sentiment': result['sentiment'],
                        'emoji': result['emoji'],
                        'polarity': result['polarity'],
                        'subjectivity': result['subjectivity'],
                        'confidence': result['confidence']
                    }
                    st.session_state.feedback_history.append(feedback_record)
                    st.session_state.total_analyses += 1

                    st.markdown("---")
                    st.markdown("### 📊 Analysis Results")

                    res_col1, res_col2, res_col3 = st.columns(3)

                    with res_col1:
                        sentiment_color = "#48bb78" if result['sentiment'] == "Positive" else "#fc8181" if result[
                                                                                                               'sentiment'] == "Negative" else "#ecc94b"
                        st.markdown(f"""
                            <div class="metric-card">
                                <h3>Sentiment</h3>
                                <div class="emoji-large">{result['emoji']}</div>
                                <h2 style="color: {sentiment_color};">{result['sentiment']}</h2>
                            </div>
                        """, unsafe_allow_html=True)

                    with res_col2:
                        st.markdown(f"""
                            <div class="metric-card">
                                <h3>Confidence</h3>
                                <div style="font-size: 48px; margin: 10px 0;">📊</div>
                                <h2 style="color: #90cdf4;">{result['confidence']}%</h2>
                            </div>
                        """, unsafe_allow_html=True)

                    with res_col3:
                        st.markdown(f"""
                            <div class="metric-card">
                                <h3>Subjectivity</h3>
                                <div style="font-size: 48px; margin: 10px 0;">🎯</div>
                                <h2 style="color: #90cdf4;">{(result['subjectivity'] * 100):.1f}%</h2>
                            </div>
                        """, unsafe_allow_html=True)

                    if 'vader_score' in result and 'textblob_score' in result:
                        st.markdown("---")
                        st.subheader("🔬 Model Scores")
                        model_col1, model_col2, model_col3 = st.columns(3)

                        with model_col1:
                            st.metric("VADER Score", f"{result['vader_score']:.3f}")

                        with model_col2:
                            st.metric("TextBlob Score", f"{result['textblob_score']:.3f}")

                        with model_col3:
                            st.metric("Ensemble", f"{result['polarity']:.3f}")

                    st.markdown("---")
                    st.subheader("📈 Polarity Meter")

                    if PLOTLY_AVAILABLE:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=result['polarity'],
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Sentiment Polarity", 'font': {'color': '#e2e8f0'}},
                            delta={'reference': 0},
                            gauge={
                                'axis': {'range': [-1, 1], 'tickcolor': '#e2e8f0'},
                                'bar': {'color': "#3182ce"},
                                'bgcolor': '#132440',
                                'steps': [
                                    {'range': [-1, -0.5], 'color': '#9b2c2c'},
                                    {'range': [-0.5, -0.1], 'color': '#c53030'},
                                    {'range': [-0.1, 0.1], 'color': '#b7791f'},
                                    {'range': [0.1, 0.5], 'color': '#2f855a'},
                                    {'range': [0.5, 1], 'color': '#48bb78'}
                                ]
                            }
                        ))
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': '#e2e8f0'},
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        polarity_normalized = (result['polarity'] + 1) / 2
                        st.progress(polarity_normalized)
                        st.markdown(f"**Polarity Score:** {result['polarity']:.3f}")

                    if keywords:
                        st.markdown("---")
                        st.subheader("🏷️ Extracted Keywords")
                        cols = st.columns(5)
                        for i, (word, count) in enumerate(keywords[:10]):
                            with cols[i % 5]:
                                st.metric(f"#{i + 1}", word, count)

                    st.success("✅ Analysis completed successfully!")
            else:
                st.warning("⚠️ Please enter feedback text to analyze.")

    with col2:
        st.markdown("""
            <div class="metric-card" style="text-align: left;">
                <h3 style="color: #63b3ed;">💡 How It Works</h3>
                <p style="color: #a0aec0;">Our AI uses ensemble learning:</p>
                <ul style="color: #a0aec0; list-style: none; padding: 0;">
                    <li>🎯 VADER - Best for short text</li>
                    <li>📚 TextBlob - Paragraph analysis</li>
                    <li>🔤 Keywords - Context detection</li>
                </ul>
                <h4 style="color: #63b3ed; margin-top: 20px;">Classification:</h4>
                <ul style="color: #a0aec0; list-style: none; padding: 0;">
                    <li>😊 Positive (Polarity > 0.1)</li>
                    <li>😐 Neutral (-0.1 to 0.1)</li>
                    <li>😞 Negative (Polarity < -0.1)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

elif analysis_mode == "📝 Batch Analysis":
    st.header("Batch Feedback Processing")

    col1, col2 = st.columns([2, 1])

    with col1:
        batch_text = st.text_area(
            "Enter Multiple Feedbacks (one per line):",
            placeholder="Amazing product, highly recommended!\nService was terrible, very disappointed.\nIt's okay, nothing special...",
            height=200,
            key="batch_feedback"
        )

        if st.button("📊 Process Batch", type="primary", use_container_width=True):
            if batch_text.strip():
                feedbacks = [line.strip() for line in batch_text.split('\n') if line.strip()]

                if feedbacks:
                    with st.spinner(f"🔄 Processing {len(feedbacks)} feedbacks..."):
                        results = []
                        for feedback in feedbacks:
                            result = analyze_sentiment(feedback)
                            results.append({
                                'Feedback': feedback[:100] + "..." if len(feedback) > 100 else feedback,
                                'Sentiment': result['sentiment'],
                                'Emoji': result['emoji'],
                                'Confidence': f"{result['confidence']}%",
                                'Polarity': result['polarity']
                            })

                            feedback_record = {
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'customer': 'Batch',
                                'text': feedback,
                                'sentiment': result['sentiment'],
                                'emoji': result['emoji'],
                                'polarity': result['polarity'],
                                'subjectivity': result['subjectivity'],
                                'confidence': result['confidence']
                            }
                            st.session_state.feedback_history.append(feedback_record)
                            st.session_state.total_analyses += 1

                        st.markdown("### 📊 Batch Results")
                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True)

                        sentiment_counts = df['Sentiment'].value_counts()

                        chart_col1, chart_col2 = st.columns(2)

                        with chart_col1:
                            st.subheader("Sentiment Distribution")
                            if PLOTLY_AVAILABLE:
                                fig = px.pie(
                                    values=sentiment_counts.values,
                                    names=sentiment_counts.index,
                                    color=sentiment_counts.index,
                                    color_discrete_map={
                                        'Positive': '#48bb78',
                                        'Neutral': '#ecc94b',
                                        'Negative': '#fc8181'
                                    },
                                    hole=0.4
                                )
                                fig.update_layout(
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    font={'color': '#e2e8f0'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.bar_chart(sentiment_counts)

                        with chart_col2:
                            st.subheader("Confidence Analysis")
                            if PLOTLY_AVAILABLE:
                                fig = px.histogram(
                                    df,
                                    x='Sentiment',
                                    color='Sentiment',
                                    color_discrete_map={
                                        'Positive': '#48bb78',
                                        'Neutral': '#ecc94b',
                                        'Negative': '#fc8181'
                                    }
                                )
                                fig.update_layout(
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    font={'color': '#e2e8f0'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.bar_chart(df['Sentiment'].value_counts())

                        st.success(f"✅ Processed {len(feedbacks)} feedbacks successfully!")
                else:
                    st.warning("⚠️ No valid feedback found.")
            else:
                st.warning("⚠️ Please enter feedback text.")

    with col2:
        st.markdown("""
            <div class="metric-card" style="text-align: left;">
                <h3 style="color: #63b3ed;">📝 Batch Guidelines</h3>
                <ul style="color: #a0aec0; list-style: none; padding: 0;">
                    <li>• One feedback per line</li>
                    <li>• Empty lines skipped</li>
                    <li>• Each line analyzed independently</li>
                </ul>
                <h4 style="color: #63b3ed; margin-top: 20px;">Output Includes:</h4>
                <ul style="color: #a0aec0; list-style: none; padding: 0;">
                    <li>📊 Individual analysis</li>
                    <li>📈 Distribution charts</li>
                    <li>💯 Confidence scores</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

else:  # History Dashboard
    st.header("📊 Analysis History Dashboard")

    if st.session_state.feedback_history:
        df_history = pd.DataFrame(st.session_state.feedback_history)

        col1, col2, col3 = st.columns(3)
        with col1:
            sentiment_filter = st.multiselect(
                "Sentiment Filter:",
                options=["Positive", "Neutral", "Negative"],
                default=["Positive", "Neutral", "Negative"]
            )

        with col2:
            if 'customer' in df_history.columns:
                customers = ['All'] + list(df_history['customer'].unique())
                customer_filter = st.selectbox("Customer Filter:", customers)

        with col3:
            if len(df_history) > 0:
                confidence_range = st.slider(
                    "Minimum Confidence %:",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=5.0
                )

        filtered_df = df_history[df_history['sentiment'].isin(sentiment_filter)]
        if 'customer' in filtered_df.columns and 'customer_filter' in locals() and customer_filter != 'All':
            filtered_df = filtered_df[filtered_df['customer'] == customer_filter]
        if 'confidence_range' in locals():
            filtered_df = filtered_df[filtered_df['confidence'] >= confidence_range]

        st.markdown("---")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

        with stat_col1:
            st.metric("Total", len(filtered_df))

        with stat_col2:
            positive_count = len(filtered_df[filtered_df['sentiment'] == 'Positive'])
            st.metric("😊 Positive", positive_count)

        with stat_col3:
            neutral_count = len(filtered_df[filtered_df['sentiment'] == 'Neutral'])
            st.metric("😐 Neutral", neutral_count)

        with stat_col4:
            negative_count = len(filtered_df[filtered_df['sentiment'] == 'Negative'])
            st.metric("😞 Negative", negative_count)

        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Sentiment Distribution")
            if PLOTLY_AVAILABLE:
                fig = px.pie(
                    filtered_df,
                    names='sentiment',
                    color='sentiment',
                    color_discrete_map={
                        'Positive': '#48bb78',
                        'Neutral': '#ecc94b',
                        'Negative': '#fc8181'
                    },
                    hole=0.3
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#e2e8f0'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                sentiment_counts = filtered_df['sentiment'].value_counts()
                st.bar_chart(sentiment_counts)

        with chart_col2:
            st.subheader("Confidence Analysis")
            if PLOTLY_AVAILABLE:
                filtered_df_copy = filtered_df.copy()
                filtered_df_copy['timestamp'] = pd.to_datetime(filtered_df_copy['timestamp'])
                fig = px.scatter(
                    filtered_df_copy,
                    x='timestamp',
                    y='confidence',
                    color='sentiment',
                    color_discrete_map={
                        'Positive': '#48bb78',
                        'Neutral': '#ecc94b',
                        'Negative': '#fc8181'
                    },
                    size='confidence',
                    hover_data=['text']
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#e2e8f0'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Install plotly for advanced charts")

        st.markdown("---")
        st.subheader("☁️ Word Cloud Analysis")
        wordcloud_fig = generate_wordcloud(st.session_state.feedback_history)
        if wordcloud_fig:
            st.pyplot(wordcloud_fig)
        else:
            st.info("Install wordcloud package for this visualization")

        st.markdown("---")
        st.subheader("📋 Detailed Feedback Log")

        for i, row in filtered_df.iterrows():
            sentiment_class = row['sentiment'].lower()
            st.markdown(f"""
                <div class="feedback-card {sentiment_class}">
                    <h4 style="color: {'#48bb78' if row['sentiment'] == 'Positive' else '#fc8181' if row['sentiment'] == 'Negative' else '#ecc94b'};">
                        {row['emoji']} {row['sentiment']} (Confidence: {row['confidence']}%)
                    </h4>
                    <p style="color: #a0aec0;"><strong>Customer:</strong> {row['customer']}</p>
                    <p style="color: #a0aec0;"><strong>Time:</strong> {row['timestamp']}</p>
                    <p style="color: #e2e8f0;"><strong>Feedback:</strong> {row['text'][:200]}...</p>
                    <p style="color: #718096;"><small>Polarity: {row['polarity']:.3f} | Subjectivity: {row['subjectivity']:.3f}</small></p>
                </div>
            """, unsafe_allow_html=True)

        if st.button("🗑️ Clear All History", type="secondary"):
            st.session_state.feedback_history = []
            st.session_state.total_analyses = 0
            st.rerun()
    else:
        st.markdown("""
            <div class="metric-card" style="text-align: center; padding: 40px;">
                <div style="font-size: 64px;">📭</div>
                <h3 style="color: #a0aec0;">No Analysis History</h3>
                <p style="color: #718096;">Start analyzing feedback to build your history</p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
        <p>📊 SentimentIQ | Professional Feedback Analysis Platform</p>
        <p style="color: #718096; font-size: 0.9em;">Powered by VADER • TextBlob • Enhanced NLP Engine</p>
        <p style="color: #4a5568; font-size: 0.8em;">© 2024 SentimentIQ Analytics. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)