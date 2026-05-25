import streamlit as st
import requests
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Configuration
API_BASE  = "http://127.0.0.1:8000"
API_URL   = f"{API_BASE}/predict"
MONGODB_URI   = "mongodb://localhost:27017/"
NEWS_API_KEY  = "59593215cd46458c9214ba33b88c2831"

# Custom CSS for modern design
st.markdown( """
<style>
    /* Main theme colors */
    :root {
        --primary-color: #FF4B4B;
        --secondary-color: #0E1117;
        --accent-color: #00D9FF;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    .main-header p {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }

    /* Card styling */
    .custom-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }

    /* Result badges */
    .result-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 0.5rem;
    }

    .fake-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }

    .real-badge {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        font-weight: 600;
    }

    /* News card */
    .news-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }

    .news-card:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: translateX(5px);
    }

    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }

    .status-online {
        background: #00ff00;
        box-shadow: 0 0 10px #00ff00;
    }

    .status-offline {
        background: #ff0000;
        box-shadow: 0 0 10px #ff0000;
    }
</style>
""", unsafe_allow_html=True )


# MongoDB connection
@st.cache_resource
def get_mongo_client () :
    try :
        client = MongoClient( MONGODB_URI, serverSelectionTimeoutMS=2000 )
        client.server_info()
        db = client["fake_news_db"]
        return db
    except Exception as e :
        return None


# Initialize session state
if 'analysis_history' not in st.session_state :
    st.session_state.analysis_history = []

# Page config
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.markdown( """
<div class="main-header">
    <h1>🔍 AI-Powered Fake News Detection</h1>
    <p>Multilingual Explainable AI System with XLM-RoBERTa & IndicBERT</p>
</div>
""", unsafe_allow_html=True )

# Sidebar
with st.sidebar :
    st.image( "https://img.icons8.com/clouds/200/000000/news.png", width=150 )

    st.markdown( "### 🎯 Navigation" )
    page = st.radio(
        "Navigation Menu",  # descriptive label
        ["🏠 Dashboard", "📝 Manual Analysis", "📰 Live News", "🗂️ Claims Database", "📊 Analytics"],
        label_visibility="collapsed"
    )

    st.markdown( "---" )

    # System status
    st.markdown( "### 🔧 System Status" )
    db = get_mongo_client()

    if db is not None :
        st.markdown( '<span class="status-indicator status-online"></span>**MongoDB Connected**',
                     unsafe_allow_html=True )
    else :
        st.markdown( '<span class="status-indicator status-offline"></span>**MongoDB Offline**',
                     unsafe_allow_html=True )

    try :
        response = requests.get( f"{API_BASE}/health", timeout=2 )
        if response.status_code == 200 :
            st.markdown( '<span class="status-indicator status-online"></span>**API Online**', unsafe_allow_html=True )
        else :
            st.markdown( '<span class="status-indicator status-offline"></span>**API Offline**',
                         unsafe_allow_html=True )
    except :
        st.markdown( '<span class="status-indicator status-offline"></span>**API Offline**', unsafe_allow_html=True )

    st.markdown( "---" )
    st.markdown( "### ℹ️ About" )
    st.info( """
    This system uses advanced AI models to detect fake news in multiple languages including English and Indian languages.

    **Features:**
    - Real-time analysis
    - Explainable AI
    - Multi-language support
    - Live news monitoring
    """ )


# Helper functions
def predict_news ( text, model ) :
    try :
        response = requests.post( API_URL, json={"text" : text, "model" : model} )
        return response.json()
    except Exception as e :
        return {"error" : str( e )}


def fetch_live_news ( topic, page_size=10 ) :
    try :
        url = "https://newsapi.org/v2/everything"
        params = {
            'q' : f'{topic} India',
            'language' : 'en',
            'sortBy' : 'publishedAt',
            'pageSize' : page_size,
            'apiKey' : NEWS_API_KEY
        }
        response = requests.get( url, params=params )
        data = response.json()

        if data.get( "status" ) == "ok" :
            return data.get( "articles", [] )
        return []
    except Exception as e :
        st.error( f"Error fetching news: {e}" )
        return []


def save_to_mongodb ( collection_name, data ) :
    db = get_mongo_client()
    if db is not None :
        try :
            collection = db[collection_name]
            collection.update_one(
                {"url" : data.get( "url", "manual_" + str( datetime.now().timestamp() ) )},
                {"$set" : data},
                upsert=True
            )
            return True
        except Exception as e :
            st.error( f"Save error: {e}" )
            return False
    return False


# ========== PAGE: DASHBOARD ==========
if page == "🏠 Dashboard" :
    st.markdown( "## 📊 Dashboard Overview" )

    # Metrics row
    col1, col2, col3, col4 = st.columns( 4 )

    with col1 :
        st.markdown( """
        <div class="metric-card">
            <div class="metric-label">Total Analyses</div>
            <div class="metric-value">{}</div>
        </div>
        """.format( len( st.session_state.analysis_history ) ), unsafe_allow_html=True )

    with col2 :
        fake_count = len( [h for h in st.session_state.analysis_history if h.get( 'prediction' ) == 'Fake'] )
        st.markdown( """
        <div class="metric-card">
            <div class="metric-label">Fake Detected</div>
            <div class="metric-value">{}</div>
        </div>
        """.format( fake_count ), unsafe_allow_html=True )

    with col3 :
        real_count = len( [h for h in st.session_state.analysis_history if h.get( 'prediction' ) == 'Real'] )
        st.markdown( """
        <div class="metric-card">
            <div class="metric-label">Real News</div>
            <div class="metric-value">{}</div>
        </div>
        """.format( real_count ), unsafe_allow_html=True )

    with col4 :
        avg_conf = sum( [h.get( 'confidence', 0 ) for h in st.session_state.analysis_history] ) / len(
            st.session_state.analysis_history ) if st.session_state.analysis_history else 0
        st.markdown( """
        <div class="metric-card">
            <div class="metric-label">Avg Confidence</div>
            <div class="metric-value">{:.0%}</div>
        </div>
        """.format( avg_conf ), unsafe_allow_html=True )

    st.markdown( "<br>", unsafe_allow_html=True )

    # Charts
    if st.session_state.analysis_history :
        col1, col2 = st.columns( 2 )

        with col1 :
            st.markdown( "### 📈 Prediction Distribution" )
            df = pd.DataFrame( st.session_state.analysis_history )
            fig = px.pie( df, names='prediction', color='prediction',
                          color_discrete_map={'Fake' : '#f5576c', 'Real' : '#00f2fe'},
                          hole=0.4 )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict( color='white' )
            )
            st.plotly_chart( fig, use_container_width=True )

        with col2 :
            st.markdown( "### 📊 Model Usage" )
            model_counts = df['model'].value_counts()
            fig = px.bar( x=model_counts.index, y=model_counts.values,
                          color=model_counts.index,
                          labels={'x' : 'Model', 'y' : 'Count'} )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict( color='white' ),
                showlegend=False
            )
            st.plotly_chart( fig, use_container_width=True )
    else :
        st.info( "📊 Start analyzing news to see your dashboard statistics!" )

# ========== PAGE: MANUAL ANALYSIS ==========
elif page == "📝 Manual Analysis" :
    st.markdown( "## 📝 Analyze News Article" )

    col1, col2 = st.columns( [3, 1] )

    with col1 :
        text_input = st.text_area(
            "Enter news article text",
            height=300,
            placeholder="Paste or type the news article here for analysis...",
            help="Enter the complete text of the news article you want to analyze"
        )

    with col2 :
        st.markdown( "### ⚙️ Settings" )
        model_choice = st.selectbox(
            "AI Model",
            ["xlmr", "indicbert"],
            help="XLM-R: Best for code-mixed text\nIndicBERT: Best for Indian languages"
        )

        save_result = st.checkbox( "💾 Save to database", value=False )

        st.markdown( "---" )
        st.markdown( "### 📚 Model Info" )
        if model_choice == "xlmr" :
            st.info( "**XLM-RoBERTa**\n\n✓ 100+ languages\n✓ Code-mixed text\n✓ Cross-lingual" )
        else :
            st.info( "**IndicBERT**\n\n✓ 12 Indian languages\n✓ Hindi optimized\n✓ Regional content" )

    if st.button( "🔍 Analyze Now", type="primary", use_container_width=True ) :
        if text_input.strip() :
            with st.spinner( "🤖 AI is analyzing..." ) :
                result = predict_news( text_input, model_choice )

                if "error" not in result :
                    st.success( "✅ Analysis Complete!" )

                    # Display result
                    prediction = result.get( "label", result.get( "prediction", "Unknown" ) )
                    confidence = result.get( "confidence", 0 )

                    if prediction == "Fake" :
                        st.markdown( f'<div class="result-badge fake-badge">🚨 FAKE NEWS DETECTED</div>',
                                     unsafe_allow_html=True )
                    else :
                        st.markdown( f'<div class="result-badge real-badge">✅ AUTHENTIC NEWS</div>',
                                     unsafe_allow_html=True )

                    st.markdown( "<br>", unsafe_allow_html=True )

                    # Metrics
                    col1, col2, col3 = st.columns( 3 )
                    with col1 :
                        st.metric( "Prediction", prediction, delta=None )
                    with col2 :
                        st.metric( "Confidence", f"{confidence:.1%}" )
                    with col3 :
                        st.metric( "Model", model_choice.upper() )

                    # Confidence bar
                    st.markdown( "### 📊 Confidence Level" )
                    st.progress( confidence )

                    # Token importance
                    if "tokens" in result and result["tokens"] :
                        st.markdown( "### 🎯 Key Words Analysis" )
                        with st.expander( "View important tokens", expanded=False ) :
                            tokens_df = pd.DataFrame( result["tokens"] )
                            st.dataframe( tokens_df, use_container_width=True )

                    # Save to history
                    st.session_state.analysis_history.append( {
                        "timestamp" : datetime.now(),
                        "text" : text_input[:100] + "...",
                        "prediction" : prediction,
                        "confidence" : confidence,
                        "model" : model_choice
                    } )

                    if save_result :
                        data = {
                            "text" : text_input,
                            "prediction" : prediction,
                            "confidence" : confidence,
                            "model" : model_choice,
                            "timestamp" : datetime.now(),
                            "source" : "manual_analysis"
                        }
                        if save_to_mongodb( "predictions", data ) :
                            st.success( "💾 Saved to database" )
                else :
                    st.error( f"❌ Error: {result['error']}" )
        else :
            st.warning( "⚠️ Please enter some text to analyze" )

# ========== PAGE: LIVE NEWS ==========
elif page == "📰 Live News" :
    st.markdown( "## 📰 Live News Analysis" )

    col1, col2, col3 = st.columns( [2, 1, 1] )
    with col1 :
        topic = st.selectbox( "📌 Select Topic",
                              ["politics", "social", "health", "technology", "business", "sports"] )
    with col2 :
        num_articles = st.slider( "📰 Articles", 5, 20, 10 )
    with col3 :
        model_for_analysis = st.selectbox( "🤖 Model", ["xlmr", "indicbert"] )

    auto_analyze = st.checkbox( "⚡ Auto-analyze all articles", value=False )

    if st.button( "🔄 Fetch Latest News", type="primary", use_container_width=True ) :
        with st.spinner( "📡 Fetching latest news..." ) :
            articles = fetch_live_news( topic, num_articles )

            if articles :
                st.success( f"✅ Fetched {len( articles )} articles" )

                for idx, article in enumerate( articles ) :
                    with st.container() :
                        st.markdown( f"""
                        <div class="news-card">
                            <h3>📰 {article.get( 'title', 'No title' )}</h3>
                        </div>
                        """, unsafe_allow_html=True )

                        col1, col2 = st.columns( [3, 1] )

                        with col1 :
                            st.write( f"**📰 Source:** {article.get( 'source', {} ).get( 'name', 'Unknown' )}" )
                            st.write( f"**📅 Published:** {article.get( 'publishedAt', 'Unknown' )}" )
                            st.write( f"**📝 Description:** {article.get( 'description', 'No description' )}" )

                            if article.get( 'url' ) :
                                st.markdown( f"[🔗 Read full article]({article['url']})" )

                        with col2 :
                            if auto_analyze or st.button( f"🔍 Analyze", key=f"analyze_{idx}" ) :
                                content = article.get( 'content' ) or article.get( 'description', '' )
                                if content :
                                    result = predict_news( content, model_for_analysis )

                                    if "error" not in result :
                                        prediction = result.get( "label", result.get( "prediction", "Unknown" ) )
                                        confidence = result.get( "confidence", 0 )

                                        if prediction == "Fake" :
                                            st.error( f"🚨 {prediction}" )
                                        else :
                                            st.success( f"✅ {prediction}" )

                                        st.metric( "Confidence", f"{confidence:.1%}" )

                                        # Save to DB
                                        data = {
                                            "title" : article.get( 'title' ),
                                            "content" : content,
                                            "source" : article.get( 'source', {} ).get( 'name' ),
                                            "url" : article.get( 'url' ),
                                            "publishedAt" : article.get( 'publishedAt' ),
                                            "prediction" : prediction,
                                            "confidence" : confidence,
                                            "model" : model_for_analysis,
                                            "analyzed_at" : datetime.now(),
                                            "topic" : topic
                                        }
                                        save_to_mongodb( "analyzed_news", data )

                        st.markdown( "---" )
            else :
                st.warning( "📭 No articles found" )

# ========== PAGE: CLAIMS DATABASE ==========
elif page == "🗂️ Claims Database" :
    st.markdown( "## 🗂️ AltNews Fact-Check Claims" )

    db = get_mongo_client()

    if db is not None :
        col1, col2 = st.columns( [1, 3] )

        with col1 :
            if st.button( "🔄 Refresh Database", type="primary", use_container_width=True ) :
                st.info( "Run `python altnews_scrapper.py` to update" )

        with col2 :
            search_query = st.text_input( "🔍 Search claims", placeholder="Enter keyword..." )

        try :
            collection = db["altnews_claims"]

            if search_query :
                claims = list( collection.find(
                    {"claim_text" : {"$regex" : search_query, "$options" : "i"}},
                    limit=50
                ) )
            else :
                claims = list( collection.find().sort( "date", -1 ).limit( 50 ) )

            if claims :
                st.info( f"📊 Showing {len( claims )} claims" )

                for claim in claims :
                    with st.expander( f"🚨 {claim.get( 'claim_text', 'No title' )[:100]}..." ) :
                        col1, col2 = st.columns( [2, 1] )

                        with col1 :
                            st.write( f"**✅ Verdict:** {claim.get( 'verdict', 'Unknown' )}" )
                            st.write( f"**🌐 Language:** {claim.get( 'language', 'Unknown' )}" )
                            st.write( f"**📌 Topic:** {claim.get( 'topic', 'Unknown' )}" )
                            st.write( f"**📅 Date:** {claim.get( 'date', 'Unknown' )}" )

                            if claim.get( 'source' ) :
                                st.markdown( f"[🔗 View on AltNews]({claim['source']})" )

                        with col2 :
                            if st.button( "🔍 Re-analyze", key=f"reanalyze_{claim.get( 'id' )}" ) :
                                result = predict_news( claim.get( 'claim_text', '' ), "xlmr" )
                                if "error" not in result :
                                    st.metric( "AI Prediction", result.get( "label", result.get( "prediction", "Unknown" ) ) )
                                    st.metric( "Confidence", f"{result.get( 'confidence', 0 ):.1%}" )
            else :
                st.warning( "📭 No claims in database" )

        except Exception as e :
            st.error( f"❌ Error: {e}" )
    else :
        st.error( "❌ MongoDB connection unavailable" )

# ========== PAGE: ANALYTICS ==========
elif page == "📊 Analytics" :
    st.markdown( "## 📊 Detailed Analytics" )

    if st.session_state.analysis_history :
        df = pd.DataFrame( st.session_state.analysis_history )

        # Top metrics
        col1, col2, col3, col4 = st.columns( 4 )
        with col1 :
            st.metric( "📈 Total", len( df ) )
        with col2 :
            fake = len( df[df['prediction'] == 'Fake'] )
            st.metric( "🚨 Fake", fake, delta=f"{fake / len( df ) * 100:.1f}%" )
        with col3 :
            real = len( df[df['prediction'] == 'Real'] )
            st.metric( "✅ Real", real, delta=f"{real / len( df ) * 100:.1f}%" )
        with col4 :
            avg_conf = df['confidence'].mean()
            st.metric( "💯 Avg Confidence", f"{avg_conf:.1%}" )

        st.markdown( "---" )

        # Charts
        col1, col2 = st.columns( 2 )

        with col1 :
            st.markdown( "### 📊 Predictions Over Time" )
            df['timestamp'] = pd.to_datetime( df['timestamp'] )
            timeline = df.groupby( [df['timestamp'].dt.date, 'prediction'] ).size().reset_index( name='count' )
            fig = px.line( timeline, x='timestamp', y='count', color='prediction',
                           color_discrete_map={'Fake' : '#f5576c', 'Real' : '#00f2fe'} )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict( color='white' )
            )
            st.plotly_chart( fig, use_container_width=True )

        with col2 :
            st.markdown( "### 📈 Confidence Distribution" )
            fig = px.histogram( df, x='confidence', nbins=20,
                                color='prediction',
                                color_discrete_map={'Fake' : '#f5576c', 'Real' : '#00f2fe'} )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict( color='white' )
            )
            st.plotly_chart( fig, use_container_width=True )

        st.markdown( "### 📋 Recent Analysis History" )
        st.dataframe( df.sort_values( 'timestamp', ascending=False ).head( 20 ),
                      use_container_width=True )

        if st.button( "🗑️ Clear History", type="secondary" ) :
            st.session_state.analysis_history = []
            st.rerun()
    else :
        st.info( "📊 No data yet. Start analyzing to see analytics!" )