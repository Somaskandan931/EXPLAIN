import streamlit as st
import requests
import pandas as pd
from pymongo import MongoClient
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000/predict"
MONGODB_URI = "mongodb://localhost:27017/"
NEWS_API_KEY = "59593215cd46458c9214ba33b88c2831"


# MongoDB connection
@st.cache_resource
def get_mongo_client () :
    try :
        client = MongoClient( MONGODB_URI )
        db = client["fake_news_db"]
        return db
    except Exception as e :
        st.error( f"MongoDB connection failed: {e}" )
        return None


# Initialize session state
if 'analysis_history' not in st.session_state :
    st.session_state.analysis_history = []

# Page styling
st.set_page_config( page_title="Fake News Detection System", layout="wide" )
st.title( "🔍 Explainable Fake News Detection System" )

# Sidebar for navigation
page = st.sidebar.radio( "Navigation", [
    "Manual Analysis",
    "Live News Feed",
    "AltNews Claims Database",
    "Analysis History"
] )


# Helper function: Predict news
def predict_news ( text, model ) :
    try :
        response = requests.post( API_URL, json={"text" : text, "model" : model} )
        return response.json()
    except Exception as e :
        return {"error" : str( e )}


# Helper function: Fetch live news
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
        else :
            st.error( f"News API error: {data.get( 'message', 'Unknown error' )}" )
            return []
    except Exception as e :
        st.error( f"Error fetching news: {e}" )
        return []


# Helper function: Save to MongoDB
def save_to_mongodb ( collection_name, data ) :
    db = get_mongo_client()
    if db is not None :
        try :
            collection = db[collection_name]
            collection.update_one(
                {"url" : data.get( "url" )},
                {"$set" : data},
                upsert=True
            )
            return True
        except Exception as e :
            st.error( f"MongoDB save error: {e}" )
            return False
    return False


# ========== PAGE 1: MANUAL ANALYSIS ==========
if page == "Manual Analysis" :
    st.header( "📝 Analyze News Article" )

    col1, col2 = st.columns( [3, 1] )

    with col1 :
        text_input = st.text_area(
            "Enter news article text",
            height=200,
            placeholder="Paste or type the news article here..."
        )

    with col2 :
        model_choice = st.selectbox(
            "Select Model",
            ["xlmr", "indicbert", "tfidf"],
            help="Choose the ML model for prediction"
        )

        save_result = st.checkbox( "Save to database", value=False )

    if st.button( "🔍 Analyze", type="primary" ) :
        if text_input.strip() :
            with st.spinner( "Analyzing..." ) :
                result = predict_news( text_input, model_choice )

                if "error" not in result :
                    # Display results
                    st.success( "Analysis Complete!" )

                    col1, col2, col3 = st.columns( 3 )
                    with col1 :
                        prediction = result.get( "prediction", "Unknown" )
                        color = "🔴" if prediction == "Fake" else "🟢"
                        st.metric( "Prediction", f"{color} {prediction}" )

                    with col2 :
                        confidence = result.get( "confidence", 0 )
                        st.metric( "Confidence", f"{confidence:.2%}" )

                    with col3 :
                        st.metric( "Model Used", model_choice.upper() )

                    # Show explanation if available
                    if "explanation" in result :
                        st.subheader( "🧠 Explanation" )
                        st.write( result["explanation"] )

                    # Save to history
                    st.session_state.analysis_history.append( {
                        "timestamp" : datetime.now(),
                        "text" : text_input[:100] + "...",
                        "prediction" : prediction,
                        "confidence" : confidence,
                        "model" : model_choice
                    } )

                    # Save to MongoDB if requested
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
                            st.success( "✅ Saved to database" )
                else :
                    st.error( f"❌ Error: {result['error']}" )
        else :
            st.warning( "⚠️ Please enter some text to analyze" )

# ========== PAGE 2: LIVE NEWS FEED ==========
elif page == "Live News Feed" :
    st.header( "📰 Live News Analysis" )

    col1, col2, col3 = st.columns( [2, 1, 1] )
    with col1 :
        topic = st.selectbox( "Select Topic", ["politics", "social", "health", "technology", "business"] )
    with col2 :
        num_articles = st.slider( "Number of articles", 5, 20, 10 )
    with col3 :
        auto_analyze = st.checkbox( "Auto-analyze", value=False )

    model_for_analysis = st.selectbox( "Model for analysis", ["xlmr", "indicbert", "tfidf"] )

    if st.button( "🔄 Fetch Latest News", type="primary" ) :
        with st.spinner( "Fetching news..." ) :
            articles = fetch_live_news( topic, num_articles )

            if articles :
                st.success( f"✅ Fetched {len( articles )} articles" )

                for idx, article in enumerate( articles ) :
                    with st.expander( f"📄 {article.get( 'title', 'No title' )}", expanded=(idx == 0) ) :
                        col1, col2 = st.columns( [3, 1] )

                        with col1 :
                            st.write( f"**Source:** {article.get( 'source', {} ).get( 'name', 'Unknown' )}" )
                            st.write( f"**Published:** {article.get( 'publishedAt', 'Unknown' )}" )
                            st.write( f"**Description:** {article.get( 'description', 'No description' )}" )

                            if article.get( 'url' ) :
                                st.markdown( f"[Read full article]({article['url']})" )

                        with col2 :
                            if auto_analyze or st.button( f"Analyze", key=f"analyze_{idx}" ) :
                                content = article.get( 'content' ) or article.get( 'description', '' )
                                if content :
                                    result = predict_news( content, model_for_analysis )

                                    if "error" not in result :
                                        prediction = result.get( "prediction", "Unknown" )
                                        confidence = result.get( "confidence", 0 )

                                        color = "🔴" if prediction == "Fake" else "🟢"
                                        st.metric( "Result", f"{color} {prediction}" )
                                        st.metric( "Confidence", f"{confidence:.2%}" )

                                        # Save to MongoDB
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
                                            "topic" : topic,
                                            "source_type" : "NewsAPI"
                                        }
                                        save_to_mongodb( "analyzed_news", data )
                                    else :
                                        st.error( "Analysis failed" )
            else :
                st.warning( "No articles found" )

# ========== PAGE 3: ALTNEWS CLAIMS DATABASE ==========
elif page == "AltNews Claims Database" :
    st.header( "🗂️ AltNews Fact-Check Claims" )

    db = get_mongo_client()

    if db is not None :
        col1, col2 = st.columns( [1, 3] )

        with col1 :
            if st.button( "🔄 Refresh from AltNews", type="primary" ) :
                with st.spinner( "Scraping AltNews..." ) :
                    try :
                        # You can trigger the scraper here or show a message
                        st.info( "Run the altnews_scrapper.py script to update the database" )
                    except Exception as e :
                        st.error( f"Error: {e}" )

        with col2 :
            search_query = st.text_input( "🔍 Search claims", placeholder="Search by keyword..." )

        # Fetch from MongoDB
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

                # Display as dataframe
                df = pd.DataFrame( claims )
                if '_id' in df.columns :
                    df = df.drop( '_id', axis=1 )

                # Display each claim
                for claim in claims :
                    with st.expander( f"🚨 {claim.get( 'claim_text', 'No title' )[:100]}..." ) :
                        col1, col2 = st.columns( [2, 1] )

                        with col1 :
                            st.write( f"**Verdict:** {claim.get( 'verdict', 'Unknown' )}" )
                            st.write( f"**Language:** {claim.get( 'language', 'Unknown' )}" )
                            st.write( f"**Topic:** {claim.get( 'topic', 'Unknown' )}" )
                            st.write( f"**Date:** {claim.get( 'date', 'Unknown' )}" )

                            if claim.get( 'source' ) :
                                st.markdown( f"[View on AltNews]({claim['source']})" )

                        with col2 :
                            if st.button( "Re-analyze", key=f"reanalyze_{claim.get( 'id' )}" ) :
                                result = predict_news( claim.get( 'claim_text', '' ), "xlmr" )
                                if "error" not in result :
                                    st.metric( "ML Prediction", result.get( "prediction" ) )
                                    st.metric( "Confidence", f"{result.get( 'confidence', 0 ):.2%}" )
            else :
                st.warning( "No claims found in database. Run the scraper first." )

        except Exception as e :
            st.error( f"Error fetching claims: {e}" )
    else :
        st.error( "MongoDB connection unavailable" )

# ========== PAGE 4: ANALYSIS HISTORY ==========
elif page == "Analysis History" :
    st.header( "📊 Analysis History" )

    if st.session_state.analysis_history :
        df = pd.DataFrame( st.session_state.analysis_history )

        # Display metrics
        col1, col2, col3 = st.columns( 3 )
        with col1 :
            st.metric( "Total Analyses", len( df ) )
        with col2 :
            fake_count = len( df[df['prediction'] == 'Fake'] )
            st.metric( "Fake News Detected", fake_count )
        with col3 :
            avg_confidence = df['confidence'].mean()
            st.metric( "Avg Confidence", f"{avg_confidence:.2%}" )

        # Display table
        st.dataframe( df, use_container_width=True )

        # Clear history button
        if st.button( "🗑️ Clear History" ) :
            st.session_state.analysis_history = []
            st.rerun()
    else :
        st.info( "No analysis history yet. Start analyzing articles!" )

# Footer
st.sidebar.markdown( "---" )
st.sidebar.info(
    "**System Status**\n\n"
    f"MongoDB: {'🟢 Connected' if get_mongo_client() is not None else '🔴 Disconnected'}\n\n"
    f"API: Localhost:8000"
)