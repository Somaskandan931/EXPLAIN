"""
Quick Start Script for Fake News Detection System
Checks configuration, models, and helps with initial setup
"""

import os
import sys
from pathlib import Path
import subprocess


def print_header ( text ) :
    """Print a formatted header"""
    print( "\n" + "=" * 60 )
    print( f"  {text}" )
    print( "=" * 60 )


def check_python_version () :
    """Check Python version"""
    print_header( "Checking Python Version" )
    version = sys.version_info
    print( f"Python version: {version.major}.{version.minor}.{version.micro}" )

    if version.major < 3 or (version.major == 3 and version.minor < 8) :
        print( "❌ Python 3.8 or higher is required" )
        return False
    print( "✓ Python version is compatible" )
    return True


def check_dependencies () :
    """Check if required packages are installed"""
    print_header( "Checking Dependencies" )

    required_packages = [
        'torch',
        'transformers',
        'fastapi',
        'uvicorn',
        'streamlit',
        'captum',
        'pymongo',
        'pandas',
        'numpy',
        'scikit-learn'
    ]

    missing = []
    for package in required_packages :
        try :
            __import__( package )
            print( f"✓ {package}" )
        except ImportError :
            print( f"❌ {package} - NOT INSTALLED" )
            missing.append( package )

    if missing :
        print( f"\n⚠️  Missing packages: {', '.join( missing )}" )
        print( "Run: pip install -r requirements.txt" )
        return False

    print( "\n✓ All dependencies are installed" )
    return True


def check_project_structure () :
    """Check if project structure is correct"""
    print_header( "Checking Project Structure" )

    base_dir = Path.cwd()
    required_dirs = [
        'app',
        'explainability',
        'continual_learning',
        'ui',
        'models',
        'data'
    ]

    required_files = [
        'app/main.py',
        'app/config.py',
        'explainability/unified_explainer.py',
        'explainability/integrated_gradients.py',
        'ui/streamlit_app.py',
        'model_loader.py'
    ]

    all_good = True

    # Check directories
    for dir_name in required_dirs :
        dir_path = base_dir / dir_name
        if dir_path.exists() :
            print( f"✓ {dir_name}/" )
        else :
            print( f"❌ {dir_name}/ - MISSING" )
            all_good = False
            # Create missing directory
            dir_path.mkdir( parents=True, exist_ok=True )
            print( f"  Created {dir_name}/" )

    # Check files
    print( "\nChecking required files:" )
    for file_name in required_files :
        file_path = base_dir / file_name
        if file_path.exists() :
            print( f"✓ {file_name}" )
        else :
            print( f"❌ {file_name} - MISSING" )
            all_good = False

    return all_good


def check_models () :
    """Check if model files exist"""
    print_header( "Checking Model Files" )

    base_dir = Path.cwd()
    models_dir = base_dir / 'models'

    model_checks = [
        ('XLM-RoBERTa', models_dir / 'xlmr_model'),
        ('IndicBERT', models_dir / 'indicbert_model'),
        ('TF-IDF Model', models_dir / 'tfidf_model.pkl'),
        ('TF-IDF Vectorizer', models_dir / 'tfidf_vectorizer.pkl')
    ]

    all_exist = True
    for name, path in model_checks :
        if path.exists() :
            print( f"✓ {name}: {path}" )
        else :
            print( f"❌ {name}: {path} - NOT FOUND" )
            all_exist = False

    if not all_exist :
        print( "\n⚠️  Some models are missing!" )
        print( "\nTo add your models:" )
        print( "1. For transformer models (XLM-R, IndicBERT):" )
        print( "   model.save_pretrained('models/xlmr_model')" )
        print( "   tokenizer.save_pretrained('models/xlmr_model')" )
        print( "\n2. For TF-IDF model:" )
        print( "   import pickle" )
        print( "   pickle.dump(model, open('models/tfidf_model.pkl', 'wb'))" )
        print( "   pickle.dump(vectorizer, open('models/tfidf_vectorizer.pkl', 'wb'))" )
        return False

    print( "\n✓ All model files are present" )
    return True


def check_mongodb () :
    """Check MongoDB connection"""
    print_header( "Checking MongoDB" )

    try :
        from pymongo import MongoClient
        client = MongoClient( "mongodb://localhost:27017/", serverSelectionTimeoutMS=2000 )
        client.server_info()  # Will throw exception if can't connect
        print( "✓ MongoDB is running and accessible" )
        return True
    except Exception as e :
        print( f"⚠️  MongoDB connection failed: {e}" )
        print( "Note: The app will work without MongoDB, but data won't persist" )
        return False


def test_model_loading () :
    """Test if models can be loaded"""
    print_header( "Testing Model Loading" )

    try :
        import config
        from model_loader import check_model_files

        exists, message = check_model_files( config )
        if exists :
            print( "✓ All model files validated" )

            # Try loading
            print( "\nAttempting to load models..." )
            from model_loader import ModelLoader

            loader = ModelLoader( config )
            print( "  Loading XLM-R model..." )
            xlmr_model, xlmr_tokenizer = loader.load_xlmr_model()
            print( "  ✓ XLM-R loaded" )

            print( "  Loading IndicBERT model..." )
            indic_model, indic_tokenizer = loader.load_indicbert_model()
            print( "  ✓ IndicBERT loaded" )

            print( "  Loading TF-IDF model..." )
            tfidf_model, tfidf_vec, X_train = loader.load_tfidf_model()
            print( "  ✓ TF-IDF loaded" )

            print( "\n✓ All models loaded successfully!" )
            return True
        else :
            print( f"❌ Model validation failed:\n{message}" )
            return False

    except Exception as e :
        print( f"❌ Model loading failed: {e}" )
        return False


def create_init_files () :
    """Create __init__.py files in all packages"""
    print_header( "Creating Package Init Files" )

    base_dir = Path.cwd()
    packages = ['app', 'explainability', 'continual_learning', 'ui']

    for package in packages :
        init_file = base_dir / package / '__init__.py'
        if not init_file.exists() :
            init_file.touch()
            print( f"✓ Created {package}/__init__.py" )
        else :
            print( f"✓ {package}/__init__.py exists" )


def print_next_steps ( all_checks_passed ) :
    """Print next steps based on check results"""
    print_header( "Summary & Next Steps" )

    if all_checks_passed :
        print( "✅ All checks passed! Your system is ready." )
        print( "\nTo start the system:" )
        print( "\n1. Start the API (Terminal 1):" )
        print( "   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000" )
        print( "\n2. Start Streamlit (Terminal 2):" )
        print( "   streamlit run ui/streamlit_app.py" )
        print( "\n3. Access the application:" )
        print( "   - API Docs: http://localhost:8000/docs" )
        print( "   - Streamlit UI: http://localhost:8501" )
    else :
        print( "⚠️  Some checks failed. Please resolve the issues above." )
        print( "\nCommon solutions:" )
        print( "1. Install dependencies: pip install -r requirements.txt" )
        print( "2. Add your trained models to the models/ directory" )
        print( "3. Make sure all required files are in place" )
        print( "\nRun this script again after fixing issues." )


def main () :
    """Run all checks"""
    print_header( "Fake News Detection System - Quick Start" )
    print( "This script will check your setup and help you get started" )

    checks = []

    # Run all checks
    checks.append( ("Python Version", check_python_version()) )
    checks.append( ("Dependencies", check_dependencies()) )
    checks.append( ("Project Structure", check_project_structure()) )

    # Create init files
    create_init_files()

    # Continue with other checks
    checks.append( ("Model Files", check_models()) )
    checks.append( ("MongoDB", check_mongodb()) )

    # Test model loading only if models exist
    if checks[-2][1] :  # If model files check passed
        checks.append( ("Model Loading", test_model_loading()) )

    # Print summary
    print_header( "Check Results" )
    for check_name, passed in checks :
        status = "✓ PASSED" if passed else "❌ FAILED"
        print( f"{check_name}: {status}" )

    all_passed = all( passed for _, passed in checks )
    print_next_steps( all_passed )

    return 0 if all_passed else 1


if __name__ == "__main__" :
    sys.exit( main() )