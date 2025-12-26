"""
Quick Start Script for Fake News Detection System
Checks configuration, models, and helps with initial setup
"""

import os
import sys
from pathlib import Path
import subprocess


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_subheader(text):
    """Print a formatted subheader"""
    print(f"\n── {text} " + "─" * (60 - len(text)))


def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current: {version.major}.{version.minor}.{version.micro}")
        print("   Please upgrade Python and try again")
        return False

    if version.minor >= 8 and version.minor < 12:
        print("✓ Python version is compatible")
    else:
        print(f"✓ Python {version.major}.{version.minor} detected (recommended: 3.8-3.11)")

    return True


def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")

    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'peft': 'PEFT (for LoRA)',
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'streamlit': 'Streamlit',
        'captum': 'Captum (XAI)',
        'pymongo': 'PyMongo',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'scikit-learn': 'Scikit-learn',
        'pydantic': 'Pydantic'
    }

    installed = []
    missing = []

    for package, full_name in required_packages.items():
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {full_name:30} ({package})")
            installed.append(package)
        except ImportError:
            print(f"❌ {full_name:30} ({package}) - NOT INSTALLED")
            missing.append(package)

    print(f"\n{len(installed)}/{len(required_packages)} packages installed")

    if missing:
        print(f"\n⚠️  Missing {len(missing)} package(s)")
        print("\n💡 To install missing packages:")
        print("   pip install -r requirements.txt")
        print("\n   Or install individually:")
        for pkg in missing:
            print(f"   pip install {pkg}")
        return False

    print("\n✅ All dependencies are installed")
    return True


def check_project_structure():
    """Check if project structure is correct"""
    print_header("Checking Project Structure")

    base_dir = Path.cwd()

    structure = {
        'directories': [
            ('app', 'Core API application'),
            ('explainability', 'XAI components'),
            ('continual_learning', 'Continual learning modules'),
            ('ui', 'User interface'),
            ('models', 'Model storage'),
            ('data', 'Dataset storage'),
        ],
        'files': [
            ('main.py', 'Main API entry point'),
            ('app/config.py', 'Configuration file'),
            ('model_loader.py', 'Model loader utility'),
            ('requirements.txt', 'Dependencies list')
        ]
    }

    all_good = True
    created_dirs = []

    # Check directories
    print_subheader("Directories")
    for dir_name, description in structure['directories']:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name:25} - {description}")
        else:
            print(f"⚠️  {dir_name:25} - MISSING (will create)")
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"   └─ Created {dir_name}/")
                created_dirs.append(dir_name)
            except Exception as e:
                print(f"   └─ Failed to create: {e}")

    # Check files
    print_subheader("Key Files")
    for file_name, description in structure['files']:
        file_path = base_dir / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
            print(f"✓ {file_name:40} ({size_str})")
        else:
            print(f"⚠️  {file_name:40} - MISSING")

    if created_dirs:
        print(f"\n📁 Created {len(created_dirs)} missing director{'y' if len(created_dirs) == 1 else 'ies'}")

    return True  # Always return True since we create missing dirs


def check_models():
    """Check if model files exist"""
    print_header("Checking Model Files")

    base_dir = Path.cwd()
    models_dir = base_dir / 'models'

    if not models_dir.exists():
        print("❌ models/ directory does not exist")
        models_dir.mkdir(exist_ok=True)
        print("   Created models/ directory")
        return False

    print_subheader("LoRA Models")

    model_status = []

    # Check XLM-RoBERTa LoRA
    xlmr_lora = models_dir / 'xlmr_lora'
    if xlmr_lora.exists() and xlmr_lora.is_dir():
        checkpoints = sorted([d for d in xlmr_lora.iterdir()
                            if d.is_dir() and d.name.startswith('checkpoint-')])

        if checkpoints:
            size = sum(f.stat().st_size for f in xlmr_lora.rglob('*') if f.is_file())
            size_mb = size / (1024 * 1024)
            print(f"✓ XLM-RoBERTa LoRA")
            print(f"   📂 Path: {xlmr_lora}")
            print(f"   📦 Size: {size_mb:.1f} MB")
            print(f"   📊 Checkpoints: {len(checkpoints)}")
            print(f"      • First: {checkpoints[0].name}")
            print(f"      • Latest: {checkpoints[-1].name}")
            model_status.append(('xlmr_lora', True))
        else:
            print(f"⚠️  XLM-RoBERTa LoRA - No checkpoints found!")
            print(f"   📂 Path: {xlmr_lora}")
            model_status.append(('xlmr_lora', False))
    else:
        print(f"❌ XLM-RoBERTa LoRA - NOT FOUND")
        print(f"   Expected at: {xlmr_lora}")
        model_status.append(('xlmr_lora', False))

    # Check IndicBERT LoRA
    indic_lora = models_dir / 'indicbert_lora'
    if indic_lora.exists() and indic_lora.is_dir():
        checkpoints = sorted([d for d in indic_lora.iterdir()
                            if d.is_dir() and d.name.startswith('checkpoint-')])

        if checkpoints:
            size = sum(f.stat().st_size for f in indic_lora.rglob('*') if f.is_file())
            size_mb = size / (1024 * 1024)
            print(f"\n✓ IndicBERT LoRA")
            print(f"   📂 Path: {indic_lora}")
            print(f"   📦 Size: {size_mb:.1f} MB")
            print(f"   📊 Checkpoints: {len(checkpoints)}")
            print(f"      • First: {checkpoints[0].name}")
            print(f"      • Latest: {checkpoints[-1].name}")
            model_status.append(('indicbert_lora', True))
        else:
            print(f"\n⚠️  IndicBERT LoRA - No checkpoints found!")
            print(f"   📂 Path: {indic_lora}")
            model_status.append(('indicbert_lora', False))
    else:
        print(f"\n❌ IndicBERT LoRA - NOT FOUND")
        print(f"   Expected at: {indic_lora}")
        model_status.append(('indicbert_lora', False))

    # Summary
    found = sum(1 for _, status in model_status if status)
    print(f"\n📊 {found}/2 LoRA model(s) found with valid checkpoints")

    if found == 0:
        print("\n❌ No models found!")
        print("\n💡 Your LoRA models should be in:")
        print("   - models/xlmr_lora/ (with checkpoint-* directories)")
        print("   - models/indicbert_lora/ (with checkpoint-* directories)")
        return False
    elif found == 1:
        print("\n⚠️  Only 1 model found - you can still run with single model")
        return True
    else:
        print("\n✅ Both LoRA models are present")
        return True


def check_mongodb():
    """Check MongoDB connection"""
    print_header("Checking MongoDB Connection")

    try:
        from pymongo import MongoClient
        from pymongo.errors import ServerSelectionTimeoutError

        print("Attempting to connect to MongoDB...")
        client = MongoClient(
            "mongodb://localhost:27017/",
            serverSelectionTimeoutMS=3000
        )

        # Test connection
        client.server_info()

        # Get database info
        db_list = client.list_database_names()
        print(f"✓ MongoDB is running at localhost:27017")
        print(f"  Available databases: {', '.join(db_list)}")

        # Check if our database exists
        if 'fake_news_db' in db_list:
            db = client['fake_news_db']
            collections = db.list_collection_names()
            print(f"  ✓ Database 'fake_news_db' exists")
            if collections:
                print(f"    Collections: {', '.join(collections)}")
        else:
            print("  ℹ️  Database 'fake_news_db' will be created on first use")

        client.close()
        return True

    except ServerSelectionTimeoutError:
        print("❌ MongoDB connection timeout")
        print("   MongoDB is not running or not accessible")
        print("\n💡 To start MongoDB:")
        print("   - Windows: Start MongoDB service from Services")
        print("   - Linux: sudo systemctl start mongod")
        print("   - macOS: brew services start mongodb-community")
        print("\n⚠️  Note: App will work without MongoDB (no data persistence)")
        return False

    except Exception as e:
        print(f"⚠️  MongoDB check failed: {e}")
        print("   App will work without MongoDB (no data persistence)")
        return False


def test_model_loading():
    """Test if models can be loaded"""
    print_header("Testing Model Loading")

    try:
        # Add project root to path
        sys.path.insert(0, str(Path.cwd()))

        print("Importing configuration...")
        from app import config
        print("✓ Config imported")

        print("\nImporting model loader...")
        from model_loader import ModelLoader
        print("✓ Model loader imported")

        # Try loading models
        print("\n" + "─" * 70)
        print("Loading LoRA models (this may take a minute)...")
        print("─" * 70)

        loader = ModelLoader(config)

        loaded_models = []

        # Load XLM-R LoRA
        print("\n1/2 Loading XLM-RoBERTa LoRA...")
        try:
            xlmr_model, xlmr_tokenizer = loader.load_xlmr_model()
            print("    ✓ XLM-RoBERTa LoRA loaded successfully")
            loaded_models.append("XLM-RoBERTa LoRA")
        except Exception as e:
            print(f"    ❌ Failed: {str(e)[:100]}")

        # Load IndicBERT LoRA
        print("\n2/2 Loading IndicBERT LoRA...")
        try:
            indic_model, indic_tokenizer = loader.load_indicbert_model()
            print("    ✓ IndicBERT LoRA loaded successfully")
            loaded_models.append("IndicBERT LoRA")
        except Exception as e:
            print(f"    ⚠️  Failed: {str(e)[:100]}")

        print("\n" + "─" * 70)
        if len(loaded_models) > 0:
            print(f"✅ Successfully loaded {len(loaded_models)} model(s):")
            for model in loaded_models:
                print(f"   • {model}")
            print("─" * 70)
            return True
        else:
            print("❌ No models could be loaded")
            print("─" * 70)
            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all files are in place and dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()
        return False


def create_init_files():
    """Create __init__.py files in all packages"""
    print_header("Creating Package Init Files")

    base_dir = Path.cwd()
    packages = ['app', 'explainability', 'continual_learning', 'ui']

    created = 0
    existing = 0

    for package in packages:
        init_file = base_dir / package / '__init__.py'
        package_dir = base_dir / package

        if not package_dir.exists():
            print(f"⚠️  {package}/ directory doesn't exist, skipping")
            continue

        if not init_file.exists():
            try:
                init_file.touch()
                print(f"✓ Created {package}/__init__.py")
                created += 1
            except Exception as e:
                print(f"❌ Failed to create {package}/__init__.py: {e}")
        else:
            existing += 1

    print(f"\n📝 Status: {existing} existing, {created} created")


def check_config_file():
    """Check if config file exists and is properly set up"""
    print_header("Checking Configuration File")

    config_path = Path.cwd() / 'app' / 'config.py'

    if not config_path.exists():
        print("❌ app/config.py not found")
        return False

    print("✓ config.py exists")

    # Try to import and check key settings
    try:
        sys.path.insert(0, str(Path.cwd()))
        from app import config

        # Required for LoRA models
        required_attrs = [
            'XLMR_MODEL_PATH',
            'INDICBERT_MODEL_PATH',
            'API_HOST',
            'API_PORT',
            'DEVICE'
        ]

        print("\nConfiguration values:")
        missing = []
        for attr in required_attrs:
            if hasattr(config, attr):
                value = getattr(config, attr)
                print(f"✓ {attr:25} = {value}")
            else:
                print(f"❌ {attr:25} - NOT FOUND")
                missing.append(attr)

        if missing:
            print(f"\n⚠️  Missing {len(missing)} required setting(s)")
            return False

        print("\n✅ Configuration is valid")
        return True

    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_next_steps(check_results):
    """Print next steps based on check results"""
    print_header("Summary & Next Steps")

    # Count passed/failed
    passed = sum(1 for _, status in check_results if status)
    failed = len(check_results) - passed

    print(f"\n📊 Results: {passed}/{len(check_results)} checks passed")

    if failed > 0:
        print(f"\n⚠️  {failed} check(s) had issues:")
        for name, status in check_results:
            if not status:
                print(f"   • {name}")

    # Need at least: Python, Dependencies, Config, and Models to run
    critical_passed = all(status for name, status in check_results
                         if name in ['Python Version', 'Dependencies', 'Configuration', 'Model Files'])

    if critical_passed:
        print("\n" + "🎉" * 35)
        print("\n✅ System is ready! You can now start the application.")
        print("\n" + "🎉" * 35)

        print("\n" + "─" * 70)
        print("🚀 How to Start the Application")
        print("─" * 70)

        print("\n📍 Step 1: Start the FastAPI Backend")
        print("   Open a terminal in this directory and run:")
        print("\n   python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000")
        print("\n   OR (if main.py is in app/):")
        print("\n   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")

        print("\n📍 Step 2: Start the Streamlit UI (Optional)")
        print("   Open a NEW terminal and run:")
        print("\n   streamlit run ui/streamlit_app.py")

        print("\n📍 Step 3: Access the Application")
        print("   - 📚 API Docs:    http://localhost:8000/docs")
        print("   - 🌐 Streamlit:   http://localhost:8501")
        print("   - ❤️  Health:      http://localhost:8000/health")

        print("\n📍 Step 4: Test with curl")
        print("   curl -X POST http://localhost:8000/predict \\")
        print("     -H 'Content-Type: application/json' \\")
        print("     -d '{\"text\": \"Breaking news story\", \"model\": \"xlmr\"}'")

    else:
        print("\n⚠️  Critical checks failed. Please fix the issues above.")
        print("\n" + "─" * 70)
        print("💡 Common Solutions")
        print("─" * 70)

        for name, status in check_results:
            if not status:
                if name == "Dependencies":
                    print("\n📦 Fix Dependencies:")
                    print("   pip install torch transformers peft fastapi uvicorn")
                    print("   pip install streamlit captum pymongo pandas numpy scikit-learn")
                elif name == "Model Files":
                    print("\n🤖 Fix Models:")
                    print("   Ensure your LoRA models are in:")
                    print("   - models/xlmr_lora/")
                    print("   - models/indicbert_lora/")
                    print("   Each should have checkpoint-* directories")
                elif name == "Configuration":
                    print("\n⚙️  Fix Configuration:")
                    print("   Check that app/config.py has:")
                    print("   - XLMR_MODEL_PATH")
                    print("   - INDICBERT_MODEL_PATH")
                    print("   - API_HOST and API_PORT")

        print("\n📋 After fixing, run this script again:")
        print("   python quickstart.py")

    print("\n" + "=" * 70 + "\n")


def main():
    """Run all checks"""
    print_header("🔍 Fake News Detection System - Quick Start")
    print("Validating your LoRA-based fake news detection setup...\n")

    checks = []

    # Run all checks
    checks.append(("Python Version", check_python_version()))
    checks.append(("Dependencies", check_dependencies()))
    checks.append(("Project Structure", check_project_structure()))
    checks.append(("Configuration", check_config_file()))

    # Create init files
    create_init_files()

    # Check models
    checks.append(("Model Files", check_models()))

    # MongoDB is optional
    mongo_status = check_mongodb()
    checks.append(("MongoDB (Optional)", mongo_status))

    # Test model loading if critical checks passed
    critical_passed = all(status for name, status in checks[:5])
    if critical_passed:
        model_test = test_model_loading()
        checks.append(("Model Loading", model_test))

    # Print summary
    print_next_steps(checks)


if __name__ == '__main__':
    main()