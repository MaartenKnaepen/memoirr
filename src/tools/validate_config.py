"""Configuration validation tool for Memoirr.

Usage:
    python -m src.tools.validate_config
    python -c "from src.tools.validate_config import main; main()"
"""

import sys
from pathlib import Path
from typing import Dict, Any

def print_validation_result(validation: Dict[str, Any]) -> None:
    """Pretty print validation results."""
    
    print("🔍 Memoirr Configuration Validation")
    print("=" * 50)
    
    if validation["is_valid"]:
        print("✅ Configuration is VALID")
    else:
        print("❌ Configuration FAILED validation")
    
    # Print issues
    if validation["issues"]:
        print(f"\n🚨 Issues Found ({len(validation['issues'])}):")
        for i, issue in enumerate(validation["issues"], 1):
            print(f"  {i}. {issue}")
    
    # Print warnings
    if validation["warnings"]:
        print(f"\n⚠️  Warnings ({len(validation['warnings'])}):")
        for i, warning in enumerate(validation["warnings"], 1):
            print(f"  {i}. {warning}")
    
    # Print suggestions
    if validation["suggestions"]:
        print(f"\n💡 Suggestions:")
        for i, suggestion in enumerate(validation["suggestions"], 1):
            print(f"  {i}. {suggestion}")
    
    print()


def check_file_paths() -> Dict[str, Any]:
    """Check important file paths and directories."""
    results = {
        "models_dir": None,
        "env_file": None,
        "available_models": []
    }
    
    # Check models directory
    models_dir = Path("models")
    results["models_dir"] = {
        "exists": models_dir.exists(),
        "path": str(models_dir.absolute()),
        "is_dir": models_dir.is_dir() if models_dir.exists() else False
    }
    
    if models_dir.exists() and models_dir.is_dir():
        results["available_models"] = [
            str(p.name) for p in models_dir.iterdir() 
            if p.is_dir() and not p.name.startswith('.')
        ]
    
    # Check .env file
    env_file = Path(".env")
    results["env_file"] = {
        "exists": env_file.exists(),
        "path": str(env_file.absolute()),
        "is_file": env_file.is_file() if env_file.exists() else False
    }
    
    return results


def print_environment_info(file_info: Dict[str, Any]) -> None:
    """Print environment and file system information."""
    print("📁 Environment Information")
    print("-" * 30)
    
    # Models directory
    models = file_info["models_dir"]
    status = "✅" if models["exists"] and models["is_dir"] else "❌"
    print(f"{status} Models directory: {models['path']}")
    
    if file_info["available_models"]:
        print(f"   Available models ({len(file_info['available_models'])}):")
        for model in file_info["available_models"][:10]:  # Show max 10
            print(f"     - {model}")
        if len(file_info["available_models"]) > 10:
            print(f"     ... and {len(file_info['available_models']) - 10} more")
    else:
        print("   No models found")
    
    # .env file
    env = file_info["env_file"]
    status = "✅" if env["exists"] and env["is_file"] else "❌"
    print(f"{status} .env file: {env['path']}")
    
    print()


def validate_specific_setting(setting_name: str) -> None:
    """Validate a specific setting with detailed output."""
    try:
        from src.core.config import get_settings
        settings = get_settings(validate=False)  # Don't validate all, just load
        
        if hasattr(settings, setting_name.lower()):
            value = getattr(settings, setting_name.lower())
            print(f"✅ {setting_name}: {value}")
        else:
            print(f"❌ {setting_name}: Setting not found")
            
    except Exception as e:
        print(f"❌ {setting_name}: Failed to load - {e}")


def main() -> None:
    """Main validation function."""
    print("🚀 Starting Memoirr configuration validation...\n")
    
    # Check file system first
    file_info = check_file_paths()
    print_environment_info(file_info)
    
    # Try to load and validate settings
    try:
        from src.core.config import validate_settings_comprehensive, Settings
        
        # Load settings without validation first
        settings = Settings()  # type: ignore[call-arg]
        print("✅ Settings loaded successfully")
        
        # Run comprehensive validation
        validation = validate_settings_comprehensive(settings)
        print_validation_result(validation)
        
        # Exit with appropriate code
        sys.exit(0 if validation["is_valid"] else 1)
        
    except ImportError as e:
        print(f"❌ Failed to import required modules: {e}")
        print("💡 Make sure you're running from the project root directory")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        print("💡 Check your .env file and model setup")
        sys.exit(1)


if __name__ == "__main__":
    main()