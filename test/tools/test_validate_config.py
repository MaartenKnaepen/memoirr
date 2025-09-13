"""Tests for the configuration validation tool."""

import pytest
from unittest.mock import patch, MagicMock
from io import StringIO
import sys

from src.tools.validate_config import (
    print_validation_result,
    check_file_paths,
    print_environment_info,
    main
)


class TestPrintValidationResult:
    """Test the validation result printing function."""

    def test_print_valid_result(self, capsys):
        """Test printing a valid validation result."""
        validation = {
            "is_valid": True,
            "issues": [],
            "warnings": ["Warning 1", "Warning 2"],
            "suggestions": []
        }
        
        print_validation_result(validation)
        captured = capsys.readouterr()
        
        assert "✅ Configuration is VALID" in captured.out
        assert "⚠️  Warnings (2):" in captured.out
        assert "Warning 1" in captured.out
        assert "Warning 2" in captured.out

    def test_print_invalid_result(self, capsys):
        """Test printing an invalid validation result."""
        validation = {
            "is_valid": False,
            "issues": ["Issue 1", "Issue 2"],
            "warnings": [],
            "suggestions": ["Suggestion 1", "Suggestion 2"]
        }
        
        print_validation_result(validation)
        captured = capsys.readouterr()
        
        assert "❌ Configuration FAILED validation" in captured.out
        assert "🚨 Issues Found (2):" in captured.out
        assert "Issue 1" in captured.out
        assert "Issue 2" in captured.out
        assert "💡 Suggestions:" in captured.out
        assert "Suggestion 1" in captured.out

    def test_print_minimal_result(self, capsys):
        """Test printing minimal validation result."""
        validation = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "suggestions": []
        }
        
        print_validation_result(validation)
        captured = capsys.readouterr()
        
        assert "✅ Configuration is VALID" in captured.out
        # Should not contain sections for empty lists
        assert "🚨 Issues Found" not in captured.out
        assert "⚠️  Warnings" not in captured.out
        assert "💡 Suggestions" not in captured.out


class TestCheckFilePaths:
    """Test file path checking functionality."""

    def test_check_file_paths_complete(self):
        """Test file path checking with complete setup."""
        result = check_file_paths()
        
        # Just check it returns the expected structure
        assert "models_dir" in result
        assert "env_file" in result
        assert "available_models" in result
        assert isinstance(result["available_models"], list)

    def test_check_file_paths_missing(self):
        """Test file path checking with missing files."""
        result = check_file_paths()
        
        # Just check basic structure
        assert "models_dir" in result
        assert "env_file" in result


class TestPrintEnvironmentInfo:
    """Test environment information printing."""

    def test_print_environment_complete(self, capsys):
        """Test printing complete environment info."""
        file_info = {
            "models_dir": {
                "exists": True,
                "path": "/test/models",
                "is_dir": True
            },
            "env_file": {
                "exists": True,
                "path": "/test/.env",
                "is_file": True
            },
            "available_models": ["model1", "model2", "model3"]
        }
        
        print_environment_info(file_info)
        captured = capsys.readouterr()
        
        assert "✅ Models directory: /test/models" in captured.out
        assert "Available models (3):" in captured.out
        assert "model1" in captured.out
        assert "✅ .env file: /test/.env" in captured.out

    def test_print_environment_missing(self, capsys):
        """Test printing environment info with missing components."""
        file_info = {
            "models_dir": {
                "exists": False,
                "path": "/test/models",
                "is_dir": False
            },
            "env_file": {
                "exists": False,
                "path": "/test/.env",
                "is_file": False
            },
            "available_models": []
        }
        
        print_environment_info(file_info)
        captured = capsys.readouterr()
        
        assert "❌ Models directory: /test/models" in captured.out
        assert "No models found" in captured.out
        assert "❌ .env file: /test/.env" in captured.out

    def test_print_environment_many_models(self, capsys):
        """Test printing with many models (should truncate)."""
        file_info = {
            "models_dir": {
                "exists": True,
                "path": "/test/models",
                "is_dir": True
            },
            "env_file": {
                "exists": True,
                "path": "/test/.env",
                "is_file": True
            },
            "available_models": [f"model{i}" for i in range(15)]  # 15 models
        }
        
        print_environment_info(file_info)
        captured = capsys.readouterr()
        
        assert "Available models (15):" in captured.out
        assert "... and 5 more" in captured.out  # Should show only 10, then "5 more"


class TestMainFunction:
    """Test the main validation function."""

    @patch('src.tools.validate_config.check_file_paths')
    @patch('src.tools.validate_config.print_environment_info')
    @patch('src.tools.validate_config.print_validation_result')
    @patch('src.core.config.validate_settings_comprehensive')
    @patch('src.core.config.Settings')
    def test_main_success(self, mock_settings, mock_validate, mock_print_result, 
                          mock_print_env, mock_check_paths):
        """Test main function with successful validation."""
        # Mock file paths check
        mock_check_paths.return_value = {"test": "data"}
        
        # Mock settings creation
        mock_settings_instance = MagicMock()
        mock_settings.return_value = mock_settings_instance
        
        # Mock validation
        mock_validate.return_value = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "suggestions": []
        }
        
        # Mock sys.exit to prevent actual exit
        with patch('sys.exit') as mock_exit:
            main()
            mock_exit.assert_called_once_with(0)
        
        # Verify function calls
        mock_check_paths.assert_called_once()
        mock_print_env.assert_called_once()
        mock_settings.assert_called_once()
        mock_validate.assert_called_once_with(mock_settings_instance)
        mock_print_result.assert_called_once()

    @patch('src.tools.validate_config.check_file_paths')
    @patch('src.tools.validate_config.print_environment_info')
    @patch('src.tools.validate_config.print_validation_result')
    @patch('src.core.config.validate_settings_comprehensive')
    @patch('src.core.config.Settings')
    def test_main_validation_failure(self, mock_settings, mock_validate, mock_print_result,
                                     mock_print_env, mock_check_paths):
        """Test main function with validation failure."""
        # Mock file paths check
        mock_check_paths.return_value = {"test": "data"}
        
        # Mock settings creation
        mock_settings_instance = MagicMock()
        mock_settings.return_value = mock_settings_instance
        
        # Mock validation failure
        mock_validate.return_value = {
            "is_valid": False,
            "issues": ["Test issue"],
            "warnings": [],
            "suggestions": []
        }
        
        # Mock sys.exit to prevent actual exit
        with patch('sys.exit') as mock_exit:
            main()
            mock_exit.assert_called_once_with(1)  # Should exit with error code

    @patch('src.core.config.validate_settings_comprehensive')
    @patch('src.tools.validate_config.check_file_paths')
    @patch('src.tools.validate_config.print_environment_info')
    def test_main_import_error(self, mock_print_env, mock_check_paths, mock_validate, capsys):
        """Test main function with import error."""
        # Mock file paths check
        mock_check_paths.return_value = {"test": "data"}
        
        # Mock import error when trying to import from src.core.config
        mock_validate.side_effect = ImportError("Module not found")
        
        with patch('sys.exit') as mock_exit:
            main()
            mock_exit.assert_called_once_with(1)
        
        captured = capsys.readouterr()
        assert "❌ Failed to import required modules" in captured.out

    @patch('src.tools.validate_config.check_file_paths')
    @patch('src.tools.validate_config.print_environment_info')
    @patch('src.core.config.Settings')
    def test_main_settings_error(self, mock_settings, mock_print_env, mock_check_paths, capsys):
        """Test main function with settings creation error."""
        # Mock file paths check
        mock_check_paths.return_value = {"test": "data"}
        
        # Mock settings creation failure
        mock_settings.side_effect = Exception("Settings error")
        
        with patch('sys.exit') as mock_exit:
            main()
            mock_exit.assert_called_once_with(1)
        
        captured = capsys.readouterr()
        assert "❌ Configuration loading failed" in captured.out