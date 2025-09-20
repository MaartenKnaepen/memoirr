"""Jinja2 template loader for system prompts.

This module handles loading and rendering Jinja2 templates for system prompts,
providing a flexible way to customize AI behavior for different tasks.

Follows Memoirr patterns: pure functions, proper error handling, caching.
"""

from pathlib import Path
from typing import Any, Optional, List
from functools import lru_cache

from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound

from src.core.logging_config import get_logger


# Cache the Jinja2 environment
@lru_cache(maxsize=1)
def get_jinja_env() -> Environment:
    """Get a cached Jinja2 environment configured for prompt templates.
    
    Returns:
        Configured Jinja2 Environment with file system loader.
    """
    prompts_dir = Path(__file__).parent
    return Environment(
        loader=FileSystemLoader(prompts_dir),
        autoescape=False,  # We want raw text output
        trim_blocks=True,
        lstrip_blocks=True,
    )


def load_prompt_template(template_name: str) -> Template:
    """Load a Jinja2 template from the prompts directory.
    
    Args:
        template_name: Name of the template file (e.g., "default_system.j2").
        
    Returns:
        Loaded Jinja2 Template object.
        
    Raises:
        FileNotFoundError: If template file doesn't exist.
        RuntimeError: If template loading fails.
    """
    logger = get_logger(__name__)
    
    try:
        env = get_jinja_env()
        template = env.get_template(template_name)
        
        logger.debug(
            "Prompt template loaded successfully",
            template_name=template_name,
            component="template_loader"
        )
        
        return template
        
    except TemplateNotFound as e:
        logger.error(
            "Prompt template not found",
            template_name=template_name,
            prompts_dir=str(Path(__file__).parent),
            error=str(e),
            component="template_loader"
        )
        raise FileNotFoundError(f"Prompt template '{template_name}' not found in src/prompts/") from e
        
    except Exception as e:
        logger.error(
            "Failed to load prompt template",
            template_name=template_name,
            error=str(e),
            error_type=type(e).__name__,
            component="template_loader"
        )
        raise RuntimeError(f"Failed to load prompt template '{template_name}': {str(e)}") from e


def render_system_prompt(
    template_name: str,
    task_type: Optional[str] = None,
    custom_instructions: Optional[str] = None,
    **kwargs: Any
) -> str:
    """Render a system prompt from a Jinja2 template.
    
    Args:
        template_name: Name of the template file to render.
        task_type: Optional task type for specialized instructions.
        custom_instructions: Optional custom instructions to include.
        **kwargs: Additional variables to pass to the template.
        
    Returns:
        Rendered system prompt string.
        
    Raises:
        FileNotFoundError: If template doesn't exist.
        RuntimeError: If template rendering fails.
    """
    logger = get_logger(__name__)
    
    try:
        template = load_prompt_template(template_name)
        
        # Prepare template variables
        template_vars = {
            "task_type": task_type,
            "custom_instructions": custom_instructions,
            **kwargs
        }
        
        # Remove None values to avoid template issues
        template_vars = {k: v for k, v in template_vars.items() if v is not None}
        
        rendered = template.render(**template_vars)
        
        logger.debug(
            "System prompt rendered successfully",
            template_name=template_name,
            task_type=task_type,
            has_custom_instructions=bool(custom_instructions),
            rendered_length=len(rendered),
            template_vars=list(template_vars.keys()),
            component="template_loader"
        )
        
        return rendered.strip()
        
    except Exception as e:
        logger.error(
            "Failed to render system prompt",
            template_name=template_name,
            task_type=task_type,
            error=str(e),
            error_type=type(e).__name__,
            component="template_loader"
        )
        raise RuntimeError(f"Failed to render system prompt: {str(e)}") from e


def list_available_templates() -> List[str]:
    """List all available prompt templates in the prompts directory.
    
    Returns:
        List of template filenames.
    """
    prompts_dir = Path(__file__).parent
    templates = []
    
    for file_path in prompts_dir.glob("*.j2"):
        templates.append(file_path.name)
    
    return sorted(templates)


def validate_template(template_name: str) -> bool:
    """Validate that a template exists and can be loaded.
    
    Args:
        template_name: Name of the template to validate.
        
    Returns:
        True if template is valid, False otherwise.
    """
    try:
        load_prompt_template(template_name)
        return True
    except (FileNotFoundError, RuntimeError):
        return False