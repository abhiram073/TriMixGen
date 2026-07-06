import yaml
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class PromptBuilder:
    """
    Standardized prompt templating engine.
    Isolates prompt engineering from dataset loading and tokenization.
    Supports dynamic registration and strict validation of placeholders.
    """
    def __init__(self, config_path: str = "configs/prompts.yaml"):
        self.templates = {}
        self._load_config(config_path)

    def _load_config(self, config_path: str):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            if 'templates' in config:
                for name, details in config['templates'].items():
                    self.register_template(
                        name=name,
                        template=details.get('template', ''),
                        required_kwargs=details.get('required_kwargs', [])
                    )
            logger.info(f"Loaded {len(self.templates)} prompt templates from {config_path}")
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Initialized empty PromptBuilder.")

    def register_template(self, name: str, template: str, required_kwargs: List[str]):
        """Dynamically registers a new template."""
        if not template:
            raise ValueError(f"Template string cannot be empty for '{name}'")
        self.templates[name] = {
            "template": template,
            "required_kwargs": required_kwargs
        }

    def validate_kwargs(self, template_name: str, kwargs: Dict[str, Any]):
        """Ensures all required placeholders are present in the provided arguments."""
        if template_name not in self.templates:
            raise KeyError(f"Template '{template_name}' not found.")
            
        required = self.templates[template_name]["required_kwargs"]
        missing = [kwarg for kwarg in required if kwarg not in kwargs]
        
        if missing:
            raise ValueError(f"Missing required kwargs for template '{template_name}': {missing}")

    def render(self, template_name: str, **kwargs) -> str:
        """
        Renders a single prompt.
        Validates arguments before rendering.
        """
        self.validate_kwargs(template_name, kwargs)
        template_str = self.templates[template_name]["template"]
        try:
            return template_str.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Template '{template_name}' has mismatched placeholders. Error: {e}")

    def batch_render(self, template_name: str, batch_kwargs: List[Dict[str, Any]]) -> List[str]:
        """Renders a batch of prompts efficiently."""
        return [self.render(template_name, **kwargs) for kwargs in batch_kwargs]

    def preview_templates(self, sample_data: Dict[str, Dict[str, Any]]) -> str:
        """
        Generates a markdown report previewing how each template renders with sample data.
        """
        report = "# Prompt Preview Report\n\n"
        for t_name, data in sample_data.items():
            if t_name in self.templates:
                rendered = self.render(t_name, **data)
                report += f"### Template: `{t_name}`\n"
                report += f"**Format:** `{self.templates[t_name]['template']}`\n"
                report += f"**Rendered:**\n```text\n{rendered}\n```\n\n"
        return report
