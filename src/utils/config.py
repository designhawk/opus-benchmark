"""Configuration management."""

import os
from pathlib import Path
from typing import Optional
import yaml
from dotenv import load_dotenv
from rich.prompt import Prompt
from rich.console import Console
import logging

logger = logging.getLogger(__name__)

load_dotenv()


class Config:
    """Manage application configuration."""

    CONFIG_DIR = Path.home() / ".config" / "opus-benchmark"
    CONFIG_FILE = CONFIG_DIR / "config.yaml"

    def __init__(self):
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self.console = Console()

    def get_api_key(self) -> str:
        """Get OpenRouter API key from environment or config."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            return api_key

        config = self._load_config()
        return config.get("api_key", "")

    def set_api_key(self, api_key: str):
        """Set OpenRouter API key."""
        config = self._load_config()
        config["api_key"] = api_key
        self._save_config(config)
        self.console.print(f"[green]API key saved successfully[/green]")

    def get_default_model(self) -> str:
        """Get default model from config."""
        config = self._load_config()
        return config.get("model", "arcee-ai/trinity-large-preview:free")

    def set_default_model(self, model: str):
        """Set default model."""
        config = self._load_config()
        config["model"] = model
        self._save_config(config)
        self.console.print(f"[green]Default model set to: {model}[/green]")

    def get_default_corpus(self) -> str:
        """Get default corpus from config."""
        config = self._load_config()
        return config.get("corpus", "OpenSubtitles")

    def set_default_corpus(self, corpus: str):
        """Set default corpus."""
        config = self._load_config()
        config["corpus"] = corpus
        self._save_config(config)
        self.console.print(f"[green]Default corpus set to: {corpus}[/green]")

    def get_default_samples(self) -> int:
        """Get default sample size from config."""
        config = self._load_config()
        return config.get("samples", 10)

    def set_default_samples(self, samples: int):
        """Set default sample size."""
        config = self._load_config()
        config["samples"] = samples
        self._save_config(config)
        self.console.print(f"[green]Default samples set to: {samples}[/green]")

    def validate(self) -> bool:
        """Validate configuration."""
        api_key = self.get_api_key()
        if not api_key:
            self.console.print(
                "[red]Error: OpenRouter API key not configured.[/red]\n"
                "Run: opus-benchmark config set-api-key"
            )
            return False
        return True

    def interactive_setup(self):
        """Interactive configuration setup."""
        self.console.print("[bold cyan]OPUS-LLM-Benchmark Configuration[/bold cyan]\n")

        if not self.get_api_key():
            self.console.print("Step 1: OpenRouter API Key")
            self.console.print(
                "Get your API key from https://openrouter.ai/ and paste it below."
            )
            api_key = Prompt.ask("API Key", password=True)
            if api_key:
                self.set_api_key(api_key)
                self.console.print()

        model = self.get_default_model()
        self.console.print(f"Step 2: Default Model (current: {model})")
        new_model = Prompt.ask(
            "Press Enter to keep current or type a new model",
            default=model,
        )
        if new_model and new_model != model:
            self.set_default_model(new_model)

        self.console.print("\n[green]Configuration complete![/green]")

    def show(self):
        """Show current configuration."""
        self.console.print("[bold]Current Configuration[/bold]\n")
        self.console.print(
            f"API Key: {'[green]Set[/green]' if self.get_api_key() else '[red]Not set[/red]'}"
        )
        self.console.print(f"Default Model: {self.get_default_model()}")
        self.console.print(f"Default Corpus: {self.get_default_corpus()}")
        self.console.print(f"Default Samples: {self.get_default_samples()}")

    def _load_config(self) -> dict:
        """Load configuration from file."""
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE) as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        return {}

    def _save_config(self, config: dict):
        """Save configuration to file."""
        try:
            with open(self.CONFIG_FILE, "w") as f:
                yaml.dump(config, f)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise
