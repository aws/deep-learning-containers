"""Global variables for documentation generation."""

from pathlib import Path

from omegaconf import OmegaConf

# Path constants
SRC_DIR = Path(__file__).parent
DOCS_DIR = SRC_DIR.parent
DATA_DIR = SRC_DIR / "data"
LEGACY_DIR = SRC_DIR / "legacy"
TABLES_DIR = SRC_DIR / "tables"
TEMPLATES_DIR = SRC_DIR / "templates"
REFERENCE_DIR = DOCS_DIR / "reference"
README_PATH = DOCS_DIR.parent / "README.md"
RELEASE_NOTES_DIR = DOCS_DIR / "releasenotes"
TUTORIALS_DIR = DOCS_DIR / "tutorials"

# Release notes configuration
RELEASE_NOTES_REQUIRED_FIELDS = ["announcements", "packages"]
GLOBAL_CONFIG_PATH = SRC_DIR / "global.yml"

AVAILABLE_IMAGES_TABLE_HEADER = "##"
RELEASE_NOTES_TABLE_HEADER = "###"
TUTORIALS_REPO = "https://github.com/aws-samples/sample-aws-deep-learning-containers"
PUBLIC_GALLERY_URL = "https://gallery.ecr.aws/deep-learning-containers"
SITE_URL = "https://aws.github.io/deep-learning-containers/"

# Load global config once at import time
global_cfg = OmegaConf.load(GLOBAL_CONFIG_PATH)
GLOBAL_CONFIG = OmegaConf.to_container(global_cfg, resolve=True)
