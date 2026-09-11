"""Core data and metric APIs for the alignment analysis pipeline."""

import sys
from pathlib import Path

# plot-skill lives outside the package; putting its scripts dir on sys.path
# here lets every `import plots.core.*` client use `import style_presets`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".agents/skills/plot-skill/scripts"))
