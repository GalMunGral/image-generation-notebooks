import glob
import nbformat
from nbconvert import HTMLExporter
from pathlib import Path

GROUPS = [
    sorted(glob.glob('diffusion/*.ipynb')),
    sorted(glob.glob('autoregressive/*.ipynb')),
]

# Merge all notebooks into one
combined = None
for group in GROUPS:
    for path in group:
        nb = nbformat.read(path, as_version=4)
        if combined is None:
            combined = nb
        else:
            combined.cells.extend(nb.cells)

# Export to HTML
Path('_site').mkdir(exist_ok=True)
body, _ = HTMLExporter().from_notebook_node(combined)
Path('_site/index.html').write_text(body)
print('wrote _site/index.html')