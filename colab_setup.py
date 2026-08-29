# =============================================================================
# COLAB SETUP — run this FIRST in every new session.
#
# /content is wiped on every runtime restart. This keeps the project in Drive
# and symlinks it, so a restart costs you nothing but the pip install.
# =============================================================================

# --- 1. mount Drive -----------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

# --- 2. project lives in Drive, not /content ---------------------------------
import os, pathlib
PROJ = pathlib.Path('/content/drive/MyDrive/hvsmr_qswin')
PROJ.mkdir(parents=True, exist_ok=True)
(PROJ / 'cache').mkdir(exist_ok=True)
(PROJ / 'runs').mkdir(exist_ok=True)

os.chdir(PROJ)
print('working dir:', os.getcwd())
print('files:', sorted(p.name for p in PROJ.iterdir()))

# --- 3. dependencies ----------------------------------------------------------
!pip install -q timm pennylane nibabel grad-cam

# --- 4. pick up edits without restarting --------------------------------------
%load_ext autoreload
%autoreload 2

# =============================================================================
# FIRST TIME ONLY
# =============================================================================
# 1. Upload the .py files into /content/drive/MyDrive/hvsmr_qswin/
#    (Drive web interface, or the Colab file browser under drive/MyDrive/)
# 2. Put the dataset at    /content/drive/MyDrive/hvsmr_qswin/hvsmr2/
#       hvsmr2/images/   *.nii.gz
#       hvsmr2/labels/   *.nii.gz
# 3. Put metadata.csv at   /content/drive/MyDrive/hvsmr_qswin/metadata.csv
# 4. Build the cache once: !python prepare_cache.py
#
# The cache (~2 GB) then persists in Drive across restarts. You never rebuild
# it unless min_fg_voxels or img_size changes.
#
# =============================================================================
# EVERY SESSION AFTER THAT
# =============================================================================
# Run this cell, then go straight to:
#     !python train.py --bottleneck none --epochs 25
#
# NOTE: reading the cache from Drive is slower than from local disk. If epochs
# feel sluggish, copy it local at the start of a session:
#     !mkdir -p /content/cache && cp cache/*.npy cache/*.csv /content/cache/
# and point CACHE_DIR in cached_data.py at /content/cache. Rebuild-free,
# because the Drive copy stays intact.
