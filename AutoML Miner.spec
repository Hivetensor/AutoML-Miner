# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

torch_data = collect_data_files('torch')
torchvision_data = collect_data_files('torchvision')
numpy_data = collect_data_files('numpy')

a = Analysis(
    ['gui_app_modern.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('data', 'data'),  # Include your data directory
        *torch_data,
        *torchvision_data,
        *numpy_data
    ],
    hiddenimports=[
        'PIL', 
        'numpy', 
        'pandas', 
        'torch', 
        'torchvision', 
        'torchvision.datasets',
        'substrateinterface',
        'cryptography',
        'automl_client.utils.resource_path',
        'automl_client.gui',
        'automl_client.gui.main_window',
        'automl_client.gui.theme',
        'automl_client.gui.components',
        'automl_client.gui.components.hex_logo',
        'automl_client.gui.screens',
        'automl_client.gui.screens.dashboard_screen',
        'automl_client.gui.screens.wallet_screen',
        'automl_client.gui.screens.mining_screen',
        'automl_client.gui.screens.settings_screen',
        'automl_client.gui.screens.base_screen'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AutoML Miner',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch='arm64',  # Changed from universal2 to arm64 for M-series Mac
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AutoML Miner',
)
app = BUNDLE(
    coll,
    name='AutoML Miner.app',
    icon=None,
    bundle_identifier=None,
) 