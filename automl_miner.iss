[Setup]
AppName=AutoML Miner
AppVersion=1.0
DefaultDirName={pf64}\AutoML Miner
DefaultGroupName=AutoML Miner
UninstallDisplayIcon={app}\AutoML Miner.exe
Compression=lzma2
SolidCompression=yes
OutputDir=installer

[Files]
Source: "dist\AutoML Miner\*"; DestDir: "{app}"; Flags: recursesubdirs

[Icons]
Name: "{group}\AutoML Miner"; Filename: "{app}\AutoML Miner.exe"
Name: "{commondesktop}\AutoML Miner"; Filename: "{app}\AutoML Miner.exe"