; CryptoAI Desktop - NSIS Installer Script
; Windows 11 Optimized

!macro customHeader
  !system "echo CryptoAI Desktop Installer Building..."
!macroend

!macro customInit
  ; Check if Python is installed (optional)
  ReadRegStr $0 HKLM "SOFTWARE\Python\PythonCore\3.11\InstallPath" ""
  StrCmp $0 "" 0 +3
    ReadRegStr $0 HKLM "SOFTWARE\Python\PythonCore\3.10\InstallPath" ""
  StrCmp $0 "" 0 +2
    MessageBox MB_ICONINFORMATION "Python 3.10+ is required for full functionality. Please install Python from python.org after completing this installation."
!macroend

!macro customInstall
  ; Create data directories
  CreateDirectory "$INSTDIR\data"
  CreateDirectory "$INSTDIR\models"
  CreateDirectory "$INSTDIR\logs"
  CreateDirectory "$INSTDIR\checkpoints"

  ; Set directory permissions
  AccessControl::GrantOnFile "$INSTDIR\data" "(BU)" "FullAccess"
  AccessControl::GrantOnFile "$INSTDIR\models" "(BU)" "FullAccess"
  AccessControl::GrantOnFile "$INSTDIR\logs" "(BU)" "FullAccess"
  AccessControl::GrantOnFile "$INSTDIR\checkpoints" "(BU)" "FullAccess"
!macroend

!macro customUnInstall
  ; Clean up data directories (with confirmation)
  MessageBox MB_YESNO "Do you want to remove all trading data, models, and logs?" IDNO SkipDataRemoval
    RMDir /r "$INSTDIR\data"
    RMDir /r "$INSTDIR\models"
    RMDir /r "$INSTDIR\logs"
    RMDir /r "$INSTDIR\checkpoints"
  SkipDataRemoval:
!macroend
