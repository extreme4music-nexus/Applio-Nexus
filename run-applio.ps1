# Check for Administrator mode / System32
if ($PWD.Path -eq "C:\Windows\System32") {
    Host.UI.RawUI.ForegroundColor = "Red"
    Write-Host "Applio does not require administrator permissions and should be run as a regular user.`n"
    Read-Host "Press Enter to exit"
    exit 1
}

# Set window title to current folder name
$host.ui.RawUI.WindowTitle = (Get-Item .).Name

# Check if the virtual environment exists
$envPath = Join-Path $PSScriptRoot "env\Scripts\Activate.ps1"
if (-not (Test-Path $envPath)) {
    Write-Host "[ERROR] Virtual environment 'env' not found or incomplete." -ForegroundColor Red
    Write-Host "Please run your installer script first to set up the environment.`n"
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate the local environment and launch Applio
& $envPath
python app.py --open

Read-Host "`nPress Enter to exit"