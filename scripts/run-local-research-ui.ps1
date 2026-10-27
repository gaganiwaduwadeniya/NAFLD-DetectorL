$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$RootDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$RuntimeDir = if ($env:NAFLD_RUNTIME_DIR) { $env:NAFLD_RUNTIME_DIR } else { Join-Path $RootDir '.runtime' }
$PythonRuntimeDir = Join-Path $RuntimeDir 'pytorch-research-v1-windows'
$StateDir = Join-Path $RuntimeDir 'nafld-local'
$LogDir = Join-Path $StateDir 'logs'
$DataDir = if ($env:LOCAL_DATA_DIR) { $env:LOCAL_DATA_DIR } else { Join-Path $StateDir 'data' }
$PythonBin = Join-Path $PythonRuntimeDir 'Scripts\python.exe'
$RequirementsFile = Join-Path $RootDir 'pytorch_service\requirements.lock.txt'
$RequirementsStamp = Join-Path $PythonRuntimeDir '.requirements.sha256'
$NodePlatformStamp = Join-Path $RuntimeDir '.node-platform-windows.txt'
$ResearchRoot = Join-Path $RootDir 'pytorch_service\research_v1'
$ModelPort = if ($env:ML_SERVICE_PORT) { [int]$env:ML_SERVICE_PORT } else { 5002 }
$AppPort = if ($env:PORT) { [int]$env:PORT } else { 3001 }
$ModelUrl = "http://127.0.0.1:$ModelPort"
$AppUrl = "http://127.0.0.1:$AppPort"
$CheckOnly = $false
$PrepareOnly = $false
$OpenBrowser = $true
$ModelProcess = $null
$AppProcess = $null

function Show-Usage {
    @'
Usage: npm run app -- [options]

Prepare and run the local NAFLD research interface from one terminal.
Native Windows mode is supported and does not require WSL.

Options:
  --check         Verify existing tools, runtime, models, and application files only.
  --prepare-only  Update dependencies and build the application, then stop.
  --no-open       Start the system without opening a browser.
  --help          Show this help.

Optional environment variables (PowerShell syntax shown):
  $env:LOCAL_DATA_DIR = 'D:\NAFLD local data'
  $env:PORT = '3001'
  $env:ML_SERVICE_PORT = '5002'
'@
}

function Write-Log([string]$Message) {
    Write-Host "[NAFLD] $Message"
}

function Stop-WithError([string]$Message) {
    throw "[NAFLD] ERROR: $Message"
}

foreach ($Argument in $args) {
    switch -Exact ($Argument) {
        '--check' { $CheckOnly = $true }
        '--prepare-only' { $PrepareOnly = $true }
        '--no-open' { $OpenBrowser = $false }
        '--help' { Show-Usage; exit 0 }
        '-h' { Show-Usage; exit 0 }
        default { Stop-WithError "Unknown option: $Argument. Run 'npm run app -- --help'." }
    }
}

function Assert-Command([string]$Name, [string]$InstallHint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Stop-WithError "Required command '$Name' is unavailable. $InstallHint"
    }
}

function Assert-ModelBundle([string]$ResearchRoot) {
    if (-not (Test-Path -LiteralPath (Join-Path $ResearchRoot 'src\nafld') -PathType Container)) {
        Stop-WithError "Research inference source is missing under $ResearchRoot\src."
    }
    if (-not (Test-Path -LiteralPath (Join-Path $ResearchRoot 'checkpoints\research-v1\binary\manifest.json') -PathType Leaf)) {
        Stop-WithError 'Binary model manifest is missing.'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $ResearchRoot 'checkpoints\research-v1\staging\manifest.json') -PathType Leaf)) {
        Stop-WithError 'Staging model manifest is missing.'
    }

    $BinaryModels = @(Get-ChildItem -LiteralPath (Join-Path $ResearchRoot 'checkpoints\research-v1\binary') -Filter 'fold-*.pt' -File)
    $StagingModels = @(Get-ChildItem -LiteralPath (Join-Path $ResearchRoot 'checkpoints\research-v1\staging') -Filter 'fold-*.pt' -File)
    if ($BinaryModels.Count -ne 5) {
        Stop-WithError "Expected 5 selected DenseNet121 checkpoints; found $($BinaryModels.Count)."
    }
    if ($StagingModels.Count -ne 5) {
        Stop-WithError "Expected 5 selected EfficientNet-B0 checkpoints; found $($StagingModels.Count)."
    }
}

function Get-RequirementsHash {
    return (Get-FileHash -LiteralPath $RequirementsFile -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Test-PythonRuntime {
    if (-not (Test-Path -LiteralPath $PythonBin -PathType Leaf)) {
        return $false
    }

    $ValidationCode = @'
import importlib.metadata as metadata
import sys
expected = {
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "timm": "1.0.29",
    "numpy": "2.1.3",
    "Pillow": "13.0.0",
    "fastapi": "0.121.3",
    "uvicorn": "0.38.0",
    "python-multipart": "0.0.20",
    "jsonschema": "4.25.1",
    "psutil": "7.1.3",
}
if sys.version_info[:2] != (3, 12):
    raise SystemExit(1)
for package, version in expected.items():
    if metadata.version(package) != version:
        raise SystemExit(1)
'@
    & $PythonBin -c $ValidationCode 2>$null
    return $LASTEXITCODE -eq 0
}

function Initialize-PythonRuntime {
    $ExpectedHash = Get-RequirementsHash
    $ActualHash = ''
    if (Test-Path -LiteralPath $RequirementsStamp -PathType Leaf) {
        $ActualHash = (Get-Content -LiteralPath $RequirementsStamp -Raw).Trim()
    }

    if (($ActualHash -eq $ExpectedHash) -and (Test-PythonRuntime)) {
        Write-Log 'Python runtime is current.'
        return
    }
    if ($CheckOnly) {
        Stop-WithError "The Python runtime is missing or out of date. Run 'npm run app' to repair it."
    }

    Assert-Command 'uv' "Install uv from https://docs.astral.sh/uv/getting-started/installation/ and reopen PowerShell."
    New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
    if (-not (Test-Path -LiteralPath $PythonBin -PathType Leaf)) {
        Write-Log 'Creating the managed native Windows Python 3.12 environment...'
        & uv venv --python 3.12 $PythonRuntimeDir
        if ($LASTEXITCODE -ne 0) { Stop-WithError 'uv could not create the Python 3.12 environment.' }
    }

    Write-Log 'Synchronizing the exact verified PyTorch runtime...'
    & uv pip install --python $PythonBin --requirements $RequirementsFile --torch-backend cu128 --exact --strict
    if ($LASTEXITCODE -ne 0) { Stop-WithError 'uv could not synchronize the locked Python dependencies.' }
    if (-not (Test-PythonRuntime)) {
        Stop-WithError 'Python packages do not match the verified inference contract after synchronization.'
    }
    Set-Content -LiteralPath $RequirementsStamp -Value $ExpectedHash -Encoding ASCII -NoNewline
    Write-Log 'Python runtime verified.'
}

function Initialize-NodeRuntime {
    Assert-Command 'node' 'Install Node.js 20 or newer from https://nodejs.org/.'
    Assert-Command 'npm.cmd' 'Install Node.js 20 or newer from https://nodejs.org/.'
    & node -e "const major=Number(process.versions.node.split('.')[0]); process.exit(major >= 20 ? 0 : 1)"
    if ($LASTEXITCODE -ne 0) { Stop-WithError 'Node.js 20 or newer is required.' }

    $PackageLockHash = (Get-FileHash -LiteralPath (Join-Path $RootDir 'package-lock.json') -Algorithm SHA256).Hash.ToLowerInvariant()
    $NodeAbi = (& node -p "process.versions.modules").Trim()
    $ExpectedPlatformStamp = "$PackageLockHash|win32|$env:PROCESSOR_ARCHITECTURE|$NodeAbi"
    $ActualPlatformStamp = ''
    if (Test-Path -LiteralPath $NodePlatformStamp -PathType Leaf) {
        $ActualPlatformStamp = (Get-Content -LiteralPath $NodePlatformStamp -Raw).Trim()
    }

    & npm.cmd ls --depth=0 --silent *> $null
    if (($LASTEXITCODE -eq 0) -and ($ActualPlatformStamp -eq $ExpectedPlatformStamp)) {
        Write-Log 'Node dependencies are current.'
        return
    }
    if ($CheckOnly) {
        Stop-WithError "Node dependencies are missing or out of date. Run 'npm run app' to repair them."
    }

    Write-Log 'Installing the locked Node dependencies...'
    & npm.cmd ci
    if ($LASTEXITCODE -ne 0) { Stop-WithError 'npm ci failed.' }
    & npm.cmd ls --depth=0 --silent *> $null
    if ($LASTEXITCODE -ne 0) { Stop-WithError 'Node dependency validation failed after npm ci.' }
    New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
    Set-Content -LiteralPath $NodePlatformStamp -Value $ExpectedPlatformStamp -Encoding ASCII -NoNewline
}

function Build-Application {
    $ServerBundle = Join-Path $RootDir 'dist\server.cjs'
    if ($CheckOnly) {
        if (-not (Test-Path -LiteralPath $ServerBundle -PathType Leaf)) {
            Stop-WithError "The production build is missing. Run 'npm run app' to build it."
        }
        Write-Log 'Existing production build found.'
        return
    }

    Write-Log 'Building the production UI and API...'
    & npm.cmd run build
    if (($LASTEXITCODE -ne 0) -or (-not (Test-Path -LiteralPath $ServerBundle -PathType Leaf))) {
        Stop-WithError 'Production application build failed.'
    }
}

function Test-ModelReady {
    try {
        $Response = Invoke-RestMethod -Uri "$ModelUrl/ready" -Method Get -TimeoutSec 2
        return $Response.ready -eq $true
    } catch {
        return $false
    }
}

function Test-ApplicationReady {
    try {
        $Response = Invoke-WebRequest -UseBasicParsing -Uri "$AppUrl/login" -Method Get -TimeoutSec 2
        return $Response.StatusCode -ge 200 -and $Response.StatusCode -lt 500
    } catch {
        return $false
    }
}

function Wait-ForModel {
    for ($Attempt = 0; $Attempt -lt 180; $Attempt++) {
        if (Test-ModelReady) { return $true }
        if ($ModelProcess -and $ModelProcess.HasExited) { return $false }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Wait-ForApplication {
    for ($Attempt = 0; $Attempt -lt 60; $Attempt++) {
        if (Test-ApplicationReady) { return $true }
        if ($AppProcess -and $AppProcess.HasExited) { return $false }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Show-LogTail([string]$Path) {
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        foreach ($Line in (Get-Content -LiteralPath $Path -Tail 80)) {
            [Console]::Error.WriteLine($Line)
        }
    }
}

function Stop-ManagedServices {
    foreach ($Process in @($AppProcess, $ModelProcess)) {
        if ($Process -and (-not $Process.HasExited)) {
            Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        }
    }
}

try {
    Set-Location $RootDir
    Assert-ModelBundle $ResearchRoot
    Write-Log 'Verified the selected 5 + 5 checkpoint bundle.'

    Initialize-PythonRuntime
    Initialize-NodeRuntime
    Build-Application

    if ($CheckOnly) {
        Write-Log 'Readiness check passed. No services were started.'
        exit 0
    }
    if ($PrepareOnly) {
        Write-Log 'Preparation completed. No services were started.'
        exit 0
    }

    New-Item -ItemType Directory -Force -Path $LogDir, $DataDir | Out-Null
    $JwtFile = Join-Path $StateDir 'jwt.secret'
    if ((-not (Test-Path -LiteralPath $JwtFile -PathType Leaf)) -or ((Get-Item -LiteralPath $JwtFile).Length -eq 0)) {
        $RandomBytes = New-Object byte[] 32
        [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($RandomBytes)
        $JwtSecret = ([System.BitConverter]::ToString($RandomBytes)).Replace('-', '').ToLowerInvariant()
        Set-Content -LiteralPath $JwtFile -Value $JwtSecret -Encoding ASCII -NoNewline
    }
    $JwtSecret = (Get-Content -LiteralPath $JwtFile -Raw).Trim()

    if (Test-ModelReady) { Stop-WithError "Port $ModelPort already has a ready model service. Stop it before using this managed launcher." }
    if (Test-ApplicationReady) { Stop-WithError "Port $AppPort is already serving an application. Stop it before using this managed launcher." }

    $ModelLog = Join-Path $LogDir 'model-service.log'
    $ModelErrorLog = Join-Path $LogDir 'model-service-error.log'
    $AppLog = Join-Path $LogDir 'application.log'
    $AppErrorLog = Join-Path $LogDir 'application-error.log'

    Write-Log 'Starting the verified model service; first load can take about one minute...'
    $env:PYTHONPATH = $RootDir
    $ModelProcess = Start-Process -FilePath $PythonBin -ArgumentList @(
        '-m', 'uvicorn', 'pytorch_service.app:app', '--host', '127.0.0.1', '--port', "$ModelPort"
    ) -WorkingDirectory $RootDir -NoNewWindow -PassThru -RedirectStandardOutput $ModelLog -RedirectStandardError $ModelErrorLog

    if (-not (Wait-ForModel)) {
        Show-LogTail $ModelLog
        Show-LogTail $ModelErrorLog
        Stop-WithError 'Model service did not become ready.'
    }
    Write-Log 'Model service is ready.'

    Write-Log 'Starting the local research dashboard...'
    $env:NODE_ENV = 'production'
    $env:PORT = "$AppPort"
    $env:LOCAL_DATA_DIR = $DataDir
    $env:USE_LOCAL_DB = 'true'
    $env:MODEL_PROVIDER = 'pytorch_research_v1'
    $env:ML_SERVICE_URL = $ModelUrl
    $env:ML_INFERENCE_TIMEOUT_MS = '60000'
    $env:STORE_UPLOAD_IMAGES = 'false'
    $env:JWT_SECRET = $JwtSecret
    $NodeExecutable = (Get-Command node).Source
    $AppProcess = Start-Process -FilePath $NodeExecutable -ArgumentList @('dist\server.cjs') -WorkingDirectory $RootDir -NoNewWindow -PassThru -RedirectStandardOutput $AppLog -RedirectStandardError $AppErrorLog

    if (-not (Wait-ForApplication)) {
        Show-LogTail $AppLog
        Show-LogTail $AppErrorLog
        Stop-WithError 'The dashboard did not become ready.'
    }

    Write-Log "System ready: $AppUrl"
    Write-Log "Logs: $LogDir"
    Write-Log 'Press Ctrl+C once to stop both services.'
    if ($OpenBrowser) { Start-Process $AppUrl }

    while ((-not $ModelProcess.HasExited) -and (-not $AppProcess.HasExited)) {
        Start-Sleep -Seconds 1
    }
    Stop-WithError "A managed service stopped unexpectedly. Review $LogDir."
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
} finally {
    if ($ModelProcess -or $AppProcess) {
        Write-Log 'Stopping local services...'
        Stop-ManagedServices
    }
}
