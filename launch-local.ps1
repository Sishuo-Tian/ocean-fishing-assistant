$ErrorActionPreference = 'Stop'
$address = 'http://127.0.0.1:8765'
$backend = $null

function Test-ModelReady {
    try {
        $health = Invoke-RestMethod -Uri "$address/health" -TimeoutSec 2
        return $health.ready -eq $true -and $health.model -eq 'Earth2Ocean ViT-B/16'
    } catch {
        return $false
    }
}

function Find-Earth2OceanPython {
    $candidates = @()
    if ($env:EARTH2OCEAN_PYTHON) {
        $candidates += $env:EARTH2OCEAN_PYTHON
    }
    if ($env:CONDA_PREFIX -and (Split-Path $env:CONDA_PREFIX -Leaf) -ieq 'Earth2Ocean') {
        $candidates += Join-Path $env:CONDA_PREFIX 'python.exe'
    }
    $condaRegistry = Join-Path $env:USERPROFILE '.conda\environments.txt'
    if (Test-Path -LiteralPath $condaRegistry -PathType Leaf) {
        foreach ($envPath in (Get-Content -LiteralPath $condaRegistry -Encoding UTF8)) {
            if ($envPath -and (Split-Path $envPath -Leaf) -ieq 'Earth2Ocean') {
                $candidates += Join-Path $envPath 'python.exe'
            }
        }
    }
    $conda = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($conda) {
        $condaRoot = Split-Path (Split-Path $conda.Source -Parent) -Parent
        $candidates += Join-Path $condaRoot 'envs\Earth2Ocean\python.exe'
    }
    foreach ($path in $candidates) {
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            return (Get-Item -LiteralPath $path).FullName
        }
    }
    if ($env:CONDA_PREFIX -and $env:CONDA_DEFAULT_ENV -and $env:CONDA_DEFAULT_ENV -ne 'base') {
        $activePython = Join-Path $env:CONDA_PREFIX 'python.exe'
        if (Test-Path -LiteralPath $activePython -PathType Leaf) {
            return $activePython
        }
    }
    $manualPath = (Read-Host '未找到 Earth2Ocean 环境。请输入该环境的 python.exe 完整路径').Trim().Trim('"')
    if ($manualPath -and (Test-Path -LiteralPath $manualPath -PathType Leaf)) {
        return (Get-Item -LiteralPath $manualPath).FullName
    }
    throw '找不到 Python。请先配置 Earth2Ocean 环境，再重新启动。'
}

try {
    if (Test-ModelReady) {
        Write-Host '本机模型已运行，正在打开网页。'
        Start-Process "$address/"
        exit 0
    }

    $server = Join-Path $PSScriptRoot 'server.py'
    $modelDir = Join-Path (Split-Path $PSScriptRoot -Parent) 'Earth2Ocean-code'
    foreach ($relativePath in @(
        'Earth2Ocean.py',
        'environment\windows_compat.py',
        'pretrained_ckpt\vitb_laion2b_s34b_b88k.bin',
        'pretrained_ckpt\depth_anything_v2_vitb.pth'
    )) {
        $requiredPath = Join-Path $modelDir $relativePath
        if (!(Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "缺少模型文件：$requiredPath。请把网页文件夹放在完整的 Earth2Ocean-code 文件夹旁边。"
        }
    }
    $python = Find-Earth2OceanPython
    $logDir = Join-Path ([System.IO.Path]::GetTempPath()) ('Earth2Ocean-web-' + $PID)
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $stderr = Join-Path $logDir 'server-error.log'
    $env:PYTHONIOENCODING = 'utf-8'
    Write-Host '正在加载 Earth2Ocean 模型，加载完成后会自动打开网页。'
    $backend = Start-Process -FilePath $python -ArgumentList @('-u', '-B', ('"' + $server + '"')) `
        -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $logDir 'server.log') -RedirectStandardError $stderr

    $deadline = (Get-Date).AddMinutes(3)
    do {
        $backend.Refresh()
        if ($backend.HasExited) {
            Get-Content -LiteralPath $stderr -Tail 12 -Encoding UTF8 | Write-Host
            throw "模型服务启动失败。请检查 Python 环境、CUDA 和端口 8765。日志：$stderr"
        }
        if (Test-ModelReady) {
            Write-Host "远洋观察台已启动 $address"
            Start-Process "$address/"
            Read-Host '使用期间请保留此窗口。按 Enter 停止本次服务' | Out-Null
            exit 0
        }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)
    throw "等待模型启动超时。日志：$stderr"
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
} finally {
    if ($backend -and !$backend.HasExited) {
        Stop-Process -InputObject $backend -ErrorAction SilentlyContinue
    }
}
