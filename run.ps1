param(
    [ValidateSet("check", "status", "prepare", "train", "eval", "cv", "baseline", "all", "package", "materials", "benchmark", "gpu", "clean-cache")]
    [string]$Step = "check",
    [switch]$Force,
    [string]$Config = "configs/local_3060.json",
    [int[]]$Batches = @(24, 48, 64),
    [int]$Steps = 20,
    [string]$Output = "materiais_artigo"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Ambiente .venv nao encontrado. Use Python 3.12 e instale requirements-local.txt."
}

Push-Location $Root
try {
    if ($Step -eq "gpu") {
        nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv -l 2
        return
    }
    if ($Step -eq "clean-cache") {
        Remove-Item -LiteralPath ".\__pycache__", ".\src\cric_pipeline\__pycache__" -Recurse -Force -ErrorAction SilentlyContinue
        Write-Output "Caches Python removidos."
        return
    }
    $env:PYTHONPATH = Join-Path $Root "src"
    $argsList = @("-m", "cric_pipeline", $Step, "--config", $Config)
    if ($Force) { $argsList += "--force" }
    if ($Step -eq "benchmark") {
        $argsList += "--batches"
        foreach ($Batch in $Batches) {
            $argsList += [string]$Batch
        }
        $argsList += "--steps"
        $argsList += [string]$Steps
    }
    if ($Step -eq "materials") {
        $argsList += "--output"
        $argsList += $Output
    }
    & $Python @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "Etapa '$Step' falhou com codigo $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
