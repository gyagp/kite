param(
  [string]$SourceRoot = 'E:\workspace\project\evo',
  [string]$TargetRoot = 'E:\workspace\project\kite\local'
)
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$source = (Resolve-Path -LiteralPath $SourceRoot).Path.TrimEnd('\')
$target = [IO.Path]::GetFullPath($TargetRoot).TrimEnd('\')
if ($source -ne 'E:\workspace\project\evo') { throw 'Source must be the explicitly selected Evo directory.' }
if (-not $target.StartsWith($workspace + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Target must be inside Kite.' }
if ([IO.Path]::GetPathRoot($source) -ne [IO.Path]::GetPathRoot($target)) { throw 'Migration requires a same-volume move.' }
New-Item -ItemType Directory -Path $target -Force | Out-Null
$large = @('.cache', '.venv', 'artifacts', 'models')
foreach ($name in $large) {
  $from = Join-Path $source $name
  $to = Join-Path $target $name
  if (-not [IO.Path]::GetFullPath($from).StartsWith($source + '\')) { throw 'Invalid source path.' }
  if (-not [IO.Path]::GetFullPath($to).StartsWith($target + '\')) { throw 'Invalid target path.' }
  if ((Test-Path -LiteralPath $from) -and (Test-Path -LiteralPath $to)) { throw "Both source and target exist: $name" }
  if (Test-Path -LiteralPath $from) {
    Move-Item -LiteralPath $from -Destination $to
    Write-Output "Relocated $name"
  }
}
foreach ($entry in Get-ChildItem -LiteralPath $source -Force) {
  $to = Join-Path $target $entry.Name
  if (-not (Test-Path -LiteralPath $to)) { Copy-Item -LiteralPath $entry.FullName -Destination $to -Recurse }
}
@{ schema = 1; legacy_root = $source; relocated_from = $source; ready = $false } |
  ConvertTo-Json | Set-Content -LiteralPath (Join-Path $target 'kite-assets.json') -Encoding utf8
Write-Output "Assets are now under $target. Original source files remain at $source."
