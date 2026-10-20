param(
    [string]$Subscription = '358ca1c3-7e57-42cf-b506-d4eb78529b04',
    [string]$ResourceGroup = 'rg-speech-tts',
    [string]$AccountName = 'speech-tts-358ca1c3-jpe',
    [string]$Region = $env:SPEECH_REGION,
    [string]$Voice = 'en-US-AvaMultilingualNeural'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$voiceDirectory = Join-Path $projectRoot 'demo\showcase'
New-Item -ItemType Directory -Path $voiceDirectory -Force | Out-Null
$destination = Join-Path $voiceDirectory 'mission-brief.wav'
$temporary = Join-Path $voiceDirectory ('.speech-' + [guid]::NewGuid().ToString('N') + '.tmp')
$transcript = 'Mission control, look at that view. An astronaut is floating above the blue Earth, with white clouds far below.'
$format = 'riff-24khz-16bit-mono-pcm'
$speechKey = $env:SPEECH_KEY
$headers = @{}
try {
    if ([string]::IsNullOrWhiteSpace($Region)) {
        $Region = az cognitiveservices account show --subscription $Subscription --resource-group $ResourceGroup --name $AccountName --query location -o tsv --only-show-errors
        if ($LASTEXITCODE -ne 0) { throw 'Could not resolve the Azure Speech region.' }
        $Region = ([string]$Region).Trim()
    }
    if ($Region -notmatch '^[a-z0-9]+$') { throw 'Invalid Azure Speech region.' }
    if ([string]::IsNullOrWhiteSpace($speechKey)) {
        # Capture the key in this process only. Never print it or write it to an asset.
        $speechKey = az cognitiveservices account keys list --subscription $Subscription --resource-group $ResourceGroup --name $AccountName --query key1 -o tsv --only-show-errors
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($speechKey)) {
            throw 'Could not retrieve the Azure Speech key. Sign in with Azure CLI or set SPEECH_KEY.'
        }
    }
    $headers = @{
        'Ocp-Apim-Subscription-Key' = ([string]$speechKey).Trim()
        'X-Microsoft-OutputFormat' = $format
    }
    $escapedVoice = [System.Security.SecurityElement]::Escape($Voice)
    $escapedText = [System.Security.SecurityElement]::Escape($transcript)
    $ssml = "<speak version='1.0' xml:lang='en-US'><voice name='$escapedVoice'><prosody rate='-3%'>$escapedText</prosody></voice></speak>"
    try {
        Invoke-WebRequest -UseBasicParsing -Method Post -Uri "https://$Region.tts.speech.microsoft.com/cognitiveservices/v1" -Headers $headers -ContentType 'application/ssml+xml' -UserAgent 'KiteDemoNarration/1.0' -Body ([System.Text.Encoding]::UTF8.GetBytes($ssml)) -OutFile $temporary -TimeoutSec 90
    } catch {
        $status = if ($_.Exception.Response) { [int]$_.Exception.Response.StatusCode } else { 'unavailable' }
        throw "Azure Speech synthesis failed (HTTP $status). Check resource access, region, and voice availability."
    }
    $wave = [System.IO.File]::ReadAllBytes($temporary)
    if ($wave.Length -le 44 -or [System.Text.Encoding]::ASCII.GetString($wave, 0, 4) -ne 'RIFF' -or [System.Text.Encoding]::ASCII.GetString($wave, 8, 4) -ne 'WAVE') {
        throw 'Azure Speech did not return a valid WAV file; existing narration was preserved.'
    }
    Move-Item -LiteralPath $temporary -Destination $destination -Force
    $metadata = @{
        provider = 'Azure AI Speech'
        voice = $Voice
        region = $Region
        format = $format
        transcript = $transcript
        generated_utc = [DateTime]::UtcNow.ToString('o')
        sha256 = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
    } | ConvertTo-Json
    [System.IO.File]::WriteAllText((Join-Path $voiceDirectory 'speech.json'), $metadata, [System.Text.UTF8Encoding]::new($false))
    Write-Output "Created Azure Speech narration using $Voice ($Region)."
} finally {
    $headers.Clear()
    $speechKey = $null
    if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary }
}
