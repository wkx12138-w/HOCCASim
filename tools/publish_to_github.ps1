# -*- coding: utf-8 -*-
<#
.SYNOPSIS
    Publish this repository to GitHub in one command.

.DESCRIPTION
    Creates the repository through the GitHub REST API (if it does not exist
    yet) and pushes the local ``main`` branch to it.  Only a personal access
    token with the ``repo`` scope (classic) or ``Contents: read and write``
    (fine-grained) is required - no ``gh`` CLI, no interactive browser flow.

.EXAMPLE
    $env:GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxx"
    .\tools\publish_to_github.ps1 -User myname

.EXAMPLE
    .\tools\publish_to_github.ps1 -User myname -Token $env:GITHUB_TOKEN -Repo hocca-simulator -Private

.NOTES
    Never commit or paste a token into a tracked file.  Use the environment
    variable and clear it afterwards:

        Remove-Item Env:GITHUB_TOKEN

    Revoke tokens at https://github.com/settings/tokens
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$User,

    [string]$Token = $env:GITHUB_TOKEN,

    [string]$Repo = "hocca-simulator",

    [string]$Description = "Reference simulator for HOCCA: communication-constrained cooperative multi-UAV search",

    [switch]$Private
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($Token)) {
    throw "No token supplied. Set `$env:GITHUB_TOKEN or pass -Token."
}

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Push-Location $repoRoot
try {
    if (-not (Test-Path (Join-Path $repoRoot '.git'))) {
        throw "No git repository at $repoRoot"
    }

    $headers = @{
        Authorization          = "Bearer $Token"
        Accept                 = "application/vnd.github+json"
        "X-GitHub-Api-Version" = "2022-11-28"
        "User-Agent"           = "hocca-simulator-publisher"
    }

    Write-Host "==> checking the token" -ForegroundColor Cyan
    $me = Invoke-RestMethod -Uri "https://api.github.com/user" -Headers $headers -Method Get
    Write-Host "    authenticated as $($me.login)"

    Write-Host "==> ensuring repository $User/$Repo exists" -ForegroundColor Cyan
    $existing = $null
    try {
        $existing = Invoke-RestMethod -Uri "https://api.github.com/repos/$User/$Repo" -Headers $headers -Method Get
    } catch {
        $existing = $null
    }

    if ($null -eq $existing) {
        $body = @{
            name        = $Repo
            description = $Description
            private     = [bool]$Private
            has_issues  = $true
            has_wiki    = $false
        } | ConvertTo-Json
        $created = Invoke-RestMethod -Uri "https://api.github.com/user/repos" -Headers $headers -Method Post -Body $body -ContentType 'application/json'
        Write-Host "    created $($created.html_url)" -ForegroundColor Green
    } else {
        Write-Host "    already exists: $($existing.html_url)"
    }

    $remote = "https://$User`:$Token@github.com/$User/$Repo.git"
    $hasRemote = (& git remote) -contains 'origin'
    if ($hasRemote) {
        & git remote set-url origin $remote
    } else {
        & git remote add origin $remote
    }

    Write-Host "==> pushing main" -ForegroundColor Cyan
    & git push -u origin main
    & git remote set-url origin "https://github.com/$User/$Repo.git"
    & git branch --set-upstream-to=origin/main main 2>$null

    $url = if ($existing) { $existing.html_url } else { $created.html_url }
    Write-Host ""
    Write-Host "done: $url" -ForegroundColor Green
    Write-Host "next:  enable GitHub Pages / add topics, then remove the token from your shell"
}
finally {
    Pop-Location
}
