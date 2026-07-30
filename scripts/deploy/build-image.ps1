param(
    [Parameter(Mandatory = $true)]
    [string]$Repository,
    [Parameter(Mandatory = $true)]
    [string]$Version,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-f0-9]{7,64}$')]
    [string]$GitCommit,
    [switch]$Push
)

$ErrorActionPreference = 'Stop'
$tag = "${Repository}:${Version}"
docker build --pull --build-arg "APP_VERSION=${Version}" `
    --label "org.opencontainers.image.revision=${GitCommit}" --tag $tag .
if ($LASTEXITCODE -ne 0) {
    throw 'Docker image build failed.'
}

if (-not $Push) {
    Write-Output "Built ${tag}. Push is required before an immutable repository digest exists."
    exit 0
}

docker push $tag
if ($LASTEXITCODE -ne 0) {
    throw 'Docker image push failed.'
}
$digest = docker image inspect --format '{{index .RepoDigests 0}}' $tag
if ($LASTEXITCODE -ne 0 -or $digest -notmatch '@sha256:[a-f0-9]{64}$') {
    throw 'Registry did not return an immutable image digest.'
}
Write-Output $digest
