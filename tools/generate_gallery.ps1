# Generate the 30-scene screenshot gallery (Theia: 256 frames, 1280×720).
# Usage: .\tools\generate_gallery.ps1 [-Renderer path\to\theia.exe] [-OutputDir screenshots]
param(
    [string]$Renderer = "build\theia.exe",
    [string]$OutputDir = "screenshots",
    [int]$Frames = 256,
    [int]$Width = 1280,
    [int]$Height = 720
)

$scenes = @(
    "ABeautifulGame",
    "bunny_shaderball",
    "camera_suzanne",
    "cornell_classic",
    "cornell_classic_rec709",
    "cornell_empty",
    "cornell_spheres",
    "cornell_suzanne",
    "cornell_textured_cube",
    "dragon_teapot",
    "openpbr_advanced",
    "openpbr_coat",
    "openpbr_dielectrics",
    "openpbr_fuzz",
    "openpbr_metals",
    "openpbr_organics",
    "openpbr_specular",
    "openpbr_thinfilm",
    "shaderball_base",
    "shaderball_checker",
    "shaderball_coat",
    "shaderball_emission",
    "shaderball_fuzz",
    "shaderball_metal",
    "shaderball_opacity",
    "shaderball_specular",
    "shaderball_subsurface",
    "shaderball_thinfilm",
    "shaderball_transmission",
    "shader_ball"
)

if (!(Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir | Out-Null }

$total = $scenes.Count
$i = 0
foreach ($scene in $scenes) {
    $i++
    Write-Host "[$i/$total] $scene ..."
    $out = Join-Path $OutputDir "$scene.png"
    & $Renderer --scene $scene --offscreen-frames $Frames --width $Width --height $Height `
        --output $out --no-validation 2>&1 | Select-String 'Saved|ERROR' | Select-Object -First 2
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "  $scene exited with code $LASTEXITCODE"
    }
}
Write-Host "Gallery complete: $total scenes -> $OutputDir"
