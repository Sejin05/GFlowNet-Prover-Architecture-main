# setup_env.ps1: Dynamically configures MinGW self-contained toolchain environment for Windows GNU target
# without requiring hardcoded absolute user paths in version-controlled config files.

$ErrorActionPreference = "Stop"

$userProfile = [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::UserProfile)
$toolchainPath = Join-Path $userProfile ".rustup\toolchains\stable-x86_64-pc-windows-gnu\lib\rustlib\x86_64-pc-windows-gnu"
$selfContainedBin = Join-Path $toolchainPath "bin\self-contained"
$selfContainedLib = Join-Path $toolchainPath "lib\self-contained"

if (Test-Path $selfContainedBin) {
    Write-Host "[+] Found self-contained MinGW toolchain at: $selfContainedBin" -ForegroundColor Green
    
    # Prepend to session PATH for current PowerShell process
    $env:PATH = "$selfContainedBin;$env:PATH"
    
    # Set Cargo environment overrides dynamically
    $gccExe = Join-Path $selfContainedBin "x86_64-w64-mingw32-gcc.exe"
    if (Test-Path $gccExe) {
        $env:CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER = $gccExe
        $env:RUSTFLAGS = "-L native=$selfContainedLib -C link-arg=-B$selfContainedBin/ --cfg getrandom_backend=""windows_legacy"""
        Write-Host "[+] Dynamically exported CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER" -ForegroundColor Cyan
        Write-Host "[+] Dynamically exported RUSTFLAGS with self-contained libs and getrandom_backend" -ForegroundColor Cyan
    }
} else {
    Write-Host "[!] Self-contained MinGW toolchain directory not found at $selfContainedBin" -ForegroundColor Yellow
    Write-Host "    If using MSVC or Linux/Docker, no environment adjustments are needed." -ForegroundColor Gray
}

Write-Host "[OK] Environment configuration completed successfully." -ForegroundColor Green
