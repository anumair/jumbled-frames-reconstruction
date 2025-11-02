# Execution Time Logging Script
# Runs all versions and logs execution times

$logFile = "EXECUTION_TIME_LOG.txt"
"=" * 80 | Out-File $logFile
"EXECUTION TIME LOG - Frame Reconstruction Pipeline" | Out-File $logFile -Append
"Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $logFile -Append
"=" * 80 | Out-File $logFile -Append
"" | Out-File $logFile -Append

# Activate virtual environment
Write-Host "`n🔧 Activating virtual environment..." -ForegroundColor Cyan
.\venv\Scripts\Activate.ps1

# Function to run and time a command
function Run-TimedCommand {
    param(
        [string]$Name,
        [string]$Command,
        [string]$Description
    )
    
    Write-Host "`n" "=" * 80 -ForegroundColor Yellow
    Write-Host "🚀 Running: $Name" -ForegroundColor Green
    Write-Host "Description: $Description" -ForegroundColor Cyan
    Write-Host "=" * 80 -ForegroundColor Yellow
    
    "`n$Name" | Out-File $logFile -Append
    "-" * 80 | Out-File $logFile -Append
    "Description: $Description" | Out-File $logFile -Append
    "Start Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $logFile -Append
    
    $startTime = Get-Date
    
    try {
        Invoke-Expression $Command
        $success = $true
    }
    catch {
        Write-Host "❌ Error: $_" -ForegroundColor Red
        "Error: $_" | Out-File $logFile -Append
        $success = $false
    }
    
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    "End Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $logFile -Append
    "Duration: $($duration.ToString('hh\:mm\:ss\.fff'))" | Out-File $logFile -Append
    "Status: $(if($success){'✅ Success'}else{'❌ Failed'})" | Out-File $logFile -Append
    "" | Out-File $logFile -Append
    
    Write-Host "`n⏱️  Duration: $($duration.ToString('hh\:mm\:ss\.fff'))" -ForegroundColor Magenta
    Write-Host "Status: $(if($success){'✅ Success'}else{'❌ Failed'})`n" -ForegroundColor $(if($success){'Green'}else{'Red'})
}

# V1: ResNet50 + 2-opt + Sliding Window Refinement
Write-Host "`n" "=" * 80 -ForegroundColor Cyan
Write-Host "📦 VERSION 1: ResNet50 + Refinement" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

Run-TimedCommand -Name "V1 - Step 1: Extract Features" `
    -Command "python src/approach_resnet_tf_refinement/v1/1_extract_resnet_features.py" `
    -Description "Extract ResNet50 features from frames"

Run-TimedCommand -Name "V1 - Step 2: Compute Similarity" `
    -Command "python src/approach_resnet_tf_refinement/v1/2_compute_similarity.py" `
    -Description "Compute similarity matrix from features"

Run-TimedCommand -Name "V1 - Step 3: Smart Starting Point" `
    -Command "python src/approach_resnet_tf_refinement/v1/3_order_frames_smart.py" `
    -Description "Find optimal starting frame using smart detection"

Run-TimedCommand -Name "V1 - Step 4: 2-opt Refinement" `
    -Command "python src/approach_resnet_tf_refinement/v1/4_refine_2opt.py" `
    -Description "Apply 2-opt local search optimization"

Run-TimedCommand -Name "V1 - Step 5: Sliding Window" `
    -Command "python src/approach_resnet_tf_refinement/v1/5_sliding_window.py" `
    -Description "Apply sliding window refinement"

Run-TimedCommand -Name "V1 - Step 6: Reconstruct Video" `
    -Command "python src/approach_resnet_tf_refinement/v1/6_reconstruct_video.py" `
    -Description "Create final reconstructed video"

# V2: V1 + Optical Flow Enhancement
Write-Host "`n" "=" * 80 -ForegroundColor Cyan
Write-Host "📦 VERSION 2: V1 + Optical Flow Enhancement" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

Run-TimedCommand -Name "V2 - Optical Flow Reconstruction" `
    -Command "python src/approach_resnet_tf_refinement/v2/reconstruct_fast_optical.py" `
    -Description "Apply optical flow refinement on V1 output"

Run-TimedCommand -Name "V2 - Construct Video" `
    -Command "python src/approach_resnet_tf_refinement/v2/construct_video.py" `
    -Description "Create V2 video from refined frames"

# V3: Reversed V2 Output
Write-Host "`n" "=" * 80 -ForegroundColor Cyan
Write-Host "📦 VERSION 3: Reversed V2 Output" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

Run-TimedCommand -Name "V3 - Reverse Frame Order" `
    -Command "python src/approach_resnet_tf_refinement/v3/reverse_video.py" `
    -Description "Reverse V2 frame order"

Run-TimedCommand -Name "V3 - Reconstruct Video" `
    -Command "python src/approach_resnet_tf_refinement/v3/reconstruct_video.py" `
    -Description "Create V3 video from reversed frames"

# Summary
Write-Host "`n" "=" * 80 -ForegroundColor Green
Write-Host "✅ ALL VERSIONS COMPLETED" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Green
Write-Host "`n📊 Execution log saved to: $logFile" -ForegroundColor Cyan

"=" * 80 | Out-File $logFile -Append
"EXECUTION COMPLETED" | Out-File $logFile -Append
"Total Script End Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $logFile -Append
"=" * 80 | Out-File $logFile -Append
