# PowerShell script to start the voice agent application
Write-Host "Starting AI Voice Agent Application..." -ForegroundColor Green

# Start Backend Agent
Write-Host "`nStarting Backend Agent..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\backend'; python -m uv run python src/agent.py dev"

# Wait a bit for backend to initialize
Start-Sleep -Seconds 3

# Start Frontend
Write-Host "Starting Frontend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\frontend'; npm run dev"

Write-Host "`n✅ Application started!" -ForegroundColor Green
Write-Host "Backend agent is running in one window" -ForegroundColor Cyan
Write-Host "Frontend will be available at http://localhost:3000" -ForegroundColor Cyan
Write-Host "`nPress Ctrl+C in each window to stop the services" -ForegroundColor Gray
