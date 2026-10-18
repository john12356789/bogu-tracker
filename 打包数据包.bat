@echo off
rem ============================================================================
rem  骨骼追踪 · 打数据包
rem
rem  双击 = 打一个完整的「骨骼追踪-数据.zip」。它就是封装这一条：
rem
rem      python src\make_bundle.py
rem
rem  要干别的就把参数拼在后面（或者直接从命令行跑同一个文件）：
rem
rem      打包数据包.bat --dry                 只列清单和体积，不写文件
rem      打包数据包.bat --out D:\x.zip        换输出位置
rem      打包数据包.bat --verify D:\x.zip     检查一个已经打好的包
rem      打包数据包.bat --fast                压缩级别 1，快但大一点
rem
rem  跟 启动.bat 一样存成 GBK，理由见那个文件的文件头。
rem ============================================================================
setlocal
cd /d "%~dp0"
title 骨骼追踪 · 打数据包

set "PYTHON=C:\Python312\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

rem 清单里要打进包的那个 Python 环境。在这里先挡一道——不然要等清单扫完
rem （几分钟）才报"环境缺失"，白等一场。
set "PIPELINE_ENV=E:\anaconda3\envs\gvhmr"

if not exist "src\make_bundle.py" (
  echo [错误] 这个脚本要放在项目根目录，里面要有 src\make_bundle.py。
  echo         当前目录：%CD%
  pause
  exit /b 1
)

rem --verify 只是读一个已有的 zip，用不到那个环境，别拿它挡人
echo %* | findstr /c:"--verify" >nul 2>nul
if errorlevel 1 (
  if not exist "%PIPELINE_ENV%\python.exe" (
    echo [错误] 找不到要打进包里的 Python 环境：%PIPELINE_ENV%
    echo         换一个：  打包数据包.bat --pipeline-env "D:\some\env"
    echo         只是想看看清单：  打包数据包.bat --dry
    pause
    exit /b 1
  )
)

echo.
echo   项目    %CD%
echo   解释器  %PYTHON%
echo   环境    %PIPELINE_ENV%
echo.
echo   要把这些压成一个 zip：
echo     src\  +  GVHMR\  +  权重 3.1 GB  +  Python 环境 6.0 GB  +  ffmpeg
echo     ->  约 6.1 GB，大约 8 分钟。中途别关窗口。
echo.

"%PYTHON%" src\make_bundle.py %*
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
  echo   完成。
) else (
  echo   失败，退出码 %RC%。上面那几行输出里有原因。
)
echo.
pause
exit /b %RC%
