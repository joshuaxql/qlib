@echo off
pushd %~dp0
if "%SPHINXBUILD%" == "" set SPHINXBUILD=sphinx-build
if "%SPHINXOPTS%" == "" set SPHINXOPTS=-W --keep-going
if "%1" == "" (
    %SPHINXBUILD% -M help . _build %SPHINXOPTS%
) else (
    %SPHINXBUILD% -M %1 . _build %SPHINXOPTS% %2 %3 %4 %5 %6 %7 %8 %9
)
set RESULT=%ERRORLEVEL%
popd
exit /b %RESULT%
