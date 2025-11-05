@echo off
echo ========================================
echo Text Removal Model Data Generation
echo ========================================
echo.
echo Configuration:
echo - Chinese samples: 630 images (60%%)
echo - English samples: 315 images (30%%)
echo - Mixed samples: 105 images (10%%)
echo - Total: 1050 images
echo - Image height: 64px (total 192px stacked)
echo - Format: bg_and_text_mask (vertical stack)
echo.
echo ========================================
echo.

set START_TIME=%time%
echo Start time: %START_TIME%
echo.

echo Generating data...
python main.py --config example_data\generate_mixed_language.py --dataset img --num_processes 4 --log_period 10

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo [SUCCESS] Data generation completed!
    echo ========================================
    echo.
    echo Start time: %START_TIME%
    echo End time: %time%
    echo.
    echo Data saved to: example_data\output\mixed_language\
    echo.
    echo Statistics:
    echo - Chinese samples: 630 images (60%%)
    echo - English samples: 315 images (30%%)
    echo - Mixed samples: 105 images (10%%)
    echo - Total: 1050 images
    echo.
) else (
    echo.
    echo ========================================
    echo [ERROR] Generation failed!
    echo ========================================
    echo Error code: %errorlevel%
    echo.
)

pause
