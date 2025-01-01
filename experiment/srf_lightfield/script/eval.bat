@echo off

set CUDA_VISIBLE_DEVICES=0
set NGPPATH=..\..\..\build
set PYTHONPATH=%PYTHONPATH%;%NGPPATH%;..\python

set DATA_ROOT=..\..\..\dataset\lf_data\lf_data
set RESULT_ROOT=..\checkpoint

for %%s in (africa, basket, ship, torch) do (
    echo "%%s eval >>>"
    python -m run ^
        --load_snapshot %RESULT_ROOT%\%%s\%%s_ckpt.msgpack ^
        --nerf_compatibility ^
        --test_transforms %DATA_ROOT%\%%s\transforms_test.json
)