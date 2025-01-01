@echo off

set CUDA_VISIBLE_DEVICES=1
set NGPPATH=..\..\..\build
set PYTHONPATH=%PYTHONPATH%;%NGPPATH%;..\python

set DATA_ROOT=..\..\..\dataset\lf_data\lf_data
set RESULT_ROOT=..\checkpoint
set N_STEPS=50000

rem statue
for %%s in (africa, basket, ship, torch) do (
    echo "%%s train >>>"
    mkdir %RESULT_ROOT%\%%s
    python -m run ^
        --scene %DATA_ROOT%\%%s\transforms_train.json ^
        --discard_saturation ^
        --test_transforms %DATA_ROOT%\%%s\transforms_test.json ^
        --network ..\config\srf_%%s.json ^
        --save_snapshot %RESULT_ROOT%\%%s\%%s_ckpt.msgpack ^
        --n_steps %N_STEPS%
)