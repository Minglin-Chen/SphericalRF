@echo off

set CUDA_VISIBLE_DEVICES=1
set NGPPATH=..\..\..\build
set PYTHONPATH=%PYTHONPATH%;%NGPPATH%;..\python

set DATA_ROOT=..\..\..\dataset\360_v2
set RESULT_ROOT=..\checkpoint_improved_3
set N_STEPS=50000


for %%s in (bicycle, flowers, garden, stump, treehill, room, kitchen, counter, bonsai) do (
    echo "%%s train >>>"
    mkdir %RESULT_ROOT%\%%s
    python -m run ^
        --scene %DATA_ROOT%\%%s\transforms_train.json ^
        --test_transforms %DATA_ROOT%\%%s\transforms_test.json ^
        --network ..\config\srf_%%s.json ^
        --save_snapshot %RESULT_ROOT%\%%s\%%s_ckpt.msgpack ^
        --n_steps %N_STEPS%
)