@echo off

set CUDA_VISIBLE_DEVICES=1
set NGPPATH=..\..\..\build
set PYTHONPATH=%PYTHONPATH%;%NGPPATH%;..\python

set DATA_ROOT=..\..\..\dataset\360_v2
set RESULT_ROOT=..\checkpoint_improved_3
set SPHERICAL_TRAJECTORY_ROOT=..\..\..\script\trajectory\results\generate_spherical_trajectory\mipnerf360


@REM set scene=bicycle
@REM python -m run ^
@REM     --load_snapshot %RESULT_ROOT%\%scene%\%scene%_ckpt.msgpack ^
@REM     --predict_transforms %SPHERICAL_TRAJECTORY_ROOT%\transforms_%scene%.json ^
@REM     --predict_width 1280 ^
@REM     --predict_height 720 ^
@REM     --predict_angle_x 56 ^
@REM     --fps 30
@REM @REM --predict_width 1237
@REM @REM --predict_height 822
@REM @REM @REM 56                     50


@REM set scene=flowers
@REM python -m run ^
@REM     --load_snapshot %RESULT_ROOT%\%scene%\%scene%_ckpt.msgpack ^
@REM     --predict_transforms %SPHERICAL_TRAJECTORY_ROOT%\transforms_%scene%.json ^
@REM     --predict_width 1280 ^
@REM     --predict_height 720 ^
@REM     --predict_angle_x 60 ^
@REM     --fps 30
@REM @REM --predict_width 1256
@REM @REM --predict_height 828
@REM 60


@REM set scene=garden
@REM python -m run ^
@REM     --load_snapshot %RESULT_ROOT%\%scene%\%scene%_ckpt.msgpack ^
@REM     --predict_transforms %SPHERICAL_TRAJECTORY_ROOT%\transforms_%scene%.json ^
@REM     --predict_width 1280 ^
@REM     --predict_height 720 ^
@REM     --predict_angle_x 70 ^
@REM     --fps 30
@REM @REM --predict_width 1297
@REM @REM --predict_height 840
@REM 68


@REM set scene=stump
@REM python -m run ^
@REM     --load_snapshot %RESULT_ROOT%\%scene%\%scene%_ckpt.msgpack ^
@REM     --predict_transforms %SPHERICAL_TRAJECTORY_ROOT%\transforms_%scene%.json ^
@REM     --predict_width 1280 ^
@REM     --predict_height 720 ^
@REM     --predict_angle_x 57 ^
@REM     --fps 30
@REM @REM @REM --predict_width 1245
@REM @REM @REM --predict_height 825
@REM @REM 57 60


@REM set scene=treehill
@REM python -m run ^
@REM     --load_snapshot %RESULT_ROOT%\%scene%\%scene%_ckpt.msgpack ^
@REM     --predict_transforms %SPHERICAL_TRAJECTORY_ROOT%\transforms_%scene%.json ^
@REM     --predict_width 1280 ^
@REM     --predict_height 720 ^
@REM     --predict_angle_x 60 ^
@REM     --fps 30
@REM @REM --predict_width 1267
@REM @REM --predict_height 832
@REM 62


@REM set scene=room
@REM python -m run ^
@REM     --load_snapshot %RESULT_ROOT%\%scene%\%scene%_ckpt.msgpack ^
@REM     --predict_transforms %SPHERICAL_TRAJECTORY_ROOT%\transforms_%scene%.json ^
@REM     --predict_width 1280 ^
@REM     --predict_height 720 ^
@REM     --predict_angle_x 52 ^
@REM     --fps 30
@REM @REM --predict_width 1557
@REM @REM --predict_height 1038
@REM 52


@REM set scene=counter
@REM python -m run ^
@REM     --load_snapshot %RESULT_ROOT%\%scene%\%scene%_ckpt.msgpack ^
@REM     --predict_transforms %SPHERICAL_TRAJECTORY_ROOT%\transforms_%scene%.json ^
@REM     --predict_width 1280 ^
@REM     --predict_height 720 ^
@REM     --predict_angle_x 52 ^
@REM     --fps 30
@REM @REM --predict_width 1558
@REM @REM --predict_height 1038


set scene=kitchen
python -m run ^
    --load_snapshot %RESULT_ROOT%\%scene%\%scene%_ckpt.msgpack ^
    --predict_transforms %SPHERICAL_TRAJECTORY_ROOT%\transforms_%scene%.json ^
    --predict_width 1280 ^
    --predict_height 720 ^
    --predict_angle_x 51 ^
    --fps 30
@REM --predict_width 1558
@REM --predict_height 1039


@REM set scene=bonsai
@REM python -m run ^
@REM     --load_snapshot %RESULT_ROOT%\%scene%\%scene%_ckpt.msgpack ^
@REM     --predict_transforms %SPHERICAL_TRAJECTORY_ROOT%\transforms_%scene%.json ^
@REM     --predict_width 1280 ^
@REM     --predict_height 720 ^
@REM     --predict_angle_x 60 ^
@REM     --fps 30
@REM @REM --predict_width 1559
@REM @REM --predict_height 1039
@REM 52