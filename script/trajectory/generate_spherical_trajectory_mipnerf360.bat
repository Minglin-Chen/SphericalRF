@echo off

set OUTPUT_ROOT=results\generate_spherical_trajectory\mipnerf360

set PYTHONPATH=..\visualization;%PYTHONPATH%

set COORD_SPEC=opengl
set CAMERA_SCALE=0.05


set scene=bicycle
python -m generate_spherical_trajectory ^
    --filename %OUTPUT_ROOT%\transforms_%scene%.json ^
    --num 150 --radius 0.8 --elevation 30.0 --translation 0.0 0.0 -0.2
python -m visualize_camera ^
    %OUTPUT_ROOT%\transforms_%scene%.json %OUTPUT_ROOT% --coord_spec %COORD_SPEC% --cam_scale %CAMERA_SCALE%


set scene=flowers
python -m generate_spherical_trajectory ^
    --filename %OUTPUT_ROOT%\transforms_%scene%.json ^
    --num 150 --radius 0.8 --elevation 15.0 --translation 0.0 0.0 -0.1
python -m visualize_camera ^
    %OUTPUT_ROOT%\transforms_%scene%.json %OUTPUT_ROOT% --coord_spec %COORD_SPEC% --cam_scale %CAMERA_SCALE%


set scene=garden
python -m generate_spherical_trajectory ^
    --filename %OUTPUT_ROOT%\transforms_%scene%.json ^
    --num 150 --radius 0.8 --elevation 15.0 --translation 0.0 0.0 -0.4
python -m visualize_camera ^
    %OUTPUT_ROOT%\transforms_%scene%.json %OUTPUT_ROOT% --coord_spec %COORD_SPEC% --cam_scale %CAMERA_SCALE%


set scene=stump
python -m generate_spherical_trajectory ^
    --filename %OUTPUT_ROOT%\transforms_%scene%.json ^
    --num 150 --radius 1.0 --elevation 30.0 --translation 0.0 0.0 -0.5
python -m visualize_camera ^
    %OUTPUT_ROOT%\transforms_%scene%.json %OUTPUT_ROOT% --coord_spec %COORD_SPEC% --cam_scale %CAMERA_SCALE%


set scene=treehill
python -m generate_spherical_trajectory ^
    --filename %OUTPUT_ROOT%\transforms_%scene%.json ^
    --num 150 --radius 0.7 --elevation 30.0 --translation 0.0 0.0 -0.2
python -m visualize_camera ^
    %OUTPUT_ROOT%\transforms_%scene%.json %OUTPUT_ROOT% --coord_spec %COORD_SPEC% --cam_scale %CAMERA_SCALE%


set scene=room
python -m generate_spherical_trajectory ^
    --filename %OUTPUT_ROOT%\transforms_%scene%.json ^
    --num 150 --radius 0.6 0.4 0.5 --elevation 30.0 --translation 0.15 0.0 -0.3
python -m visualize_camera ^
    %OUTPUT_ROOT%\transforms_%scene%.json %OUTPUT_ROOT% --coord_spec %COORD_SPEC% --cam_scale %CAMERA_SCALE%


set scene=counter
python -m generate_spherical_trajectory ^
    --filename %OUTPUT_ROOT%\transforms_%scene%.json ^
    --num 150 --radius 0.7 --elevation 30.0 --translation 0.0 0.0 -0.4
python -m visualize_camera ^
    %OUTPUT_ROOT%\transforms_%scene%.json %OUTPUT_ROOT% --coord_spec %COORD_SPEC% --cam_scale %CAMERA_SCALE%


set scene=kitchen
python -m generate_spherical_trajectory ^
    --filename %OUTPUT_ROOT%\transforms_%scene%.json ^
    --num 150 --radius 0.7 --elevation 20.0 --translation 0.0 0.0 -0.4
python -m visualize_camera ^
    %OUTPUT_ROOT%\transforms_%scene%.json %OUTPUT_ROOT% --coord_spec %COORD_SPEC% --cam_scale %CAMERA_SCALE%


set scene=bonsai
python -m generate_spherical_trajectory ^
    --filename %OUTPUT_ROOT%\transforms_%scene%.json ^
    --num 150 --radius 0.8 --elevation 30.0 --translation 0.0 0.0 -0.4
python -m visualize_camera ^
    %OUTPUT_ROOT%\transforms_%scene%.json %OUTPUT_ROOT% --coord_spec %COORD_SPEC% --cam_scale %CAMERA_SCALE%