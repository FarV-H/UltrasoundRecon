clc
clear

%% 编译cuda函数库
mexcuda -setup C++
mexcuda -dynamic -I../CUDA/include/ ../CUDA/src/mex_project_3d.cu ../CUDA/src/project_3d.cu ../CUDA/src/types.cu -outdir ../MEX
mexcuda -dynamic -I../CUDA/include/ ../CUDA/src/mex_backproject_3d.cu ../CUDA/src/backproject_3d.cu ../CUDA/src/types.cu -outdir ../MEX

disp('Compilation complete')

%% 添加 MEX 编译库路径
addpath('../MEX');

disp('Add path complete')
