clear
clc

%% 模拟断层数据
mode = '4_tilt';
[data, geo] = CreateData(mode);

% as(data)

%% 模拟探测器位置
distance = linspace(20, 480, 8);
Points = SetPoint(distance, geo);

% 绘制探测器位置
p = zeros(geo.nx, geo.ny, geo.nz);
for ii = 1:size(Points, 1)
    for jj = 1:size(Points, 2)
        for kk = 1:size(Points, 3)
            p(Points(ii, jj, 1), Points(ii, jj, 2), Points(ii, jj, 3)) = 1;
        end
    end
end

source = reshape(permute(Points, [3, 2, 1]), [size(Points, 3), size(Points, 2) * size(Points, 1)]);
reciver = zeros(3, size(Points, 2), size(Points, 2) * size(Points, 1));
reciver(:, :, 1:size(Points, 2)) = repmat(permute(Points(2, :, :), [3, 2, 1]), [1, 1, size(Points, 2)]);
reciver(:, :, size(Points, 2) + 1:size(Points, 2) * 2) = repmat(permute(Points(1, :, :), [3, 2, 1]), [1, 1, size(Points, 2)]);
reciver(:, :, 2 * size(Points, 2) + 1:size(Points, 2) * 3) = repmat(permute(Points(4, :, :), [3, 2, 1]), [1, 1, size(Points, 2)]);
reciver(:, :, 3 * size(Points, 2) + 1:size(Points, 2) * 4) = repmat(permute(Points(3, :, :), [3, 2, 1]), [1, 1, size(Points, 2)]);

%% 仿真投影数据
disp 'Starting Projection Simulation !'
t = tic;
proj = SimProj(source, reciver, data, geo);
disp 'Projection Simulation Complete !'
toc(t);


%% 反投影重建

disp 'Starting Backproject !'
t = tic;
img = BackProject(geo, proj, source, reciver);
disp 'Backproject Complete !'
toc(t);


%% SIRT 迭代重建
IR_times = 10;
result = zeros(size(repmat(img, [1, 1, 1, 1 + IR_times])));
result(:, :, :, 1) = img;

Smooth_Kernel = fspecial("average", [5, 5]);

figure
mse = linspace(0, 0, IR_times);
for irt = 1:IR_times
    disp([num2str(irt), ' / ', num2str(IR_times)])
    disp("IR: Projection .")
    d_proj = SimProj(source, reciver, img, geo);
    
    d_proj = proj - d_proj;
    
    disp("IR: BackProjection .")
    
    d_img = BackProject(geo, d_proj, source, reciver);
    
    img = img + d_img * 0.5;
    img = imfilter(img, Smooth_Kernel);
% %     a = uint8(img * 255 / (max(img(:)) - min(img(:))));
% %     imshow(a,'Colormap',jet(255));
%     disp([irt, ' / ', IR_times])
    result(:, :, :, irt + 1) = img;
    mse(irt) = sum(d_proj, 'all');
end
plot(mse)
save('4_tilt.mat', "result", "mse", "data")
