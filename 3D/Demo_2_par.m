clear
clc

%% 模拟断层数据 探测器位置
scan_mode = '2_parallel';
data_type = 'Section';
[data, geo, Points] = CreateData(scan_mode, data_type);

% % 绘制图像
% figure
% slice(data, [50], [], [100 30]);
% shading flat
% view(-20, 25)
% colormap jet
% caxis([0, 10])

% 绘制探测器位置
% p = zeros(geo.nx, geo.ny, geo.nz);
% for ii = 1:size(Points, 1)
%     for jj = 1:size(Points, 2)
%         for kk = 1:size(Points, 3)
%             p(Points(ii, jj, 1), Points(ii, jj, 2), Points(ii, jj, 3)) = 1;
%         end
%     end
% end
% as(p)

source = reshape(permute(Points(1:2, :, :), [3, 2, 1]), [size(Points(1:2, :, :), 3), size(Points(1:2, :, :), 2), size(Points(1:2, :, :), 1)]);
% source = source(:, [1, 32]);
reciver = zeros(3, size(Points(1:2, :, :), 2), size(Points(1:2, :, :), 1));
reciver(:, :, 1) = permute(Points(2, :, :), [3, 2, 1]);
reciver(:, :, 2) = permute(Points(1, :, :), [3, 2, 1]);
% reciver = reciver(:, :, [1,32]);

%% 仿真投影数据
disp 'Starting Projection Simulation !'
t = tic;
proj = Project3D(source, reciver, data, geo);
toc(t);
disp 'Projection Simulation Complete !'


%% 反投影重建

disp 'Starting Backproject !'
t = tic;
img = BackProject3D(geo, proj, source, reciver);
toc(t);
disp 'Backproject Complete !'


%% SIRT 迭代重建
IR_times = 10;
result = zeros(size(repmat(img, [1, 1, 1, 1 + IR_times])));
result(:, :, :, 1) = img;

Smooth_Kernel = fspecial("average", [5, 5]);

% figure
mse = linspace(0, 0, IR_times);

for irt = 1:IR_times
    disp(' ');
    disp([num2str(irt), ' / ', num2str(IR_times)]);
    disp("IR: Projection .");
    d_proj = Project3D(source, reciver, img, geo);
    
    d_proj = proj - d_proj;
    
    disp("IR: BackProjection .");
    d_img = BackProject3D(geo, d_proj, source, reciver);
    
    img = img + d_img * 0.5;
    img = imfilter(img, Smooth_Kernel);
%     a = uint8(img * 255 / (max(img(:)) - min(img(:))));
%     imshow(a,'Colormap',jet(255));
    result(:, :, :, irt + 1) = img;
    mse(irt) = mean(d_proj, 'all');
end
plot(mse)
save('2_par.mat', 'result', 'mse', 'data');
