clear
clc

%% 模拟断层数据 探测器位置
scan_mode = '2_parallel';
data_type = 'Section';
[image, geo, Points] = CreateData(scan_mode, data_type);
% for ii = 1:size(image, 3)
%     image(:, :, ii) = image(:, :, 1);
% end

% % 绘制图像
% figure
% slice(image, [50], [], [100 30]);
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

%% 仿真投影数据
disp 'Starting Data Simulation !'
t = tic;
data(:, :, :, 1) = Forward3D(squeeze(Points(1, :, :)), image, geo);
data(:, :, :, 2) = Forward3D(squeeze(Points(2, :, :)), image, geo);
toc(t);
disp 'Projection Data Complete !'


% %% TFM 全聚焦重建

disp 'Starting Total Focusing Method (TFM) !'
t = tic;
recon_1 = TFD3D(squeeze(Points(1, :, :)), data(:, :, :, 1), geo);
recon_2 = TFD3D(squeeze(Points(2, :, :)), data(:, :, :, 2), geo);
recon = recon_1 + recon_2;
toc(t);
disp 'Recon Complete !'

as(image);
as(recon);
% 
% 
% %% SIRT 迭代重建
% IR_times = 10;
% result = zeros(size(repmat(img, [1, 1, 1, 1 + IR_times])));
% result(:, :, :, 1) = img;
% 
% Smooth_Kernel = fspecial("average", [5, 5]);
% 
% % figure
% mse = linspace(0, 0, IR_times);
% 
% for irt = 1:IR_times
%     disp(' ');
%     disp([num2str(irt), ' / ', num2str(IR_times)]);
%     disp("IR: Projection .");
%     d_proj = Project3D(source, reciver, img, geo);
%     
%     d_proj = proj - d_proj;
%     
%     disp("IR: BackProjection .");
%     d_img = BackProject3D(geo, d_proj, source, reciver);
%     
%     img = img + d_img * 0.5;
%     img = imfilter(img, Smooth_Kernel);
% %     a = uint8(img * 255 / (max(img(:)) - min(img(:))));
% %     imshow(a,'Colormap',jet(255));
%     result(:, :, :, irt + 1) = img;
%     mse(irt) = mean(d_proj, 'all');
% end
% plot(mse)
% save('2_par.mat', 'result', 'mse', 'data');
% % 
% plot(abs(squeeze(data(1, 1, :, 1))))
% hold on
% plot(abs(squeeze(data(1, 2, :, 1))))
% plot(abs(squeeze(data(1, 3, :, 1))))
% plot(abs(squeeze(data(1, 4, :, 1))))
% plot(abs(squeeze(data(1, 5, :, 1))))
% plot(abs(squeeze(data(1, 6, :, 1))))
% plot(abs(squeeze(data(1, 7, :, 1))))
% plot(abs(squeeze(data(1, 8, :, 1))))
% plot(abs(squeeze(data(1, 9, :, 1))))
% plot(abs(squeeze(data(1, 10, :, 1))))
% plot(abs(squeeze(data(1, 11, :, 1))))
% plot(abs(squeeze(data(1, 12, :, 1))))
% plot(abs(squeeze(data(1, 13, :, 1))))
% plot(abs(squeeze(data(1, 14, :, 1))))
% plot(abs(squeeze(data(1, 15, :, 1))))
% plot(abs(squeeze(data(1, 16, :, 1))))


A=fspecial('average',[3,3]);
Y=imfilter(recon,A);

