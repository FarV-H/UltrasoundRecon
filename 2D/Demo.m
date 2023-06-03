clear
clc

%% 模拟断层数据
[mask, data, nx, ny, dx, dy] = CreateData();

imtool(data, [])

%% 模拟探测器位置
distance = [107.5:7.5:220, 235:15:460];
Points = SetPoint(distance, dx, dy);

% p = zeros(512);
% for ii = 1:size(Points, 1)
%     for jj = 1:size(Points, 2)
%         p(Points(ii, jj, 1), Points(ii, jj, 2)) = 1;
%     end
% end

%% 仿真投影数据
proj = zeros(32, 64);
A = zeros(32, 512, 512, 64);
for ii = 1:32
    [A(:, :, :, ii), proj(:, ii)] = SimProj(squeeze (Points(1,ii,:)), squeeze (Points(2, :, :)), data, mask);
end
for ii = 1:32
    [A(:, :, :, ii + 32), proj(:, ii + 32)] = SimProj(squeeze (Points(2,ii,:)), squeeze (Points(1, :, :)), data, mask);
end


%% 反投影重建
img = zeros(512, 512);
norm = squeeze(sum(A, [1, 4]));
norm(norm == 0) = -1;
% 
% a = uint8(img * 255 / (max(img(:)) - min(img(:))));
% imwrite(a, jet(255), 'test.gif', 'gif', 'DelayTime', 0.05, 'Loopcount', inf);

for ii = 1:64
    for jj = 1:32
        if(sum(sum(A(jj, :, :, ii)))~=0)
            img = img + (norm ~= -1) .* squeeze(A(jj, :, :, ii) * proj(jj, ii)) / sum(A(jj, :, :, ii), 'all') ./ norm;
        end
    end
    a = uint8(img * 255 / (max(img(:)) - min(img(:))));
%     imshow(a,'Colormap',jet(255));
    imwrite(a, jet(255), 'test.gif', 'gif', 'WriteMode', 'append');
end

%% SIRT 迭代重建
result = zeros(512, 512, 11);
result(:, :, 1) = img;

Smooth_Kernel = fspecial("average", [5, 5]);

figure
for irt = 1:10
    d_proj = zeros(32, 64);
    for ii = 1:32
        [~, d_proj(:, ii)] = SimProj(squeeze (Points(1,ii,:)), squeeze (Points(2, :, :)), img, mask);
    end
    for ii = 1:32
        [~, d_proj(:, ii + 32)] = SimProj(squeeze (Points(2,ii,:)), squeeze (Points(1, :, :)), img, mask);
    end
    
    d_proj = proj - d_proj;
    
    for ii = 1:64
        for jj = 1:32
           if(sum(sum(A(jj, :, :, ii)))~=0)
                d_img = (norm ~= -1) .* squeeze(A(jj, :, :, ii) * proj(jj, ii)) / sum(A(jj, :, :, ii), 'all') ./ norm;
            end
        end
%     a = uint8(img * 255 / (max(img(:)) - min(img(:))));
%         imshow(a,'Colormap',jet(255));
%     imwrite(a, jet(255), 'test.gif', 'gif', 'WriteMode', 'append');
    end
    
    img = img + d_img * 0.5;
    img = imfilter(img, Smooth_Kernel);
    a = uint8(img * 255 / (max(img(:)) - min(img(:))));
    imshow(a,'Colormap',jet(255));
    disp(irt)
    result(:, :, irt + 1) = img;
end

for kk = 1:11
    mse(kk) = sum((result(:, :, kk) - data).^2, 'all') / (512 * 512);
end
plot(mse)
