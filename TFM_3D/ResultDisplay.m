res = recon;
% res(result<1) = 1;
figure
set(gcf,'position',[200,200,350,600])
slice(res(:, :, :, 1), 110, 110, 30);
shading flat
view(-25, 15)
caxis([8000, 15000])

figure
set(gcf,'position',[200,200,350,600])
slice(data, 110, 110, 30);
shading flat
view(-25, 15)
caxis([1, 4])

figure
slice(data, [65], [], 65);
shading flat
view(-20, 25)
colormap jet
caxis([0, 3])

X_fft = fftn(X);
X_fft = fftshift(X_fft);
[nx, ny,nz] = meshgrid(1:128, 1:128, 1:256);
w = sqrt((nx - 64.5).^2 + (ny - 64.5).^2 + (nz - 128.5).^2);
X_filter = abs(ifftn(X_fft .* w));
% subplot(1,2,2);
% slice(X_filter, [35, 95], 95, 30);
% shading flat
% view(-20, 25)
% caxis([1, 4])
% 
% ree  = X_filter;
% ree(X_filter<1)  = 1;

res = recon;
% res(res<2) = 2;
for ii = 1:256
    a = uint8((res(:, :, ii) - min(min(res(:, :, ii)))) * 255 / (max(max(res(:, :, ii))) - min(min(res(:, :, ii)))));
%     a(a<25) = 0;
    if(ii == 1)
        imwrite(a, jet(255), 'test_8.gif', 'gif', 'DelayTime', 0.02, 'Loopcount', inf);
    else
        imwrite(a, jet(255), 'test_8.gif', 'gif', 'WriteMode', 'append','DelayTime',0.02);
    end
end