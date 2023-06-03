function [e_t] = Forward2D(Points, image, geo)

% % 设定输出信号并计算频谱
% f_t = 1;
% F_w = fft(f_t);

e_t = zeros(size(Points, 1), size(Points, 1), geo.L);
E_w = zeros(1, geo.L);

% 选择发射器与探测器
for mm = 1:size(Points, 1)
    for nn = 1:size(Points, 1)
%         [X, Y] = meshgrid(1:size(image, 1), 1:size(image, 2));
%         d_t = sqrt((X * geo.dx - Points(mm)).^2 + (Y * geo.dy) .^ 2);
%         d_r = sqrt((X * geo.dx - Points(nn)).^2 + (Y * geo.dy) .^ 2);
%         w = 2 * pi * geo.f * (1:geo.L) / geo.L;
%         k = w ./ geo.c;
%         temp_dt = (-1 ./ sqrt(8 * pi * 1i * repmat(d_t, [1, 1, geo.L]).*permute(repmat(k', [1, size(image)]), [2, 3, 1]))) .* exp(-1i * repmat(d_t, [1, 1, geo.L]).*permute(repmat(k', [1, size(image)]), [2, 3, 1]));
%         temp_dr = (-1 ./ sqrt(8 * pi * 1i * repmat(d_r, [1, 1, geo.L]).*permute(repmat(k', [1, size(image)]), [2, 3, 1]))) .* exp(-1i * repmat(d_r, [1, 1, geo.L]).*permute(repmat(k', [1, size(image)]), [2, 3, 1]));
%         E_w = sum(repmat(image, [1, 1, geo.L]) .* temp_dt .* temp_dr, [1, 2]);
        for ii = 1:size(image, 1)
            for jj = 1:size(image, 2)
                if(image(ii, jj) ~= 0)
                    d_t = sqrt((ii * geo.dx - Points(mm, 1))^2 + (jj * geo.dy - Points(mm, 2)) ^ 2);
                    d_r = sqrt((ii * geo.dx - Points(nn, 1))^2 + (jj * geo.dy - Points(nn, 2)) ^ 2);
                    w = 2 * pi * geo.f * (1:geo.L) / geo.L;
                    k = w ./ geo.c;
                    E_w = E_w + image(ii, jj) * (-1 ./ (8 * pi * sqrt(d_t * d_r) * k)) .* exp(-1i * k * (d_t + d_r));
                end
            end
        end
        e_t(mm, nn, :) = abs(ifft(E_w));
        E_w = zeros(1, geo.L);
        fprintf('(%d, %d)', mm, nn);
    end
    fprintf('\n');
end
end