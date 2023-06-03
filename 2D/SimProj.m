function  [A, proj]= SimProj(source, reciecer, data, mask)
% 确定扫描范围内像素数及具体像素位置
k = 0;
for ii = 1:size(data, 1)
    for jj = 1:size(data, 2)
        if mask(ii, jj) == 1
            k = k + 1;
            pixel(k, 1) = ii;
            pixel(k, 2) = jj;
            data_mask(k) = data(ii, jj);
        end
    end
end

weight = zeros(size(pixel, 1), size(reciecer, 1));
proj = zeros(size(reciecer, 1));
%分别计算每个像素的路径矩阵
for nn = 1:size(pixel, 1)
    for ii = 1:size(reciecer, 1)
%         if ii == 7
%             ii;
%         end
        a1 = (pixel(nn, 1) - 0.5 - source(1)) * (reciecer(ii, 2) - source(2)) / (reciecer(ii, 1) - source(1)) + source(2);
        a2 = (pixel(nn, 1) + 0.5 - source(1)) * (reciecer(ii, 2) - source(2)) / (reciecer(ii, 1) - source(1)) + source(2);
        b1 = (pixel(nn, 2) - 0.5 - source(2)) * (reciecer(ii, 1) - source(1)) / (reciecer(ii, 2) - source(2)) + source(1);
        b2 = (pixel(nn, 2) + 0.5 - source(2)) * (reciecer(ii, 1) - source(1)) / (reciecer(ii, 2) - source(2)) + source(1);
        k = 0;
        if((a1 <= pixel(nn, 2) + 0.5 && a1 > (pixel(nn, 2) - 0.5)))
            k = k + 1;
            p(k, 1) = pixel(nn, 1) - 0.5;
            p(k, 2) = a1;
        end
        if((a2 < pixel(nn, 2) + 0.5 && a2 >= (pixel(nn, 2) - 0.5)))
            k = k + 1;
            p(k, 1) = pixel(nn, 1) + 0.5;
            p(k, 2) = a2;
        end
        if((b1 < pixel(nn, 1) + 0.5 && b1 >= (pixel(nn, 1) - 0.5)))
            k = k + 1;
            p(k, 1) = b1;
            p(k, 2) = pixel(nn, 2) - 0.5;
        end
        if((b2 <= pixel(nn, 1) + 0.5 && b2 > (pixel(nn, 1) - 0.5)))
            k = k + 1;
            p(k, 1) = b2;
            p(k, 2) = pixel(nn, 2) + 0.5;
        end

        if(k == 2)
            weight(nn, ii) = sqrt((p(1, 1)-p(2, 1))^2 + (p(1, 2)-p(2, 2))^2);
        else
            weight(nn, ii) = 0;
        end
        
        % 路径平行或垂直
%         if(source(1) == reciecer(ii, 1) || source(2) == reciecer(ii, 2))
%             weight(nn, ii) = 1;
%         end
    end
end
proj = data_mask * weight;

A = zeros(size(reciecer, 1), size(data, 1), size(data, 2));
for nn = 1:size(reciecer, 1)
    k = 0;
    for ii = 1:size(data, 1)
        for jj = 1:size(data, 2)
            if mask(ii, jj) == 1
                k = k + 1;
                A(nn, ii, jj) = A(nn, ii, jj) + weight(k, nn);
            end
        end
    end
    %imshow(squeeze(A(nn, :, :)), [])
end

end