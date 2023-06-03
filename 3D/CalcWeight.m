function [weight] = CalcWeight(source, reciecer, geo)
%定义坐标矩阵
idx_x = zeros(1, ceil(norm(source - reciecer) * 8));
idx_y = zeros(1, ceil(norm(source - reciecer) * 8));
idx_z = zeros(1, ceil(norm(source - reciecer) * 8));

n = 0;
l =  norm(reciecer - source);
for ii = 1:size(geo.mask, 1)
    for jj = 1:size(geo.mask, 2)
        for kk = 1:size(geo.mask, 3)
            d = norm(cross(reciecer - source, [ii; jj; kk] - source)) / l;
            if(d < sqrt(3 / 4) && norm([ii; jj; kk] - source) <= l && norm([ii; jj; kk] - reciecer) <= l)
                n = n + 1;
                idx_x(n) = ii;
                idx_y(n) = jj;
                idx_z(n) = kk;
            end
        end
    end
end
idx_x = idx_x(1:n);
idx_y = idx_y(1:n);
idx_z = idx_z(1:n);

temp_points = single(zeros(6, size(idx_x, 2), 3));

temp_points(1, :, 1) = idx_x + 0.5;
temp_points(1, :, 2) = single((source(1) * reciecer(2) - source(2) * reciecer(1) + (idx_x + 0.5)...
                            * source(2) - (idx_x + 0.5) * reciecer(2)) / (source(1) - reciecer(1)));
temp_points(1, :, 3) = single((source(1) * reciecer(3) - source(3) * reciecer(1) + (idx_x + 0.5)...
                            * source(3) - (idx_x + 0.5) * reciecer(3)) / (source(1) - reciecer(1)));

temp_points(2, :, 1) = idx_x - 0.5;
temp_points(2, :, 2) = single((source(1) * reciecer(2) - source(2) * reciecer(1) + (idx_x - 0.5)...
                             * source(2) - (idx_x - 0.5) * reciecer(2)) / (source(1) - reciecer(1)));
temp_points(2, :, 3)  = single((source(1) * reciecer(3) - source(3) * reciecer(1) + (idx_x - 0.5)...
                             * source(3) - (idx_x - 0.5) * reciecer(3)) / (source(1) - reciecer(1)));

temp_points(3, :, 1) = single((source(2) * reciecer(1) - source(1) * reciecer(2) + (idx_y + 0.5)...
                            * source(1) - (idx_y + 0.5) * reciecer(1)) / (source(2) - reciecer(2)));
temp_points(3, :, 2) = idx_y + 0.5;
temp_points(3, :, 3) = single((source(2) * reciecer(3) - source(3) * reciecer(2) + (idx_y + 0.5)...
                            * source(3) - (idx_y + 0.5) * reciecer(3)) / (source(2) - reciecer(2)));

temp_points(4, :, 1) = single((source(2) * reciecer(1) - source(1) * reciecer(2) + (idx_y - 0.5)...
                            * source(1) - (idx_y - 0.5) * reciecer(1)) / (source(2) - reciecer(2)));
temp_points(4, :, 2) = idx_y - 0.5;
temp_points(4, :, 3) = single((source(2) * reciecer(3) - source(3) * reciecer(2) + (idx_y - 0.5)...
                            * source(3) - (idx_y - 0.5) * reciecer(3)) / (source(2) - reciecer(2)));

temp_points(5, :, 1) = single((source(3) * reciecer(1) - source(1) * reciecer(3) + (idx_z + 0.5)...
                            * source(1) - (idx_z + 0.5) * reciecer(1)) / (source(3) - reciecer(3)));
temp_points(5, :, 2) = single((source(3) * reciecer(2) - source(2) * reciecer(3) + (idx_z + 0.5)...
                            * source(2) - (idx_z + 0.5) * reciecer(2)) / (source(3) - reciecer(3)));
temp_points(5, :, 3) = idx_z + 0.5;

temp_points(6, :, 1) = single((source(3) * reciecer(1) - source(1) * reciecer(3) + (idx_z - 0.5)...
                            * source(1) - (idx_z - 0.5) * reciecer(1)) / (source(3) - reciecer(3)));
temp_points(6, :, 2) = single((source(3) * reciecer(2) - source(2) * reciecer(3) + (idx_z - 0.5)...
                            * source(2) - (idx_z - 0.5) * reciecer(2)) / (source(3) - reciecer(3)));
temp_points(6, :, 3) = idx_z - 0.5;

temp_points(isinf(temp_points)) = 0;
temp_points(isnan(temp_points)) = 0;

p = zeros(2, size(idx_x, 2), 3) - 1;

for kk = 1:size(temp_points, 1)
    val_mask = squeeze(repmat((squeeze(temp_points(kk, :, 1)) <= idx_x + 0.5) .* (squeeze(temp_points(kk, :, 1)) >= idx_x - 0.5)...
             .* (squeeze(temp_points(kk, :, 2)) <= idx_y + 0.5) .* (squeeze(temp_points(kk, :, 2)) >= idx_y - 0.5)...
             .* (squeeze(temp_points(kk, :, 3)) <= idx_z + 0.5) .* (squeeze(temp_points(kk, :, 3)) >= idx_z - 0.5), [1, 1, 3]));
    
    temp_mask = (p == -1);
    temp = squeeze(p(1, :, :));
    temp = temp .* (~(val_mask .* squeeze(temp_mask(1, :, :)))) + squeeze(temp_points(kk, :, :)) .* (val_mask .* squeeze(temp_mask(1, :, :)));
    p(1, :, :) = temp;
    temp = squeeze(p(2, :, :));
    temp = temp .* (~(val_mask .* squeeze(temp_mask(2, :, :)))) + squeeze(temp_points(kk, :, :)) .* (val_mask .* squeeze(temp_mask(2, :, :)));
    temp(repmat(squeeze((temp(:, 1) == permute(squeeze(p(1, :, 1)), [2, 1]))...
        .* (temp(:, 2) == permute(squeeze(p(1, :, 2)), [2, 1]))...
        .* (temp(:, 3) == permute(squeeze(p(1, :, 3)), [2, 1]))), [1, 3]) == true) = -1;
    p(2, :, :) = temp;
end

temp = p(1, :, :);
temp(p(2, :, :) == -1) = -1;
p(1, :, :) = temp;
p(p == -1) = 0;
p_weight = sqrt((squeeze(p(1, :, 1)) - squeeze(p(2, :, 1))).^2 +...
                      (squeeze(p(1, :, 2)) - squeeze(p(2, :, 2))).^2 + ...
                      (squeeze(p(1, :, 3)) - squeeze(p(2, :, 3))).^2);

weight = zeros(size(geo.mask));
for ii = 1:size(idx_x, 2)
    weight(idx_x(ii), idx_y(ii), idx_z(ii)) = p_weight(ii);
end
weight = geo.mask .* weight;
end