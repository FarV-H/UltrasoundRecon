function [Points] = SetPoint(distance, dx, dy)
Points = zeros(2, 32, 2);
for ii = 1:2
    for jj = 1:32
        if ii == 1
            Points(ii, jj, 1) = 1;
            Points(ii, jj, 2) = int32(distance(jj) / dy + 1);
        end
        if ii == 2
            Points(ii, jj, 1) = int32(sqrt(distance(jj)^2 / 2) / dx + 1);
            Points(ii, jj, 2) = int32(sqrt(distance(jj)^2 / 2) / dy + 1);
        end
    end
end
end
