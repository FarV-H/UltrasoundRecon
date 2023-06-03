function [Points] = SetPoint(distance_1, geo)

switch geo.scan_mode
    case '1_parallel'
        Points = zeros(size(distance_1, 2), 2);
        for jj = 1:size(distance_1, 2)
                Points(jj, 1) = 0;
                Points(jj, 2) = distance_1(mod((jj - 1), size(distance_1, 2)) + 1);
%                 Points(jj, 3) = distance_2(fix((jj - 1) / size(distance_1, 2)) + 1);
        end

    case '2_parallel'
        Points = zeros(2, size(distance_1, 2), 2);
        for jj = 1:size(distance_1, 2)
                Points(1, jj, 1) = 0;
                Points(1, jj, 2) = distance_1(mod((jj - 1), size(distance_1, 2)) + 1);
                Points(2, jj, 1) = geo.sx;
                Points(2, jj, 2) = distance_1(mod((jj - 1), size(distance_1, 2)) + 1) ;
        end

    otherwise
        disp (['geo.scan_mode is error : ', geo.scan_mode])
end
end
