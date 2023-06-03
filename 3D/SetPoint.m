function [Points] = SetPoint(distance_1, distance_2, geo)

switch geo.scan_mode
    case '2_tilt'
        Points = zeros(2, size(distance_1, 2) * size(distance_2, 2), 3);
        for ii = 1:size(Points, 1)
            for jj = 1:size(Points, 2)
                if ii == 1
                    Points(ii, jj, 1) = 1;
                    Points(ii, jj, 2) = int32(distance_1(fix((jj - 1) / size(distance_1, 2)) + 1) / geo.dy + 1);
                    Points(ii, jj, 3) = int32(distance_2(mod((jj - 1), size(distance, 2)) + 1) / geo.dz + 1);
                end
                if ii == 2
                    Points(ii, jj, 1) = int32(sqrt(distance_1(fix((jj - 1) / size(distance_1, 2)) + 1)^2 / 2) / geo.dx + 1);
                    Points(ii, jj, 2) = int32(sqrt(distance_1(fix((jj - 1) / size(distance_1, 2)) + 1)^2 / 2) / geo.dy + 1);
                    Points(ii, jj, 3) = int32(distance_2(mod((jj - 1), size(distance_2, 2)) + 1) / geo.dz + 1);
                end
            end
        end
    
    case '4_tilt'
        Points = zeros(4, size(distance_1, 2) * size(distance_2, 2), 3);
        for ii = 1:size(Points, 1)
            for jj = 1:size(Points, 2)
                if ii == 1
                    Points(ii, jj, 1) = 1;
                    Points(ii, jj, 2) = int32(distance_1(fix((jj - 1) / size(distance_1, 2)) + 1) / geo.dy + 1);
                    Points(ii, jj, 3) = int32(distance_2(mod((jj - 1), size(distance_2, 2)) + 1) / geo.dz + 1);
                end
                if ii == 2
                    Points(ii, jj, 1) = int32(sqrt(distance_1(fix((jj - 1) / size(distance_1, 2)) + 1)^2 / 2) / geo.dx + 1);
                    Points(ii, jj, 2) = int32(sqrt(distance_1(fix((jj - 1) / size(distance_1, 2)) + 1)^2 / 2) / geo.dy + 1);
                    Points(ii, jj, 3) = int32(distance_2(mod((jj - 1), size(distance_2, 2)) + 1) / geo.dz + 1);
                end
                if ii == 3
                    Points(ii, jj, 2) = 1;
                    Points(ii, jj, 1) = int32(distance_1(fix((jj - 1) / size(distance_1, 2)) + 1) / geo.dy + 1);
                    Points(ii, jj, 3) = int32(distance_2(mod((jj - 1), size(distance_2, 2)) + 1) / geo.dz + 1);
                end
                if ii == 4
                    Points(ii, jj, 1) = int32(sqrt(distance_1(fix((jj - 1) / size(distance_1, 2)) + 1)^2 / 2) / geo.dx + 1);
                    Points(ii, jj, 2) = int32(sqrt(distance_1(fix((jj - 1) / size(distance_1, 2)) + 1)^2 / 2) / geo.dy + 1);
                    Points(ii, jj, 3) = int32(distance_2(mod((jj - 1), size(distance_2, 2)) + 1) / geo.dz + 1);
                end
            end
        end
        
    case '2_parallel'
        Points = zeros(2, size(distance_1, 2) * size(distance_2, 2), 3);
        for ii = 1:size(Points, 1)
            for jj = 1:size(Points, 2)
                if ii == 1
                    Points(ii, jj, 1) = 0;
                    Points(ii, jj, 2) = distance_1(mod((jj - 1), size(distance_1, 2)) + 1);
                    Points(ii, jj, 3) = distance_2(fix((jj - 1) / size(distance_1, 2)) + 1);
                end
                if ii == 2
                    Points(ii, jj, 1) = geo.sx;
                    Points(ii, jj, 2) = distance_1(mod((jj - 1), size(distance_1, 2)) + 1) ;
                    Points(ii, jj, 3) = distance_2(fix((jj - 1) / size(distance_1, 2)) + 1);
                end
            end
        end
        
    case '4_parallel'
        Points = zeros(4, size(distance_1, 2) * size(distance_2, 2), 3);
        for ii = 1:size(Points, 1)
            for jj = 1:size(Points, 2)
                if ii == 1
                    Points(ii, jj, 1) = 1;
                    Points(ii, jj, 2) = int32(distance_1(fix((jj - 1) / size(distance_1, 2)) + 1) / geo.dy + 1);
                    Points(ii, jj, 3) = int32(distance_2(mod((jj - 1), size(distance_2, 2)) + 1) / geo.dz + 1);
                end
                if ii == 2
                    Points(ii, jj, 1) = geo.nx;
                    Points(ii, jj, 2) = int32(distance_1(fix((jj - 1) / size(distance_1, 2)) + 1) / geo.dy + 1);
                    Points(ii, jj, 3) = int32(distance_2(mod((jj - 1), size(distance_2, 2)) + 1) / geo.dz + 1);
                end
                if ii == 3
                    Points(ii, jj, 1) = int32(distance_1(fix((jj - 1) / size(distance_1, 2)) + 1) / geo.dy + 1);
                    Points(ii, jj, 2) = 1;
                    Points(ii, jj, 3) = int32(distance_2(mod((jj - 1), size(distance_2, 2)) + 1) / geo.dz + 1);
                end
                if ii == 4
                    Points(ii, jj, 1) = int32(distance_1(fix((jj - 1) / size(distance_1, 2)) + 1) / geo.dy + 1);
                    Points(ii, jj, 2) = geo.ny;
                    Points(ii, jj, 3) = int32(distance_2(mod((jj - 1), size(distance_2, 2)) + 1) / geo.dz + 1);
                end
            end
        end
    otherwise
        disp (['geo.scan_mode is error : ', geo.scan_mode])
end
end
