function [data, geo, Points] = CreateData(scan_mode, data_type)

geo.nx = int32(128);
geo.ny = int32(geo.nx);
geo.nz = int32(256);
geo.sx = 50.0;
geo.sy = 50.0;
geo.sz = 100.0;
geo.dx = double(geo.sx) / double(geo.nx);
geo.dy = double(geo.sy) / double(geo.ny);
geo.dz = double(geo.sz) / double(geo.nz);
geo.num_channels_x = int32(4);
geo.num_channels_y = int32(8);
geo.num_channels = int32(geo.num_channels_x * geo.num_channels_y);
geo.num_views = int32(geo.num_channels_x * geo.num_channels_y);
geo.num_panel = int32(2);
geo.scan_mode = scan_mode;
geo.data_type = data_type;

data = single(zeros(geo.nx, geo.ny, geo.nz));
geo.mask = true(geo.nx, geo.ny, geo.nz);

switch geo.data_type
    case 'Cylinder'         % 柱体
        for ii = 1:geo.nx
            for jj = 1:geo.ny
                for kk = 1:geo.nz
                    data(ii, jj, kk) = 1;
                    geo.mask(ii, jj, kk) = true;
                    if ((ii * geo.dx - 70)^2 + (jj * geo.dy - 370)^2 < 225 && kk * geo.dz < 150)
                        data(ii, jj, kk) = 2;
                    end
                    if ((ii * geo.dx - 250)^2 + (jj * geo.dy - 150)^2 < 900 && kk * geo.dz < 250)
                        data(ii, jj, kk) = 3;
                    end
                    if ((ii * geo.dx - 350)^2 + (jj * geo.dy - 300)^2 < 625 && kk * geo.dz < 350)
                        data(ii, jj, kk) = 4;
                    end
                end
            end
        end
        % 模拟探测器位置
        distance = linspace(1, 49, 4);
        Points = SetPoint(distance, distance, geo);
    
    case 'Section'          % 切面
        for ii = 1:geo.nx
            for jj = 1:geo.ny
                for kk = 1:geo.nz
                    data(ii, jj, kk) = 2;
                    d1 = abs(ii * geo.dx + 0.5 * jj * geo.dy - 0.4375 * kk * geo.dz - 40) / sqrt(1^2 + 0.5^2 + 0.4375^2);
                    if (d1 <= 0.5)
                        data(ii, jj, kk) = 4;
                    end
%                     d2 = abs(ii * geo.dx  + 0.6 * jj - kk * geo.dz + 60) / sqrt(1^2 + 0.6^2 + 1^2);
%                     if (d2 <= 0.25 && kk >= (130 - ii * geo.dx  + 0.47 * jj * geo.dy) / (0.88 * geo.dz))
%                         data(ii, jj, kk) = 8;
%                     end
                end
            end
        end
        % 模拟探测器位置
        distance_1 = linspace(2, 48, geo.num_channels_x);
        distance_2 = linspace(2, 98, geo.num_channels_y);
        Points = SetPoint(distance_1, distance_2, geo);
        
    otherwise
        disp (['geo.data_type is error : ', geo.data_type])
end

end
