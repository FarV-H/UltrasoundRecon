function [data, geo, Points] = CreateData(scan_mode, data_type)

geo.c = 3800;                                   % 波速      m/s
geo.f = 20e7;                                   % 采样频率  Hz
geo.T = 1/geo.f;                                % 间隔时间  s
geo.L =  1e-03 * geo.f;                         % 信号长度  L
geo.t = (0:geo.L - 1) * geo.T;                  % Time Vector 秒(0 ~ 0.1s)
geo.nx = int32(128);
geo.ny = int32(geo.nx);
geo.nz = int32(256);
geo.sx = 0.05;                                  % m
geo.sy = 0.05;                                  % m
geo.sz = 0.1;                                   % m
geo.dx = double(geo.sx) / double(geo.nx);       % m
geo.dy = double(geo.sy) / double(geo.ny);       % m
geo.dz = double(geo.sz) / double(geo.nz);       % m
geo.num_channels_x = int32(4);
geo.num_channels_y = int32(8);
geo.num_channels   = int32(geo.num_channels_x * geo.num_channels_y);
geo.num_panel      = int32(2);
geo.scan_mode      = scan_mode;
geo.data_type      = data_type;

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
                    data(ii, jj, kk) = 0;
                    d1 = abs(ii - 0.4 * jj - 0.1 * kk - 2) / sqrt(1^2 + 0.4^2 + 0.1^2);
                    if (d1 <= 0.0005)
                        data(ii, jj, kk) = 10;
                    end
%                     d2 = abs(ii * geo.dx  + 0.6 * jj - kk * geo.dz + 60000) / sqrt(1^2 + 0.6^2 + 1^2);
%                     if (d2 <= 250 && kk >= (130 - ii * geo.dx  + 0.47 * jj * geo.dy) / (0.88 * geo.dz))
%                         data(ii, jj, kk) = 10;
%                     end
                end
            end
        end
%         data(60, 100, 120) = 10;
%         data(9:11, 9:11) = 10;
%         data(19:21, 19:21) = 10;
%         data(29:31, 29:31) = 10;
%         data(39:41, 39:41) = 10;
%         data(49:51, 49:51) = 10;
%         data(59:61, 59:61) = 10;
%         data(69:71, 69:71) = 10;
%         data(79:81, 79:81) = 10;
%         data(89:91, 89:91) = 10;
%         data(99:101, 99:101) = 10;
%         data(109:111, 109:111) = 10;
%         data(119:121, 119:121) = 10;
        % 模拟探测器位置
        distance_1 = linspace(0.002, 0.048, geo.num_channels_x);
        distance_2 = linspace(0.002, 0.098, geo.num_channels_y);
        Points = SetPoint(distance_1, distance_2, geo);
        
    otherwise
        disp (['geo.data_type is error : ', geo.data_type])
end

end
