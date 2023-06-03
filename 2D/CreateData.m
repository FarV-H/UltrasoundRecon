function [mask, data, nx, ny, dx, dy] = CreateData()
nx = 512;
ny = nx;

dx = 500 / nx;
dy = 500 / ny;

data = zeros(nx, ny);
mask = zeros(nx, ny);

for ii = 1:nx
    for jj = 1:ny
        if (ii < jj)
            data(ii, jj) = data(ii, jj) + 1;
            mask(ii, jj) = 1;
        end
        if ((ii - 80)^2 + (jj - 180)^2 <= 2500)
            data(ii, jj) = data(ii, jj) + 0.5 - sqrt((ii - 80)^2 + (jj - 180)^2) / 100;
        end
    end
end

end
