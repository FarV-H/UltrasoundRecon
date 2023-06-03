function [image] = BackProject3D(geo, proj, source, reciecer)

geo.interp = int32(1);

image = mex_backproject_3d(source, reciecer, proj, geo);

%% 插值
idx = find(image);
[x, y, z] = ind2sub(size(image), idx);
for i = 1:size(x)
    v(i) = double(image(x(i), y(i), z(i)));
end
[xq,yq,zq] = meshgrid(1:size(image, 1), 1:size(image, 2), 1:size(image, 3));
image = permute(griddata(x,y,z,v,xq,yq,zq, 'natural'), [2, 1, 3]);
clear n x y z v d xq yq zq

image(isnan(image)) = 0;
image(isinf(image)) = 0;

end