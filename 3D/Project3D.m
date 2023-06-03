function  [proj]= Project3D(source, reciver, data, geo)
% t = tic;
% proj = zeros(size(reciecer, 2), size(reciecer, 3));
% 
% for ii = 1:size(reciecer, 3)
%     proj_view = zeros(1, size(reciecer, 2));
%     
%     for jj = 1:size(reciecer, 2)
%         weight = CalcWeight(squeeze(source(:, ii)), squeeze(reciecer(:, jj, ii)), geo);
%         proj_view(jj) = sum(data .* weight, 'all');
%     end
%     proj(:, ii) = proj_view;
% end
% toc(t);
% data = ones(size(data));
proj(:, :, 1) = mex_project_3d(source(:, :, 1), reciver(:, :, 1), data, geo);
proj(:, :, 2) = mex_project_3d(source(:, :, 2), reciver(:, :, 2), data, geo);

end
