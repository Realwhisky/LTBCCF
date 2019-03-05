function x = get_CNfeatures(im, config, cos_window)  
%GET_FEATURES
%  从图像中提取密集特征

    cell_size=config.features.cell_size;     % 特征cell的大小 LCT设置为4*4大小
    nwindow=config.features.window_size;     % 特征窗的大小
    nbins=config.features.nbins;             % 特征bin方向的大小
    
    %HOG features, from Piotr's Toolbox
    x = double(fhog(single(im) / 255, cell_size, config.features.hog_orientations));
    x(:,:,end) = [];  %remove all-zeros channel ("truncation feature")移除所有零通道截断特征

    % pixel intensity histogram, from Piotr's Toolbox                % 像素峰值直方图
    h1=histcImWin(im,nbins,ones(nwindow,nwindow),'same');            % 
    h1=h1(cell_size:cell_size:end,cell_size:cell_size:end,:);        %

    % intensity ajusted hitorgram                                    % 像素值调整直方图
    
    im= 255-calcIIF(im,[cell_size,cell_size],32);
    h2=histcImWin(im,nbins,ones(nwindow,nwindow),'same');
    h2=h2(cell_size:cell_size:end,cell_size:cell_size:end,:);

    x=cat(3,x,h1,h2);                                                % 联结得到的 n*1维特征向量
        	
	%process with cosine window if needed                            % 用consine窗进行处理
	if ~isempty(cos_window),
		x = bsxfun(@times, x, cos_window);                           % 计算x与cos_window的点积
	end
	
end
