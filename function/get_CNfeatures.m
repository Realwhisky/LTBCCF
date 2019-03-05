function x = get_CNfeatures(im, config, cos_window)  
%GET_FEATURES
%  ��ͼ������ȡ�ܼ�����

    cell_size=config.features.cell_size;     % ����cell�Ĵ�С LCT����Ϊ4*4��С
    nwindow=config.features.window_size;     % �������Ĵ�С
    nbins=config.features.nbins;             % ����bin����Ĵ�С
    
    %HOG features, from Piotr's Toolbox
    x = double(fhog(single(im) / 255, cell_size, config.features.hog_orientations));
    x(:,:,end) = [];  %remove all-zeros channel ("truncation feature")�Ƴ�������ͨ���ض�����

    % pixel intensity histogram, from Piotr's Toolbox                % ���ط�ֱֵ��ͼ
    h1=histcImWin(im,nbins,ones(nwindow,nwindow),'same');            % 
    h1=h1(cell_size:cell_size:end,cell_size:cell_size:end,:);        %

    % intensity ajusted hitorgram                                    % ����ֵ����ֱ��ͼ
    
    im= 255-calcIIF(im,[cell_size,cell_size],32);
    h2=histcImWin(im,nbins,ones(nwindow,nwindow),'same');
    h2=h2(cell_size:cell_size:end,cell_size:cell_size:end,:);

    x=cat(3,x,h1,h2);                                                % ����õ��� n*1ά��������
        	
	%process with cosine window if needed                            % ��consine�����д���
	if ~isempty(cos_window),
		x = bsxfun(@times, x, cos_window);                           % ����x��cos_window�ĵ��
	end
	
end
