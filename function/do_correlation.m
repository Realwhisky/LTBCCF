function [ pos, max_response ] = do_correlation( im, pos, window_sz, cos_window, config, model)
% *******************************����λ���˲���Rc �� ��Ӧ�÷�***********************************
% if size(im,3) > 1, im = rgb2gray(im); end

cell_size=config.features.cell_size;                             % HOG cell size

patch = get_subwindow(im, pos, window_sz);                       % ѵ��������f--patch������get_subwindow.m���ɴ�С��ͬ��--ѵ��������
             
zf = fft2(get_features(patch,config,cos_window));	             % ��ȡ�ڶ�֡ Ԥ��λ�� �����ͼƬ����	���Ѿ�����cos������

%kzf = gaussian_correlation(zf, model.xf, config.kernel_sigma); % ��˹��K��z��x����xf��������patch������zf�Ǵ�Ԥ����������
                                                                 % config.kernel_sigma��run_tracker�ﶨ���=1
 
 kzf = linear_correlation(zf, model.xf);   % *******��������Ѹ�˹�˸ĳ������Ժ� ��Ϊ�˱��� ��ά�� ��˹�˷������ٵ� ƽ���ӿռ�Լ������*********   

                                                                 
response =fftshift(real(ifft2(model.alphaf .* kzf)));            % ��������Ӧ������ʵ��
                                                                 % Y = fftshift(X) ͨ������Ƶ�����ƶ����������ģ��������и���Ҷ�任 X
                                                                 % �����źŵ�Ƶ�ʷ���ʱ������Ƶ����ƽ�Ƶ����Ļ���а���
                                                      
max_response=max(response(:));                                   % �ҵ������Ӧ

[vert_delta, horiz_delta] = find(response == max_response, 1);   % �� response �в��ҵ�1�� response == max_response��Ԫ��
                                                                 % �õ������ĵ�X��Y����仯
                                                                                                                                                                                                           
pos = pos + cell_size * [vert_delta - floor(size(zf,1)/2)-1, horiz_delta - floor(size(zf,2)/2)-1]; 

                                                                 % λ�ø���
                                                      
end

