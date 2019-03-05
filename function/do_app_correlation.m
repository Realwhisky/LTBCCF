function [ pos,responseapp, max_response ] = do_app_correlation( im_gray, pos, app_sz, config, model, projection_matrix, cos_window_app, currentScaleFactor)

           % if size(im,3) > 1, im = rgb2gray(im); end

           cell_size=config.features.cell_size;                             % HOG cell size
           y=config.y;
           interp_sz = size(y) * cell_size; 
           lambda = config.lambda; 

           [xt_npca, xt_pca] = get_move_subwindow_learn(im_gray, pos, app_sz, currentScaleFactor);            
            xt = feature_projection(xt_npca, xt_pca, projection_matrix, cos_window_app);
            xtf = fft2(xt);
            
            responsef = sum(model.hf_num .* xtf, 3) ./ (model.hf_den + lambda);
            responsef = resizeDFT2(responsef, interp_sz);
            responseapp = ifft2(responsef, 'symmetric');
            max_response=max(responseapp(:));  
            
            
            [row, col] = find(responseapp == max_response, 1);            
            disp_row = mod(row - 1 + floor((interp_sz(1)-1)/2), interp_sz(1)) - floor((interp_sz(1)-1)/2);
            disp_col = mod(col - 1 + floor((interp_sz(2)-1)/2), interp_sz(2)) - floor((interp_sz(2)-1)/2);
            translation_vec = round([disp_row, disp_col] * currentScaleFactor);
            pos = pos + translation_vec;                                                     

                                                                 % Î»ÖÃ¸üÐÂ
                                                      
end

