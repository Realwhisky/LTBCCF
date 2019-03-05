# LTBCCF

Visual object tracking -- Long-term object tracking method based on background constraints and convolutional features （LTBCCF）

        A long-term tracking algorithm can hardly work without the help of detecting model.
        Here is a brief introduction of my algorithm.
        It can be devided into two parts,one is tracking stage,the other is re-detecting stage :

## Tracking stage

![demo](https://github.com/Realwhisky/LTBCCF_algorithm/blob/master/utility/tracking%20%20stage.png)

From the beginning we get the interest region of one frame from the groundtruth,then extract feature like histogram of gradient（HOG），raw pixel，CN features，after getting those features ,we train correlation filter by using the mentioned methed before.When the new frame comes,we use filter detect the nearby area and find target position through response map.

During tracking stage,we also train a memory filter in the meantime,memory filter merely learns from core part of search window,we train a scale filter based this filter.And memory filter can also check whether the tracking was success or not.If failed,the tracker will active detector.


## Detecting stage

![demo](https://github.com/Realwhisky/LTBCCF_algorithm/blob/master/utility/detecting%20stage.png)

Our re-detecting model is based on a CNN model --- VGGNet，this model is pre-trained on ImageNet and Coco dataset.We use this model for feature extarcting.

When first frame comes, we use VGGNet to extract the 3rd,4th,5th convolution feature map and resize them into same size.After dimension reduction of those features,we start training mutiple correlation filters,when target is missing,the detector will be actived,and it will working in a bigger search window and find our target again.


    (Besides better anti-jamming performance, it is real-time with a average speed of 35.4FPS)


## Platform

    Intel i5-8300H
    8 GB DDR4
    Nvidia GeForce GTX 1060
    VS2013 + Mattlab2016a + Cuda7.5
    Piotr_toolbox + Vlfeat_toolbox + MatConvNet1.0-beta24 


## EFFECT  
(a comparison with other mainstream algorithms)

![demo](https://github.com/Realwhisky/LTBCCF_algorithm/blob/master/utility/demo_girl2.gif)


## Conclusion


A long-term target tracking method, based on background constraints and convolutional fea-tures(LTBCCF), is proposed to solve the target loss problem caused by background aliasing and oc-clusion in long-term object tracking.Firstly, the feature of input image is fused and dimensionally reduced to enhance the performance of target feature discrimination and reduce the complexity of feature computation. Secondly, background constraints are introduced into the filter training process, which makes the filter more focused on the target response to improve the anti-jamming ability. Finally, by setting memory filter and the Peak to Sidelobe Ratio detection, the tracker can judge whether the target is missing or not. If the target is lost, a convolutional features filter is introduced to re-detect the target. The experimental results on 50 video sequences of Visual Tracking Benchmark dataset show that the proposed algorithm achieves a total accuracy score of 92.1% and a total success rate of 63.6% in complex scenes such as background aliasing, fast motion and severe occlusion. It is superior to most existing tracking algorithms and has a long time robust tracking effect.


## References

[1] J. F. Henriques, R. Caseiro, P. Martins, J. Batista, "High-Speed Tracking with Kernelized Correlation Filters",TPAMI 2015.

[2] Martin Danelljan, Gustav Häger, Fahad Khan, Michael Felsberg. "Discriminative Scale Space Tracking",TPAMI 2017. 

[3] Ma C , Huang J B , Yang X , et al. "Hierarchical Convolutional Features for Visual Tracking",ICCV 2015.

[4] Y. Wu, J. Lim, M.-H. Yang, "Online Object Tracking: A Benchmark", CVPR 2013.Website: http://visual-tracking.net/

[5] P. Dollar, "Piotr's Image and Video Matlab Toolbox (PMT)". Website: http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html
