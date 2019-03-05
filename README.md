# My-LTBCCF-algorithm
Visual object tracking -- Long-term object tracking method based on background constraints and convolutional features （LTBCCF）

A long-term tracking algorithm can hardly do without the help of detecting model,Here is a brief introduction of my algorithm.It can be devided into two parts,one is tracking stage,the other is re-detecting stage :
TRACKING
![demo](https://github.com/Realwhisky/LTBCCF_algorithm/blob/master/utility/tracking%20%20stage.png)
DETECTING
![demo](https://github.com/Realwhisky/LTBCCF_algorithm/blob/master/utility/detecting%20stage.png)
EFFECT
![demo](https://github.com/Realwhisky/LTBCCF_algorithm/blob/master/utility/demo_girl2.gif)
(It is real-time with a average speed of 35.4FPS)

A long-term target tracking method, based on background constraints and convolutional fea-tures(LTBCCF), is proposed to solve the target loss problem caused by background aliasing and oc-clusion in long-term object tracking.Firstly, the feature of input image is fused and dimensionally reduced to enhance the performance of target feature discrimination and reduce the complexity of feature computation. Secondly, background constraints are introduced into the filter training process, which makes the filter more focused on the target response to improve the anti-jamming ability. Finally, by setting memory filter and the Peak to Sidelobe Ratio detection, the tracker can judge whether the target is missing or not. If the target is lost, a convolutional features filter is introduced to re-detect the target. The experimental results on 50 video sequences of Visual Tracking Benchmark dataset show that the proposed algorithm achieves a total accuracy score of 92.1% and a total success rate of 63.6% in complex scenes such as background aliasing, fast motion and severe occlusion. It is superior to most existing tracking algorithms and has a long time robust tracking effect.
