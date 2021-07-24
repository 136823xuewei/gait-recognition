# Gait Recognition  
Aiming at detecting the pattern of human walking movement, gait recognition takes advantage of the time-serial data and can identify a person distantly. The time-serial data, which is usually presented in video form, always has a limitation in frame rate, which intrinsically affects the performance of the recognition models. In order to increase the frame rate of gait videos, we propose a new kind of generative adversarial networks(GAN) named Frame-GAN to reduce the gap between adjacent frames. Inspired by the recent advances in metric learning, we also propose a new effective loss function named Margin Ratio Loss(MRL) to boost the recognition molde.  

## Getting Started  
These instructions will give you a copy of the project up and running on your local machine for development and testing purposes.  

### Prerequisites 
* Gait databases: [CASIA-B](http://www.cbsr.ia.ac.cn/china/Gait%20Databases%20CH.asp) and [OU-ISIR](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitLP.html)  
* Linux 
* Nvidia k40 GPU and Nvidia P100 GPU
* Python3.6.5 and Tensorflow1.0  

## License 
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/136823xuewei/gait-recognition/blob/main/LICENSE) file for details
