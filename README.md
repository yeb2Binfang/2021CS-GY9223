<p align="center">
<a href="http://usm.md/?lang=en"><img src="https://user-images.githubusercontent.com/68700549/118066006-eaf92080-b36b-11eb-9116-9f8e02a79534.png" align="center" height="100"></a>
</p>

<div align="center"> 
  
## New York University

 </div>
 
<div align="center"> 
  

# Interpretable  Machine  Learning  Approach  to  Human  Emotion Recognition  and  Visualization 

#### Project  Report  for  CS-GY  9223  Visualization  for  Machine  Learning


### Course  Instructor:  Dr.  Claudio  Silva

#### Vahan Babushkin, Binfang Ye


#### Please check [project report](https://github.com/vbabushkin/2021CS-GY9223/blob/main/REPORT/CS_GY_9223_PROJECT_REPORT.pdf) for more details


<!--## ABSTRACT-->
</div>
<div align="justify"> 
<!--Analyzing the biological signals our brain generates in response to external visual cues might shed the light on the elicited neurophysiological modifications in the nervous system that leads to the  emergence of biological states known as emotions. Currently there is no analytical  model that can fully describe  the processes in the human brain associated with emotions and can provide a reliable approach for reverse engineering of the biological signals into emotional states. However, several Machine Learning and Deep Learning technics are capable to infer emotions from biological signals with high accuracy. In comparison to analytical model, the trained ML model or network acts as a black box, i. e. it does not provide enough information about the hidden processes in human brain that are associated with the emotional state. The convolutional networks offer a  prostpective approach towards creating interpretable models for human emotion recogintion. The feature maps generated in the hidden layers might shed a light on the bilogical signal features that can be associated with positive or negative emotions. In this project we focus on interpretablility of the models for human emotion recognition from the electroencephalogram (EEG) recordings of neural activations elicited by visual stimuli. We conduct a visual analysis of the feature maps extracted by the hidden layers of two CNN models and conclude on frequency ranges that the model uses to differentiate between four emotion categories. Then we propose a modification of the currently existing model leading to the improvement of recognition accuracy. We also investigate the original data in order to eliminate outliers affecting the classification accuracy. Finally, we propose a pipeline for analysis of the EEG data for emotion classification.-->
</div>

<div align="center"> 
 
## INTRODUCTION
</div>
<div align="justify"> 
  
<!--Emotions play an essential role in our everyday lives since they affect our social interactions, decision making, perception and cognition, performance, and intelligence and many other aspect of our lives. From the perspective of neurobiology, human emotions are associated with different biological states originated as a response to neurophysiological modifications in the neural system elicited by external stimuli, and are associated with thoughts, feelings, behavioral responses, and a degree of pleasure or displeasure [1]. From a psychological point of view the emotion consists of three components – arousal, affect and feeling [2], which can be measured from the biological data. The psychological arousal of emotion is usually associated with the modified biological signals such as heart rate, skin conductance, and pupil dilation [3]. The behavioral demonstration of emotion (affect) is related to facial expression, gesticulation or change in the voice modulation. Conscious experience (feeling) of the emotion can be detected by analyzing changes in brain signals from electroencephalography (EEG) records [4]. Also, feelings can be estimated through self-reporting such as the self-assessment manikin [5]. However, these estimates are relatively subjective.-->

<!--The concepts of valence and arousal act as dimensions in models for human emotions categorization. The commonly used circumplex model is demonstrated on Figure 1, where emotions  are projected to the valence-arousal coordinates. The valence estimates the extent of the pleasantness of perceiving the stimulus and the arousal represents the degree of   awareness induced by the stimulus. In most of the affective computing studies the valence and arousal are as-signed high/low levels, thus distinguishing between positive valence  high arousal (PVHA), positive valence low arousal(PVLA), negative valence low arousal (NVLA) and negative valence high arousal (NVHA) [6] [7] [8]. In this project we follow the terminology in [3] where authors differentiate between Positive/Negative Valence (in most studies knownas High/Low Valence) and High/Low Arousal.-->
</div>


<div align="justify"> 
  
Analyzing the biological signals our brain generates in response to external visual cues might shed the light on the elicited neurophysiological modifications in the nervous system that leads to the  emergence of biological states known as emotions. Currently there is no analytical  model that can fully describe  the processes in the human brain associated with emotions and can provide a reliable approach for reverse engineering of the biological signals into emotional states. However, several Machine Learning and Deep Learning technics are capable to infer emotions from biological signals with high accuracy. In comparison to analytical model, the trained ML model or network acts as a black box, i. e. it does not provide enough information about the hidden processes in human brain that are associated with the emotional state. The convolutional networks offer a  prostpective approach towards creating interpretable models for human emotion recogintion. The feature maps generated in the hidden layers might shed a light on the bilogical signal features that can be associated with positive or negative emotions. In this project, we use the EEG data for human emotion recognition according to the classical circumplex model(see Fig. 1). The main focus of the project is in on the interpretability of the models for human emotion recognition. The main work we have done is that we visualize the SVM boundary and the CNN model offered in [3]. For CNN, we visualize the hidden layers to study which features  are responsible for the particular emotion category. It allows to make a conclusion about frequency ranges that the modeluses to differentiate between four emotion categories. We propose a modified architecture of CNN model that improves the recognition accuracy after we conducted the visual analysis of old CNN model and the new CNN model. A thorough analysis of original datadetermined presence of outliers in spectral power density at the beginning of each trial, that affects the classification accuracy. These outliers were not reported in [3] and their removal increases the accuracy of the model. Finally, we propose a pipeline for analysis of the EEG data for emotion classification.
</div>


<p align="center">
<img src="https://github.com/vbabushkin/2021CS-GY9223/blob/main/FIGURES/Emotion_class.png" align="center" height="300"></p>
<div align="center"> 
  
Fig. 1: 4 Emotion classes: Positive Valence High Arousal (PVHA), Negative Valence High Arousal (NVHA), Negative Valence Low Arousal (NVLA), Positive Valence Low Arousal (PVLA).

</div>

<div align="justify"> 
  
<!-- The multiple studies demonstrate that it is possible to achieve high accuracy for human emotions classification using Machine Learning. The popular Machine Learning  approaches for human emotion recognition from EEG data include K-Nearest Neighbor (KNN), Bayesian Network (BN), Artificial Neural Network (ANN), and Support Vector Machine (SVM). For instance, the SVM model adapted for a multi-class classification was used for recognition of four music-induced emotional responses from EEG recordings with  anaccuracy of 90.52%  using the one-against-one scheme and the accuracy of 82.37% with all-together scheme [11]. Other studies focusing on classifying the valence and arousal with SVM reported accuracies of 32% and 37% [12]. Usually the high accuracies are achieved while recognizing emotions with the SVM classifier from the offline data. However, for real-time emotion recognition, the classification accuracy is usually low. For example, in [13] the SVM classifier was used for online emotion recognition and achieved average accuracy of 70.5%. Therefore, when we classify the emotional EEG data, several factors should be considered since it will affect the accuracy of emotions recognition such as different experiment environments, preprocessing techniques, feature selection, and length of the dataset [9] [10].-->

<!--The multivariate character of the EEG data pose several challenges for emotion classification including noise and low generalization due to high dimensionality. Some studies use the Independent Component Analysis (ICA) for decomposing the data into independent components [4], or Empirical Mode Decomposition and then Genetic Algorithms to extract important statistical features [14]. Focusing on specific frequency bands and features allows to significantly increase the accuracy of classifier, e.g. applying SVM to five  frequency features extracted from each channel of the EEG records lead to average accuracy of 55.72% for and 60.23% for classifying valence and arousal [15]. Also,  incorporation of different modalities into the model such as audio/visual features, extracted from video stimulus increases the accuracy to 58.16% for valence and 61.35% for arousal [15].-->

<!--The Convolutional Neural Networks (CNNs) gain popularity for emotion recognition from the EEG data in recent years due to their capability to simplify the data  preporcessing stage and avoid engineering new features – the CNNs can learn the hidden dependencies in raw data by progressively encoding the features from primitive to more complex ones in subsequent layers. This property of CNNs enables detection of the local trends and extract scale-invariant features for neighboring points, such as frequency variations  patterns in nearby electrodes. It allows to capture the existing relationship between emotional states and the EEG data. On the other hand, the drawbacks of CNNs is the need of  large arrays of data for training which can be compensated by aditional preporcassing of the input data, or adding another modalities.-->
</div> 

<div align="center"> 
 
## VISUAL DATA ANALYSIS
</div>

### Spectral data visualization

<div align="justify"> 
The averaged plots of time course of spectral densities for each of the bands are shown on Fig. 2. The first observation is that the magnitude of the spectral power densities remains relatively stable, except of the first 500 msec (10 timebins). This observation confirms that the emotion data is weakly depend on the time for short time intervals  (e.g. it takes minutes or even hours to calm down). We can also notice a significant increase in the spectral power density at the beginning on the trial, which can be explained  either by extensive firing of neurons on perceiving the new visual stimulus or, by the noise introduced with the trigger, whichis a more plausible explanation. Furhter analysis sugests thatthis perturbation in spectral power density does not containsignificant information about emotional states and its removalincreases the accuracy of the model.
</div>

<p align="center">
<img src="https://github.com/vbabushkin/2021CS-GY9223/blob/main/FIGURES/The%20%20spectral%20%20densities.png" align="center" height="200">
</p>
<div align="center"> 
  
Fig. 2: The spectral densities for four emotion categories averaged over trials and channels. Notice that the magnitude of the emotion data does not strictly depend on the time except of the first 10 timebins (∼500msec).

</div>

<div align="justify"> 
The plots of spectral power versus all frequencies aredemonstrated on Fig. 3. The spectral power increases with the frequency and demonstrates similar behavior for all fouremotion categories. The peak at 50Hz reflects the effect of applying the notch  filter. Interestingly, that the magnitude of the spectral power density for NVLA prevails for lower frequencies, particularly in Beta band (13-31 Hz), while the magnitudes for  NVHA and PVHA become more prevalentin Low Gamma (31Hz – 50Hz) and High Gamma (51Hz – 79Hz) bands.
</div>

<p align="center">
<a href="http://usm.md/?lang=en"><img src="https://github.com/vbabushkin/2021CS-GY9223/blob/main/FIGURES/Power%20spectral%20density%20averaged.png" align="center" height="300"></a>
</p>
<div align="center"> 
  
Fig. 3: Power spectral density averaged over 160 timepointsand 59 sensors.

</div>

### VISUALIZING LAYERS OF THE CNN NETWORK

<div align="justify"> 
For old and new CNN model, we have visualized the hidden layers, dense layers, and maximum activation plot. We have specific analysis about the graphs in the report to explain why the new model works better for the EEG data we collected. In the readme file, we will show the dense layers to compare the old CNN modela with new CNN model. 
</div>

#### Old CNN Model
<div align="justify"> 
 
Fig 4. shows the CNN old model (81% average accuracy). For CNN model, we visualized the hidden layers to learn which features are responsible for the particular emotion category.  
</div>
<p align="center">
<a href="http://usm.md/?lang=en"><img src="https://github.com/vbabushkin/2021CS-GY9223/blob/main/FIGURES/Old-model.png" align="center" height="150"></a>
</p>

<div align="center"> 
  
Fig. 4: CNN old model

</div>

#### New CNN Model
<div align="justify"> 
Fig 5. shows the CNN new model (85% average accuracy). 
</div>

<p align="center">
<a href="http://usm.md/?lang=en"><img src="https://github.com/vbabushkin/2021CS-GY9223/blob/main/FIGURES/New-model.png" align="center" height="150"></a>
</p>

<div align="center"> 
  
Fig. 5: CNN new model

</div>

#### Dense Layers of Old and New CNN model
<div align="justify"> 
  
Visualization with activation maximizationof the last layer shows that there is a clear difference betweenthe four classes, more pronounced for the new model. But in general,  the NVLA is characterized by activations in alpha(8-12 Hz) and beta (13Hz–30Hz) bands for channels from 25 to 45 while the NVHA can be described by the activations in lower  gamma and in high gamma for channels between 0 and 10 and 45 and 59 (see Fig. 6). For PVLA we have the activated regions in lower/higher gamma – more wider and blurred in old  model(from channels from 10 to 50)but sharper in the new model, where the lower gamma activations are observed in channels between 55 and 59 and higher gamma – for channels between 10 and 30. For PVHA the activation moslty happens in beta for channels from 10 to 25 and in high gamma for channels 50-59, but the old model does not capture the beta  band activations. The relative obscurity of visualized activations by an old model was another objective for reviewing the model’s archi-tecture. The removal of the maxpooling  layer in old model on Fig. 4.) results in more clear activation maximization visualizations for the last dense layer. Also, increasing the model width, i. e. the number of filters in each convolution layer from 32 to 64 leads to the imporvement of the classification accuracy.
</div>

<p align="center">
<a href="http://usm.md/?lang=en"><img src="https://github.com/vbabushkin/2021CS-GY9223/blob/main/FIGURES/dense_layer.png" align="center" height="300"></a>
</p>

<div align="center"> 
  
Fig. 6: Left four pictures show the visualization of dense layers with activation maximization for each class of the old CNN model. Right four pictures show the visualization of dense layers with activation maximization for each class of the new CNN model.

</div>

### VISUALIZING SVM MODEL
<div align="justify"> 
  
We were also interested in investigating how the SVM model, proposed in [3] is capable of discerning between 4 types of emotions. For this purpose, we visualized the decision  boundary(shown in Fig. 7) drawn by the SVM model with RBF function with γ= 0.1 for the 80 × 160 × 59 × 2671 tensor averaged across the time and three frequency bins, corresponding to beta,  lower gamma, and higher gammabands and then flattened to get a vector of length 177 for eachtrial. We can notice that despite of close proximity of samples from four emotion categories, the SVM model is capable of creating a very complex boundary.   
</div>

<p align="center">
<a href="http://usm.md/?lang=en"><img src="https://github.com/vbabushkin/2021CS-GY9223/blob/main/FIGURES/SVM.png" align="center" height="300"></a>
</p>
<div align="center"> 
  
Fig. 7: Visualization of decision boundaries drawn by SVMwith RBF (γ= 0.1) for first 5 principal components.

</div>

<div align="center"> 
 
## CONCLUSION
</div>

<div align="justify"> 
  

The main focus of this project is to interpret the emotion classification in CNN model and to offer an improved archi-tecture. We visualized feature maps extracted by the old CNN model and noticing that the max pooling layer decreases the"resolution" of the final feature map we decided to remove it from the architecture. In the meantime, the feature  maps of the convolution layers show that the filters learn patterns of the activations of sensors for different frequencies. Some filters consider activations of single sensors,  others groups of sensors. To capture the variety of activation patterns we decided to increase the number of filters (and it is also recommended for increasing the prediction accuracy of the model). The final convolution layer of a new model(Fig. 5)) is capable of better discerning between four emotioncategories, which results in the increase of  the accuracy of the model. Also, the activation maximization of the dense layer of the new model(Fig.  6), while showing similar patterns as of the old one (Fig. 6) provides more clear idea about which frequency ranges are responsible for eliciting different types of emotions. Thus, we conclude from Fig. 6 that the NVLA is characterized by activations  in alpha and beta bands, while NVHA and PVHA can be described by the activations in lower gamma and in high gamma bands. Notice that we cannot make similar conclusions about activation of brain areas (under correponding sensors) due to the poor spatial resolutions of EEG.
</div>

<div align="center"> 

## LIBRARIES USED

</div>

<div align="justify"> 

Jorge Piazentin Ono, Sonia Castelo, Roque Lopez, Enrico Bertini, Juliana Freire, Claudio Silva, "PipelineProfiler : A Visual Analytics Tool for the Exploration of AutoML Pipelines." IEEE Transactions on Visualization and Computer Graphics. 2021 ; Vol. 27, No. 2. pp. 390-400., https://pypi.org/project/pipelineprofiler/

Matthias Feurer, Katharina Eggensperger, Stefan Falkner, Marius Lindauer and Frank Hutter,"Auto-Sklearn 2.0: The Next Generation," arXiv:2007.04074 [cs.LG], 2020 https://arxiv.org/abs/2007.04074, github: https://github.com/automl/auto-sklearn

Hendrik Jacob van Veen, Nathaniel Saul, David Eargle, & Sam W. Mangham. (2019, October 14). Kepler Mapper: A flexible Python implementation of the Mapper algorithm (Version 1.4.1). Zenodo. http://doi.org/10.5281/zenodo.4077395, https://pypi.org/project/kmapper/

F. Chollet, "Keras," 2015, https://keras.io

Raghavendra Kotikalapudi et al., "keras-vis", 2017, https://github.com/raghakot/keras-vis

Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean,
Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, Manjunath Kudlur,
Josh Levenberg, Rajat Monga, Sherry Moore, Derek G. Murray, Benoit Steiner, Paul Tucker,
Vijay Vasudevan, Pete Warden, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng, Google Brain, "Tensorflow: A system for large-scale machine learning", 2016, 
In 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16) (pp. 265–283)

F. Pedregosa, V. Gael, G. Alexandre, M. Vincent, T. Bertrand, G. Olivier, M. Blondel, P. Prettenhofer, W. Ron, D. Vincent, V. Jake, P. Alexandre, C. David and B. Matthieu, "Scikit-learn: Machine learning in Python.," Journal of Machine Learning Research, p. 2825–2830, 2011. 


</div>

<div align="center"> 

## REFERENCES

</div>

<div align="justify"> 

[1] J. Panksepp, “Affective neuroscience the foundations of human and animal emotions,” New York, 2004.

[2] L. F. Barrett, B. Mesquita, K. N. Ochsner, and J. J. Gross,  “The experience  of  emotion,” Annu. Rev. Psychol.,  vol.  58,  pp.  373–403,2007.

[3]  V.  Babushkin,  W.  Park,  M.   Hassan  Jamil,  H.  Alsuradi,  and  M.  Eid, “EEG-based classification of the intensity of emotional responses,” in 10th International IEEE EMBS Conference on Neural Engineering,2021.

[4]  A. Bhardwaj, A. Gupta, P. Jain, A. Rani, and J. Yadav, “Classification of  human  emotions  from  EEG  signals  using  SVM  and  LDA  classifiers,” in 2015 2nd International Conference on Signal Processing andIntegrated Networks (SPIN), 2015, pp. 180–185.

[5]  M.   M.   Bradley   and   P.   J.   Lang,   “Measuring   emotion:   the   self-assessment manikin and the semantic differential,”Journal of behavior therapy and experimental psychiatry, vol. 25, no. 1, pp. 49–59, 1994.

[6]  N.  Dar,  M.  Akram,  S.  Khawaja,  and  A.  Pujari,  “CNN  and  LSTM-based emotion charting using physiological signals,” Sensors, vol. 20,p. 4551, 08 2020.

[7]  Y.  Zhao,  X.  Cao,  J.  Lin,  D.  Yu,  and  X.  Cao,  “Multimodal  emotion recognition model using physiological signals,” 2019.

[8]  S. Siddharth, T.-P. Jung, and T. J. Sejnowski, “Utilizing deep learning towards  multimodal  bio-sensing  and  vision-based  affective  computing,” 2019.

[9]  A.  T.  Sohaib,  S.  Qureshi,  J.  Hagelbäck,  O.  Hilborn,  and  P Jerčić, “Evaluating classifiers for emotion recognition using EEG,” in Foundations of Augmented Cognition, D. D. Schmorrow and C. M. Fidopiastis, Eds. Berlin, Heidelberg: Springer Berlin Heidelberg, 2013, pp.492–501.

[10]  P.  Rani,  C.  Liu,  N.  Sarkar,  and  E.  Vanman,  “An  empirical  study  ofmachine  learning  techniques  for  affect  recognition  in  human–robot interaction, ”Pattern Analysis and Applications, vol. 9, 05 2006.

[11]  Y.-P. Lin, C.-H. Wang, T.-L. Wu, S.-K. Jeng, and J. Chen, “EEG-based emotion recognition in music listening: A comparison of schemes for multiclass support vector machine,” 04 2009, pp. 489–492.

[12]  R. Horlings, D. Datcu, and L. Rothkrantz, “Emotion recognition using brain activity,” 01 2008, p. 6.

[13]  V. Anh, M. Van, B. Ha Bang, and T. Huynh Quyet, “A real-time model based support vector machine for emotion recognition through EEG,” 11 2012, pp. 191–196.

[14]  P.  Petrantonakis  and  L.  Hadjileontiadis,  “Emotion  recognition  frombrain signals using hybrid adaptive filtering and higher order crossings analysis,” Affective Computing, IEEE Transactions on, vol. 1, pp. 81–97, 07 2010.

[15]  Y.  Zhu,  S.  Wang,  and  Q.  Ji,  “Emotion  recognition  from  users’  EEG signals with the help of stimulus videos,” vol. 2014, 07 2014, pp. 1–6.

[16]  H. Mei and X. Xu, “EEG-based emotion classification using convolutional neural network,” in 2017 International Conference on Security, Pattern Analysis, and Cybernetics (SPAC), 2017, pp. 130–135.

[17]  R.  Alhalaseh  and  S.  Alasasfeh,  “Machine-learning-based  emotionrecognition system using EEG signals,”Computers, vol. 9, no. 4, 2020.

[18]  Y.-H. Kwon, S.-B. Shin, and S. Kim, “Electroencephalography based fusion two-dimensional   (2D)-convolution   neural   networks   (CNN) model for emotion recognition system,” Sensors (Basel, Switzerland), vol. 18, 2018.

[19]  S. Koelstra, C. Muhl, M. Soleymani, J. Lee, A. Yazdani, T. Ebrahimi, T.  Pun,  A.  Nijholt,  and  I.  Patras,  “Deap:  A  database  for  emotionanalysis ;using physiological signals,” IEEE Transactions on Affective Computing, vol. 3, no. 1, pp. 18–31, 2012.

[20]  J.  Liu,  G.  Wu,  Y.  Luo,  S.  Qiu,  S.  Yang,  W.  Li,  and  Y.  Bi,  “EEG-based  emotion  classification  using  a  deep  neural  network  and  sparse autoencoder,” Frontiers in Systems Neuroscience, vol. 14, p. 43, 2020.

[21]  N. Liu, Y. Fang, L. Li, L. Hou, F. Yang, and Y. Guo, “Multiple feature fusion for automatic emotion recognition using EEG signals,” 04 2018,pp. 896–900.

[22]  L.  van  der  Maaten  and  G.  Hinton,  “Viualizing  data  using  t-SNE,” Journal of Machine Learning Research,  vol.  9,  pp.  2579–2605,  112008.

[23]  A. Bilal, A. Jourabloo, M. Ye, X. Liu, and L. Ren, “Do convolutional neural networks learn class hierarchy?” IEEE Transactions on Visual-ization and Computer Graphics, vol. 24, no. 1, pp. 152–162, 2018.

[24]  Z.  Gao,  X.  Cui,  W.  Wan,  and  Z.  Gu,  “Recognition  of  emotional states  using  multiscale  information  analysis  of  high  frequency  EEG oscillations,” Entropy,   vol.   21,   no.   6,   2019.   [Online].   Available: https://www.mdpi.com/1099-4300/21/6/609

</div>
