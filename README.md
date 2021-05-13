<p align="center">
<a href="http://usm.md/?lang=en"><img src="https://user-images.githubusercontent.com/68700549/118066006-eaf92080-b36b-11eb-9116-9f8e02a79534.png" align="center" height="100"></a>
</p>

<div align="center"> 
  
## New York Univeristy

 </div>
 
<div align="center"> 
  

# Interpretable  Machine  Learning  Approach  to  Human  Emotion Recognition  and  Visualization 

#### Project  Report  for  CS-GY  9223  Visualization  for  Machine  Learning


### Course  Instructor:  Dr.  Claudio  Silva

#### Vahan Babushkin, Binfang Ye

###### (code repository)

## ABSTRACT
</div>
<div align="justify"> 
Analyzing the biological signals our brain generates in response to external visual cues might shed the light on the elicited neurophysiological modifications in the nervous system that leads to the  emergence of biological states known as emotions. Currently there is no analytical  model that can fully describe  the processes in the human brain associated with emotions and can provide a reliable approach for reverse engineering of the biological signals into emotional states. However, several Machine Learning and Deep Learning technics are capable to infer emotions from biological signals with high accuracy. In comparison to analytical model, the trained ML model or network acts as a black box, i. e. it does not provide enough information about the hidden processes in human brain that are associated with the emotional state. The convolutional networks offer a  prostpective approach towards creating interpretable models for human emotion recogintion. The feature maps generated in the hidden layers might shed a light on the bilogical signal features that can be associated with positive or negative emotions. In this project we focus on interpretablility of the models for human emotion recognition from the electroencephalogram (EEG) recordings of neural activations elicited by visual stimuli. We conduct a visual analysis of the feature maps extracted by the hidden layers of two CNN models and conclude on frequency ranges that the model uses to differentiate between four emotion categories. Then we propose a modification of the currently existing model leading to the improvement of recognition accuracy. We also investigate the original data in order to eliminate outliers affecting the classification accuracy. Finally, we propose a pipeline for analysis of the EEG data for emotion classification.
</div>

<div align="center"> 
 
## INTRODUCTION
</div>
<div align="justify"> 
Emotions play an essential role in our everyday lives since they affect our social interactions, decision making, perception and cognition, performance, and intelligence and many other aspect of our lives. From the perspective of neurobiology, human emotions are associated with different biological states originated as a response to neurophysiological modifications in the neural system elicited by external stimuli, and are associated with thoughts, feelings, behavioral responses, and a degree of pleasure or displeasure [1]. From a psychological point of view the emotion consists of three components – arousal, affect and feeling [2], which can be measured from the biological data. The psychological arousal of emotion is usually associated with the modified biological signals such as heart rate, skin conductance, and pupil dilation [3]. The behavioral demonstration of emotion (affect) is related to facial expression, gesticulation or change in the voice modulation. Conscious experience (feeling) of the emotion can be detected by analyzing changes in brain signals from electroencephalography (EEG) records [4]. Also, feelings can be estimated through self-reporting such as the self-assessment manikin [5]. However, these estimates are relatively subjective.
 
The concepts of valence and arousal act as dimensions in models for human emotions categorization. The commonly used circumplex model is demonstrated on Figure 1, where emotions  are projected to the valence-arousal coordinates. The valence estimates the extent of the pleasantness of perceiving the stimulus and the arousal represents the degree of   awareness induced by the stimulus. In most of the affective computing studies the valence and arousal are as-signed high/low levels, thus distinguishing between positive valence  high arousal (PVHA), positive valence low arousal(PVLA), negative valence low arousal (NVLA) and negative valence high arousal (NVHA) [6] [7] [8]. In this project we follow the terminology in [3] where authors differentiate between Positive/Negative Valence (in most studies knownas High/Low Valence) and High/Low Arousal.
</div>
<p align="center">
<a href="http://usm.md/?lang=en"><img src="https://github.com/vbabushkin/2021CS-GY9223/blob/main/Figures/Emotion_class.png" align="center" height="300"></a>
</p>
<div align="center"> 
  
Fig.1: 4 Emotion classes: Positive Valence High Arousal (PVHA), Negative Valence High Arousal (NVHA), Negative Valence Low Arousal (NVLA), Positive Valence Low Arousal (PVLA).

</div>

<div align="justify"> 
In this project, we use the EEG data for human emotion recognition according to the classical circumplex model(see Fig. 1). The main focus of the project is in on the interpretability of the models for human emotion recognition. We visualize the hidden layers of the Convolutional NeuralNetworks model offered in [3] to study which features  are responsible for the particular emotion category. It allows to make a conclusion about frequency ranges that the modeluses to differentiate between four emotion categories.  We propose a modified architecture of CNN model that improves the recognition accuracy. A thorough analysis of original datadetermined presence of outliers in spectral power  density at the beginning of each trial, that affects the classification accuracy. These outliers were not reported in [3] and their removal increases the accuracy of the model.  Finally, we propose a pipeline for analysis of the EEG data for emotion classification.
</div>

<div align="center"> 
 
## RELATED-WORK
</div>

<div align="justify"> 
  
<!-- The multiple studies demonstrate that it is possible to achieve high accuracy for human emotions classification using Machine Learning. The popular Machine Learning  approaches for human emotion recognition from EEG data include K-Nearest Neighbor (KNN), Bayesian Network (BN), Artificial Neural Network (ANN), and Support Vector Machine (SVM). For instance, the SVM model adapted for a multi-class classification was used for recognition of four music-induced emotional responses from EEG recordings with  anaccuracy of 90.52%  using the one-against-one scheme and the accuracy of 82.37% with all-together scheme [11]. Other studies focusing on classifying the valence and arousal with SVM reported accuracies of 32% and 37% [12]. Usually the high accuracies are achieved while recognizing emotions with the SVM classifier from the offline data. However, for real-time emotion recognition, the classification accuracy is usually low. For example, in [13] the SVM classifier was used for online emotion recognition and achieved average accuracy of 70.5%. Therefore, when we classify the emotional EEG data, several factors should be considered since it will affect the accuracy of emotions recognition such as different experiment environments, preprocessing techniques, feature selection, and length of the dataset [9] [10].-->

The multivariate character of the EEG data pose several challenges for emotion classification including noise and low generalization due to high dimensionality. Some studies use the Independent Component Analysis (ICA) for decomposing the data into independent components [4], or Empirical Mode Decomposition and then Genetic Algorithms to extract important statistical features [14]. Focusing on specific frequency bands and features allows to significantly increase the accuracy of classifier, e.g. applying SVM to five  frequency features extracted from each channel of the EEG records lead to average accuracy of 55.72% for and 60.23% for classifying valence and arousal [15]. Also,  incorporation of different modalities into the model such as audio/visual features, extracted from video stimulus increases the accuracy to 58.16% for valence and 61.35% for arousal [15].
</div> 
