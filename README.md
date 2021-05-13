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
Emotions play an essential role in our everyday livessince they affect our social interactions, decision making, perception and cognition, performance, and intelligenceand many other aspect of our lives. From the perspective ofneurobiology, human emotions are associated with differentbiological states originated as a response to neurophysiological  modifications in the nervous system elicited by external stimuli, and are associated with thoughts, feelings, behavioral responses, and a degree of pleasure or displeasure [1]. From a psychological point of view the emotion consists of three components – arousal, affect and feeling [2], which can be measured from the biological data. The psychological arousal of emotion is usually associated with the modified biological signals such as heat rate, skin conductance, and pupil dilation [3]. The behavioral demonstration of emotion (affect) is related to facial expression, gesticulation or changein the voice modulation. Conscious experience (feeling) of the emotion can be detected by analyzing changes in brain signals from electroencephalography (EEG) records [4]. Also, feelings can be estimated through self-reporting such as the self-assessment manikin [5]. However,  these estimates are relatively subjective.
</div>
