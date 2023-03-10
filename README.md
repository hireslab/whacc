![](./pictures/whacc-logo-v1.png) <br />

WhACC is a tool for automated touched image classification. 

Many neuroscience labs (e.g. [Hires Lab](https://www.hireslab.org/)) use tasks that involve whisker active touch against thin movable poles to study diverse questions of sensory and motor coding. Since neurons operate at temporal resolutions of milliseconds, determining precise whisker contact periods is essential. Yet, accurately classifying the precise moment of touch is time-consuming and labor intensive. 

## [Walkthrough: Google CoLab](https://colab.research.google.com/drive/1HqkzE-Wih89DKwrOWplp58UrbNMP1KPS?usp=sharing)

![](./pictures/trial_animation.gif) <br />
*Single example trial lasting 4 seconds. Example video (left) along with whisker traces, decomposed components, and spikes recorded from L5 (right). How do we identify the precise millisecond frame when touch occurs?*

![](./pictures/ResNetV2_2048_features_clustered.gif) <br />
*Original 2048 output features extracted from the penultimate layer of the initial ResNet50 V2 model, clustered for emphasize*


## Flow diagram of WhACC video pre-processing and design implementation

![](./pictures/WhACC_figure_1.png) <br />

## Touch frame scoring and variation in human curation

![](./pictures/WhACC_figure_2.png) <br />

## Data selection and model performance

![](./pictures/WhACC_figure_3.png) <br />

## Feature engineering and selection

![](./pictures/WhACC_figure_5.png) <br />

## WhACC shows expert human level performance

![](./pictures/WhACC_figure_4.png) <br />

## WhACC can be retrained on a small subset to account for data drift over time or different datasets (see GUI below)

![](./pictures/WhACC_figure_6.png) <br />

## WhACC GUI: used to curate automatically selected subset of data for optimal performance
![](./pictures/WhACC_GUI_Curator.png) <br />

## Use left and right arrows to move through images, use up to label as touch (green) and down to label as not-touch (red)
![](./pictures/curation_GUI.gif) <br />


## Code contributors:
WhACC code and software was originally developed by Phillip Maire and Jonathan Cheung in the laboratory of [Samuel Andrew Hires](https://www.hireslab.org/). 
