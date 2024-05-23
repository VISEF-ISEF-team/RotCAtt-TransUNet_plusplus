# RotCAtt-TransUNet++
Cardiovascular disease remains a predominant global health concern, responsible for a significant portion of mortality worldwide. Accurate segmentation of cardiac medical imaging data is pivotal in mitigating fatality rates associated with cardiovascular conditions. However, existing state-of-the-art (SOTA) neural networks, including both CNN-based and Transformer-based approaches, exhibit limitations in practical applicability due to their inability to effectively capture inter-slice connections alongside intra-slice information. This deficiency is particularly pronounced in datasets featuring intricate, long-range details along the z-axis, such as coronary arteries in axial views. To address these challenges, we present RotCAtt-TransUNet++, a novel architecture tailored for robust segmentation of complex cardiac structures. Our approach incorporates several key innovations, including redesigned feature extraction with dense downsampling, integration of transformer layers with a specialized rotatory attention mechanism, and augmentation with channel-wise attention gates within the decoder. By synergistically leveraging these advancements, our model adeptly captures both inter-slice and intra-slice information, facilitating accurate segmentation of sophisticated cardiac structures. Experimental results demonstrate the superior performance of our proposed model compared to existing SOTA approaches in this domain. Through meticulous design and integration of advanced techniques, RotCAtt-TransUNet++ stands poised to significantly enhance the efficacy of cardiac medical imaging segmentation, offering promising avenues for improved diagnosis and treatment of cardiovascular diseases.

[Read paper: RotCAtt-TransUNet++: Sophisticated Cardiac Segmentation without Ignoring Minuscule Details](RotCAtt_TransUNet_plusplus.pdf)


## Model Architecture
![img](imgs/RotTrans%20Architecture.png)


|  |  |
|----------|----------|
**Rotatory attention mechanism** | **Channel-wise attention gate** |
|![img](imgs/Rotatory%20Attention%20Mechanism.png) | ![img](imgs/Channel_wise%20Attention.png) |

