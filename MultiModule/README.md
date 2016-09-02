### Introduction
This folder is for multi-module learning. Not only for proposal writing, but also for the project with Germany. 

### [ICML2016][1]
- [Learning from Multiway Data: Simple and Efficient Tensor Regression][2]

### [CVPR2016][3]
- [Learning Attributes Equals Multi-Source Domain Generalization][4]
	- 2 cited
- [MDL-CW: A Multimodal Deep Learning Framework With Cross Weights][5]
- [Discriminative Multi-Modal Feature Fusion for RGBD Indoor Scene Recognition][6]
- [Multimodal Spontaneous Emotion Corpus for Human Behavior Analysis][7]
- [Temporal Multimodal Learning in Audiovisual Speech Recognition][8]
- [Learning Multi-Domain Convolutional Neural Networks for Visual Tracking][9]
	- 16 cited

### Related Work
- [Multimodal deep learning][10]
	- ICML2011, 457 cited
- [Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding][11]
	- Jun 2016, Trevor Darrell
- [Cross-modality Consistent Regression for Joint Visual-Textual Sentiment Analysis of Social Multimedia][12]
- [Audiovisual fusion: Challenges and new approaches][13]
	- Overview, 4 cited, 2015
- [Methodologies for cross-domain data fusion: An overview][14]
	- Overview from MS, 19 cited, 2015

- [Improved multimodal deep learning with variation of information][15]
	- NIPS2014, 22 cited
- [Multimodal sparse representation learning and applications][16]
	- arxiv, Nov 2015,
- [Deep fragment embeddings for bidirectional image sentence mapping][17]
	- NIPS2014, 130 cited
	- Image Caption related
		- [Deep visual-semantic alignments for generating image descriptions][18], 416 cited
		- [Show and tell: A neural image caption generator][19], 442 cited
		- [Show, attend and tell: Neural image caption generation with visual attention][20], 306 cited

### Ideas
- About modality fusion: we can make use of **RL** to let the model itself determine **at which level the features from different modalities are fused**.
- 老板的考虑：
	- 融合图像与音频，相互之间的作用
	- 数据不足时从其他途径补充，用来增强模型的能力：比如我们要识别药品，但是可以从训练其他类型瓶子开始，捕捉到类似的特征，然后在小数据集上做类似于fine tune的工作











[1]:	http://jmlr.org/proceedings/papers/v48/
[2]:	http://jmlr.org/proceedings/papers/v48/yu16.pdf
[3]:	http://cvpr2016.thecvf.com/program/main_conference
[4]:	http://arxiv.org/abs/1605.00743
[5]:	http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Rastegar_MDL-CW_A_Multimodal_CVPR_2016_paper.html
[6]:	http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Zhu_Discriminative_Multi-Modal_Feature_CVPR_2016_paper.html
[7]:	http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Zhang_Multimodal_Spontaneous_Emotion_CVPR_2016_paper.html
[8]:	http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Hu_Temporal_Multimodal_Learning_CVPR_2016_paper.html
[9]:	http://arxiv.org/abs/1510.07945
[10]:	http://machinelearning.wustl.edu/mlpapers/paper_files/ICML2011Ngiam_399.pdf
[11]:	http://arxiv.org/abs/1606.01847
[12]:	https://www.semanticscholar.org/paper/Cross-modality-Consistent-Regression-for-Joint-You-Luo/0fc0a5a1480c833e0bf3ea15566c82454b6ae834/pdf
[13]:	http://ivpl.ece.northwestern.edu/sites/default/files/saraieee_proc.pdf
[14]:	https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Methods20for20Cross-Domain20Data20fusion.pdf
[15]:	http://papers.nips.cc/paper/5279-improved-multimodal-deep-learning-with-variation-of-information
[16]:	http://arxiv.org/abs/1511.06238
[17]:	http://papers.nips.cc/paper/5281-deep-fragment-embeddings-for-bidirectional-image-sentence-mapping
[18]:	http://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Karpathy_Deep_Visual-Semantic_Alignments_2015_CVPR_paper.html
[19]:	http://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Vinyals_Show_and_Tell_2015_CVPR_paper.html
[20]:	http://www.jmlr.org/proceedings/papers/v37/xuc15.pdf