### Introduction
This folder is for attention related works. 

### Works
1. [Attend, infer and Repeat][1]
	- ICML2016, Google
	- A framework for efficient inference in **structured image models** that explicitly **reason about objects**. **Probabilistic inference** with **rnn**. 

2. [RNN with attention for OCR in wild][2]
	- CVPR 2016
	- RNN with **attention model**, no need for lexicon.   

3. [Attentive Pooling][3] 
	- A two-way attention mechanism for discriminative model training. 
	- related work: [attention based CNN][4]

4. [Look, listen and learn][5]
	- AAAI 2016, from ShangTang technology. 
	- **Multi-modal speech recognition** by LSTM. 
	- [basic code][6], [updated code][7]
	- [data][8]

5. [Generate Headlines with RNN][9]

6. [Selective Search for word-spotting][10]
	- object proposals techniques emerge as an alternative to the traditional text detectors.
	- [code][11]

7. [Listen, attend and spell][12]
	- arxiv 2015
	- Composed of a listener and a speller
	- Listener: a pyramidal recurrent network encoder accepts filder bank spectra as inputs
	- Speller: an attention based recurrent netowrk decoder that emits characters as outputs. 


 8. [Listen, attend and walk][13]
	- arxiv 2015, 4 cited
	- translates natural language instructions to action sequences

9. [Playing atari][14]
	- NIPS2013, by Google Deepmind, 186 cited
	- deep learning + reinforcement learning 

10. [multiple object recognition][15]
	- Attention based RNN trained with reinforcement learning to attend the mose relevant regions of the input image.
	- ICLR 2015

11. [DRAW][16]
	- A recurrent network for image generation 
	- ICML2015, 100 cited, from Google Deepmind

12. [Spatial transform network][17]
	- NIPS2015, 49 cited

13. [Social LSTM][18]
	- From Li Fei Fei
	- We should pay attention to the work of [Alexandre Alahi][19]

14. [Show attend and tell][20]
	- Bengio, NIPS2015, 189 cited

[1]:	http://arxiv.org/abs/1603.08575
[2]:	http://arxiv.org/abs/1603.03101
[3]:	http://arxiv.org/abs/1602.03609
[4]:	http://arxiv.org/pdf/1512.05193.pdf
[5]:	http://arxiv.org/pdf/1602.04364v1.pdf
[6]:	https://github.com/jimmy-ren/lstm_speaker_naming_aaai16
[7]:	https://github.com/jimmy-ren/vLSTM
[8]:	https://github.com/jimmy-ren/lstm_speaker_naming_aaai16
[9]:	http://rsarxiv.github.io/2016/04/24/%E8%87%AA%E5%8A%A8%E6%96%87%E6%91%98%EF%BC%88%E4%BA%94%EF%BC%89/
[10]:	http://arxiv.org/abs/1604.02619
[11]:	https://github.com/lluisgomez/TextProposals
[12]:	https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/44926.pdf
[13]:	http://arxiv.org/abs/1506.04089
[14]:	http://arxiv.org/abs/1312.5602
[15]:	http://arxiv.org/abs/1412.7755
[16]:	http://arxiv.org/abs/1502.04623
[17]:	http://papers.nips.cc/paper/5854-spatial-transformer-networks
[18]:	http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf
[19]:	http://web.stanford.edu/~alahi/pub.htm
[20]:	http://arxiv.org/abs/1502.03044