### Introduction
This folder is specially for the memory network. 

### Related papers:
- [Memory Networks][1]
	- ICLR2015
- [End-To-End Memory Networks][2]
	- NIPS2015, 78 cited
- [Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks][3]
	- 108 cited, 2.19 2015 arxiv
- [Large-scale Simple Question Answering with Memory Networks][4] 
	- 28 cited, 6.5 2015 arxiv
- [Evaluating Prerequisite Qualities for Learning End-to-End Dialog Systems][5] 
	- 7 cited, 11.22 2015 arxiv
- [The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations][6] 
	- 19 cited, 11.7 2015 arxiv
- [Dialog-based Language Learning][7] 
	- 4.20 2016 arxiv
- [Learning End-to-End Goal-Oriented Dialog][8]
	- 2 cited, 5.24 2016 arxiv
- [Stacked attention networks for image question answering][9]
	- 26 cited, 2015 arxiv
	- [python][10]

### [ICML2016][11]
- [Learning to Generate with Memory][12]
- [Ask Me Anything: Dynamic Memory Networks for Natural Language Processing][13]
	- 71 cited
- [Meta-Learning with Memory-Augmented Neural Networks][14]
	- 3 cited
- [Associative Long Short-Term Memory][15]
	- 4 cited
- [Recurrent Orthogonal Networks and Long-Memory Tasks][16]
- [Dynamic Memory Networks for Visual and Textual Question Answering][17]
	- 17 cited
- [Control of Memory, Active Perception, and Action in Minecraft][18]
	- 2 cited

### Works to recover
1. Dynamic Memory Network
	- python [code][19], 32 star, 9 folk
	- python [code][20], 154 star, 43 folk

### Works recovered
1. End-to-end
	- [tensorflow][21]
	- [python][22]
	- on **bigeye7**

### Faculty
- [Marco Zorzi][23]
	- [Learning Orthographic Structure With Sequential Generative Neural Networks][24]
	- **Pay attention** to this project: **GENMOD**
	- [EuroScience Open Forum 2016][25]: July 26

### Ideas
- 2016.8.21
	- About MemN2N. Actually the attention used here is on **sentence level**, what if we detail it into **phrase level**? Such as the recently proposed **structure attention work**.
- 2016.8.30
	- About the attention model: what about a mask directly connect the question and the final answer. For example, consider the following two questions:
		- **What** kind of tree is it?
		- **Where** is the tree? 
		- Thus, given **where**, we should select those position-related words; given **what**, we should select those class-related words.










[1]:	http://arxiv.org/abs/1410.3916
[2]:	http://papers.nips.cc/paper/5846-end-to-end-memory-networks
[3]:	http://arxiv.org/abs/1502.05698
[4]:	http://arxiv.org/abs/1506.02075
[5]:	http://arxiv.org/abs/1511.06931
[6]:	http://arxiv.org/abs/1511.02301
[7]:	http://arxiv.org/abs/1604.06045
[8]:	http://arxiv.org/abs/1605.07683
[9]:	http://arxiv.org/abs/1511.02274
[10]:	https://github.com/zcyang/imageqa-san
[11]:	http://jmlr.org/proceedings/papers/v48/
[12]:	http://arxiv.org/abs/1602.07416
[13]:	http://arxiv.org/abs/1506.07285
[14]:	http://jmlr.org/proceedings/papers/v48/santoro16.pdf
[15]:	http://arxiv.org/abs/1602.03032
[16]:	http://jmlr.org/proceedings/papers/v48/henaff16.pdf
[17]:	http://arxiv.org/abs/1603.01417
[18]:	http://arxiv.org/abs/1605.09128
[19]:	https://github.com/swstarlab/DynamicMemoryNetworks
[20]:	https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano
[21]:	https://github.com/carpedm20/MemN2N-tensorflow
[22]:	https://github.com/vinhkhuc/MemN2N-babi-python
[23]:	https://scholar.google.com/citations?hl=zh-CN&user=MgF3uIMAAAAJ&view_op=list_works&sortby=pubdate
[24]:	http://s3.amazonaws.com/academia.edu.documents/42328867/Learning_Orthographic_Structure_With_Seq20160207-26129-1rlm2pc.pdf?AWSAccessKeyId=AKIAJ56TQJRTWSMTNPEA&Expires=1468932324&Signature=daiAMmhswxs4FH6URXaOBLqBR5g%3D&response-content-disposition=inline%3B%20filename%3DLearning_Orthographic_Structure_With_Seq.pdf
[25]:	http://www.esof.eu/