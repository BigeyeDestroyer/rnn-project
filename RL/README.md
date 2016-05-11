### Introduction
This folder is for resources related to reinforcemnet learning. 

### Resources
- [github resources][1]
- [Q-learning][2]
- [DeepMind-RL][3]
- [tensorflow tutorial][4]

### Books and tutorials
- [Reinforcement Learning: An Introduction][5]
	- An classical intuitive intro to the field. 
	- [matlab code][6]

- [Reinforcement Learning and Dynamic Programming using Function Approximators][7]
	- A practical book that explains some state-of-the-art algorithms. 
	- 2010.

- [From Bandits to Monte-Carlo Tree Search: The Optimistic Principle Applied to Optimization and Planning][8]
	- Important **nonconvex optimization** methods 
	- Covering the interesting topics such as **Monte-Carlo Tree Search** and **Bandits**.
	- 2014.

### Papers to recover
- image caption with **hard attention**
	- [policy Gradient][9]

	- [recurrent models of visual attention][10]
		- NIPS2014, 96 cited
		- code: [Tensorflow][11]
		- realized in the folder  **visual-attention**
		- related technique papers: [approx gradient][12], [deep POMDP][13]

	- [multiple object recognition][14]: ICLR2015, 51 cited

	- [show attend and tell][15]
		-  ICML2015, 189 cited
		- [code][16]

- rnn with **additional memory**
	- [programmer interpreter][17]
		- Neural Programmer-Interpreter (NPI): A recurrent and compositional neural network that learns to represent and execute programs.
		- Best paper for ICLR 2016 from Google DeepMind.
		- code: [Tensorflow][18]

	- [Neural Turing Machine][19]
		- 2014, 111 cited, by Alex Graves
		- code: [theano][20], [Tensorflow][21] 

	- [Pointer Network][22]
		- NIPS2015, 20 cited
		- code: [theano][23], [Tensorflow][24]
		- Pay attention to this guy: [Oriol Vinyals][25]

	- [stackRNN][26]
		- NIPS2015, 24 cited
		- code: [C++][27], [python][28]

	- Another 4 highly-related papers
		- [RL turing machines][29]: 2015 arxiv, 25 cited
		- [learning simple algorithms][30]: 2015 arxiv, 3 cited
		- [neural random access][31]: 2015 arxiv, 5 cited
		- [neural programmer][32]: 2015 ICLR, 8 cited

[1]:	https://github.com/BigeyeDestroyer/deepRL/tree/resource
[2]:	http://mnemstudio.org/path-finding-q-learning-tutorial.htm
[3]:	http://www.infoq.com/cn/articles/atari-reinforcement-learning
[4]:	https://github.com/pkmital/tensorflow_tutorials
[5]:	http://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html
[6]:	http://waxworksmath.com/Authors/N_Z/Sutton/sutton.html
[7]:	https://orbi.ulg.ac.be/bitstream/2268/27963/1/book-FA-RL-DP.pdf
[8]:	https://hal.archives-ouvertes.fr/hal-00747575v5/document
[9]:	http://www.scholarpedia.org/article/Policy_gradient_methods
[10]:	http://arxiv.org/abs/1406.6247
[11]:	https://github.com/seann999/tensorflow_mnist_ram
[12]:	http://incompleteideas.net/sutton/williams-92.pdf
[13]:	http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/Wierstra_ICANN_2007_%5B0%5D.pdf
[14]:	http://arxiv.org/abs/1412.7755
[15]:	http://arxiv.org/abs/1502.03044
[16]:	https://github.com/kelvinxu/arctic-captions
[17]:	http://arxiv.org/pdf/1511.06279v4.pdf
[18]:	https://github.com/carpedm20/NPI-tensorflow
[19]:	http://arxiv.org/abs/1410.5401
[20]:	https://github.com/shawntan/neural-turing-machines
[21]:	https://github.com/carpedm20/NTM-tensorflow
[22]:	http://papers.nips.cc/paper/5866-pointer-networks
[23]:	https://github.com/vshallc/PtrNets
[24]:	https://github.com/ikostrikov/TensorFlow-Pointer-Networks
[25]:	https://scholar.google.com/citations?hl=zh-CN&user=NkzyCvUAAAAJ&view_op=list_works&sortby=pubdate
[26]:	http://papers.nips.cc/paper/5857-inferring-algorithmic-patterns-with-stack-augmented-recurrent-nets
[27]:	https://github.com/facebook/Stack-RNN
[28]:	https://github.com/DoctorTeeth/diffmem
[29]:	http://arxiv.org/abs/1505.00521
[30]:	http://arxiv.org/abs/1511.07275
[31]:	http://arxiv.org/abs/1511.06392
[32]:	http://arxiv.org/abs/1511.04834