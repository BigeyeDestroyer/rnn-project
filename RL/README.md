### Introduction
This folder is for resources related to reinforcemnet learning. 

### Resources
- [github resources][1]
- [Q-learning][2]
- [DeepMind-RL][3]

### Books and tutorials
- [Reinforcement Learning: An Introduction][4]
	- An classical intuitive intro to the field. 
	- [matlab code][5]

- [Reinforcement Learning and Dynamic Programming using Function Approximators][6]
	- A practical book that explains some state-of-the-art algorithms. 
	- 2010.

- [From Bandits to Monte-Carlo Tree Search: The Optimistic Principle Applied to Optimization and Planning][7]
	- Important **nonconvex optimization** methods 
	- Covering the interesting topics such as **Monte-Carlo Tree Search** and **Bandits**.
	- 2014.

### Papers to recover
- image caption with **hard attention**
	- [policy Gradient][8]

	- [recurrent models of visual attention][9]
		- NIPS2014, 96 cited
		- code: [Tensorflow][10]
		- realized in the folder  **visual-attention**
		- related technique papers: [approx gradient][11], [deep POMDP][12]

	- [multiple object recognition][13]: ICLR2015, 51 cited 

	- [show attend and tell][14]
		-  ICML2015, 189 cited
		- [code][15]

- rnn with **additional memory**
	- [programmer interpreter][16]
		- Neural Programmer-Interpreter (NPI): A recurrent and compositional neural network that learns to represent and execute programs.
		- Best paper for ICLR 2016 from Google DeepMind.
		- code: [Tensorflow][17]

	- [Neural Turing Machine][18]
		- 2014, 111 cited, by Alex Graves
		- code: [theano][19], [Tensorflow][20] 

	- [Pointer Network][21]
		- NIPS2015, 20 cited
		- code: [theano][22], [Tensorflow][23]
		- Pay attention to this guy: [Oriol Vinyals][24]

	- [stackRNN][25]
		- NIPS2015, 24 cited
		- code: [C++][26], [python][27]

	- Another 4 highly-related papers
		- [RL turing machines][28]: 2015 arxiv, 25 cited
		- [learning simple algorithms][29]: 2015 arxiv, 3 cited
		- [neural random access][30]: 2015 arxiv, 5 cited
		- [neural programmer][31]: 2015 ICLR, 8 cited

[1]:	https://github.com/BigeyeDestroyer/deepRL/tree/resource
[2]:	http://mnemstudio.org/path-finding-q-learning-tutorial.htm
[3]:	http://www.infoq.com/cn/articles/atari-reinforcement-learning
[4]:	http://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html
[5]:	http://waxworksmath.com/Authors/N_Z/Sutton/sutton.html
[6]:	https://orbi.ulg.ac.be/bitstream/2268/27963/1/book-FA-RL-DP.pdf
[7]:	https://hal.archives-ouvertes.fr/hal-00747575v5/document
[8]:	http://www.scholarpedia.org/article/Policy_gradient_methods
[9]:	http://arxiv.org/abs/1406.6247
[10]:	https://github.com/seann999/tensorflow_mnist_ram
[11]:	http://incompleteideas.net/sutton/williams-92.pdf
[12]:	http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/Wierstra_ICANN_2007_%5B0%5D.pdf
[13]:	http://arxiv.org/abs/1412.7755
[14]:	http://arxiv.org/abs/1502.03044
[15]:	https://github.com/kelvinxu/arctic-captions
[16]:	http://arxiv.org/pdf/1511.06279v4.pdf
[17]:	https://github.com/carpedm20/NPI-tensorflow
[18]:	http://arxiv.org/abs/1410.5401
[19]:	https://github.com/shawntan/neural-turing-machines
[20]:	https://github.com/carpedm20/NTM-tensorflow
[21]:	http://papers.nips.cc/paper/5866-pointer-networks
[22]:	https://github.com/vshallc/PtrNets
[23]:	https://github.com/ikostrikov/TensorFlow-Pointer-Networks
[24]:	https://scholar.google.com/citations?hl=zh-CN&user=NkzyCvUAAAAJ&view_op=list_works&sortby=pubdate
[25]:	http://papers.nips.cc/paper/5857-inferring-algorithmic-patterns-with-stack-augmented-recurrent-nets
[26]:	https://github.com/facebook/Stack-RNN
[27]:	https://github.com/DoctorTeeth/diffmem
[28]:	http://arxiv.org/abs/1505.00521
[29]:	http://arxiv.org/abs/1511.07275
[30]:	http://arxiv.org/abs/1511.06392
[31]:	http://arxiv.org/abs/1511.04834