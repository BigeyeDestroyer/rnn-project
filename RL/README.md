### Introduction
This folder is for resources related to reinforcemnet learning. 

### Resources
- [github resources][1]
- [Q-learning][2]
- [DeepMind-RL][3]
- [tensorflow tutorial][4]
	- **Anaconda to install tensorflow**, we should pay attention to [ipython environment][5]

### Books and tutorials
- [2015NIPS tutorial on RL][6]
- [Reinforcement Learning: An Introduction][7]
	- An classical intuitive intro to the field. 
	- [matlab code][8]

- [Reinforcement Learning and Dynamic Programming using Function Approximators][9]
	- A practical book that explains some state-of-the-art algorithms. 
	- 2010.

- [From Bandits to Monte-Carlo Tree Search: The Optimistic Principle Applied to Optimization and Planning][10]
	- Important **nonconvex optimization** methods 
	- Covering the interesting topics such as **Monte-Carlo Tree Search** and **Bandits**.
	- 2014.

### Papers to recover
- image caption with **hard attention**
	- [policy Gradient][11]

	- [recurrent models of visual attention][12]
		- NIPS2014, 96 cited
		- code: [Tensorflow][13]
		- realized in the folder  **visual-attention**
		- related technique papers: [approx gradient][14], [deep POMDP][15]

	- [multiple object recognition][16]: ICLR2015, 51 cited

	- [show attend and tell][17]
		-  ICML2015, 189 cited
		- [code][18]

- rnn with **additional memory**
	- [programmer interpreter][19]
		- Neural Programmer-Interpreter (NPI): A recurrent and compositional neural network that learns to represent and execute programs.
		- Best paper for ICLR 2016 from Google DeepMind.
		- code: [Tensorflow][20]

	- [Neural Turing Machine][21]
		- 2014, 111 cited, by Alex Graves
		- code: [theano][22], [Tensorflow][23] 

	- [Pointer Network][24]
		- NIPS2015, 20 cited
		- code: [theano][25], [Tensorflow][26]
		- Pay attention to this guy: [Oriol Vinyals][27]

	- [stackRNN][28]
		- NIPS2015, 24 cited
		- code: [C++][29], [python][30]

	- Another 4 highly-related papers
		- [RL turing machines][31]: 2015 arxiv, 25 cited
		- [learning simple algorithms][32]: 2015 arxiv, 3 cited
		- [neural random access][33]: 2015 arxiv, 5 cited
		- [neural programmer][34]: 2015 ICLR, 8 cited

[1]:	https://github.com/BigeyeDestroyer/deepRL/tree/resource
[2]:	http://mnemstudio.org/path-finding-q-learning-tutorial.htm
[3]:	http://www.infoq.com/cn/articles/atari-reinforcement-learning
[4]:	https://github.com/pkmital/tensorflow_tutorials
[5]:	http://stackoverflow.com/questions/33960051/unable-to-import-a-module-from-python-notebook-in-jupyter
[6]:	https://nips.cc/Conferences/2015/Schedule?event=4890
[7]:	http://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html
[8]:	http://waxworksmath.com/Authors/N_Z/Sutton/sutton.html
[9]:	https://orbi.ulg.ac.be/bitstream/2268/27963/1/book-FA-RL-DP.pdf
[10]:	https://hal.archives-ouvertes.fr/hal-00747575v5/document
[11]:	http://www.scholarpedia.org/article/Policy_gradient_methods
[12]:	http://arxiv.org/abs/1406.6247
[13]:	https://github.com/seann999/tensorflow_mnist_ram
[14]:	http://incompleteideas.net/sutton/williams-92.pdf
[15]:	http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/Wierstra_ICANN_2007_%5B0%5D.pdf
[16]:	http://arxiv.org/abs/1412.7755
[17]:	http://arxiv.org/abs/1502.03044
[18]:	https://github.com/kelvinxu/arctic-captions
[19]:	http://arxiv.org/pdf/1511.06279v4.pdf
[20]:	https://github.com/carpedm20/NPI-tensorflow
[21]:	http://arxiv.org/abs/1410.5401
[22]:	https://github.com/shawntan/neural-turing-machines
[23]:	https://github.com/carpedm20/NTM-tensorflow
[24]:	http://papers.nips.cc/paper/5866-pointer-networks
[25]:	https://github.com/vshallc/PtrNets
[26]:	https://github.com/ikostrikov/TensorFlow-Pointer-Networks
[27]:	https://scholar.google.com/citations?hl=zh-CN&user=NkzyCvUAAAAJ&view_op=list_works&sortby=pubdate
[28]:	http://papers.nips.cc/paper/5857-inferring-algorithmic-patterns-with-stack-augmented-recurrent-nets
[29]:	https://github.com/facebook/Stack-RNN
[30]:	https://github.com/DoctorTeeth/diffmem
[31]:	http://arxiv.org/abs/1505.00521
[32]:	http://arxiv.org/abs/1511.07275
[33]:	http://arxiv.org/abs/1511.06392
[34]:	http://arxiv.org/abs/1511.04834