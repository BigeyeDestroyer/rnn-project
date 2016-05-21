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
- [2015NIPS tutorial on Monte Carlo][7]
- [Reinforcement Learning: An Introduction][8]
	- An classical intuitive intro to the field. 
	- [matlab code][9]

- [Reinforcement Learning and Dynamic Programming using Function Approximators][10]
	- A practical book that explains some state-of-the-art algorithms. 
	- 2010.

- [From Bandits to Monte-Carlo Tree Search: The Optimistic Principle Applied to Optimization and Planning][11]
	- Important **nonconvex optimization** methods 
	- Covering the interesting topics such as **Monte-Carlo Tree Search** and **Bandits**.
	- 2014.

### Papers to recover
- image caption with **hard attention**
	- [policy Gradient][12]

	- [recurrent models of visual attention][13]
		- NIPS2014, 96 cited
		- code: [Tensorflow][14]
		- realized in the folder  **visual-attention**
		- related technique papers: [approx gradient][15], [deep POMDP][16]

	- [multiple object recognition][17]: ICLR2015, 51 cited

	- [show attend and tell][18]
		-  ICML2015, 189 cited
		- [code][19]

- rnn with **additional memory**
	- [One-shot Learning with Memory-Augmented Neural Networks][20]
		- from **Google Deepmind**
		- Inheritated from **turing machine**
	- [programmer interpreter][21]
		- Neural Programmer-Interpreter (NPI): A recurrent and compositional neural network that learns to represent and execute programs.
		- Best paper for ICLR 2016 from Google DeepMind.
		- code: [Tensorflow][22]

	- [Neural Turing Machine][23]
		- 2014, 111 cited, by Alex Graves
		- code: [theano][24], [Tensorflow][25] 

	- [Memory Networks][26]
		- 2014, 81 cited, from **FaceBook**
		- code, [matlab][27], [tensorflow][28], [theano][29]

	- [Pointer Network][30]
		- NIPS2015, 20 cited
		- code: [theano][31], [Tensorflow][32]
		- Pay attention to this guy: [Oriol Vinyals][33]

	- [stackRNN][34]
		- NIPS2015, 24 cited
		- code: [C++][35], [python][36]

	- Another 4 highly-related papers
		- [RL turing machines][37]: 2015 arxiv, 25 cited
		- [learning simple algorithms][38]: 2015 arxiv, 3 cited
		- [neural random access][39]: 2015 arxiv, 5 cited
		- [neural programmer][40]: 2015 ICLR, 8 cited

### ICML2016 reinforcement learning
- [Doubly Robust Off-policy Value Evaluation for Reinforcement Learning][41]
	- Study the problem of **off-policy problem** in RL. 
	- Extend **doubly robust estimator** to get **unbiased** and **lower variance** results. 
	- Work from **Microsoft**

- [Smooth Imitation Learning][42]
	- The goal is to **train a policy** that can **imitate  human behavior** in a **dynamic and continuous environment** 
	- Work from **Caltech**

- [The Knowledge Gradient for Sequential Decision Making with Stochastic Binary Feedbacks][43]
	- **Small samples** and **time-consuming observation** environment 
	- Work from **Princeton**

- [Benchmarking **Deep Reinforcement Learning** for Continuous Control][44]
	- Continuous control
	- Work from **Berkeley**

- [Asynchronous Methods for **Deep Reinforcement Learning**][45]
	- Asynchronous gradient descent 
	- Work from **GoogleDeepmind**

- [Dueling Network Architectures for **Deep Reinforcement Learning**][46]
	- A new network architecture 
	- Work from **GoogleDeepmind**
	- 10 cited already 

- [Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning][47]
	- A new way of **predicting the performance** of RL
	- Work from **CMU**

- [Hierarchical Decision Making In Electricity Grid Management][48]
	- Algorithm that alternates between **Slow-time policy improvement** and **fast-time value function approximation** 
	- Work from **Israel**
	- 9 cited

- [Improving the Efficiency of **Deep Reinforcement Learning** with Normalized Advantage Functions and Synthetic Experience][49]
	- **Reduce sample complexity** of **deep reinforcement learning** 
	- Work from **GoogleDeepmind**

[1]:	https://github.com/BigeyeDestroyer/deepRL/tree/resource
[2]:	http://mnemstudio.org/path-finding-q-learning-tutorial.htm
[3]:	http://www.infoq.com/cn/articles/atari-reinforcement-learning
[4]:	https://github.com/pkmital/tensorflow_tutorials
[5]:	http://stackoverflow.com/questions/33960051/unable-to-import-a-module-from-python-notebook-in-jupyter
[6]:	https://nips.cc/Conferences/2015/Schedule?event=4890
[7]:	https://nips.cc/Conferences/2015/Schedule?event=4887
[8]:	http://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html
[9]:	http://waxworksmath.com/Authors/N_Z/Sutton/sutton.html
[10]:	https://orbi.ulg.ac.be/bitstream/2268/27963/1/book-FA-RL-DP.pdf
[11]:	https://hal.archives-ouvertes.fr/hal-00747575v5/document
[12]:	http://www.scholarpedia.org/article/Policy_gradient_methods
[13]:	http://arxiv.org/abs/1406.6247
[14]:	https://github.com/seann999/tensorflow_mnist_ram
[15]:	http://incompleteideas.net/sutton/williams-92.pdf
[16]:	http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/Wierstra_ICANN_2007_%5B0%5D.pdf
[17]:	http://arxiv.org/abs/1412.7755
[18]:	http://arxiv.org/abs/1502.03044
[19]:	https://github.com/kelvinxu/arctic-captions
[20]:	http://arxiv.org/abs/1605.06065
[21]:	http://arxiv.org/pdf/1511.06279v4.pdf
[22]:	https://github.com/carpedm20/NPI-tensorflow
[23]:	http://arxiv.org/abs/1410.5401
[24]:	https://github.com/shawntan/neural-turing-machines
[25]:	https://github.com/carpedm20/NTM-tensorflow
[26]:	http://arxiv.org/abs/1410.3916
[27]:	https://github.com/facebook/MemNN
[28]:	https://github.com/carpedm20/MemN2N-tensorflow
[29]:	https://github.com/vinhkhuc/MemN2N-babi-python
[30]:	http://papers.nips.cc/paper/5866-pointer-networks
[31]:	https://github.com/vshallc/PtrNets
[32]:	https://github.com/ikostrikov/TensorFlow-Pointer-Networks
[33]:	https://scholar.google.com/citations?hl=zh-CN&user=NkzyCvUAAAAJ&view_op=list_works&sortby=pubdate
[34]:	http://papers.nips.cc/paper/5857-inferring-algorithmic-patterns-with-stack-augmented-recurrent-nets
[35]:	https://github.com/facebook/Stack-RNN
[36]:	https://github.com/DoctorTeeth/diffmem
[37]:	http://arxiv.org/abs/1505.00521
[38]:	http://arxiv.org/abs/1511.07275
[39]:	http://arxiv.org/abs/1511.06392
[40]:	http://arxiv.org/abs/1511.04834
[41]:	http://arxiv.org/abs/1511.03722
[42]:	http://hoangminhle.github.io/
[43]:	https://arxiv.org/abs/1510.02354
[44]:	https://arxiv.org/abs/1604.06778
[45]:	https://arxiv.org/abs/1602.01783
[46]:	http://arxiv.org/abs/1511.06581
[47]:	http://arxiv.org/abs/1604.00923
[48]:	http://arxiv.org/abs/1603.01840
[49]:	http://arxiv.org/abs/1603.00748