### Introduction
This folder is for resources related to reinforcemnet learning. 

### Resources
- [github resources][1]
- [Q-learning][2]
- [DeepMind-RL][3]
- [tensorflow tutorial][4]
	- **Anaconda to install tensorflow**, we should pay attention to [ipython environment][5]
- [David Silver][6]

### Books and tutorials
- [2015NIPS tutorial on RL][7]
- [2015NIPS tutorial on Monte Carlo][8]
- [Reinforcement Learning: An Introduction][9]
	- An classical intuitive intro to the field. 
	- [matlab code][10]

- [Reinforcement Learning and Dynamic Programming using Function Approximators][11]
	- A practical book that explains some state-of-the-art algorithms. 
	- 2010.

- [From Bandits to Monte-Carlo Tree Search: The Optimistic Principle Applied to Optimization and Planning][12]
	- Important **nonconvex optimization** methods 
	- Covering the interesting topics such as **Monte-Carlo Tree Search** and **Bandits**.
	- 2014.

### Papers to recover
- image caption with **hard attention**
	- [policy Gradient][13]

	- [recurrent models of visual attention][14]
		- NIPS2014, 96 cited
		- code: [Tensorflow][15]
		- realized in the folder  **visual-attention**
		- related technique papers: [approx gradient][16], [deep POMDP][17]

	- [multiple object recognition][18]: ICLR2015, 51 cited

	- [show attend and tell][19]
		-  ICML2015, 189 cited
		- [code][20]

- rnn with **additional memory**
	- [One-shot Learning with Memory-Augmented Neural Networks][21]
		- from **Google Deepmind**
		- Inheritated from **turing machine**
	- [programmer interpreter][22]
		- Neural Programmer-Interpreter (NPI): A recurrent and compositional neural network that learns to represent and execute programs.
		- Best paper for ICLR 2016 from Google DeepMind.
		- code: [Tensorflow][23]

	- [Neural Turing Machine][24]
		- 2014, 111 cited, by Alex Graves
		- code: [theano][25], [Tensorflow][26] 

	- [Memory Networks][27]
		- 2014, 81 cited, from **FaceBook**
		- code, [matlab][28], [tensorflow][29], [theano][30]

	- [Pointer Network][31]
		- NIPS2015, 20 cited
		- code: [theano][32], [Tensorflow][33]
		- Pay attention to this guy: [Oriol Vinyals][34]

	- [stackRNN][35]
		- NIPS2015, 24 cited
		- code: [C++][36], [python][37]

	- Another 4 highly-related papers
		- [RL turing machines][38]: 2015 arxiv, 25 cited
		- [learning simple algorithms][39]: 2015 arxiv, 3 cited
		- [neural random access][40]: 2015 arxiv, 5 cited
		- [neural programmer][41]: 2015 ICLR, 8 cited

### ICML2016 reinforcement learning
- [Doubly Robust Off-policy Value Evaluation for Reinforcement Learning][42]
	- Study the problem of **off-policy problem** in RL. 
	- Extend **doubly robust estimator** to get **unbiased** and **lower variance** results. 
	- Work from **Microsoft**

- [Smooth Imitation Learning][43]
	- The goal is to **train a policy** that can **imitate  human behavior** in a **dynamic and continuous environment** 
	- Work from **Caltech**

- [The Knowledge Gradient for Sequential Decision Making with Stochastic Binary Feedbacks][44]
	- **Small samples** and **time-consuming observation** environment 
	- Work from **Princeton**

- [Benchmarking **Deep Reinforcement Learning** for Continuous Control][45]
	- Continuous control
	- Work from **Berkeley**

- [Asynchronous Methods for **Deep Reinforcement Learning**][46]
	- Asynchronous gradient descent 
	- Work from **GoogleDeepmind**

- [Dueling Network Architectures for **Deep Reinforcement Learning**][47]
	- A new network architecture 
	- Work from **GoogleDeepmind**
	- 10 cited already 

- [Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning][48]
	- A new way of **predicting the performance** of RL
	- Work from **CMU**

- [Hierarchical Decision Making In Electricity Grid Management][49]
	- Algorithm that alternates between **Slow-time policy improvement** and **fast-time value function approximation** 
	- Work from **Israel**
	- 9 cited

- [Improving the Efficiency of **Deep Reinforcement Learning** with Normalized Advantage Functions and Synthetic Experience][50]
	- **Reduce sample complexity** of **deep reinforcement learning** 
	- Work from **GoogleDeepmind**

[1]:	https://github.com/BigeyeDestroyer/deepRL/tree/resource
[2]:	http://mnemstudio.org/path-finding-q-learning-tutorial.htm
[3]:	http://www.infoq.com/cn/articles/atari-reinforcement-learning
[4]:	https://github.com/pkmital/tensorflow_tutorials
[5]:	http://stackoverflow.com/questions/33960051/unable-to-import-a-module-from-python-notebook-in-jupyter
[6]:	http://www0.cs.ucl.ac.uk/staff/d.silver/web/Home.html
[7]:	https://nips.cc/Conferences/2015/Schedule?event=4890
[8]:	https://nips.cc/Conferences/2015/Schedule?event=4887
[9]:	http://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html
[10]:	http://waxworksmath.com/Authors/N_Z/Sutton/sutton.html
[11]:	https://orbi.ulg.ac.be/bitstream/2268/27963/1/book-FA-RL-DP.pdf
[12]:	https://hal.archives-ouvertes.fr/hal-00747575v5/document
[13]:	http://www.scholarpedia.org/article/Policy_gradient_methods
[14]:	http://arxiv.org/abs/1406.6247
[15]:	https://github.com/seann999/tensorflow_mnist_ram
[16]:	http://incompleteideas.net/sutton/williams-92.pdf
[17]:	http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/Wierstra_ICANN_2007_%5B0%5D.pdf
[18]:	http://arxiv.org/abs/1412.7755
[19]:	http://arxiv.org/abs/1502.03044
[20]:	https://github.com/kelvinxu/arctic-captions
[21]:	http://arxiv.org/abs/1605.06065
[22]:	http://arxiv.org/pdf/1511.06279v4.pdf
[23]:	https://github.com/carpedm20/NPI-tensorflow
[24]:	http://arxiv.org/abs/1410.5401
[25]:	https://github.com/shawntan/neural-turing-machines
[26]:	https://github.com/carpedm20/NTM-tensorflow
[27]:	http://arxiv.org/abs/1410.3916
[28]:	https://github.com/facebook/MemNN
[29]:	https://github.com/carpedm20/MemN2N-tensorflow
[30]:	https://github.com/vinhkhuc/MemN2N-babi-python
[31]:	http://papers.nips.cc/paper/5866-pointer-networks
[32]:	https://github.com/vshallc/PtrNets
[33]:	https://github.com/ikostrikov/TensorFlow-Pointer-Networks
[34]:	https://scholar.google.com/citations?hl=zh-CN&user=NkzyCvUAAAAJ&view_op=list_works&sortby=pubdate
[35]:	http://papers.nips.cc/paper/5857-inferring-algorithmic-patterns-with-stack-augmented-recurrent-nets
[36]:	https://github.com/facebook/Stack-RNN
[37]:	https://github.com/DoctorTeeth/diffmem
[38]:	http://arxiv.org/abs/1505.00521
[39]:	http://arxiv.org/abs/1511.07275
[40]:	http://arxiv.org/abs/1511.06392
[41]:	http://arxiv.org/abs/1511.04834
[42]:	http://arxiv.org/abs/1511.03722
[43]:	http://hoangminhle.github.io/
[44]:	https://arxiv.org/abs/1510.02354
[45]:	https://arxiv.org/abs/1604.06778
[46]:	https://arxiv.org/abs/1602.01783
[47]:	http://arxiv.org/abs/1511.06581
[48]:	http://arxiv.org/abs/1604.00923
[49]:	http://arxiv.org/abs/1603.01840
[50]:	http://arxiv.org/abs/1603.00748