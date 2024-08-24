This repository is for the first advanced machine learning study group in Qishi. The following is our syllabus:

### Lecture 0: Introduction
- Understand the overall structure and goals of the course.
- Identify connections between different models.
- Recognize real-world applications of the models discussed.
  
### Lecture 1: Lasso and Regularization
- Develop a comprehensive understanding of Lasso and its role in regularization.
- Explore how Lasso performs feature selection by enforcing sparsity.
- Understand implicit regularization and its effects on model complexity.
- Examine the relationship between Lasso and boosting techniques.
- Learn efficient computational methods like coordinate descent for solving Lasso problems.
  
### Lecture 2: Principal Component Analysis(PCA)
- Understand the mathematical foundation of PCA.
- Learn how PCA reduces dimensionality by identifying key components that capture the most variance.
- Explore the connection between PCA and Variational Autoencoders (VAEs) in terms of data representation and dimensionality reduction.
  
### Lecture 3: Proabilistic Graphcial Model
- Understand the structure and types of probabilistic graphical models (PGMs), including Bayesian and Markov networks.
- Learn how PGMs represent complex distributions and dependencies using graphs.
- Explore the principles of Bayesian machine learning and how PGMs facilitate reasoning under uncertainty.
  
### Lecture 4: Expectation-Maximization(EM) Algorithm 1
- Understand the fundamentals of the Expectation-Maximization (EM) algorithm.
- Learn the detailed mathematical derivation of the EM algorithm.
- Explore the proofs of inequalities used in EM, such as Jensen's inequality.
- Apply EM to simple applications like estimating parameters with missing data.
- Discuss other common applications of the EM algorithm.
  
### Lecture 5: EM Algorithm 2 and Gaussian Mixture Model
- Explore advanced applications of the EM algorithm, including the MM (Minorization-Maximization) algorithm.
- Use Gaussian Mixture Models (GMMs) as a key example to understand the practical implementation of EM.
- Delve into the iterative logic and convergence properties of EM in complex scenarios.
  
### Lecture 6: Variational Inference
- Understand the core principles of Variational Inference (VI) and its role in approximating complex posterior distributions.
- Learn how VI formulates inference as an optimization problem, using techniques like the Evidence Lower Bound (ELBO).
- Explore the application of VI in various machine learning algorithms, including its use in Variational Autoencoders (VAEs).
- Study key concepts such as mean-field approximation and how neural networks can replace traditional integration methods.
- Develop a strong foundation in VI to enhance understanding of subsequent topics and its pervasive role in statistical learning and machine learning frameworks.
  
### Lecture 7: State Space Model 1: Hidden Markov Model
- Understand the concept of state space models and their application in modeling time series data.
- Learn how state space models are used for analysis and prediction of time series within a Bayesian framework.
- Use Hidden Markov Models (HMMs) as a key example to understand the analysis of discrete time series.
- Study the algorithms for inference in HMMs, such as the Forward-Backward algorithm and Viterbi algorithm, to perform state estimation and sequence prediction.
  
### Lecture 8: State Space Model 2: Kalman Filter
- Understand the fundamentals of the Kalman Filter, including its use in estimating the state of linear dynamic systems.
- Learn the detailed mathematical derivation of the Kalman Filter, focusing on key concepts like prediction and update steps.
- Identify and tackle mathematical challenges in the derivation, such as matrix algebra and recursive estimation.
- Explore advanced versions of the Kalman Filter, including the Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF) for handling nonlinear systems.

### Lecture 9: MCMC 1: Basic sampling method：Importance Sampling，Gibbs Sampling, MH mehtod
- Understand the concept of sampling and its importance in approximating complex distributions.
- Learn simple sampling methods such as inversion and importance sampling, and discuss their advantages and limitations.
- Introduce Markov Chain Monte Carlo (MCMC) as a method to overcome limitations of basic sampling.
- Explore basic MCMC algorithms, including Gibbs Sampling and the Metropolis-Hastings (MH) method.

### Lecture 10: State Spcae Model 3: Particle Filter
- Understand the fundamentals of Particle Filters and their application to nonlinear and non-Gaussian systems.
- Learn how Particle Filters approximate probability distributions using a set of weighted samples.
- Explore the core steps: prediction, update, and resampling, and understand their roles in the filtering process.
- Discuss the Sequential Importance Resampling (SIR) algorithm and its significance in Particle Filters.
- Explore recent advancements and practical algorithms in Particle Filters, such as the Auxiliary Particle Filter and the Particle Metropolis-Hastings method.
  
### Lecture 11: MCMC 2: Advanced MCMC methods：SMC and HMC
- Delve into advanced MCMC techniques like Hamiltonian Monte Carlo (HMC), Sequential Monte Carlo (SMC), and Langevin Monte Carlo (LMC).
- Explore the benefits of advanced methods in efficiently sampling high-dimensional distributions.
- Apply these advanced techniques to complex inference problems in modern machine learning contexts.
  
### Lecture 12: Variational Auto Encoder(VAE)
- Understand the role of VAEs in combining deep learning with traditional machine learning through Variational Inference.
- Explore the mathematical foundation of VAEs, focusing on the Evidence Lower Bound (ELBO) and its optimization.
- Reparameterization Trick: Learn how this trick enables efficient backpropagation through stochastic layers by reparametrizing the sampling process.
- Learn how VAEs encode data into a latent space and decode it to generate new samples.
- View VAEs from a probabilistic graphical model standpoint, where the encoder approximates the posterior and the decoder models the likelihood.
- Analyze the VAE loss, combining reconstruction loss and KL divergence to ensure data fidelity and latent space regularization.
- Discuss practical variations of VAEs, such as Conditional VAEs (CVAE)
- Discuss the limitations of VAEs, such as blurriness in generated samples, and introduce GANs as a solution for generating sharper images.
  
### Lecture 13: Generative Adversarial Network(GAN)
- Understand the basic framework of GANs, consisting of a generator and a discriminator in a minimax game.
- Explore the loss functions for the generator and discriminator, and how they are derived from the Jensen-Shannon divergence.
- Understand why GANs are difficult to train, focusing on issues like instability and convergence problems.
- Analyze the mode collapse phenomenon, where the generator produces limited diversity, and discuss statistical reasons for its occurrence.
- Techniques like feature matching, minibatch discrimination, and spectral normalization.
- Use of Wasserstein GANs (WGAN) to improve training stability by using the Earth Mover's distance.
- Transition to diffusion models as an alternative generative approach that addresses some GAN limitations, providing more stable training and diverse outputs.
  
### Lecture 14: Diffusion Model 1: Theory
- Understand how diffusion models generate data by reversing a diffusion process.
- DDPM (Denoising Diffusion Probabilistic Models)
- DDIM (Denoising Diffusion Implicit Models)
  
### Lecture 15: Diffusion Model 2: Application
- Understand how CLIP (Contrastive Language–Image Pretraining) helps bridge text and image modalities.
- Explore Stable Diffusion's ability to produce high-quality images efficiently.
- Examine recent advancements that combine diffusion models with large language models (LLMs).

### Reference Books:
1. Pattern Recognition and Machine Learning, Bishop
2. The Elements of Statistical Learning, Hastie
3. Computer-Age Statistical Inference, Hastie
4. Probabilistic Machine Learning: Introduction, Murphy 
5. Probabilistic Machine Learning: Advanced Topics, Murphy
6. Probabilistic Deep Learning with Python, Keras and Tensorflow Probability, Sick
7. Time Series Analysis by State Spcae Methods, Durbin
8. MCMC from Scratch, Hanada
9. Statistical Inference, Casella

### Reference Links:
1. 白板推导机器学习：https://www.bilibili.com/video/BV1aE411o7qd/?spm_id_from=333.999.0.0&vd_source=648125ac3e2fb6b31759a16531ce143b
2. 徐亦达机器学习：https://space.bilibili.com/327617676/channel/series
3. 线代启示录： https://ccjou.wordpress.com
4. 科学空间：https://spaces.ac.cn

