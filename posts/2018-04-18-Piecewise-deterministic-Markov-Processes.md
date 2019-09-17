---
layout: post
title:  "Piecewise Deterministic Markov Processes"
date:   2018-04-18 15:15:00
categories: Notes MCMC 
author: Bradley
comments: true
---
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML.js"></script>
# Notes on Piecewise Deterministic Markov Processes


## General Framework for PDMP

The state space is defined to be the $$\mathcal{Z}= \mathbb{R}^{n}$$ and in the continuous-time setting we have $$\{z_{t}, t \geq 0 \}$$. 
A $$\mathcal{Z}$$-valued continuous time PDMP process is defined as a c\`{a}dl\`{a}d process with deterministic dynamics. This is defined as follows:
- An ODE with drift $$\phi : \mathcal{Z} \rightarrow \mathcal{Z}$$$$\begin{equation}\frac{dz_{t}}{dt} = \phi(z_{t})\end{equation}$$
- Event rate $$\lambda : \mathcal{Z} \rightarrow [0, \infty)$$ with probability of an event in the time period $$[t, t + \epsilon]$$ is $$p(E) = \lambda(z_{t}) \epsilon +  \mathcal{O}(\epsilon)$$.
- A Markov transition kernel $$Q$$ from $$\mathcal{Z}$$ to $$\mathcal{Z}$$, where the state at event time $$t$$ is given by $$z_{t} \sim Q(z_{t-}, \cdot)$$. Where $$z_{t-}$$ is the state process just before the event.

**PDMP Algorithm**


![png](http://www.robots.ox.ac.uk/~bradley/images/ctpdmp.png)

To each PDMP we have an associated generator, which is defined as:
$$\begin{equation}
   \mathcal{L}f(z) = \lim\limits_{\epsilon \rightarrow 0}\frac{\mathbb{E}[f(z_{t+\epsilon}) | z_{t} = z ] - f(z)}{\epsilon}
\end{equation}$$
which, under suitable regularity conditions can be written as: 
$$\begin{equation}
   \mathcal{Z}f(z) = <\phi(z), \nabla f(z)> + \lambda(z)\int [f(z') - f(z)]Q(z, dz')
\end{equation}$$

## Poisson Processes
Required in many PD-MCMC situations to simulate the rate $$\lambda(t)$$ dynamics. 

### Homogeneous Poisson processes

#### Code to Generate Homogeneous Poission Process
```python
t = 0
event_range = 1000
E_lam = []
rates = [7,10,100]
event_times = []
event_times.append(t)
unif = dists.Uniform(0,1)
for rate in rates:
    for i in range(event_range):
        u = unif.sample()
        temp = -torch.log(u)/rate
        E_lam.append(temp)
        # print(E_lam)
        event_times.append(event_times[i] + E_lam[i])
        # print('Debug: The event times: {}'.format(event_times))
    for i in range(len(E_lam)):
        E_lam[i] = E_lam[i].numpy()
    plt.plot(event_times[1:], E_lam[:], label = str(rate))
    E_lam = []
    event_times = [t]
# plt.plot(event_times[1:], E_lam[:])
plt.title('An example of a homegeneous poission process, for different rates')
plt.legend()
plt.show()
```

**An example of a homogeneous poission process for different rates**


![png](http://www.robots.ox.ac.uk/~bradley/images/hompospro)

### Non-Homogeneous Poisson Process
A non-homogeneous Poisson process is similar to an ordinary Poisson process, except the average rate of arrivals is allowed to vary with time. The positive of having a average rate that varies with time, is that we can model applications that generate random points more faithfully. However, the cost of this is that we lost the property of stationary increments. 

## From PDMP to PD-MCMC
Assume we have a given target density $$\rho(dz)$$ on a Borel measurable set $$(\mathcal{Z}, \mathcal{BZ})$$ and we want to sample from it, using a PDMP, how do we do this?

We would need to satisfy global and local criteria, to ensure the ergodic nature of MCMC, such as invariance.

We assume that  $$\rho(z)$$ is continuously differentiable and proportional to the canonical distribution as follows:
$$\begin{equation}
   \rho(z) \propto \exp(-H(z))
\end{equation}$$
and invariance w.r.t $$\rho$$ will be satisfied if:
$$\begin{equation}
   \int \rho(dz)\mathcal{L}f(z) = 0
\end{equation}$$
for all functions in the domain of the generator. 

### Sufficient conditions for global and local methods

### Global methods
 The following sufficient conditions on $$\phi, \lambda$$ and $$Q$$ ensure that $$\rho$$ invariance. 

- $$\exists \hspace{0.5mm} S:\mathcal{Z} \rightarrow \mathcal{Z} \hspace{0.5mm} s.t \hspace{0.5mm} \rho \circ S = \rho$$
- Event rate $$\lambda$$ satisfies : $$\lambda(S(z)) - \lambda(z) = \nabla \cdot \phi(z) -  <\nabla H(z), \phi(z)>$$
- The kernel $$Q$$ satisfies : $$\int \rho(dz)\lambda(z) Q(z, dz') = \rho(S(dz'))\lambda(S(z'))$$


### Local methods 
Assuming that $$H(z)$$ can be decomposed as $$H(z)  = \sum_{i=1}^{n} H_{i}(z)$$ and so each $$H_{i}(z) $$ only depends on a subset of the components of $$z$$. We may be interested in developing a  transition kernel which is a mixture of $$n$$ kernels performing local updates.  This can be done by introducing an event rate of the form $$\lambda(z) = \sum_{i=1}^{n} \lambda_{i}(z)$$ and a transition kernel of the form $$Q(z, dz') = \sum_{i=1}^{n}\frac{\lambda_{i}(z)}{\lambda(z)}Q_{i}(z,dz')$$ where the $$Q_{i}$$ are Markov transition kernels. If we define $$[n] := {1,2, \ldots, n}$$ and to simulate the event times of the resulting PDMP we associate a clock to each index $$i \in [n]$$. 

Sufficient local conditions on $$\phi, \{\lambda_{i} ; i \in [n]\} $$ and $$\{Q_{i} ; i \in [n]\}$$
   - $$\exists S:\mathcal{Z} \rightarrow \mathcal{Z} \hspace{0.5mm} s.t \hspace{0.5mm} \rho \circ S = \rho$$
   - Event rate $$ \lambda_{i} : i \in [n]\}$$ satisfies : $$\sum_{i=1}^{n}{\lambda_{i}(S(z)) - \lambda_{i}(z)} = \nabla \cdot \phi(z) - <\nabla H(z), \phi(z)>$$
   - $$\forall i \in [n]$$, the transition kernel $$Q_{i}$$ satisfies : $$\int \rho(dz)\lambda_{i}(z) Q_{i}(z, dz') = \rho(S(dz'))\lambda_{i}(S(z'))$$
\end{align}
### Existing PD-MCMC algorithms

All existing algorithms are based on the following framework:
- A distribution of interest $$\pi(x) \propto \exp(-U(x))$$
- An auxiliary distribution $$\phi(v)$$
- The phase space $$z= (x,v)$$ creates a density $$\rho(z) = \pi(x)\psi(v)$$. 
- $$v \epsilon \mathcal{V}$$, where $$\mathcal{V}$$ can be either $$\mathbb{R}^{d}$$ or the unit hypersphere $$\mathbb{S}^{d-1}$$
- The linear dynamical system is then considered: $$\begin{equation}\phi(z) = \phi(v,0_{d})\end{equation}$$
- With flow: $$\begin{equation} \Phi_{t}(z) = (x + vt , v) \end{equation}$$ and so $$\nabla \cdot \phi = 0$$.
- They also rely on time reversibility, i.e $$S(z) = (x,-v)$$ and so we have:
- $$\begin{equation}\lambda(x,-v) - \lambda(x,v) = -<\nabla U(x),v>\end{equation}$$ 

### Bouncy particle sampler

The bouncy particle sampler exploits any decomposable energy function U. That is $$\begin{equation} 
U(x) = \sum_{i=1}^{m}U_{i}(x)
\end{equation}$$. 
For $$\lambda_{ref} > 0$$, it uses the event rate
$$\begin{equation}
   \lambda(z) = \lambda_{ref} + \sum_{i=1}^{m} \max(0,<\nabla U_{i}(x),v>)
\end{equation}$$
It also relies on the transition kernel:
$$\begin{equation}
   Q(z,dz') = \frac{\lambda_{ref}}{\lambda(z)}\delta_{x}(dx')\psi(dv') + \sum_{i=1}^{m}\frac{\max(0,<\nabla U_{i}(x),v>)}{\lambda(z)}\delta_{x}(dx')\delta_{R_{\nabla U_{i}(x)}v}(dv')
\end{equation}$$
where $$R_{\nabla W}(x)v$$ represents an orthogonal vector field: 
\begin{equation}
\label{eq:bounce}
   R_{\nabla W(x)}v = v - 2\frac{<\nabla W(x),v>}{||\nabla W(x)||^{2}} \nabla W(x)
\end{equation}

The above equation for $$  R_{\nabla W(x)}v$$ corresponds to a bounce, as its structure is the same as a Newtonian collision to the place perpendicular to $$\nabla W$$ at $$x$$. 
The $$\psi$$ function is taken to be either a normal distribution on $$\mathbb{R}^{d}$$ or uniform distribution on $$\mathbb{S}^{d-1}$$. Uses equations (10-11) for the event rate and transition matrix when $$n = m + 1$$. When $$i \in [m]$$ it uses $$\lambda_{i} = \max(0,<\nabla U_{i}(x),v>)$$ and $$Q_{i}(z,dz') = \delta_{x}(dx')\delta_{R_{\nabla U_{i}(x)}v}(dv')$$ locally, and globally $$\lambda_{n}(z) = \lambda_{ref} $$ and $$Q_{n}(z,dz') = \delta_{x}(dx')\psi(dv')$$.

**Bouncy Particle Algorithm**


![](http://www.robots.ox.ac.uk/~bradley/images/bps.png)

### The Zig-Zag sampler
- $$\psi \sim Uniform(-1,1)$$ on $$\{-1,1\}^{d} $$
- Event rate $$\begin{equation}
        \lambda_{i}(z) = \lambda_{ref,i}
        \end{equation}$$
- Transition kernel $$\begin{equation}
Q_{i}(z,dz') = \delta_{x}(dx')\delta_{-v_{i}}(dv_{i}')\prod_{j \neq i}\delta_{dv'_{j}}
\end{equation}$$


## Embedding discreteness into continuous models using Stats. phys. ideas

Inspired by the paper ``How can statistical mechanics contribute to the social sciences''

### The Framework
 Let $$i \in \{1,\ldots, I\}$$ represent the set of individuals, $$i$$. We state that $$\forall$$  $$ i $$ $$\exists$$ $$ w_{i} \sim Bern(p_{i})$$ where $$w_{i}$$ represents the individuals choice and $$p_{i}$$ is the probability that the individual chooses that choice. 
Let us state that each individual would like to maximize their utility, represented by the utility function $$U$$ or payoff. 
$$\begin{equation}
\max U_{i} = max_{w_{i} \in \{-1,1\}}h_{i}w_{i} + \epsilon_{i}(w_{i})
\end{equation}$$ 
where

- $$h_{i}$$ measures observable difference in payoff of the two choices.
- $$\epsilon_{i}(w_i)$$ denotes the individual specific random and unobservable component to the pay off associated with the two choices.

and we assume that the difference of random choices is logistically distributed as: 
$$\begin{equation}
\epsilon_{i}(-1) - \epsilon_{i}(1) \leq z \sim \frac{1}{1 + \exp(-\beta_{i}z)^{i}}
\end{equation}$$
and that each individual is independent of the other.

But if we make this assumption how do we then introduce an interaction between different players in the game?

### Introducing Interaction 
One way to introduce interactions is by adding additional terms to the pay off. I.e let $$\mathbb{E}_{i}(x)$$ denote the \textit{subjective} expected value assigned by $$i$$ to $$x$$. 
Thus the payoff function becomes:
$$\begin{equation}
\max_{w_i \in \{-1, 1\}} h_{i}w_i - \mathbb{E}_{i}[\sum_{j \neq i} \frac{J_{i,j}}{2}(w_{i} - w_j)^{2}] + \epsilon_{i}(w_{i})
\end{equation}$$ 
Where $$J_{i,j}$$ represents the interacting aspect of decision making. It states that \textit{each} agent has beliefs about the behaviour of others in a reference group and that these beliefs influence the payoff associated with the agents decisions. 
I.e if $$J_{i,j} \geq 0 $$ implies that the agent $$i$$ experiences some incentive to conform to the behaviour of agent $$j$$ (vice versa).\\

Taking this view point enables one to construct probabilities regarding the choices available. 

$$\begin{equation}
p(w_i | h_i, \mathbb{E}_{i}(w_{j}) \forall j) \propto \exp(\beta_{i}h_i w_i + \sum_{j \neq i}\beta_{i}J_{i,j}w_i \mathbb{E}_{i}(w_j))
\end{equation}$$  
Which enables us to construct the joint probability, that is equivalent to the Gibb's measure, and enables us to express the joint probability over all choices of individual $$i$$ and how they are influence by the choices of another individual $$j$$.\\
We also make an additional assumption, and state that the system is closed. This essentially means that we assume each individual is a rational thinker. This enables us to change the subjective expectation, to a true expectation (I do not understand this way of thinking) conditioned on all influences within the model. 
I.e $$\mathbb{E}_{i}(w_{j}) \mapsto \mathbb{E}(w_j | h_i, \beta_{i}, J_{i,j})$$   $$  \forall i,j $$
Thus the mathematical expectations of individual choices are characterized by the set of $$I$$ equations, that describe the expected value of each choice. 
$$\begin{equation}
   \mathbb{E}(w_{i}) = \tanh(\beta_{i} h_i + \sum_{j \neq i} \beta_{i} J_{i,j} \mathbb{E}(w_j)
\end{equation}$$
where $$\tanh(x)  = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}$$. Thus, we have encoded the binary choices, $$w_{i} \in \{-1,1\}$$ into a continuous expression.

### Notes 
This could potentially be used to re-write many binary models.  
