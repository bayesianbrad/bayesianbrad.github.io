---
layout: post
title:  "Quantum Mechanics for the Machine Learner"
comments: true
---

These notes are based on a reading group that I was asked to do for the [OXCSML](http://csml.stats.ox.ac.uk/learning/) group at Oxford University and I thought that I may as well share them with the community as hopefully they will be of use to others :).

### Can we have a little more hype please?

Ok, let's face it, two of the biggest hyped fields right now are Quantum computing and Machine learning, sometimes called *ai*, although I've yet to be convinced by the 
*ai* part yet.
Nonetheless, I have a unique perspective on both fields as my background is in quantum information science and I retrained, thanks to the [AIMS program](https://www.eng.ox.ac.uk/aims-cdt/) at Oxford University, to become a Statistical Machine learning researcher, also called Statistics researcher, but, you know, that is not cool anymore... #FundingGoals


### Formalism

We have to start somewhere, for me one of the trickiest parts from moving from quantum mechanics to machine learning was notations, despite the fact that both rely heavily on linear algebra, they are  stylistically quite different, so lets make sure that we are all on the same page.

The first set of notations that we shall go through are for describing the quantum state, or just state and operator formalisms of quantum mechanics. 
A state, in purely semantic
terms is just a vector on $$\mathbb{C}^{D x 1}$$ i.e it is just a row vector defined in the complex domain, we will get to why this is later. Cool. 
It can be both discrete, or continuous and it has a very powerful meaning in quantum mechanics, 
that is, the state describes *absolutely* everything about the system. You can think of it like the ultimate neural network (NN), is that a Bayesian neural network? #HeTriedToMakeAJoke in the sense that it describes this surface, usually very complicated, that it tells us about all the possible outcomes that a particular quantum state can output, in fact even tells us what the probabilities associated with those outcomes are too!
The difference here is that the NN is only as descriptive as the data it has access too, whereas the quantum state is analytically defined, not heuristically.

[Add a picture of a loss surface, one defined by maths another defined by data]
There are lots of interpretations about what this state tells us, you can read about them [here](https://en.wikipedia.org/wiki/Interpretations_of_quantum_mechanics), it is both philosophically and physically interesting, so if you are interested I would have a look!
<img src="https://imgs.xkcd.com/comics/quantum.png" style="width: 300px; max-width: 100%; height: auto; float: right" alt="XKCD quantum" />
</p>

**Add nice hand drawn windows tablet thing here**




