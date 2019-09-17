---
layout: post
title:  "How to use loader and finder objects in Python"
date:   2017-12-31 20:32:00
# categories: Python Programming
# author: Bradley
comments: true
---
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML.js"></script>
In our current project we are creating a fully fledge probabilistic programming environment (**PPE**) in `python` for implementing an improved Hamiltonian Monte Carlo (**HMC**) algorithm. This enables one to perform inference (inference is just another way of saying *prediction*, or to *infer* something in the future) in models that have a mixture of discrete, continuous and branching latent variables, or simply either just continuous or just discrete latent variables. Branching is more complicated and we shall leave that discussion for another time. Just in case you were unaware, HMC can only be used with continuous latent variables, but is a highly effective inference algorithm for dealing with models of a high dimensionality.

## Modelling

One aspect of our PPE is that it takes a directed graphical model written in `clojure` and compiles it into an interface format in `python` code. One could equally just write the model in `python` according to the interface, however, this requires substantially more work as you would have to manually write out the log of the probability density function, among other things. Imagine that we have the following simple model:  
<p>
$$
\begin{align}
 x &\sim \mathcal{N}(1,\sqrt 5) \tag{1}\\
 x | y=7 &= \mathcal{N}(7| 2, \sqrt 2) \tag{2}
\end{align}
$$
</p>
We can write this model in `clojure` as follows:
```clojure
        (let [x (sample (normal 1.0 5.0))]
            (observe (normal x 2.0) 7.0)
        x)
```
which you may save as `model.clj` or `one_gaussian.clj` or whatever else, as long as you keep the `.clj` suffix. This then needs compiling into `python`. To avoid the user having to install multiple different dependencies, having to manually run a separate script to compile the output and then import the outputted `python` script which matches our designed interface. We instead built our own `importer module`, which implements a `finder` and a `loader` and then compiles the `clojure` code to `python` code and imports that as module into the users main inference script. 

## The imports module 

Here is the current `importer` module, located at `pyfo.pyfoppl.foppl.imports` :
```python
from importlib.abc import Loader as _Loader, MetaPathFinder as _MetaPathFinder
from .compiler import compile
from .model_generator import Model_Generator

import sys
PATH = sys.path[0]

def compile_module(module, input_text):
    graph, expr = compile(input_text)
    model_gen = Model_Generator(graph)
    code = model_gen.generate_class()
    exec(code, module.__dict__)
    module.graph = graph
    module.code = code
    return module

class Clojure_Loader(_Loader):

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        with open(module.__name__) as input_file:
            input_text = '\n'.join(input_file.readlines())
            compile_module(module, input_text)

class Clojure_Finder(_MetaPathFinder):

    def find_module(self, fullname, path=PATH):
        return self.find_spec(fullname, path)

    def find_spec(self, fullname, path, target = None):
        import os.path
        from importlib.machinery import ModuleSpec
        fullname = fullname.split(sep='.')[-1]
        if '.' in fullname:
            raise NotImplementedError()

        if os.path.exists(fullname + ".clj"):
            return ModuleSpec(os.path.realpath(fullname + ".clj"), Clojure_Loader())
        else:
            return None

sys.meta_path.append(Clojure_Finder())
```
## Preamble

Before we walk through this, let us first see what the user has to do to invoke the PPE. We assume that the user has 
`pip` installed `pyfo`, so it exists on the `PYTHONPATH`.

Let us say that the users working directory is: `/Users/bradley/Documents/Models/`
Within that directory they create the `one_gaussian.clj` script. The user would then write the following `python` script, which would be saved within this directory:
```python
from pyfo.pyfoppl.foppl import imports # this uses a loader and finder module.
import one_gaussian # when we do this `imports` is triggered, compiles the mode; automatically and loads it as a module.
from pyfo.inference.dhmc import DHMCSampler as dhmc  # inference algorithm

burn_in = 100
n_samples = 10 ** 3
stepsize_range = [0.03,0.15]
n_step_range = [10, 20]

dhmc_    = dhmc(one_gaussian.model, n_chains)
stats = dhmc_.sample(n_samples, burn_in, stepsize_range, n_step_range) # MCMC sampling begins
samples = stats['samples'] # returns dataframe of all samples.
means = stats['means'] # returns dictionary key:value, where key - parameter , value = mean of parameter
```
The user would then run this script, say `gauss_model.py` either via the terminal or their favourite IDE. 

They require no knowledge of how to compile their model and they require no knowledge of how the inference is performed. We believe that this adds a lot more flexibility to `pyfo`, as it requires very little knowledge to get up and running. It simply requires the user to write the model and run. As compared to languages such as: `Stan`, `PyMC3` and `pyro`, which require the user to understand more technical aspects of the process. Although, they are incredibly well developed and powerful packages, especially the former two.

## How the pyfo.pyfoppl.foppl.imports module works

Now that we have introduced the basic premice of the outcome, let us now delve into the details of how the `imports` module works. 

When `imports` is invoked in the users script, it invokes `python`'s `sys.meta_path()`, which takes the `Clojure_finder()` class as an argument. The following description of `sys.meta_path` is taken largely from `python` docs, however, it explains it better than I could have myself. 
`sys.meta_path` contains a list of meta path finder objects. These finders are queried in order to see if they know how to handle the named module (the `one_guassian` file in this case). Meta path finders must implement a method called `find_spec()` which takes three arguments: a **name**, an **import** **path**, and (optionally) a **target** **module**. The meta path finder can use any strategy it wants to determine whether it can handle the named module or not.

If the meta path finder knows how to handle the named module, it returns a `spec` object. If it cannot handle the named module, it returns `None`. If `sys.meta_path` processing reaches the end of its list without returning a `spec`, then a `ModuleNotFoundError` is raised. Any other exceptions raised are simply propagated up, aborting the import process.

The `find_spec()` method of meta path finders is called with two or three arguments. The first is the fully qualified name of the module being imported, here that would be `fullname = /Users/bradley/Documents/Models/` The second argument is the path entries to use for the module search, which in this instance is `path = sys.path[0]`. For top-level modules, the second argument is `None`, but for submodules or subpackages, the second argument is the value of the parent package’s `__path__` attribute. If the appropriate `__path__` attribute cannot be accessed, a `ModuleNotFoundError` is raised. The third argument is an existing module object that will be the target of loading later. The import system passes in a target module only during reload.

## The first piece of magic

As the module doesn't know where the user has stored the `one_gaussian.clj` model and as we thought it would be too laborious for the user to enter the `absolute path`, we instead take advantage of the `sys.path` function. This function actually creates a list, where the list contains as a first element the path to the script in which the python interpreter is first invoked, in this instance the first element of `sys.path[0] =Users/bradley/Documents/Models/`. This path does not include the script name, just the directory to the script.
Using this technique enables us to tell the `finder`, through the `find_spec()` function, where to look for the module, the `.clj` file. 

Now within the `Clojure_Finder()` class, if our `finder` finds the module, that is `if os.path.exists(fullname +'.clj')` evaluates to `true`, then we are returning `ModuleSpec(<absolute_path_to_model>)` and an instantiation of the class `Clojure_Loader()` to the users original `python` script saved in `Users/bradley/Documents/Models/`. Where `ModuleSpec()` returns all the required information for the `loader` to load the module.

## The final piece of magic

When `import one_gaussian` is called it automatically invokes `Clojure_Loader()`, as this class inherits a `_loader` object. This then invokes the method `exec_module(self, module)`, where `module = ModuleSpec()`, which has an attribute `__name__`, the absolute path to the module. We then read in the `clojure` code within the `module`, our model, and pass this to the `compile_module()` function which generates `python` code from the `clojure` script, that fits the interface that we have designed. We could have returned the module before passing it through `compile_module()`, but in our use case we need to compile the `clojure` code. This is then returns the compiled code as the module. 

I hope this is useful for anyone from a non-coding background like me, as it is only recently that I have had to learn how to do things such as this.  If anything is unclear, or you see any typos, please feel free to e-mail me :-). 
