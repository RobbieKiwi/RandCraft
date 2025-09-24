# RandCraft

RandCraft is a Python library for object-oriented combination and manipulation of univariate random variables, built on top of the [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html) module.


## Usage Example
Have you ever wanted to add together random variables but can't be bothered working out an analytical solution?
Randcraft makes it simple.

```python
from randcraft import make_normal, make_coin_flip

coin_flip = make_coin_flip()
# <RandomVariable(discrete): mean=0.5, var=0.25>
norm = make_normal(mean=0, std_dev=0.2)
# <RandomVariable(normal): mean=0.0, var=0.04>
combined = coin_flip + norm 
# <RandomVariable(mixture): mean=0.5, var=0.29>
combined.sample_one()
# 0.8678903828104276
combined.plot()
```
![Double normal](images/double_normal.png)


## Features

- **Object-oriented random variables:** Wrap and combine distributions as Python objects.
- **Distribution composition:** Add, multiply, and transform random variables.
- **Sampling and statistics:** Easily sample from composed distributions and compute statistics.
- **Extensible:** Supports custom distributions via subclassing.
- **Integration with scipy.stats:** Use any frozen continuous distribution from scipy stats

## Supported Distributions

RandCraft currently supports the following distributions:

- Normal, Uniform, Beta, Gamma + any other parametric continuous distribution from scipy.stats
- Discrete
- DiracDelta
- Gaussian kde distribution from provided observations
- Mixture distributions
- Anonymous distribution functions based on a provided sampler function

Distributions can all be combined arbitrarily with addition and subtraction.
The library will simplify the new distribution analytically where possible, and use numerical approaches otherwise.

You can also extend RandCraft with your own custom distributions.

## Installation

```bash
pip install randcraft
```

## API Overview

- `make_normal`, `make_uniform` ...etc: Create a random variable
- Addition subtraction with constants or other RVs: `+`, `-`
- `.sample_numpy(size)`: Draw samples
- `.get_mean()`, `.get_variance()`: Get statistics
- `.cdf(x)`: Evaluate cdf at points
- `.ppf(x)`: Evaluate inverse of cdf at points
- `.plot()`: Take a look at your distribution

## More Examples
### Combining dice rolls
```python
from randcraft.constructors import make_die_roll

die = make_die_roll(sides=6)
# <RandomVariable(discrete): mean=3.5, var=2.92>
three_dice = die.multi_sample(3)
# <RandomVariable(discrete): mean=10.5, var=8.75>
three_dice.cdf(10.0)
# 0.5
three_dice.ppf(0.5)
# 10.0
```

### Using arbitrary parametric continuous distribution from scipy.stats
```python
from scipy.stats import uniform
from randcraft.constructors import make_scipy

rv = make_scipy(uniform, loc=1, scale=2)
# <RandomVariable(scipy-uniform): mean=2.0, var=0.333>
b = rv.scale(2.0)
# <RandomVariable(scipy-uniform): mean=4.0, var=1.33>
```

### Kernel density estimation and combination
You have observations of two independent random variables. You want to use kernal density estimation to create continuous random variables for each and then add them together.
```python
import numpy as np
from randcraft.observations import make_gaussian_kde

observations_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
observations_b = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
rv_a = make_gaussian_kde(observations=observations_a, bw_method=0.1)
# <RandomVariable(multi): mean=3.0, var=2.42>
rv_b = make_gaussian_kde(observations=observations_b)
# <RandomVariable(multi): mean=0.5, var=0.676>
rv_joined = rv_a + rv_b
# <RandomVariable(multi): mean=3.5, var=3.1>
```
Uses `gaussian_kde` by `scipy.stats` under the hood. You also have the option to pass arguments for `gaussian_kde`, or provide your own kernel as a `RandomVariable`.

### Mixing continuous and discrete variables
You have observations of two independent random variables. You want to use kernal density estimation to create continuous random variables for each and then add them together.
```python
from randcraft.constructors import make_normal, make_uniform, make_discrete
from randcraft.misc import mix_rvs

rv1 = make_normal(mean=0, std_dev=1)
# <RandomVariable(scipy-norm): mean=0.0, var=1.0>
rv2 = make_uniform(low=-1, high=1)
# <RandomVariable(scipy-uniform): mean=-0.0, var=0.333>
combined = rv1 + rv2
# <RandomVariable(multi): mean=0.0, var=1.33>
discrete = make_discrete(values=[1, 2, 3])
# <RandomVariable(discrete): mean=2.0, var=0.667>

# Make a new rv which has a random chance of drawing from one of the other 4 rvs
mixture = mix_rvs([rv1, rv2, combined, discrete])
# <RandomVariable(mixture): mean=0.5, var=1.58>
mixture.plot()
```
![Mixture](images/mixture.png)



## Extending RandCraft

You can create custom random variable classes by subclassing the base RV class and implementing required methods.

## Known limitations

The library is designed to work with univariate random variables only. Multi-dimensional rvs or correlations etc are not supported.

## License

MIT License

## Acknowledgements

Built on [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html).