# RandCraft

RandCraft is a Python library for object-oriented combination and manipulation of univariate random variables, built on top of the [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html) module.

## Features

- **Object-oriented random variables:** Wrap and combine distributions as Python objects.
- **Distribution composition:** Add, multiply, and transform random variables.
- **Sampling and statistics:** Easily sample from composed distributions and compute statistics.
- **Extensible:** Supports custom distributions via subclassing.
- **Integration with scipy.stats:** Leverages the full power of scipy's probability distributions.

## Supported Distributions

RandCraft currently supports distributions available in `scipy.stats`, including:

- Normal (`norm`)
- Uniform (`uniform`)
- ...more coming soon!

Also included are:
- Anon: Anonymous distribution function based on a provided sampler function
- Discrete
- DiracDelta

Distributions can all be combined arbitrarily with addition and subtraction.
The library will simplify the new distribution analytically where possible, and use numerical approaches otherwise.

Mixture distributions are also supported.

You can also extend RandCraft with your own custom distributions.

## Installation

```bash
pip install randcraft
```

## Usage Example

```python
from randcraft import make_normal

# Create two random variables
rv1 = make_normal(mean=10.0, std_dev=10.0)
rv2 = make_normal(mean=20.0, std_dev=10.0)

# Combine them (e.g., sum)
rv_sum = rv1 + rv2
rv_sum.get_mean()
# 30.0
rv_sum.get_variance()
# 200.0

# Sample from the combined distribution
samples = rv_sum.sample_one()
# 31.2541231
```

## API Overview

- `make_normal`, `make_uniform` ...etc: Create a random variable.
- Arithmetic operation on individual RVs with constants: `+`, `-`, `*`, `/`, `**`
- Arithmetic operations: `+`, `-` between RVs.
- `.sample_numpy(size)`: Draw samples.
- `.get_mean()`, `.get_variance()`: Get statistics.
- `.get_chance_that_rv_is_le(x)`: Evaluate cdf at point
- `.get_value_that_is_at_le_chance(x)`: Evaluate inverse cdf at point

## Extending RandCraft

You can create custom random variable classes by subclassing the base RV class and implementing required methods.

## License

MIT License

## Acknowledgements

Built on [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html).