# CodonMoE

CodonMoE is a Python package that implements Adaptive Mixture of Codon Reformative Experts (CodonMoE) for mRNA analyses.

## Installation

You can install CodonMoE using below command:

```bash
# pip install codonmoe
python setup.py install
```

## API Reference

### CodonMoE

```python
CodonMoE(input_dim, num_experts=4, dropout_rate=0.1)
```

Parameters:
- `input_dim`: Dimension of the features from backbones
- `num_experts`: Number of expert networks in the Mixture of Experts
- `dropout_rate`: Dropout rate for regularization

### mRNAModel

```python
mRNAModel(base_model, codon_moe)
```

Parameters:
- `base_model`: The base model to be integrated with CodonMoE
- `codon_moe`: The Adaptive Mixture of Codon Reformative Experts

## API Tests

```bash
python -m unittest tests/test_codon_moe.py
```
