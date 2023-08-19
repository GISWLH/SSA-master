Codes to fill data gaps in GRACE&GFO observations, applicable for either spherical harmonic coefficients or gridded observations.

The code consists of two parts. “SSA-filling-a”, which only fills the gaps with id==3, needs two parameters, MM and KK. “SSA-filling-b”, which fills the residual gaps (id==4), also needs two parameter ranges, Mlist and Klist, and the optimal parameters will be searched within these ranges. If the readers want to apply the code to their own data, please refer to "example_C30.mat” for the format of the input series. If the dataset has been sampled uniformly (the data gaps are filled by NaN) and there is only one kind of data gaps, then only “SSA-filling-b” is sufficient. The users are advised to refer to the paper for the detailed explanation. To run the example, see the master program 'main_SSA_gap_filling.m'.

Reference:
"Filling the data gaps within GRACE missions using Singular Spectrum Analysis"
Journal of Geophysical Research: Solid earth
Shuang Yi, Nico Sneeuw
https://doi.org/10.1029/2020JB021227
---
Shuang Yi, shuangyi.geo@gmail.com, 05/12/2021
