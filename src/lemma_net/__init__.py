"""LEMMA's network (Guo et al., IJCAI 2023), copied from github.com/csguoh/LEMMA at commit
1240ef3a61fb594b87b70822526493e1d42b3f90 under its Apache-2.0 licence (LICENSE.txt).

Only the files the network needs are here, in their own package so their module names cannot clash
with ours (both projects have `model`, `dataset`, `utils`...). Changes: import lines in lemma.py,
Position_aware_module.py and backbone.py; CharsetMapper taken out of LEMMA's setup.py into
charset.py; line endings. The network code itself is unchanged. Use it through lemma_teacher.py.
"""
