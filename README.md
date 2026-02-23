# Hyphal Network: Apex-Hypha interactions

This github repository is linked to the article *'Direct Evidence of Apex–Hypha Interactions During Vegetative Growth of Fungal Thallus via Comprehensive Network and Trajectory Extraction'*.

The aim of this repository is to bring together the code (in Python 3) and files needed to analyse the interactions between apices and hyphae within the mycelial network of *Podospora anserina*.

This repository contains 2 python code files and an `Experiments` folder containing the *Network* objects corresponding to the different experiments analysed.
The data for each network is in the form of two files. A “name_coordinate.txt” file and a “name_branches.txt” file. 
The first lists each node, its identifier, and its coordinates (spatial and temporal) in the form (n,x,y,t) where n is the node identifier, x and y the spatial coordinate of the node and t its temporal coordinate. 
The second lists each link in the network, also indicating the branch to which the link belongs in the form (b,u,v), where u and v are two node identifiers and b is a branch identifier.

The first python code, named `Reseau.py` contain the definition of the Reseau and Branche classes previously described [in](https://www.sciencedirect.com/science/article/pii/S0006349524040633) *"Full identification of a growing and branching network's spatio-temporal structures"*, T. Chassereau, F. Chapeland-Leclerc et E. Herbert, 2025, DOI:10.1016/j.bpj.2024.12.002
The second python code, named `ApexHyphaInteractions.ipynb` contain the new analysis made here.
