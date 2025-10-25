"""A crystal graph coordinator for CGNN."""
#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import argparse
import pandas as pd
#import pbc
#from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from sklearn.cluster import KMeans
import sys
import os
import glob
from data import crystal_Data
import tqdm
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius

import torch
from joblib import Parallel, delayed

import warnings

CC_CUTOFF = 0.03
RADIUS_FACTOR = 3
radius=10
max_num_nbr=8
coef=3
parser = argparse.ArgumentParser(description='Crystal Graph')
parser.add_argument('--data_options', type=str,default='/home/zww/ZWW/program/cgnn-attention_symmetric/tools', metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
parser.add_argument('--radius', type=float, default=8.5,help='the cutoff radius')
args=parser.parse_args()


def get_structure(m):
    if m['nsites'] == len(m['structure']):
        return m['structure']
    else:
        s = m['structure'].get_primitive_structure()
        for _ in range(10):
            if m['nsites'] == len(s):
                return s
            else:
                s = s.get_primitive_structure()
        raise NameError('The primitive structure could not be got for {}'.format(m['material_id']))

def load_materials(filepath):
    try:
        data = np.load(filepath,allow_pickle=True)['materials']
    except UnicodeError:
        data = np.load(filepath,encoding='latin1',allow_pickle=True)['materials']
    return data

#def get_all_neighors(radius,include_index=True)

def process(data_path):
    element_embedding=crystal_Data(args.data_options, args.radius)
    materials = load_materials(data_path)
    material_ids = [m['material_id'] for m in materials]
    structures = [get_structure(m) for m in materials]
    data_ac = []
    data_nbr = []
    feature_nbr=[]
    data_image=[]
    atom_embeding=[]
    remove_material = []
    for i, geom in enumerate(structures):
        geom_structure=Structure(lattice=geom.lattice,
                                 species=geom.species,
                                 coords=geom.frac_coords,
                                 coords_are_cartesian=False,
                                 to_unit_cell=True,
                                 charge=None)
        # atom_fea = np.vstack([element_embedding.ari.get_atom_fea(geom_structure[i].specie.number)
        #                       for i in range(len(geom_structure))])#get the every element embedding in primitive cell
        cutoffs = RADIUS_FACTOR*sum(filter(None, [geom_structure[i].specie.atomic_radius for i in range(len(geom_structure))]))/len(geom_structure)
        #all_nbrs=geom_structure.get_all_neighbors(r=cutoffs, include_index=True)
        all_nbrs = geom_structure.get_symmetric_neighbor_list(r=cutoffs, sg=None)

        #all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_idx=[]
        nbr_fea=[]
        nbr_iamge=[]
        for nbr in all_nbrs:
            nbr_idx.append(list(map(lambda x: x[2], nbr)))
            nbr_fea.append(list(map(lambda x: x[1], nbr)))
            nbr_iamge.append(list(map(lambda x: x[3], nbr)))

        nbr_fea = np.concatenate(nbr_fea)
        is_non_graph=False
        if len(nbr_fea) == 0:
            # warnings.warn('{} not find enough neighbors to build graph. '
            #               'If it happens frequently, consider increase '
            #               'radius.'.format(material_ids[i]))
            remove_material.append(material_ids[i])
            is_non_graph = True
        if is_non_graph:
            continue
        else:
            nbr_fea = element_embedding.gdf.expand(nbr_fea).astype('float32')
            data_ac.append(geom.atomic_numbers)
            data_nbr.append(nbr_idx)
            feature_nbr.append(nbr_fea)
            data_image.append(nbr_iamge)
            #atom_embeding.append(atom_fea)
        #data_ac, data_nbr, feature_nbr, atom_embeding=np.array(data_ac), np.array(data_nbr), np.array(feature_nbr), np.array(atom_embeding)
        '''
        material_ids: the index of material
        data_ac: the atomic numbers of meterial
        feature_nbr: the distance between the current node and neighor nodes
        data_nbr:
        '''
    #nbr_fea_mean=np.mean(nbr_fea)
    material_ids = [j for j in material_ids if j not in remove_material]
    graph_path = data_path.replace('mp_data', 'mp_graph_symmetric')#
    np.savez_compressed(graph_path, graph_names=material_ids,
                        graph_nodes=data_ac, graph_edges=data_nbr,distance_nbr=feature_nbr, graph_image=data_image)

def main(data_dir, num_cpus):
    if not os.path.isdir(data_dir):
        print('Not found the data directory: {}'.format(data_dir))
        exit(1)

    #data_files = sorted(glob.glob(os.path.join(data_dir, 'mp_data_*.npz')))
    #data_files=data_files[0]
    # for path in data_files:
    #     process(path)
    path ='/home/zww/ZWW/program/cgnn-attention_symmetric/OQMD/mp_data_646683.npz'
    process(path)
    #
    # path ='/data/zww/data/mp_data_000.npz'
    # process(path)
    #Parallel(n_jobs=num_cpus, verbose=10)([delayed(process)(path) for path in data_files])
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Crystal Graph Coordinator.')
    parser.add_argument('--data_dir', metavar='PATH', type=str, default='/home/zww/ZWW/program/cgnn-attention_symmetric/OQMD',
                        help='The path to a data directory (default: data)')
    parser.add_argument('--num_cpus', metavar='N', type=int, default=8,
                        help='The number of CPUs used for processing (default: -1)')
    options = vars(parser.parse_args())

    main(**options)