"""
Ce fichier contient les définitions des classes Réseau, Brindille et Branche.
Ces trois classes ont été développées pour analyser les réseaux de champignons
de Podospora anserina en définissant de manière unique un graphe dynamique pour
chaque expérience.
Pour plus de détail, se référer à l'article :
"Full identification of a growing and branching network's spatio-temporal 
structures", T. Chassereau, F. Chapeland-Leclerc et E. Herbert, 2024-25

Version réduite ici qui ne permet pas la reconstruction à partir des images mais 
permet de travailler avec les objets Reseau pour l'analyse ici souhaitée.
"""

import numpy as np #For everything
import matplotlib.pyplot as plt #For visualisation
import networkx as nx #For graph object
from tqdm import tqdm #For nice loading bar
import os  #For creating folder
import pickle #For saving object

class Reseau():
    """
    Définition de la classe Reseau.
    
    Chaque instance de cette classe regroupe l'ensemble des informations 
    nécessaires à l'analyse d'une expérience à savoir l'ensemble des images
    en niveaux de gris au cours du temps, le temps de début d'analyse et
    d'arrêt, les deux images binarisées correspondantes et le graphe spatial
    associé à la dernière image.
    Pour plus d'information sur la génération du graphe spatial, se 
    référer aux fichiers 'TotalReconstruction.py' et 'Vectorisation.py'.
    """
    #float|int : Length treshold between 2 nodes over which it is considered false
    SEUIL_LONG:float = 10 # in hyphal diameter unit
    #int : Seuil de "latence" avant classification en branche latérale
    SEUIL_LAT:int = 5 #frames
    #float|int : Seuil longueur départ de branche
    SEUIL_DEPART:float = 2 # in hyphal diameter unit
    #int : Seuil nombre de boucle lors du calcul de la dynamique 
    #      pour considérer un cas suspect.
    SEUIL_BOUCLE_DYNAMIQUE:int = 40

    """
    ===========================================================================
    Declaration et representation
    """
    def __init__(self,
                 name:str, #str, name of the experiment
                 g:nx.Graph, #Networkx graph of the experiment
                 imgs:dict[int,str], #{frame: img_path} Sequence of images
                 manip_dir:str, #str, Experiment's folder
                 output_dir:str, #str, Output folder
                 first, #First image Binarized
                 last, #Last image Binarized
                 start:int,end:int, #ints, idx of start, and end of experiment
                 brindilles:list = None,#Brindilles
                 branches:list = None,#Branches
                 diameter:float = 7#float, Hyphal diamater in pixels
                 ):
        """
        Définit l'initialisation d'une instance de Réseau.
        """
        self.name = name
        self.g = g
        self.imgs = imgs
        self.first = first
        self.last = last
        self.manip_dir = manip_dir
        self.output_dir = output_dir
        self.start = start
        self.end = end
        #Initialisation of some characteristic of the Reseau
        self.brindilles = brindilles if brindilles is not None else [] #List of twigs
        self.branches = branches if branches is not None else []#List of Branches
        self.source = None
        self.diameter = diameter
        self.n2x = nx.get_node_attributes(g,"x")
        self.n2y = nx.get_node_attributes(g,"y")
        self.n2t = {}
        #Initialisation des dossiers de sortie
        directories = ["","GraphesVisu",
                       "PotentialErrors","Binarization"]
        for directory in directories:
            if not os.path.exists(output_dir+directory):
                os.mkdir(output_dir+directory)

    def __repr__(self) -> str:
        """
        Définit ce qui s'affiche lorsque qu'on utilise 'print' avec le réseau
        comme argument.
        """
        repr = "\n"
        repr += "-"*80
        repr += f"\nReseau {self.name}\n"
        repr += f"\tStart frame {self.start}\n"
        repr += f"\tEnd frame {self.end}\n"
        repr += "-"*80
        repr += "\n"
        return repr

    """
    ===========================================================================
    Utilitaires
    """
    def save(self,suffix:str):
        """
        Save the network in gpickle format
        """
        file = self.output_dir+self.name+"_"+suffix+".gpickle"
        with open(file,"wb") as f:
            pickle.dump(self,f)
        return self
    
    def network_at(self, f):
        """
        Renvoie le sous réseau g_frame extrait du réseau entier self.g contenant
        uniquement les points vérifiant t<=f
        """
        g_frame = self.g.copy()
        toKeep = [n for n in g_frame.nodes if self.n2t[n]<=f]
        g_frame = g_frame.subgraph(max(nx.connected_components(g_frame.subgraph(toKeep)),
                                          key = len)).copy()
        return g_frame

    def image_at(self, f:int):
        """
        Return the image corresponding to frame f open as numpy array
        """
        image = np.asarray(Image.open(self.imgs.format(f)))
        return image
    

    def classification_nature_branches(self,threshold:float):
        """
        Classify each branch according to the latency before branching
        """
        for b in self.branches:
            b.nature = "Apical" if b.get_tstart()-b.t[0] <= threshold else "Lateral"
        return self

    def convert2txt(self,path2file:str)->nx.Graph: 
        """
        Convert the network to a txt file of the form:
        U,V,XU,YU,TU,XV,YV,TV,B
        With (U,V) the edges X_,Y_,T_ the coordinates of the point and B the 
        index of the corresponding branch.
        In the header, informations can be found about the source and its position.
        """
        g_prune = self.prune_d2()
        data = []
        for b in self.branches:
            noeuds= [n for n in b.noeuds if n in g_prune]
            for u,v in zip(noeuds[:-1],noeuds[1:]):
                data.append([u,v,self.n2x[u],self.n2y[u],self.n2t[u],self.n2x[v],self.n2y[v],self.n2t[v],b.index])
        data = np.array(data)
        np.savetxt(path2file,data,fmt="%d",
                   header=f"#Spore = {self.source} XS = {int(self.n2x[self.source])}, YS = {int(self.n2y[self.source])}\n"+
                           "#U,V,XU,YU,TU,XV,YV,TV,B",
                   delimiter=",")
        return self
    """
    ===========================================================================
    Visualisation
    """
    def show_at(self,t:float, ax)->None:
        """
        Draw the network at the instant 't' on the axe 'ax'
        """
        degree2size = {1:10,2:2,3:14}
        degree2color = {1:"cyan",2:"slategrey",3:"orangered"}
        subg = self.g.subgraph((n for n in self.g.nodes if self.n2t[n]<= t))
        nx.draw(subg,pos={n:(self.n2x[n],self.n2y[n])
                          for n in subg.nodes},
                ax=ax,
                node_size=[degree2size.get(d,20) 
                           for n,d in subg.degree],
                node_color=[degree2color.get(d,"red") 
                            for n,d in subg.degree])
        return None

    @property
    def times(self)->np.ndarray:
        """
        Return the instant from self.start to self.end
        """
        times = np.arange(self.start,self.end+1)
        return times
    
    @property
    def Nbranches(self)->np.ndarray:
        """
        Return Nbranches the total number of branches at time t 
        for t ranging from self.start to self.end
        """
        N_t0 = np.array([b.get_tstart() for b in self.branches])
        Nbranches = np.array([np.sum(N_t0<=t) for t in self.times])
        return Nbranches
    
    @property
    def total_length(self)->np.ndarray:
        """
        Return total_length the length of the reseau
        for t ranging from self.start to self.end
        """
        L_edges = np.sqrt([(self.n2x[u]-self.n2x[v])**2+
                        (self.n2y[u]-self.n2y[v])**2
                        for u,v in self.g.edges])
        t_edges = np.array([max(self.n2t[u],self.n2t[v]) 
                            for u,v in self.g.edges])
        total_length = np.array([np.sum(L_edges[np.where(t_edges<=t)])
                                 for t in self.times])
        return total_length


"""
===========================================================================
Brindilles
"""
class Brindille():
    """
    Définition de la classe des brindilles.
    """
    def __init__(self,
                index:int,
                noeuds:list[int],
                n2x:dict[int,float],
                n2y:dict[int,float],
                n2t:dict[int,float],
                inBranches:list = [],
                confiance:float = 0):
        self.index = index #index de cette brindille
        self.noeuds = noeuds 
        self.n2x = n2x
        self.n2y = n2y 
        self.n2t = n2t 
        self.inBranches = inBranches #liste des index des branches
                                        #contenant cette brindille
        self.confiance = confiance
    
    def __repr__(self) -> str:
        repr = f"Brindille {self.index} - {len(self.noeuds)} noeuds"
        return repr
    
    def abscisse_curviligne(self)->np.ndarray:
        """
        Renvoie la liste des abscisses curvilignes des noeuds de la branche
        """
        pos = np.array([[self.n2x[n],self.n2y[n]] for n in self.noeuds])
        abscisse = np.sqrt(np.sum((pos[1:,:]-pos[:-1,:])**2,axis=-1))
        abscisse = np.cumsum(abscisse)
        abscisse = np.insert(abscisse,0,0)
        return abscisse
    
    def get_tstart(self)->float:                                        
        """
        Renvoie la coordonnée t correspondant au début de la brindille.
        """
        tstart = self.n2t[self.noeuds[1]]
        return tstart

    def get_tend(self)->float:
        """
        Renvoie la coordonnée t correspondant à la fin de la brindille.
        """
        tt = [self.n2t[n] for n in self.noeuds]
        tend = np.max(tt)
        return tend

    def is_growing_at(self, t:float)->bool:
        """
        Renvoie si oui ou non la branche est en train de croître 
        à l'instant t passé en argument.
        """
        return self.get_tstart()<=t<=self.get_tend()
    
    def detection_latence(self, seuil:int = 4)->bool:
        """
        Départ avec latence -> True 
        Départ sans latence -> False
        """
        bLatence = bool(self.get_tstart()-self.n2t[self.noeuds[0]] < seuil)
        return bLatence
    
    def unit_vector(self, end, r = np.inf):
        """
        Calcule le vecteur unitaire au niveau de l'extrémité spécifiée 
        mesuré avec un rayon r.
        """
        abscisse = self.abscisse_curviligne()
        if end not in [self.noeuds[0],self.noeuds[-1]]:
            raise ValueError(f"{end} is not an end of twig {self.index}")
        if self.noeuds[0] == end:
            kstop = next((k for k,s in enumerate(abscisse) if s > r),-1)
            u1 = end
            u2 = self.noeuds[kstop]
        else:
            abscisse = reversed(abscisse - abscisse[-1])
            kstop = next((len(self.noeuds)-k-1 for k,s in enumerate(abscisse) 
                        if s < -r),0)
            u1 = self.noeuds[kstop]
            u2 = end
        vect = np.array([self.n2x[u2]-self.n2x[u1],
                            self.n2y[u2]-self.n2y[u1]])
        unit_vect = vect/np.linalg.norm(vect)
        return unit_vect
    
    def reverse(self):
        """
        Reverse the twig.
        """
        self.noeuds.reverse()
        self.confiance = - self.confiance
        return self
    
    def calcul_confiance(self, seuil:float)->float:
        """
        Calcule la confiance dans l'orientation de la brindille
        """
        tt = np.array([self.n2t[n] for n in self.noeuds])
        dtt = tt[1:]-tt[:-1]
        #Incrément
        dtt = np.where(dtt>0, 1, dtt)
        #Décrément
        dtt = np.where(dtt<0,-1, dtt)
        #dtt = np.where(np.abs(dtt) != 1, 0, dtt)
        nP = np.sum(np.abs(dtt)+dtt)//2
        nM = np.sum(np.abs(dtt)-dtt)//2
        self.confiance = (nP-nM)/(nP+nM)*np.tanh((nP+nM)/seuil) if nP+nM else 0.
        return self.confiance
    
    def get_apex_at(self, t:float, index:bool = False)->int:
        """
        Return the current 'apex' on the twig|branch at time t
        if t>tEnd then the apex is the last node of the twig|branch
        if index == True then return also the index of apex in b.noeuds 
        """
        tt = [self.n2t[n] for n in self.noeuds]
        if self.get_tstart()>t:
            raise ValueError("La branche n'est pas encore apparu à ce moment.")
        iapex = next((i for i,ti in enumerate(tt[1:])
                    if ti > t),
                    len(tt)-1)
        apex = self.noeuds[iapex]
        if index:
            return (iapex,apex)
        return apex

    def get_all_apex(self)->tuple[list[int],list[int]]:
        """
        Return all the successive apex at the different growth time
        """
        tstart = int(self.get_tstart())
        tend = int(self.get_tend())
        time = list(range(tstart,tend+1))
        tt = self.t
        nnoeuds = len(self.noeuds)
        ntime = len(time)
        apex = []
        i=0
        it = 0
        while i < nnoeuds-1:
            if tt[i+1]>time[it]:
                apex.append(self.noeuds[i])
                it += 1
            else:
                i += 1
        time = list(range(tstart,tend+1))
        apex.append(self.noeuds[-1])
        while len(apex)<ntime:#Strange, should only occur with branches/twigs made of 2 nodes
            apex.append(self.noeuds[-1])#It's a fix but not a good one...
        return time, apex
    
    @property
    def x(self)->list[float]:
        """
        Return the list of x coordinate of each node in twig.nodes
        """
        return [self.n2x[n] for n in self.noeuds]
    
    @property
    def y(self)->list[float]:
        """
        Return the list of y coordinate of each node in twig.nodes
        """
        return [self.n2y[n] for n in self.noeuds]
    
    @property
    def t(self)->list[float]:
        """
        Return the list of t coordinate of each node in twig.nodes
        """
        return [self.n2t[n] for n in self.noeuds]
    
    @property
    def s(self)->np.ndarray:
        """
        Return the list of s, arc length of each node in twig.nodes
        """
        return self.abscisse_curviligne()

    @property
    def theta(self,radius:float = 7)->np.ndarray:
        """
        Return the list of theta, the orientation of the twig.
        Orientation is absolute.
        """
        x = np.array(self.x)
        y = np.array(self.y)
        thetas = np.zeros_like(x)
        #Cas général
        radius_squared = radius*radius
        for i,n in enumerate(self.noeuds):
            x0,y0 = x[i],y[i]
            r2 = (x-x0)*(x-x0)+(y-y0)*(y-y0)
            im = next((i-j for j,rsq in enumerate(r2[i::-1])if rsq>radius_squared ),
                    0)
            ip = next((j for j,rsq in enumerate(r2[i:],start=i) if rsq>radius_squared),
                      -1)
            im = next((i-j for j,rsq in enumerate(r2[i::-1])if rsq>radius_squared ),
                      0)
            thetas[i] = np.arctan2(y[ip]-y[im],x[ip]-x[im])
        return np.unwrap(thetas)
    
    def tangent(self,noeud)->np.ndarray:
        """
        Return the unit tangent vector of the twig at node 'noeud'.
        """
        if noeud not in self.noeuds:
            raise ValueError(f"{noeud} is not in {self}.\nThe argument 'noeud' must be inside twig/branch.noeuds")
        n2i = {n:i for i,n in enumerate(self.noeuds)}
        i = n2i[noeud]
        ip = min(i+1,len(self.noeuds)-1)
        im = max(i-1,0)
        dx = self.x[ip]-self.x[im]
        dy = self.y[ip]-self.y[im]
        direction = np.array([dx,dy])/np.sqrt(dx*dx+dy*dy)
        return direction

"""
===========================================================================
Branches
"""
class Branche(Brindille):
    """
    Définition de la classe secondaire des branches.
    Hérite de la classe secondaire des brindilles.
    """
    def __init__(self,
                    index:int, #int, index de la branche
                    noeuds:list[int], #Liste des noeuds de la branche
                    n2x:dict[int,float],
                    n2y:dict[int,float],
                    n2t:dict[int,float],
                    brindilles:list[int], #Liste des index des brindilles
                    nature:str, #nature de la branche : Apical,Lateral,Initial,...
                    ending:str, #raison de l'arret de croissance : d1, Fusion ?
                    list_overlap:list = None #Liste des chevauchements
                    ):
        self.index = index
        self.noeuds = noeuds
        self.n2x = n2x
        self.n2y = n2y
        self.n2t = n2t
        self.brindilles = brindilles
        self.nature = nature
        self.ending = ending
        self.list_overlap = list_overlap if list_overlap else []
    
    def __repr__(self) -> str:
        repr = f"Branche {self.index} - {len(self.noeuds)} noeuds"
        return repr

    def grow(self,brindille):
        """
        Prolongation de la branche par la brindille.
        """
        self.noeuds = [*self.noeuds,*brindille.noeuds[1:]]
        self.brindilles.append(brindille.index)
        return self

    def normes_vitesses(self, seuil_lat = 4):
        """
        Calcule et renvoie les normes des vitesses de la branche et la liste des
        instants correspondant.
        seuil_lat permet de ne pas prendre en compte les vitesses lentes
        due à une latence.
        Renvoie ([],[]) s'il n'est pas possible de définir une vitesse sur la branche
        """
        times,vitesses = self.vecteurs_vitesses(seuil_lat = seuil_lat)
        vs = []
        if len(vitesses):
            vs = np.sqrt(np.sum(vitesses*vitesses,axis=1))
        return times,vs

    def vecteurs_vitesses(self, seuil_lat = 4):
        """
        Calcule les vecteurs vitesses de croissance de la branche.
        Renvoie sous la forme d'un tuple la liste des instants t correspondant 
        et la liste des vecteurs vitesses.
        Renvoie [] s'il n'est pas possible de définir une vitesse sur la 
        branche.
        """
        vecteursV = []
        times = []
        pos = np.array([[self.n2x[n],self.n2y[n]] for n in self.noeuds])
        abscisse = self.s
        tt = np.array(self.t)
        index = [i for i,t in enumerate(tt[:-1]) if t != tt[i+1]]
        #index : liste des index où il y a variation de t
        if index:
            index.append(len(tt)-1)
            for i0,i in zip(index[:-1],index[1:]):
                deltaT = tt[i]-tt[i0]
                deltaS = abscisse[i]-abscisse[i0]
                deltaPos = pos[i,:]-pos[i0,:]
                if 0<deltaT<seuil_lat:
                    normedPos = np.linalg.norm(deltaPos)
                    direction = deltaPos/normedPos if normedPos else deltaPos
                    vecteursV.append(deltaS/deltaT*direction)
                    times.append((tt[i0]+tt[i])*.5)
        times = np.array(times)
        vecteursV = np.array(vecteursV)
        return times,vecteursV

    def positions_vitesses(self):     
        """
        Renvoie la liste des vecteurs vitesses et des positions 
        correspondantes de la branche.
        """
        dX,dY,dT = self.n2x,self.n2y,self.n2t
        pos = np.array([[dX[n],dY[n]] for n in self.noeuds])
        tstart = int(self.get_tstart())
        tend = int(self.get_tend())
        _,apex = self.get_all_apex()
        apex.insert(0,self.noeuds[0])
        temps = np.array([t-0.5 for t in range(tstart,tend+1)])
        if len(apex)<2:
            return np.empty(shape=(1,3)),np.empty(shape=(1,2))
        positions = np.zeros(shape=(len(apex)-1,3))
        vecteursV = np.zeros(shape=(len(apex)-1,2))
        positions[:,2] = np.arange(tstart,tend+1)-.5
        abscisse = self.abscisse_curviligne()
        n2i = {n:i for i,n in enumerate(self.noeuds)}
        positions_apex = np.array([pos[n2i[a]] for a in apex])
        positions[:,:2] = (positions_apex[:-1]+positions_apex[1:])*.5
        vecteursV = (positions_apex[1:]-positions_apex[:-1])
        cart_normes = np.linalg.norm(vecteursV,axis=1)
        curv_normes = np.array([abscisse[n2i[a1]]-abscisse[n2i[a2]]
                                for a1,a2 in zip(apex[1:],apex[:-1])])
        filtre = np.where(cart_normes>0)
        vecteursV[filtre,0] *= curv_normes[filtre]/cart_normes[filtre]
        vecteursV[filtre,1] *= curv_normes[filtre]/cart_normes[filtre]
        #Moyenne roulante 
        #positions = (positions[1:,:]+positions[:-1,:])/2
        #vecteursV = (vecteursV[1:,:]+vecteursV[:-1,:])/2
        return positions,vecteursV
    
    def apex(self)->list[int]:
        """
        Renvoie la liste des apex successifs de la branche
        """
        temps = [self.n2t[n] for n in self.noeuds]
        apex = [self.noeuds[i] for i in range(1,len(self.noeuds)-1) if temps[i] < temps[i+1]]
        apex.append(self.noeuds[-1])
        return apex

