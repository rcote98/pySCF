#!/usr/bin/env python3

# pySCF.
# Performs closed-shell Hartree-Fock SCF calculations using s-type orbitals only.
#
# Author:   Raúl Coterillo (raulcote98@gmai.com)
# Version:  November 2021

from math import erf, floor
import numpy as np
import argparse
import sys, os
import time

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def main() -> None:

    """ Main program loop. """

    # parse command line arguments
    p = argparse.ArgumentParser(
        prog="pySCF.py",
        description="Performs a closed-shell Hartree-Fock SCF calculation using s-type orbitals only.",
        epilog="That's all, folks!"
        )
    p.add_argument("input_file", type=str, nargs=1, action="store",
                help="input file for the program")
    p.add_argument("-r", "-R", "--read-integrals", dest="read_ints", action="store_true",
                    help="read integrals from the input file (default: calculate integrals instead)")
    p.add_argument("-w", "-W", "--write-integrals", dest="write_ints", action="store_true",
                    help="write integrals to a file called 'integrals.txt' (default: do not write)")
    p.add_argument("-s", "-S", "-SCF", "--write-SCF", dest="write_SCF", action="store_true",
                    help="write SCF matrices to a file called 'SCF.txt' (default: do not write)")
    p.add_argument("-b", "-B", "--external-basis", dest="basis", type=str, nargs=1, action="store", default=None,
                    help="reads basis from Gaussian basis file (.gbs file from basissetexchange.org)")
    p.add_argument("-i", "-I", "--max-iterations", dest="maxits", type=int, nargs=1, action="store", default=[100],
                    help="sets the maximum number of SCF cycles (default: 100)")
    p.add_argument("-e", "-E", "--energy-threshold", dest="Ethresh", type=float, nargs=1, action="store", default=[1e-8],
                    help="sets the energy convergence threshold (default: 1E-8 Ha)")
    p.add_argument("-p", "-P", "--density-threshold", dest="Pthresh", type=float, nargs=1, action="store", default=[1e-8],
                    help="sets the density matrix convergence threshold (default: 1E-8)")
    args = p.parse_args()

    # print the program header
    HF.print_header()                 

    # format the command line arguments
    INPUT_FILE      = args.input_file[0]
    if args.basis is not None: 
        BASIS_FILE  = args.basis[0]
    else:
        BASIS_FILE  = None

    MAXITS          = args.maxits[0]        # max number of SCF iterations
    ETHRESH         = args.Ethresh[0]       # energy convergence threshold
    PTHRESH         = args.Pthresh[0]       # density convergence treshold

    READ_INTEGRALS  = args.read_ints        # read integrals from input?
    WRITE_INTEGRALS = args.write_ints       # write integrals to file?
    WRITE_SCF       = args.write_SCF        # write SCF matrices to file?

    # check if the input files exist
    if not os.path.isfile(INPUT_FILE):
        print(f"ERROR: Input file '{INPUT_FILE}' does not exist.")
        sys.exit()
    if BASIS_FILE is not None and not os.path.isfile(BASIS_FILE):
        print(f"ERROR: Basis file '{BASIS_FILE}' does not exist.")
        sys.exit()

    # initialize the program (read input information)
    sim = HF(INPUT_FILE, basis_file=BASIS_FILE)

    # print the information from the input file
    sim.print_sim_info()           

    # run scf calculation
    sim.run_single_point(
                        max_iter=MAXITS,
                        e_thresh=ETHRESH,
                        p_thresh=PTHRESH,
                        read_integrals=READ_INTEGRALS,
                        write_integrals=WRITE_INTEGRALS,
                        write_SCF=WRITE_SCF
                        )         

    # print timer summary
    sim.tm.print_summary()                  

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class HF():

    """
    A class to perform Hartree-Fock calculations.

    Attributes
    ----------
    input_file : str
        input file of the simulation

    basis_file : str        
        Gaussian external basis set file 

    tm : Timer
        internal class that stores all execution time information  

    nats : int
        number of atoms

    atnums : np.array[nats]
        atomic numbers

    labels : list[nats]
        atomic labels

    positions : np.array[nats,3]
        atomic positions (in Angstrom)

    charge : int
        overall system charge

    nels : int
        number of electrons

    norbs : int 
        number of basis functions

    max_nprim : int
        max number of primitives per basis function

    basis_centers : nd.array[norbs]
        atomic centers of the basis functions

    basis_nprims : nd.array[norbs]
        number of primitives per basis function

    basis_values : nd.array[norbs, max_nprim, 2]
        exponents and contraction coefficients of the primitives

    S : np.array[norbs, norbs]
        overlap matrix

    T : np.array[norbs, norbs]
        kinetic energy integral matrix
    
    V : np.array[norbs, norbs]
        potential energy integral matrix
    
    O : np.array[norbs, norbs, norbs, norbs]
        two-electron integral matrix

    Methods
    -------
    __init__(input_file:str, basis_file:str=None) -> None:
        Initializes the class and parses all the input file information.
    
    print_header(textwidth=TEXTW, textshift=TEXTS) -> None:
        prints the program name and authors

    print_sim_info(self, textwidth:int=TEXTW, textshift=TEXTS) -> None:
        Prints a summary of the information in the input file.

    bohr_geometry(self) -> np.array:
        returns the system's geometry (self.position) in Bohr

    distance_bohr(self, i:int, j:int) -> float:
        return the distance between two atoms, in Bohr

    distance_angstrom(self, i:int, j:int) -> float:
        return the distance between two atoms, in Angstrom

    read_integrals(self) -> list:
        reads the integrals from the input file

    calc_integrals(self) -> list:
        calculate the integrals using the basis set information

    write_integrals(self, fname:str, prec=16) -> None:
        writes the integrals to an external file

    nuclear_repulsion(self) -> float:
        calculates the inter-nuclear repulsion energy

    SCF_setup(self) -> None:
        creates the transformation matrix and a guess for the density matrix

    SCF_step(self) -> None:    
        performs an self-consistent field step

    print_orbitals(self, textwidth:int=TEXTW, textshift:int= TEXTS,) -> None:
        print orbital energies and coefficients 
    
    print_population(self, textwidth:int=TEXTW, textshift:int= TEXTS,) -> None:
        prints a Mulliken/Lowdin population analysis of the orbitals

    write_SCF(self, fname:str, prec=16) -> None:
        writes the SCF matrices (X, P, Hc, F, etc) to an external file

    run_single_point(self,  max_iter:int        = 100,
                            e_thresh:float      = 1e-8,
                            p_thresh:float      = 1e-8,
                            textwidth:int       = TEXTW,
                            textshift:int       = TEXTS,
                            read_integrals:bool = False,
                            write_integrals:bool= False,
                            write_SCF:bool      = False) -> None:
        
        performs a complete single point calculation.
    
    Classes
    -------
    Timer()
        registers the execution time of the different functions
    
    """

    # default printing parameters
    TEXTW = 72  # text width
    TEXTS = 4   # text shift

    def __init__(self, input_file:str, basis_file:str=None) -> None:

        """
        Initializes the class and parses all the input file information.

        Parameters
        ----------
            input_file : str
                input file containing all the SCF information

            input_file : str
                input file containing all the SCF information
        """

        
        self.tm = self.Timer(self.TEXTW, self.TEXTS)
        self.input_file = input_file
        self.basis_file = basis_file

        # parse input file
        with open(input_file, "r") as f:

            self.tm.start("read_input")

            # read number of atoms
            f.readline() # number of atoms
            self.nats = int(f.readline().split()[0])

            # read geometry
            f.readline() # Atom labels, atom number, coords (Angstrom)

            self.labels     = list(np.zeros(self.nats)) 
            self.atnums     = np.zeros((self.nats), dtype=int)
            self.positions  = np.zeros((self.nats, 3), dtype=float)

            for at in range(self.nats):
                line = f.readline().split()
                self.labels[at]         = line[0]
                self.atnums[at]         = int(line[1])
                self.positions[at,:]    = [float(x) for x in line[2:]]

            # read charge
            f.readline() # Overall charge
            self.charge = int(f.readline().split()[0])
            self.nels   = np.sum(self.atnums) - self.charge

            if self.nels%2 != 0:
                print("WARNING")

            if basis_file is None:

                # read number of basis functions
                f.readline() # Number of basis funcs
                self.norbs  = int(f.readline().split()[0])

                # read max number of primitives
                f.readline() # Maximum number of primitives
                self.max_nprim  = int(f.readline().split()[0])

                # read basis
                f.readline() # Basis set: Func no, At label, Z, Atom no //  nprim // (zeta cjk)

                self.basis_centers = np.zeros(self.norbs, dtype=int)
                self.basis_nprims  = np.zeros(self.norbs, dtype=int)
                self.basis_values  = np.zeros((self.norbs, self.max_nprim, 2), dtype=float)

                for orb in range(self.norbs):                                   # iterate over basis functions
                    
                    line                     = f.readline().split()             # read line
                    self.basis_centers[orb]  = int(line[3]) - 1                 # atomic center
                    self.basis_nprims[orb]   = int(f.readline().split()[0])     # number of primitives

                    for p in range(self.basis_nprims[orb]):
                        self.basis_values[orb, p, :] = [float(x) for x in f.readline().split()[:2]]

            else:

                # read basis info
                with open(basis_file, "r") as b:
                    
                    for _ in range(12): # skip the file header
                        b.readline()

                    basis = {}
                    while True:

                        line = b.readline()

                        # check if EOF has been reached
                        if line == "":
                            break


                        norb  = 0
                        label, _ = line.split()

                        basis[label] = {} 
                        end_of_atom  = False

                        # read atom info
                        while not end_of_atom:

                            line  = b.readline()

                            # check if end of atom is reached                            
                            if "*" in line:              
                                end_of_atom = True
                            
                            # otherwise, read a new orbital
                            else:

                                # read orbital information
                                basis[label][norb] = []    
                                orb_type, nprim, _ = line.split()
                                
                                # if the orbital is s-type, keep it
                                if orb_type == "S":

                                    # read the primitives for the orbital    
                                    for p in range(int(nprim)):
                                        data = [float(x.replace('D', 'E')) for x in b.readline().split()]
                                        basis[label][norb].append(data)
                                
                                # if it is not s-type, discard it
                                else:
                                    for p in range(int(nprim)):
                                        b.readline()
                                    basis[label].pop(norb)

                                norb += 1

                # adapt to the original data structures

                self.norbs      = 0
                self.max_nprim  = 0

                for at in self.labels:
                    self.norbs += len(basis[at])
                    for orb in basis[at].keys():
                        if len(basis[at][orb]) > self.max_nprim:
                            self.max_nprim = len(basis[at][orb])

                self.basis_centers = np.zeros(self.norbs, dtype=int)
                self.basis_nprims  = np.zeros(self.norbs, dtype=int)
                self.basis_values  = np.zeros((self.norbs, self.max_nprim, 2), dtype=float)

                orb_index = 0
                for at in range(self.nats):
                    lab = self.labels[at]
                    for orb in basis[lab].keys():

                        self.basis_centers[orb_index]   = at
                        self.basis_nprims[orb_index]    = len(basis[lab][orb])

                        for p in range(self.basis_nprims[orb_index]):
                            
                            self.basis_values[orb_index, p, 0] = basis[lab][orb][p][0]
                            # apply normalization
                            N = np.power(2*self.basis_values[orb_index, p, 0]/np.pi, 3./4.) 
                            self.basis_values[orb_index, p, 1] = N*basis[lab][orb][p][1]

                        orb_index += 1

        self.n_1e_ints = int(self.norbs*(self.norbs+1)/2)
        self.n_2e_ints = int(self.norbs*(self.norbs+1)*(self.norbs**2+self.norbs+2)/8)

        self.tm.stop("read_input")

        pass

    def print_header(textwidth=TEXTW, textshift=TEXTS) -> None:

        sh = textshift
        
        print()
        print(" "*sh + "*"*textwidth)
        print("")
        print(" "*sh + " "*int((textwidth-34)/2) + r'              _____  _____ ______ ' + " "*int((textwidth-34)/2))
        print(" "*sh + " "*int((textwidth-34)/2) + r'             / ____|/ ____|  ____|' + " "*int((textwidth-34)/2))
        print(" "*sh + " "*int((textwidth-34)/2) + r' _ __  _   _| (___ | |    | |__   ' + " "*int((textwidth-34)/2))
        print(" "*sh + " "*int((textwidth-34)/2) + r'|  _ \| | | |\___ \| |    |  __|  ' + " "*int((textwidth-34)/2))
        print(" "*sh + " "*int((textwidth-34)/2) + r'| |_) | |_| |____) | |____| |     ' + " "*int((textwidth-34)/2))
        print(" "*sh + " "*int((textwidth-34)/2) + r'| .__/ \__, |_____/ \_____|_|     ' + " "*int((textwidth-34)/2))
        print(" "*sh + " "*int((textwidth-34)/2) + r'| |     __/ |                     ' + " "*int((textwidth-34)/2))
        print(" "*sh + " "*int((textwidth-34)/2) + r'|_|    |___/                      ' + " "*int((textwidth-34)/2))
        print("")
        print(" "*sh + f"{'a Python-based Hartree-Fock SCF implementation':^{textwidth}}")
        print("")
        print(" "*sh + f"{'Raúl Coterillo Ruisánchez':^{textwidth}}")
        print(" "*sh + f"{'raulcote98@gmail.com':^{textwidth}}")
        print(" "*sh + f"{'':^{textwidth}}")
        print(" "*sh + f"{'November 2021':^{textwidth}}")
        print()
        print(" "*sh + "*"*textwidth)
        print()

    def print_sim_info(self, textwidth:int=TEXTW, textshift=TEXTS) -> None:

        """ 
        Prints a summary of the information in the input file.
        
        Parameters
        ----------
            textwidth : int (Default 80)
                number of characters per row
        """

        sh    = textshift
        half  = int(textwidth/2)
        quart = int(textwidth/4)
        eigth = int(textwidth/8)

        print(" "*sh + f"{'~#~#~#~ INPUT FILE INFORMATION ~#~#~#~':^{textwidth}}")
        print("")
        print(" "*sh + f"{'Input file:':{quart}}{self.input_file:>{quart+half}}")
        if not (self.basis_file is None): 
            print(" "*sh + f"{'Basis file:':{quart}}{self.basis_file:>{quart+half}}")
        print("")
        print(" "*sh + f"{'Number of Atoms:':{half}}{self.nats:>{half}}")
        print(" "*sh + f"{'Input  Geometry:':<{textwidth}}")
        print("")
        print(" "*sh + "-"*textwidth)
        print(" "*sh + f"{'Index':^{eigth}}{'Label':^{eigth}}{'x (Ang)':^{quart}}{'y (Ang)':^{quart}}{'z (Ang)':^{quart}}")
        print(" "*sh + "-"*textwidth)
        for at in range(self.nats):
            print(" "*sh + f"{at:^{eigth}}{self.labels[at]:^{eigth}}{self.positions[at][0]:^{quart}.6f}{self.positions[at][1]:^{quart}.6f}{self.positions[at][2]:^{quart}.6f}")
        print(" "*sh + "-"*textwidth)
        print("")
        print(" "*sh + f"{'Number of Electrons:':{half}}{self.nels:{half}}")
        print(" "*sh + f"{'Overall Charge:':{half}}{self.charge:{half}}")
        print("")
        print(" "*sh + f"{'Number of Basis Functions:':{half}}{self.norbs:{half}}")
        print(" "*sh + f"{'Number of Primitive Gaussians:':{half}}{np.sum(self.basis_nprims):{half}}")
        print(" "*sh + f"{'Basis Set Specification:':<{textwidth}}")
        print("")
        print(" "*sh + "-"*textwidth)
        print(" "*sh + f"{'Orbital':^{eigth}}{'At.Lab':^{eigth}}{'At.Ind':^{eigth}}{'N.Prim':^{eigth}}{'Zeta':^{quart}}{'N':^{quart}}")
        print(" "*sh + "-"*textwidth)
        for orb in range(self.norbs):
            print(" "*sh + f"{orb:^{eigth}}{self.labels[self.basis_centers[orb]]:^{eigth}}{self.basis_centers[orb]:^{eigth}}{self.basis_nprims[orb]:^{eigth}}{'':^{half}}")
            for p in range(self.basis_nprims[orb]):
                print(" "*sh + f"{'':^{half}}{self.basis_values[orb, p, 0]:^{quart}.8f}{self.basis_values[orb, p, 1]:^{quart}.8f}")
        print(" "*sh + "-"*textwidth)
        print("")

    # ========================================================================================= #
    #                                                                                           #
    #   Functions related to with system's geometry.                                            #
    #                                                                                           #
    # ========================================================================================= #

    def bohr_geometry(self) -> np.array:
        """ Returns the system's geometry array, in bohr. """
        return self.positions/0.529177210903

    def distance_bohr(self, i:int, j:int) -> float:
        """ Returns the distance between two atoms, in bohr. """
        return np.linalg.norm(self.positions[i,:] - self.positions[j,:])/0.529177210903
 
    def distance_angstrom(self, i:int, j:int) -> float:
        """ Returns the distance between two atoms, in angstrom. """
        return np.linalg.norm(self.positions[i,:] - self.positions[j,:])

    # ========================================================================================= #
    #                                                                                           #
    #   Functions related with matrix element integral calculations.                            #
    #                                                                                           #
    # ========================================================================================= #

    def read_integrals(self) -> list:

        """
        Reads the integrals from the input file.
        
        Returns
        -------
            S : np.array[norbs, norbs]
                overlap matrix

            T : np.array[norbs, norbs]
                kinetic energy integral matrix
            
            V : np.array[norbs, norbs]
                potential energy integral matrix
            
            O : np.array[norbs, norbs, norbs, norbs]
                two-electron integral matrix
        """

        self.tm.start("read_integrals")

        K = self.norbs

        self.S = np.zeros((K, K), dtype=float)        # overlap matrix
        self.T = np.zeros((K, K), dtype=float)        # kinetic integrals
        self.V = np.zeros((K, K), dtype=float)        # nuclear integrals
        self.O = np.zeros((K, K, K, K), dtype=float)  # two-electron integrals

        # parse input file
        with open(self.input_file, "r") as f:

            # skip all lines until "Overlap integrals" is read
            while True:
                if "Overlap" in f.readline():
                    break

            self.tm.start("1e_ints", parent="read_integrals")

            # read overlap integrals
            for j in range(self.norbs):
                for i in range(j+1):
                    self.S[i,j] = float(f.readline().split()[2])
                    # symmetry ops
                    self.S[j,i] = self.S[i,j]

            # read kinetic integrals
            f.readline() # Kinetic integrals
            for j in range(self.norbs):
                for i in range(j+1):
                    self.T[i,j] = float(f.readline().split()[2])
                    # symmetry ops
                    self.T[j,i] = self.T[i,j]

            # read nuclear integrals
            f.readline() # Nuclear Attraction integrals
            for j in range(self.norbs):
                for i in range(j+1):
                    self.V[i,j] = float(f.readline().split()[2])
                    # symmetry ops
                    self.V[j,i] = self.V[i,j]

            self.tm.stop("1e_ints", parent="read_integrals")
            self.tm.start("2e_ints", parent="read_integrals")
            
            # read two electron integrals
            f.readline() # Two-Electron integrals

            for i in range(K):
                for j in range(K):
                    if i >= j:
                        for k in range(K):
                            for l in range(K):
                                if k >= l:
                                    if i*(i+1)/2.+j >= k*(k+1)/2+l:
                                    
                                        self.O[i,j,k,l] = float(f.readline().split()[4])

                                        # symmetry ops
                                        self.O[j,i,k,l] = self.O[i,j,k,l] # i <-> j
                                        self.O[i,j,l,k] = self.O[i,j,k,l] # l <-> k 
                                        self.O[j,i,l,k] = self.O[i,j,k,l] # i <-> j, l<->k
                                        
                                        self.O[k,l,i,j] = self.O[i,j,k,l] # ij <-> kl
                                        self.O[k,l,j,i] = self.O[i,j,k,l] # ij <-> kl, i <-> j
                                        self.O[l,k,i,j] = self.O[i,j,k,l] # ij <-> kl, l <-> k
                                        self.O[l,k,j,i] = self.O[i,j,k,l] # ij <-> kl, i <-> j, l<->k
   
        self.tm.stop("2e_ints", parent="read_integrals")
        self.tm.stop("read_integrals")

        return self.S, self.T, self.V, self.O

    def calc_integrals(self) -> list:

        """ 
        Calculates the required integrals using the structure and basis set information.
        
        Returns
        -------
            S : np.array[norbs, norbs]
                overlap matrix

            T : np.array[norbs, norbs]
                kinetic energy integral matrix
            
            V : np.array[norbs, norbs]
                potential energy integral matrix
            
            O : np.array[norbs, norbs, norbs, norbs]
                two-electron integral matrix
         """

        self.tm.start("calc_integrals")

        K       = self.norbs
        geom    = self.bohr_geometry()                  # atomic positions in bohr
        charge  = self.atnums                           # atomic charges 
        nprims  = self.basis_nprims                     # nº of primitives of each contracted basis function
        values  = self.basis_values                     # primitive coefficients of each contracted basis function
        centers = self.basis_centers                    # center of each contracted basis function
        
        self.S  = np.zeros((K, K), dtype=float)         # overlap matrix
        self.T  = np.zeros((K, K), dtype=float)         # kinetic integrals
        self.V  = np.zeros((K, K), dtype=float)         # nuclear integrals
        self.O  = np.zeros((K, K, K, K), dtype=float)   # two-electron integrals

        self.tm.start("1e_ints", parent="calc_integrals")

        # precalculate some terms to speed up later iterations
        dist    = np.zeros((K,K), dtype=float)                                      # interatomic distances
        zeta    = np.zeros((K, self.max_nprim, K, self.max_nprim), dtype=float)     # zeta terms
        xi      = np.zeros((K, self.max_nprim, K, self.max_nprim), dtype=float)     # xi terms
        normc   = np.zeros((K, self.max_nprim, K, self.max_nprim), dtype=float)     # normalizations
        gausp   = np.zeros((K, self.max_nprim, K, self.max_nprim, 3), dtype=float)  # gaussian products

        # iterate over required primitive pairs        
        for i in range(K):             
                for j in range(i+1):

                    # intercenter distance squared
                    dist[i,j] = np.power(self.distance_bohr(centers[i],centers[j]),2)
                    dist[j,i] = dist[i,j]

                    for a in range(nprims[i]):          
                        for b in range(nprims[j]):

                            # zeta
                            zeta[i,a,j,b]   = values[i,a,0] + values[j,b,0]
                            zeta[j,b,i,a]   = zeta[i,a,j,b]

                            # xi
                            xi[i,a,j,b]     = values[i,a,0]*values[j,b,0]/zeta[i,a,j,b]
                            xi[j,b,i,a]     = xi[i,a,j,b] 

                            # d (normalization coefficients)
                            normc[i,a,j,b]  = values[i,a,1]*values[j,b,1]
                            normc[j,b,i,a]  = normc[i,a,j,b]

                            # gaussian product centers 
                            gausp[i,a,j,b]  = (values[i,a,0]*geom[centers[i]] + values[j,b,0]*geom[centers[j]])/zeta[i,a,j,b]


        # iterate over the required one-center integrals
        for i in range(K):             
                for j in range(i+1):
                
                    # iterate over primitives
                    for a in range(nprims[i]):          
                        for b in range(nprims[j]):

                            S_iajb  = np.exp(-xi[i,a,j,b]*dist[i,j])*np.sqrt(np.power((np.pi/zeta[i,a,j,b]),3))
                            T_iajb  = xi[i,a,j,b]*(3 - 2*xi[i,a,j,b]*dist[i,j])*S_iajb

                            self.S[i,j] += normc[i,a,j,b]*S_iajb
                            self.T[i,j] += normc[i,a,j,b]*T_iajb

                            for n in range(self.nats):  # iterate over nuclei       
                                
                                V_iajbN = -2*charge[n]*np.sqrt(zeta[i,a,j,b]/np.pi)*S_iajb 
                                
                                # check if basis funcs are on the same atom for the Bois function
                                if centers[i] == centers[j] == n:
                                    bois0   = 1
                                else:
                                    x       = zeta[i,a,j,b]*np.power(np.linalg.norm(geom[n] - gausp[i,a,j,b]),2) 
                                    bois0   = 0.5*np.sqrt(np.pi/x)*erf(np.sqrt(x))
                                
                                self.V[i,j] += normc[i,a,j,b]*V_iajbN*bois0

                    # symmetry ops
                    self.S[j,i] = self.S[i,j]
                    self.T[j,i] = self.T[i,j]
                    self.V[j,i] = self.V[i,j]

        self.tm.stop("1e_ints", parent="calc_integrals")
        self.tm.start("2e_ints", parent="calc_integrals")

        # iterate over the required two-center integrals        
        for i in range(K):
                for j in range(K):
                    if i >= j:
                        for k in range(K):
                            for l in range(K):
                                if k >= l:
                                    if i*(i+1)/2.+j >= k*(k+1)/2+l:

                                        # iterate over primitives
                                        for a in range(nprims[i]):          
                                            for b in range(nprims[j]):
                                                for c in range(nprims[k]):
                                                    for d in range(nprims[l]):
                                                       
                                                        K_iajb = np.sqrt(2)*np.pi**(5./4.)/zeta[i,a,j,b]*np.exp(-xi[i,a,j,b]*dist[i,j])
                                                        K_kcld = np.sqrt(2)*np.pi**(5./4.)/zeta[k,c,l,d]*np.exp(-xi[k,c,l,d]*dist[k,l])
                                                        
                                                        gausp_dist = np.linalg.norm(gausp[i,a,j,b] - gausp[k,c,l,d])

                                                        # Boys function check
                                                        if gausp_dist < 1e-5:
                                                            bois0 = 1
                                                        else:
                                                            rho     = zeta[i,a,j,b]*zeta[k,c,l,d]/(zeta[i,a,j,b]+zeta[k,c,l,d])  
                                                            x       = rho*np.power(gausp_dist,2) 
                                                            bois0   = 0.5*np.sqrt(np.pi/x)*erf(np.sqrt(x))

                                                        self.O[i,j,k,l] += normc[i,a,j,b]*normc[k,c,l,d]*K_iajb*K_kcld*bois0/np.sqrt(zeta[i,a,j,b]+zeta[k,c,l,d])

                                        # symmetry ops
                                        self.O[j,i,k,l] = self.O[i,j,k,l] # i <-> j
                                        self.O[i,j,l,k] = self.O[i,j,k,l] # l <-> k 
                                        self.O[j,i,l,k] = self.O[i,j,k,l] # i <-> j, l<->k
                                        
                                        self.O[k,l,i,j] = self.O[i,j,k,l] # ij <-> kl
                                        self.O[k,l,j,i] = self.O[i,j,k,l] # ij <-> kl, i <-> j
                                        self.O[l,k,i,j] = self.O[i,j,k,l] # ij <-> kl, l <-> k
                                        self.O[l,k,j,i] = self.O[i,j,k,l] # ij <-> kl, i <-> j, l<->k

        self.tm.stop("2e_ints", parent="calc_integrals")
        self.tm.stop("calc_integrals")

        return self.S, self.T, self.V, self.O

    def write_integrals(self, fname:str, prec=16) -> None:

        """
        Writes the stored integrals in a file. 

        Parameters
        ----------
            fname : str
                name of the file where the integrals will be written
            prec : int (default: 16)
                decimal precision with which to write the matrix values
        """

        self.tm.start("write_integrals")

        K = self.norbs

        # create output file
        with open(fname, "w") as f:

            self.tm.start("1e_ints", parent="write_integrals")

            # write overlap integrals
            f.write("Overlap integrals:\n")
            for j in range(self.norbs):
                for i in range(j+1):
                    f.write(f"{i:6d}{j:6d}{'':6}{self.S[i,j]:>18.{prec}f}\n")

            # write kinetic integrals
            f.write("\nKinetic Integrals:\n")
            for j in range(self.norbs):
                for i in range(j+1):
                    f.write(f"{i:6d}{j:6d}{'':6}{self.T[i,j]:>18.{prec}f}\n")

            # write nuclear integrals
            f.write("\nNuclear Integrals:\n")
            for j in range(self.norbs):
                for i in range(j+1):
                    f.write(f"{i:6d}{j:6d}{'':6}{self.T[i,j]:>18.{prec}f}\n")

            self.tm.stop("1e_ints", parent="write_integrals")
            self.tm.start("2e_ints", parent="write_integrals")
            
            # write two-electron integrals
            f.write("\nTwo-Electron Integrals:\n")

            for i in range(K):
                for j in range(K):
                    if i >= j:
                        for k in range(K):
                            for l in range(K):
                                if k >= l:
                                    if i*(i+1)/2.+j >= k*(k+1)/2+l:
                                        f.write(f"{i:6d}{j:6d}{k:6d}{l:6d}{'':6}{self.O[i,j,k,l]:>18.{prec}f}\n")
            
        self.tm.stop("2e_ints", parent="write_integrals")
        self.tm.stop("write_integrals")

    # ========================================================================================= #
    #                                                                                           #
    #   Functions related with the SCF energy calculation                                       #
    #                                                                                           #
    # ========================================================================================= #

    def nuclear_repulsion(self) -> float:

        """ 
        Calculates the nuclear repulsion energy. 
        
        Returns
        -------
            the nuclear repulsion energy, in Ha
        """

        self.V_nuc = 0
        for i in range(self.nats-1):
            for j in range(i+1, self.nats):
                self.V_nuc += self.atnums[i]*self.atnums[j]/self.distance_bohr(i,j)
        return self.V_nuc

    def SCF_setup(self) -> None:

        """
        Sets up all the SCF reusable variables, such as 
        the integrals and the transformation matrix, as
        class attributes.
        """

        K = self.norbs

        # diagonalize overlap matrix, obtaining
        # eigenvalue (to be) matrix A and eigenvectors U
        eigs, U = np.linalg.eigh(self.S)

        # calculate the square root of the A matrix
        alpha = np.diag(1/np.sqrt(eigs))

        # build the transformation matrix X, transforming back A to S
        self.X = np.matmul(np.matmul(U,alpha), np.transpose(U))

        # build the core hamiltonian
        self.Hc = self.T + self.V

        # take it as the Fock operator (no two electron term yet)
        self.F = self.Hc

        # transform the Fock operator to the orthogonalized basis
        self.Hc0 = np.matmul(np.transpose(self.X), np.matmul(self.F, self.X))

        # calculate the transformed coefficient matrix,
        # obtaining the orbital energies in the process
        self.orbital_energies, self.C0 = np.linalg.eigh(self.Hc0)

        # get back the coefficient matrix
        self.C = np.matmul(self.X,self.C0)

        # initial density matrix
        self.P = np.zeros((K,K))
        for i in range(K):
            for j in range(K):
                for a in range(int(self.nels/2)):
                    self.P[i, j] += self.C[i, a]*self.C[j,a]

        # initial energy
        self.E = 0
        for i in range(K):
            for j in range(K):
                self.E += self.P[i,j]*(self.Hc[i,j]+self.F[i,j])

    def SCF_step(self) -> None:

        """
        Performs a self-consistent field step, meaning:
            - it recalculates the density matrix P
            - builds the G matrix
            - builds and transforms the Fock matrix F 
            - diagonalizes it and obtains the orbitals
            - calculates the electronic energy
        """

        K = self.norbs

        # new density matrix
        self.P = np.zeros((K,K))
        for i in range(K):
            for j in range(K):
                for a in range(int(self.nels/2)):
                    self.P[i, j] += self.C[i, a]*self.C[j,a]

        # build the G matrix, using the density matrix 
        # P and the two-electron integrals
        G = np.zeros((K,K))
        for i in range(K):
            for j in range(K):
                    for k in range(K): # lda
                        for l in range(K): #sigma
                           G[i,j] += self.P[k,l]*(2*self.O[i,j,k,l] - self.O[i,k,j,l])

        # build Fock operator
        self.F  = self.Hc + G

        # transform the Fock operator to the orthogonal basis
        self.F0 = np.matmul(np.transpose(self.X), np.matmul(self.F, self.X))

        # calculate the transformed coefficient matrix,
        # obtaining the orbital energies in the process
        self.orbital_energies, self.C0 = np.linalg.eigh(self.F0)

        # get back the coefficient matrix
        self.C = np.matmul(self.X,self.C0)

        # calculate the total energy 
        self.E = 0
        for i in range(K):
            for j in range(K):
                self.E += self.P[i,j]*(self.Hc[i,j]+self.F[i,j])

    # ========================================================================================= #
    #                                                                                           #
    #   Printing functions related with the SCF energy calculation                              #
    #                                                                                           #
    # ========================================================================================= #

    def print_orbitals(self, textwidth:int=TEXTW, textshift:int= TEXTS) -> None:

        """ Prints orbital information, i.e the energies and coefficients. """

        sh    = textshift
        third = int(textwidth/3)
        quart = int(textwidth/4)

        K        = self.norbs    
        eners    = self.orbital_energies
        eners_ev = eners*27.211386245988

        print(" "*sh + f"{'~#~#~#~ ORBITAL INFORMATION ~#~#~#~':^{textwidth}}", flush=True)
        print("")
        print(" "*sh + f"{'Energies':^{textwidth}}", flush=True)
        print(" "*sh + "-"*textwidth)
        print(" "*sh + f"{'Orbital':^{third}}{'Energy (Ha)':^{third}}{'Energy(eV)':^{third}}")
        print(" "*sh + "-"*textwidth)
        for orb in range(K):
            print(" "*sh + f"{orb:^{third}}{eners[orb]:^{third}.8E}{eners_ev[orb]:^{third}.8E}")
        print(" "*sh + "-"*textwidth)
        nrows   = floor(K/3)
        rest    = K%3
        print("")
        print(" "*sh + f"{'Coefficients':^{textwidth}}\n", flush=True)
        print(" "*sh + "-"*textwidth)
        for line in range(nrows):
            print(" "*sh + f"{'Orbital':^{quart}}{line*3:^{quart}}{line*3+1:^{quart}}{line*3+2:^{quart}}")
            print(" "*sh + "-"*textwidth)
            for orb in range(K):
                print(" "*sh + f"{orb:^{quart}}{self.C[orb, line*3]:^{quart}.8f}{self.C[orb, line*3+1]:^{quart}.8f}{self.C[orb, line*3+2]:^{quart}.8f}")
            print(" "*sh + "-"*textwidth)
        if rest != 0:
            header = " "*sh + f"{'Orbital':^{quart}}"
            for r in range(rest):
                header += f"{nrows*3+r:^{quart}}"
            print(header)
            print(" "*sh + "-"*int((rest+1)*quart))
            for orb in range(K):
                l = " "*sh + f"{orb:^{quart}}"
                for r in range(rest):
                    l += f"{self.C[orb, nrows*3+r]:^{quart}.8f}"
                print(l)
            print(" "*sh + "-"*int((rest+1)*quart))
        print("")

        pass

    def print_population(self, textwidth:int=TEXTW, textshift:int= TEXTS,) -> None:

        """ Performs a population (charge) analysis on the system, and prints the results. """

        sh    = textshift
        third = int(textwidth/3)
        sixth = int(textwidth/6)

        K     = self.norbs 
        labs  = self.labels

        # calculate mulliken population by orbital
        mull_pop = np.diag(2*np.matmul(self.P, self.S))
        lowd_pop = np.diag(np.matmul(np.linalg.inv(self.X), np.matmul(2*np.matmul(self.P, self.X), self.S)))
        
        # add contributions by atom
        mull_ele = np.zeros(self.nats)
        lowd_ele = np.zeros(self.nats)
        for o in range(K):
            at = self.basis_centers[o]
            mull_ele[at] += mull_pop[o] 
            lowd_ele[at] += lowd_pop[o] 
        
        # calculate charges
        mull_ch = self.atnums - mull_ele
        lowd_ch = self.atnums - lowd_ele
            
        print(" "*sh + f"{'~#~#~#~ POPULATION ANALYSIS ~#~#~#~':^{textwidth}}", flush=True)
        print("")
        print(" "*sh + "-"*textwidth)
        print(" "*sh + f"{'Atom':^{third}}{'Mulliken':^{third}}{'Löwdin':^{third}}")
        print(" "*sh + f"{'-'*(third-4):^{third}}{'-'*(third-4):^{third}}{'-'*(third-4):^{third}}")
        print(" "*sh + f"{'Index':^{sixth}}{'Label':^{sixth}}{'Pop':^{sixth}}{'Charge':^{sixth}}{'Pop':^{sixth}}{'Charge':^{sixth}}")
        print(" "*sh + "-"*textwidth)
        for at in range(self.nats):
            print(" "*sh + f"{at:^{sixth}}{labs[at]:^{sixth}}{mull_ele[at]:^{sixth}.4f}{mull_ch[at]:^{sixth}.4f}{lowd_ele[at]:^{sixth}.4f}{lowd_ch[at]:^{sixth}.4f}")
        print(" "*sh + "-"*textwidth)
        print("")

    def write_SCF(self, fname:str, prec=16) -> None:

        """
        Writes the SCF matrices (X, P, F, Hc, etc) in a file. 

        Parameters
        ----------
            fname : str
                file where to write the matrices
            prec : int (default: 16)
                decimal precision with which to write the matrix values
        """

        K = self.norbs

        # create output file
        with open(fname, "w") as f:

            # write transformation matrix
            f.write("Transformation Matrix X:\n")
            for i in range(self.norbs):
                for j in range(i+1):
                    f.write(f"{i+1:6d}{j+1:6d}{'':6}{self.X[i,j]:>18.{prec}f}\n")

            # write core hamiltonian
            f.write("\nCore Hamiltonian:\n")
            for i in range(self.norbs):
                for j in range(i+1):
                    f.write(f"{i+1:6d}{j+1:6d}{'':6}{self.Hc[i,j]:>18.{prec}f}\n")

            # write core hamiltonian in orthogonal basis
            f.write("\nCore Hamiltonian (Orthogonal Basis):\n")
            for i in range(self.norbs):
                for j in range(i+1):
                    f.write(f"{i+1:6d}{j+1:6d}{'':6}{self.Hc0[i,j]:>18.{prec}f}\n")

            # write last fock matrix
            f.write("\nFock Matrix:\n")
            for i in range(self.norbs):
                for j in range(i+1):
                    f.write(f"{i+1:6d}{j+1:6d}{'':6}{self.F[i,j]:>18.{prec}f}\n")

            # write last fock matrix in orthogonal basis
            f.write("\nFock Matrix (Orthogonal Basis):\n")
            for i in range(self.norbs):
                for j in range(i+1):
                    f.write(f"{i+1:6d}{j+1:6d}{'':6}{self.F0[i,j]:>18.{prec}f}\n")

            # write last density matrix
            f.write("\nDensity Matrix:\n")
            for i in range(self.norbs):
                for j in range(i+1):
                    f.write(f"{i+1:6d}{j+1:6d}{'':6}{self.P[i,j]:>18.{prec}f}\n")

    # ========================================================================================= #
    #                                                                                           #
    #   Functions that perform a given process of interest, producing output along the way      #
    #                                                                                           #
    # ========================================================================================= #

    def run_single_point(self,  max_iter:int        = 100,
                                e_thresh:float      = 1e-8,
                                p_thresh:float      = 1e-8,
                                textwidth:int       = TEXTW,
                                textshift:int       = TEXTS,
                                read_integrals:bool = False,
                                write_integrals:bool= False,
                                write_SCF:bool      = False) -> None:

        """
        Performs a complete single point calculation 
        
        Parameters
        ----------
            max_iter : int (default: 100)
                maximum number of SCF iterations
            e_thresh : float (default: 1e-8)
                convergence threshold for the energy
            p_thresh : float (default: 1e-8)
                convergence threshold for the density matrix
            read_integrals: bool (default: False)
                whether to read integrals from the input (or calculate them)
            write_integrals: bool (default: False)
                whether to write integrals to an external file (integrals.txt)
            write_SCF: bool (default: False)
                whether to write SCF matrices to an external file (SCF.txt)
        """

        tm    = self.tm  
        sh    = textshift
        half  = int(textwidth/2)
        quart = int(textwidth/4)

        tm.start("single_point_scf")

        print(" "*sh + f"{'~#~#~#~ HF SINGLE-POINT SCF ~#~#~#~':^{textwidth}}", flush=True)
        print("")
        print(" "*sh + f"{'CONVERGENCE CRITERIA:':<{textwidth}}", flush=True)
        print("")
        print(" "*sh + f"{'Energy  Threshold  -> ':<{half}}{e_thresh:<{half}.6E}", flush=True)
        print(" "*sh + f"{'Density Threshold  -> ':<{half}}{p_thresh:<{half}.6E}", flush=True)
        print(" "*sh + f"{'Maximum Iterations -> ':<{half}}{max_iter:<{half}}", flush=True)             
        print("")
        print(" "*sh + f"{'Number of 1-electron integrals:':{half}}{self.n_1e_ints:<{half}}")
        print(" "*sh + f"{'Number of 2-electron integrals:':{half}}{self.n_2e_ints:<{half}}")

        # obtain the integrals
        if read_integrals: # read integrals from input file
            print("\n"+ " "*sh + f"{'Reading Integrals from Input File...':{half}}", end="", flush=True)
            tm.start("read_ints", parent="single_point_scf")
            self.read_integrals()
            t = tm.stop("read_ints", parent="single_point_scf")
            print(f"{'DONE!':>{quart}}{f'{t:.6f} s':>{quart}}", flush=True)
        else: # calculate integrals using geometry and basis set information
            print("\n"+ " "*sh + f"{'Calculating Integrals...':{half}}", end="", flush=True)
            tm.start("calc_ints", parent="single_point_scf")
            self.calc_integrals()
            t = tm.stop("calc_ints", parent="single_point_scf")
            print(f"{'DONE!':>{quart}}{f'{t:.6f} s':>{quart}}", flush=True)
    
        if write_integrals:
            print("\n"+ " "*sh + f"{'Writing Integrals to File...':{half}}", end="", flush=True)
            tm.start("write_ints", parent="single_point_scf")
            self.write_integrals("integrals.txt")
            t = tm.stop("write_ints", parent="single_point_scf")
            print(f"{'DONE!':>{quart}}{f'{t:.6f} s':>{quart}}\n", flush=True)
            print(" "*sh + f"{'Saved to integrals.txt':^{textwidth}}", flush=True)

        # set up non-iterative quantities
        print("\n"+ " "*sh + f"{'Initial Matrix Setup...':{half}}", end="", flush=True)
        tm.start("scf_setup", parent="single_point_scf")
        self.SCF_setup()
        t = tm.stop("scf_setup", parent="single_point_scf")
        print(f"{'DONE!':>{quart}}{f'{t:.6f} s':>{quart}}\n", flush=True)

        # start scf iterations
        print(" "*sh + f"{'SCF CYCLE:':<{textwidth}}")
        print(" "*sh + "-"*textwidth)
        print(" "*sh + f"{'Iteration':^{quart}}{'Energy':^{quart}}{'dE':^{quart}}{'max(dP)':^{quart}}")
        print(" "*sh + "-"*textwidth)
        converged = False
        for it in range(max_iter):

            # save current E and P
            self.E_old = self.E
            self.P_old = self.P

            # perform SCF step
            self.tm.start("scf_step", parent="single_point_scf")
            self.SCF_step()
            self.tm.stop("scf_step", parent="single_point_scf")            
            
            # calculate E and P differences
            self.dE = self.E - self.E_old
            self.dP = np.max(np.abs(self.P - self.P_old))

            # print iteration info
            print(" "*sh + f"{it+1:^{quart}}{self.E:^{quart}.8E}{self.dE:^{quart}.8E}{self.dP:^{quart}.8E}")

            # check convergence
            if (np.abs(self.dE) < e_thresh) and (self.dP < p_thresh):
                converged = True
                break        

        print(" "*sh + "-"*textwidth)
        self.last_iterations = it + 1
        self.last_converged  = converged
        t = tm.stop("single_point_scf")

        # check if SCF procedure converged
        if converged:
            print("\n" + " "*sh + f"{'SCF CYCLE CONVERGED!':^{half}}{f'{it+1} iteration in {t:.6f} s':^{half}}")
        
            # calculate nuclear repulsion energy
            print("\n" + " "*sh + f"{'Nuclear Repulsion Calculation...':{half}}", end="")
            self.tm.start("nuc_rep", parent="single_point_scf")
            self.nuclear_repulsion()
            t = self.tm.stop("nuc_rep", parent="single_point_scf")
            print(f"{'DONE!':>{quart}}{f'{t:.6f} s':>{quart}}")

            if write_SCF: # write SCF matrices if needed
                print("\n"+ " "*sh + f"{'Writing SCF Matrices to File...':{half}}", end="", flush=True)
                tm.start("write_SCF", parent="single_point_scf")
                self.write_SCF("SCF.txt")
                t = tm.stop("write_SCF", parent="single_point_scf")
                print(f"{'DONE!':>{quart}}{f'{t:.6f} s':>{quart}}\n", flush=True)
                print(" "*sh + f"{'Saved to SCF.txt':^{textwidth}}", flush=True)

            # print out energy summary
            print("\n" + " "*sh + f"{'       Electronic Energy -> ':>{half}}{self.E:>{quart}.8E}{' Ha':<{quart}}")
            print(" "*sh + f"{'Nuclear Repulsion Energy -> ':>{half}}{self.V_nuc:>{quart}.8E}{' Ha':<{quart}}")
            print("\n" + " "*sh + f"{'            Total Energy -> ':>{half}}{self.E+self.V_nuc:>{quart}.8E}{' Ha':<{quart}}" + "\n")

            # print orbital coefficients
            tm.start("orb_print", parent="single_point_scf")
            self.print_orbitals()
            tm.stop("orb_print", parent="single_point_scf")

            # print population charge analysis
            tm.start("pop_analysis", parent="single_point_scf")
            self.print_population()
            tm.stop("pop_analysis", parent="single_point_scf")

        else:
            print("\n" + " "*sh + f"{'SCF CONVERGENCE FAILED!':^{half}}{f'{it+1} iteration in {t:.6f} s':^{half}}")
            print("\n" + " "*sh + f"{'You should check your input files,':^{textwidth}}")
            print(" "*sh + f"{'increase the number of iterations or':^{textwidth}}")
            print(" "*sh + f"{'decrease the convergence thresholds.':^{textwidth}}\n")

    # ========================================================================================= #
    #                                                                                           #
    #   Class that measures execution times within the different functions                      #
    #                                                                                           #
    # ========================================================================================= #

    class Timer():

        """
        Subclass that keeps track of function execution times. 

        Attributes
        ----------
        times : dict
            timer information

        counter : func        
            function used to measure the time, in seconds

        Methods
        -------
        __init__(self, textwidth:int, textshift:int, counter=time.perf_counter) -> None:
            Initializes the class
        
        def start(self, timer_name:str, parent:str=None) -> None:
            starts a timer

        def stop(self, timer_name:str, parent:str=None) -> float:
            stops a timer and returns delta with last start

        def print_summary(self) -> None:
            print a summary of the timers
        """

        def __init__(self, textwidth:int, textshift:int, counter=time.perf_counter) -> None:

            self.times      = {}
            self.counter    = counter
            self.textwidth  = textwidth
            self.textshift  = textshift

        def start(self, timer_name:str, parent:str=None) -> None:

            if parent is None:
                if timer_name not in self.times.keys():
                    self.times[timer_name] = [[], [], {}]
                self.times[timer_name][0].append(self.counter())
            else:
                if parent not in self.times.keys():
                    print(f"ERROR: Parent timer {parent} does not exist.")
                    sys.exit()
                if timer_name not in self.times[parent][2].keys():
                    self.times[parent][2][timer_name] = [[], []]
                self.times[parent][2][timer_name][0].append(self.counter())

        def stop(self, timer_name:str, parent:str=None) -> float:

            if parent is None:
                self.times[timer_name][0].append(self.counter())
                delta = self.times[timer_name][0][-1] - self.times[timer_name][0][-2]
                self.times[timer_name][1].append(delta)
            else:
                if parent not in self.times.keys():
                    print(f"ERROR: Parent timer {parent} does not exist.")
                    sys.exit()
                self.times[parent][2][timer_name][0].append(self.counter())
                delta = self.times[parent][2][timer_name][0][-1] - self.times[parent][2][timer_name][0][-2]
                self.times[parent][2][timer_name][1].append(delta)

            return delta

        def print_summary(self) -> None:

            sh    = self.textshift
            quart = int(self.textwidth/4)

            print(" "*sh + f"{'~#~#~#~ TIMES SUMMARY ~#~#~#~':^{self.textwidth}}")
            print("")
            print(" "*sh + "-"*self.textwidth)
            print(" "*sh + f"{'Timer':<{quart}}{'Cycles':^{quart}}{'Average (s)':^{quart}}{'Total (s)':^{quart}}")
            print(" "*sh + "-"*self.textwidth)
            for timer in self.times:
                deltas = self.times[timer][1]
                print(" "*sh + f"{timer:<{quart}}{len(deltas):^{quart}}{np.mean(deltas):^{quart}.8f}{np.sum(deltas):^{quart}.8f}")
                for child in self.times[timer][2]:
                    deltas = self.times[timer][2][child][1]
                    print(" "*sh + f"{f'   -> {child}':<{quart}}{len(deltas):^{quart}}{np.mean(deltas):^{quart}.8f}{np.sum(deltas):^{quart}.8f}")
            print(" "*sh + "-"*self.textwidth + "\n")

if __name__ == "__main__":
    main()