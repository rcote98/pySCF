#!/usr/bin/env python3

""" Performs Hartree-Fock quantum chemistry calculations.

Functions:
    - main()
    - print_header()

Classes:
    - HF()
    - Timer()
"""

from math import erf
from typing import Text
import numpy as np
import argparse
import sys, os
import time

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def main() -> None:

    """ Main program loop. """

    p = argparse.ArgumentParser(
        prog="pySCF.py",
        description="Performs a Hartree-Fock SCF calculation.",
        epilog="That's all, folks!"
        )
    p.add_argument("input_file", type=str, nargs=1, action="store",
                help="input file for the program")
    p.add_argument("-b, --external-basis", dest="basis", type=str, nargs=1, action="store", default=None,
                    help="reads basis from Gaussian basis file (.gbs)")
    p.add_argument("-r, --read-integrals", dest="read_ints", action="store_true",
                    help="read integrals from the input file")
    p.add_argument("-w, --write-integrals", dest="write_ints", action="store_true",
                    help="write integrals to a file called 'integrals.txt'")
    args = p.parse_args()

    # print the program header
    HF.print_header()                 

    # check if the input file exists
    if not os.path.isfile(args.input_file[0]):
        print(f"ERROR: Input file '{args.input_file[0]}' does not exist.")
        sys.exit()

    # get command line input
    INPUT_FILE      = args.input_file[0]
    if args.basis is not None: 
        BASIS_FILE = args.basis[0]
    else:
        BASIS_FILE = None
    READ_INTEGRALS  = args.read_ints
    WRITE_INTEGRALS = args.write_ints

    # initialize the program (read input information)
    sim = HF(INPUT_FILE, basis_file=BASIS_FILE)

    # print the information from the input file
    sim.print_sim_info()           

    # run scf calculation
    sim.run_single_point(
                       read_integrals=READ_INTEGRALS,
                       write_integrals=WRITE_INTEGRALS
                       )         

    # print timer summary
    sim.tm.print()                  

    pass

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class HF():

    """
    A class to perform Hartree-Fock calculations.

    Attributes
    ----------
    input_file : str
        input file of the simulation

    nats : int
        number of atoms

    atnums : np.array
        atomic numbers

    labels : np.array
        atomic labels

    positions : np.array
        atomic positions

    charge : int
        overall system charge

    nels : int
        number of electrons

    norbs : int 
        number of basis functions

    max_nprim : int
        max number of primitives per basis function

    bspec : dict{norbs: nd.array}
        description of the basis functions

    Methods
    -------
    __init__(input_file:str) -> None:
        Initializes the class and parses all the input file information.
    
    print_sim_info(textwidth:int=80) -> None:
        Prints a summary of the information in the input file.

    read_integrals(input_file:str) -> list:

    calc_integrals() -> list:





    """

    # default printing parameters
    TEXTW = 65  # text width
    TEXTS = 4   # text shift

    def __init__(self, input_file:str, basis_file:str=None) -> None:

        """
        Initializes the class and parses all the input file information.

        Parameters
        ----------
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

            self.labels     = np.zeros((self.nats), dtype=str) 
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
                
                # TODO parse external Gaussian basis set

                pass


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
        print(" "*sh + f"{'October 2021':^{textwidth}}")
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

        print(" "*sh + f"{'~ INPUT FILE INFORMATION ~':^{textwidth}}")
        print("")
        print(" "*sh + f"{'Input file:':{quart}}{self.input_file:>{quart+half}}")
        if self.basis_file is not None: print(print(" "*sh + f"{'Basis file:':{quart}}{self.basis_file:>{quart+half}}"))
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

        Parameters
        ----------
            input_file : str
                input file containing all the SCF information
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

            total_1e = int(K*(K+1)/2)

            # read overlap integrals
            for j in range(self.norbs):
                for i in range(j+1):
                    self.S[i,j] = float(f.readline().split()[2])
                    self.S[j,i] = self.S[i,j]

            # read kinetic integrals
            f.readline() # Kinetic integrals
            for j in range(self.norbs):
                for i in range(j+1):
                    self.T[i,j] = float(f.readline().split()[2])
                    self.T[j,i] = self.T[i,j]

            # read nuclear integrals
            f.readline() # Nuclear Attraction integrals
            for j in range(self.norbs):
                for i in range(j+1):
                    self.V[i,j] = float(f.readline().split()[2])
                    self.V[j,i] = self.V[i,j]

            self.tm.stop("1e_ints", parent="read_integrals")
            self.tm.start("2e_ints", parent="read_integrals")
            
            # read two electron integrals
            f.readline() # Two-Electron integrals

            m = 0
            total = int(K*(K+1)*(K**2+K+2)/8)
            for i in range(K):
                for j in range(K):
                    if i >= j:
                        for k in range(K):
                            for l in range(K):
                                if k >= l:
                                    if i*(i+1)/2.+j >= k*(k+1)/2+l:
                                    
                                        self.O[i,j,k,l] = float(f.readline().split()[4])
                                        self.O[j,i,k,l] = self.O[i,j,k,l] # i <-> j
                                        self.O[i,j,l,k] = self.O[i,j,k,l] # l <-> k 
                                        self.O[j,i,l,k] = self.O[i,j,k,l] # i <-> j, l<->k
                                        
                                        self.O[k,l,i,j] = self.O[i,j,k,l] # ij <-> kl
                                        self.O[k,l,j,i] = self.O[i,j,k,l] # ij <-> kl, i <-> j
                                        self.O[l,k,i,j] = self.O[i,j,k,l] # ij <-> kl, l <-> k
                                        self.O[l,k,j,i] = self.O[i,j,k,l] # ij <-> kl, i <-> j, l<->k

                                        m+=1

        print("m=", m)
        self.tm.stop("2e_ints", parent="read_integrals")
        self.tm.stop("read_integrals")

        return self.S, self.T, self.V, self.O

    def calc_integrals(self, progress=False) -> None:

        """ Calculates the integrals using the structure and basis set information. """

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
        dist    = np.zeros((K,K), dtype=float) 
        zeta    = np.zeros((K, self.max_nprim, K, self.max_nprim), dtype=float)
        xi      = np.zeros((K, self.max_nprim, K, self.max_nprim), dtype=float)
        normc   = np.zeros((K, self.max_nprim, K, self.max_nprim), dtype=float)
        gausp   = np.zeros((K, self.max_nprim, K, self.max_nprim, 3), dtype=float)

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
            input_file : str
                input file containing all the SCF information
        """

        self.tm.start("write_integrals")

        K = self.norbs

        # create output file
        with open(fname, "w") as f:

            self.tm.start("1e_ints", parent="write_integrals")

            print("dtype:", self.S.dtype)

            # write overlap integrals
            f.write("Overlap integrals:\n")
            for j in range(self.norbs):
                for i in range(j+1):
                    f.write(f"{i:6d}{j:6d}{'':6}{self.S[i,j]:>18.{prec}f}\n")

            # write kinetic integrals
            f.write("\nKinetic Integrals\n")
            for j in range(self.norbs):
                for i in range(j+1):
                    f.write(f"{i:6d}{j:6d}{'':6}{self.T[i,j]:>18.{prec}f}\n")

            # write nuclear integrals
            f.write("\nNuclear Integrals\n")
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

        return self.S, self.T, self.V, self.O

    # ========================================================================================= #
    #                                                                                           #
    #   Functions related with the SCF energy calculation                                       #
    #                                                                                           #
    # ========================================================================================= #

    def nuclear_repulsion(self) -> float:

        """ Calculates the nuclear repulsion energy. """

        self.V_nuc = 0
        for i in range(self.nats-1):
            for j in range(i+1, self.nats):
                self.V_nuc += self.atnums[i]*self.atnums[j]/self.distance_bohr(i,j)
        return self.V_nuc

    def scf_setup(self) -> None:

        """
        Set up all the SCF reusable variables, such as 
        the integrals and the transformation matrix.

        Parameters
        ----------
            read : bool (default: True)
                read the integrals from the input file?
        """

        K = self.norbs

        # starting energy and density matrix, just
        # to calculate difference with first step
        self.E = 0
        self.P = np.zeros((K,K))

        # diagonalize overlap matrix, obtaining
        # eigenvalue (to be) matrix A and eigenvectors U
        eigs, U = np.linalg.eigh(self.S)

        # calculate the square root of the A matrix
        alpha = np.diag(1/np.sqrt(eigs))

        # build the transformation matrix X, transforming back A to S
        self.X = np.matmul(np.matmul(U,alpha), np.transpose(U))

        # build the core hamiltonian
        self.Hc = self.T + self.V

        # take it as the Fock opeator
        self.F = self.Hc

        # transform the Fock operator to the orthogonal basis
        Fp = np.matmul(np.linalg.inv(self.X), np.matmul(self.F, self.X))

        # calculate the transformed coefficient matrix,
        # obtaining the orbital energies in the process
        self.orbital_energies, Cp = np.linalg.eigh(Fp)

        # get back the coefficient matrix
        self.C = np.matmul(self.X,Cp)

    def scf_step(self) -> None:

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
        Fp = np.matmul(np.linalg.inv(self.X), np.matmul(self.F, self.X))

        # calculate the transformed coefficient matrix,
        # obtaining the orbital energies in the process
        self.orbital_energies, Cp = np.linalg.eigh(Fp)

        # get back the coefficient matrix
        self.C = np.matmul(self.X,Cp)

        # calculate the total energy 
        self.E = 0
        for i in range(K):
            for j in range(K):
                self.E += self.P[i,j]*(self.Hc[i,j]+self.F[i,j])

        pass

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
                                debug:bool          = False) -> None:

        tm    = self.tm  
        sh    = textshift
        half  = int(textwidth/2)
        quart = int(textwidth/4)

        tm.start("single_point_scf")

        print(" "*sh + f"{'~ HF SINGLE-POINT SCF ~':^{textwidth}}", flush=True)
        print("")
        print(" "*sh + f"{'CONVERGENCE CRITERIA:':<{textwidth}}", flush=True)
        print("")
        print(" "*sh + f"{'Energy  Threshold  -> ':<{half}}{e_thresh:<{half}.6E}", flush=True)
        print(" "*sh + f"{'Density Threshold  -> ':<{half}}{p_thresh:<{half}.6E}", flush=True)
        print(" "*sh + f"{'Maximum Iterations -> ':<{half}}{max_iter:<{half}}", flush=True)        

        # obtain the integrals
        if read_integrals:
            print("\n"+ " "*sh + f"{'Reading Integrals from Input File...':{half}}", end="", flush=True)
            tm.start("read_ints", parent="single_point_scf")
            self.read_integrals()
            t = tm.stop("read_ints", parent="single_point_scf")
            print(f"{'DONE!':>{quart}}{f'{t:.6f} s':>{quart}}", flush=True)
        else:
            print("\n"+ " "*sh + f"{'Calculating Integrals...':{half}}", end="", flush=True)
            tm.start("calc_ints", parent="single_point_scf")
            self.calc_integrals(progress=True)
            t = tm.stop("calc_ints", parent="single_point_scf")
            print(f"{'DONE!':>{quart}}{f'{t:.6f} s':>{quart}}", flush=True)
    
        if write_integrals:
            print("\n"+ " "*sh + f"{'Writing Integrals to File...':{half}}", end="", flush=True)
            tm.start("write_ints", parent="single_point_scf")
            self.write_integrals("integrals.txt")
            t = tm.stop("write_ints", parent="single_point_scf")
            print(f"{'DONE!':>{quart}}{f'{t:.6f} s':>{quart}}", flush=True)

        # set up non-iterative quantities
        print("\n"+ " "*sh + f"{'Initial Matrix Setup...':{half}}", end="", flush=True)
        tm.start("scf_setup", parent="single_point_scf")
        self.scf_setup()
        t = tm.stop("scf_setup", parent="single_point_scf")
        print(f"{'DONE!':>{quart}}{f'{t:.6f} s':>{quart}}\n", flush=True)

        # start scf iterations
        print(" "*sh + f"{'SCF CYCLE:':<{textwidth}}\n")
        print(" "*sh + "-"*textwidth)
        print(" "*sh + f"{'Iteration':^{quart}}{'Energy':^{quart}}{'dE':^{quart}}{'dP':^{quart}}")
        print(" "*sh + "-"*textwidth)

        converged = False
        for it in range(max_iter):

            self.E_old = self.E
            self.P_old = self.P

            self.tm.start("scf_step", parent="single_point_scf")
            self.scf_step()
            self.tm.stop("scf_step", parent="single_point_scf")
            
            self.dE = self.E - self.E_old
            self.dP = np.max(np.abs(self.P - self.P_old))

            print(" "*sh + f"{it+1:^{quart}}{self.E:^{quart}.8E}{self.dE:^{quart}.8E}{self.dP:^{quart}.8E}")

            if (np.abs(self.dE) < e_thresh) and (self.dP < p_thresh):
                converged = True
                break        

        print(" "*sh + "-"*textwidth)
        self.last_iterations = it + 1
        self.last_converged  = converged
        t = tm.stop("single_point_scf")

        if converged:
            print("\n" + " "*sh + f"{'SCF CYCLE CONVERGED!':^{half}}{f'{it+1} iteration in {t:.6f} s':^{half}}")
        else:
            print("\n" + " "*sh + f"{'SCF CONVERGENCE FAILED!':^{half}}{f'{it+1} iteration in {t:.6f} s':^{half}}")

        # calculate nuclear repulsion energy
        print("\n" + " "*sh + f"{'Nuclear Repulsion Calculation...':{half}}", end="")
        self.tm.start("nuc_rep", parent="single_point_scf")
        self.nuclear_repulsion()
        t = self.tm.stop("nuc_rep", parent="single_point_scf")
        print(f"{'DONE!':>{quart}}{f'{t:.6f} s':>{quart}}")

        print("\n" + " "*sh + f"{'       Electronic Energy -> ':>{half}}{self.E:>{quart}.8E}{' Ha':<{quart}}")
        print(" "*sh + f"{'Nuclear Repulsion Energy -> ':>{half}}{self.V_nuc:>{quart}.8E}{' Ha':<{quart}}")
        print("\n" + " "*sh + f"{'            Total Energy -> ':>{half}}{self.E+self.V_nuc:>{quart}.8E}{' Ha':<{quart}}" + "\n")

    # ========================================================================================= #
    #                                                                                           #
    #   Class that measures execution times within the different functions                      #
    #                                                                                           #
    # ========================================================================================= #

    class Timer():

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

        def print(self) -> None:

            sh    = self.textshift
            quart = int(self.textwidth/4)

            print(" "*sh + f"{'~ TIMES SUMMARY ~':^{self.textwidth}}")
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