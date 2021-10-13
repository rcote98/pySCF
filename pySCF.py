#!/usr/bin/env python3

""" Performs Hartree-Fock quantum chemistry calculations.

Functions:
    - main()
    - print_header()

Classes:
    - Timer()
    - HF()
"""

from math import erf
import numpy as np
import argparse
import sys, os
import time

TEXTW = 65
TEXTS = 4

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def main() -> None:

    """ Main program loop. """

    p = argparse.ArgumentParser(
        prog="pySCF.py",
        description="Performs a Hartree-Fock SCF calculation.",
        epilog="That's all, folks!"
        )
    p.add_argument("input_file", type=str, nargs=1, action="store",
                help="input file for the program.")
    p.add_argument("--read-integrals", action="store_true",
                    help="read integrals from the input file.")
    args = p.parse_args()

    print_header()                 # print the program header

    # check if the input file exists
    if not os.path.isfile(args.input_file[0]):
        print(f"ERROR: Input file '{args.input_file[0]}' does not exist.")
        sys.exit()

    sim = HF(args.input_file[0])    # initialize the program
    sim.print_sim_info()           # print the information from the input file

    """
    sim.read_integrals()
    print("read integrals:")
    print(sim.O)
    print("")
    sim.calc_integrals()
    print("calculated integrals:")
    print(sim.O)
    """

    sim.calc_integrals()

    sim.run_single_point()          # run scf calculation

    sim.tm.print()                  # print timers

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
    


    def __init__(self, input_file:str) -> None:

        """
        Initializes the class and parses all the input file information.

        Parameters
        ----------
            input_file : str
                input file containing all the SCF information
        """

        
        self.tm = Timer()
        self.input_file = input_file
        
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

            self.tm.stop("read_input")

        pass

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
        print("")
        print(" "*sh + f"{'Number of Atoms:':{half}}{self.nats:>{half}}")
        print(" "*sh + f"{'Input  Geometry:':<{textwidth}}")
        print("")
        print(" "*sh + "-"*textwidth)
        print(" "*sh + f"{'Index':^{eigth}}{'Label':^{eigth}}{'x':^{quart}}{'y':^{quart}}{'z':^{quart}}")
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
            
        self.tm.stop("2e_ints", parent="read_integrals")
        self.tm.stop("read_integrals")

        return self.S, self.T, self.V, self.O

    def calc_integrals(self) -> None:

        """ Calculates the integrals using the structure and basis set information. """

        self.tm.start("calc_integrals")

        K       = self.norbs
        geom    = self.positions*1.8897161646320724    # atomix positions in bohr
        charge  = self.atnums
        nprims  = self.basis_nprims
        values  = self.basis_values
        centers = self.basis_centers
        
        self.S  = np.zeros((K, K), dtype=float)        # overlap matrix
        self.T  = np.zeros((K, K), dtype=float)        # kinetic integrals
        self.V  = np.zeros((K, K), dtype=float)        # nuclear integrals
        self.O  = np.zeros((K, K, K, K), dtype=float)  # two-electron integrals

        self.tm.start("1e_ints", parent="calc_integrals")

        # iterate over the required one-center integrals
        for j in range(K):             
                for i in range(j+1):
                
                    # iterate over primitives
                    for k in range(nprims[i]):          
                        for l in range(nprims[j]):

                            zeta    = values[i,k,0] + values[j,l,0]
                            xi      = values[i,k,0]*values[j,l,0]/zeta
                            dist    = self.distance_bohr(centers[i],centers[j])**2

                            d_ijkl  = values[i,k,1]*values[j,l,1]
                            S_ikjl  = np.exp(-xi*dist)*(np.pi/zeta)**(1.5)
                            T_ikjl  = xi*(3 - 2*xi*dist)*S_ikjl

                            self.S[i,j] += d_ijkl*S_ikjl
                            self.T[i,j] += d_ijkl*T_ikjl

                            for n in range(self.nats):  # iterate over nuclei       
                                
                                V_ikjlN = -2*charge[n]*np.sqrt(zeta/np.pi)*S_ikjl 
                                
                                # check if basis funcs are on the same atom for the Bois function
                                if centers[i] == centers[j] == n:
                                    bois0   = 1
                                else:
                                    g_ctr   = (values[i,k,0]*geom[centers[i]] + values[j,l,0]*geom[centers[j]])/zeta
                                    x       = zeta*np.linalg.norm(geom[n] - g_ctr)**2 
                                    bois0   = 0.5*np.sqrt(np.pi/x)*erf(np.sqrt(x))
                                
                                self.V[i,j] += d_ijkl*V_ikjlN*bois0

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

                                                        # normalization
                                                        d_iajbkcld = values[i,a,1]*values[j,b,1]*values[k,c,1]*values[l,d,1]

                                                        # bra part
                                                        zeta1   = values[i,a,0]+values[j,b,0]
                                                        xi1     = values[i,a,0]*values[j,b,0]/zeta1
                                                        dist1   = self.distance_bohr(centers[i],centers[j])**2
                                                       
                                                        # ket part
                                                        zeta2   = values[k,c,0]+values[l,d,0]
                                                        xi2     = values[k,c,0]*values[l,d,0]/zeta2
                                                        dist2   = self.distance_bohr(centers[k],centers[l])**2
                                                       
                                                        K_ijab = np.sqrt(2)*np.pi**(5./4.)/zeta1*np.exp(-xi1*dist1)
                                                        K_klcd = np.sqrt(2)*np.pi**(5./4.)/zeta2*np.exp(-xi2*dist2)

                                                        rho     = zeta1*zeta2/(zeta1+zeta2) 
                                                        g_ctr1  = (values[i,a,0]*geom[centers[i]] + values[j,b,0]*geom[centers[j]])/zeta1
                                                        g_ctr2  = (values[k,c,0]*geom[centers[k]] + values[l,d,0]*geom[centers[l]])/zeta2
                                                        g_dist = np.linalg.norm(g_ctr1 - g_ctr2)

                                                        # Boys function check
                                                        if g_dist < 1e-5:
                                                            bois0 = 1
                                                        else:
                                                            rho     = zeta1*zeta2/(zeta1+zeta2)  
                                                            x       = rho*g_dist**2 
                                                            bois0   = 0.5*np.sqrt(np.pi/x)*erf(np.sqrt(x))

                                                        self.O[i,j,k,l] += d_iajbkcld*K_ijab*K_klcd*bois0

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

    def distance_bohr(self, i:int, j:int) -> float:
        return np.linalg.norm(self.positions[i,:] - self.positions[j,:])*1.8897161646320724
 
    def distance_angstrom(self, i:int, j:int) -> float:
        return np.linalg.norm(self.positions[i,:] - self.positions[j,:])

    def scf_setup(self, read:bool=True) -> None:

        """
        Set up all the SCF reusable variables, such as 
        the integrals and the transformation matrix.

        Parameters
        ----------
            read : bool (default: True)
                read the integrals from the input file?
        """

        # obtain the integrals
        if read:
            self.read_integrals()
        else:
            self.calc_integrals()

        # initial value of the energy
        self.E = 0

        # initial guess for density matrix
        self.P = np.identity(self.norbs)

        # diagonalize overlap matrix
        self.eigs, self.U = np.linalg.eigh(self.S)

        # build X matrix
        sd = np.diag(self.eigs)
        sd12 = np.diag(1/np.sqrt(self.eigs))

        self.X = np.matmul(self.U,sd12)

        pass

    def scf_step(self) -> None:

        K = self.norbs

        # build the G matrix
        G = np.zeros((K,K))
        for u in range(K):
            for v in range(K):
                for l in range(K):
                    for s in range(K):
                        G[u,v] += self.P[l,s]*(self.O[u,v,s,l] - 0.5*self.O[u,l,s,v])

        # build Fock operator
        Hc = self.T + self.V
        F  = self.T + self.V + G

        # diagonalize Fock operator
        Fp = np.matmul(np.linalg.inv(self.X), np.matmul(F, self.X))

        # coefficient matrix
        energies, Cp = np.linalg.eigh(Fp)
        C = np.matmul(self.X,Cp)

        # new total energy
        self.E = 0
        for u in range(K):
            for v in range(K):
                self.E += 0.5*self.P[u,v]*(Hc[u,v]+F[u,v])

        # new density matrix
        self.P = np.zeros((K,K))
        for l in range(K):
            for s in range(K):
                for i in range(int(self.nels/2)):
                    self.P[l, s] += 2*C[l, i]*np.conjugate(C)[s,i]

        pass

    def nuclear_repulsion(self) -> float:

        """ Calculates the nuclear repulsion energy. """

        self.V_nuc = 0
        for i in range(self.nats-1):
            for j in range(i+1, self.nats):
                self.V_nuc += self.atnums[i]*self.atnums[j]/self.distance_bohr(i,j)
        return self.V_nuc

    def run_single_point(self,
                        max_iter:int   = 100,
                        e_thresh:float = 1e-8,
                        p_thresh:float = 1e-8,
                        textwidth:int = TEXTW,
                        textshift:int = TEXTS) -> None:

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
  
        # set up one non-iterative quantities
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
            
            self.dE = np.abs(self.E - self.E_old)
            self.dP = np.max(np.abs(self.P - self.P_old))

            print(" "*sh + f"{it+1:^{quart}}{self.E:^{quart}.8E}{self.dE:^{quart}.8E}{self.dP:^{quart}.8E}")

            if (self.dE < e_thresh) and (self.dP < p_thresh):
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

        print("\n" + " "*sh + f"{'       Electronic Energy -> ':>{half}}{self.E:>{quart}.8E}{'Ha':<{quart}}")
        print(" "*sh + f"{'Nuclear Repulsion Energy -> ':>{half}}{self.V_nuc:>{quart}.8E}{'Ha':<{quart}}")
        print("\n" + " "*sh + f"{'            Total Energy -> ':>{half}}{self.E+self.V_nuc:>{quart}.8E}{'Ha':<{quart}}" + "\n")


class Timer():

    def __init__(self, counter=time.perf_counter) -> None:

        self.times = {}
        self.counter = counter

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

    def print(self, textwidth:int=TEXTW, textshift=TEXTS) -> None:

        sh    = textshift 
        half  = int(textwidth/2)
        quart = int(textwidth/4)

        print(" "*sh + f"{'~ TIMES SUMMARY ~':^{textwidth}}")
        print("")
        print(" "*sh + "-"*textwidth)
        print(" "*sh + f"{'Timer':<{quart}}{'Cycles':^{quart}}{'Average (s)':^{quart}}{'Total (s)':^{quart}}")
        print(" "*sh + "-"*textwidth)
        for timer in self.times:
            deltas = self.times[timer][1]
            print(" "*sh + f"{timer:<{quart}}{len(deltas):^{quart}}{np.mean(deltas):^{quart}.8f}{np.sum(deltas):^{quart}.8f}")
            for child in self.times[timer][2]:
                deltas = self.times[timer][2][child][1]
                print(" "*sh + f"{f'   -> {child}':<{quart}}{len(deltas):^{quart}}{np.mean(deltas):^{quart}.8f}{np.sum(deltas):^{quart}.8f}")
        print(" "*sh + "-"*textwidth + "\n")


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

if __name__ == "__main__":
    main()