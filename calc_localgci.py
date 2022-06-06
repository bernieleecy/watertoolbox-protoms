# Written by Gregory A. Ross and Matteo Aldeghi

import numpy as np
import glob
import pickle
import sys, os, re
from collections import OrderedDict
import matplotlib

if 'DISPLAY' not in os.environ:
    matplotlib.use('agg')
import matplotlib.pyplot as plt

protomshome = os.environ["PROTOMSHOME"]
sys.path.append(protomshome + "/tools")
sys.path.append(protomshome + '/python/protomslib')
import simulationobjects
from simulationobjects import ResultsFile
import protomslib.free_energy.gcmc_free_energy_base as calc_gci
import argparse


# ============================================================================================================
# Parse Options
# ============================================================================================================
def ParseOptions():
    parser = argparse.ArgumentParser(description="Program to plot GCMC titration within discrete sites.")
    mode = parser.add_mutually_exclusive_group(required=True)

    parser.add_argument('-d', '--directories', nargs="+",
                        help="the directories containing the GCMC simulation data, default=None", default=None)
    parser.add_argument('-f', '--pdbfiles', help="the input PDB-files, default=all.pdb", default="all.pdb")
    mode.add_argument('-b', '--boxes', nargs="+",
                      help="the pdb files of the box(es) within which you wish to analysis the water content,default=None",
                      default=None)
    mode.add_argument('-c', '--clusters',
                      help="a pdb file of clusters file that will be used to determine the center of the hydration spheres, default=None",
                      default=None)
    parser.add_argument('-re', '--resultsfiles', nargs="+", help="the input results, default=results",
                        default="results")
    parser.add_argument('-r', '--residue', help="the name of the residue to extract, default='wat'", default="wa1")
    parser.add_argument('-a', '--atom', help="the name of the atom to extract, default='o00'", default="o00")
    parser.add_argument('-s', '--skip', help="the number of initial snapshots that will be discarded when fitting",
                        type=int, default=0)
    parser.add_argument('-o', '--out',
                        help="the name of the pickle data package that will save the processed PDB data and ANNs, default='data.pickle'",
                        default="data.pickle")
    parser.add_argument('-i', '--input',
                        help="the name of the pickle data package that contains the pre-processed and saved ANNss, default= None",
                        default=None)
    parser.add_argument('--hydr',
                        help="Hydration free energy for the water model used (kcal/mol). Default = -6.3 kcal/mol",
                        type=float, default=-6.3)
    parser.add_argument('--calc', action='store',
                        help="Choose whether to calculate GCI for individual sites/boxes ('ind'), for multiple sites/boxes togethr ('mult'), or both ('all'). Default is 'ind'.",
                        default='ind', choices=['ind', 'mult', 'all'])
    parser.add_argument('--radius', help="Radius of the hydration sites in Angstroms, default=1.4", type=float,
                        default=1.4)
    parser.add_argument('--maxsites', type=int,
                        help="Maximum number of hydration sites to consider; the highest occupied sites will be considered first. Default is all.",
                        default=9999)
    parser.add_argument('--bootstraps',
                        help="The number of bootstrap samples performed on the free energy estimates, default=None",
                        type=int, default=None)
    parser.add_argument('--steps', nargs="+",
                        help="the number of steps to be fitted for each titration box. The input is a list of integers.",
                        default=None, type=int)
    parser.add_argument('--fit_options',
                        help="additional options to be passed to the artificial neural network as quoted string",
                        action=ParseFitDict, type=str, default=ParseFitDict.default)
    args = parser.parse_args()

    # Hydration free energy needs to be negative
    if args.hydr > 0.0:
        raise InputError('--hydr', 'The input hydration free energy must be negative')

    # Define the mode: boxes or clusters
    if args.boxes is not None:
        args.mode = "boxes"
    elif args.clusters is not None:
        args.mode = "clusters"

    return args


# =============================================================================================================
# MAIN: Read in the GCMC results data from multiple GCMC output folders and calculating the mean number of
# "on" waters, after skipping a specified number of frames.
# ============================================================================================================
def main(args):
    # -----
    # BOXES
    # -----
    if args.mode == "boxes":
        # Parse boxes
        # -----------
        boxes = [read_box(b) for b in args.boxes]
        args.num_inputs = len(boxes)

        # Define the steps
        # ----------------
        if args.steps is not None:
            if args.num_inputs != len(args.steps):
                raise InputError('--steps', 'Number of supplied steps does not equal number of supplied boxes.')
        else:
            print('WARNING: --steps not defined: a single step per box is assumed.')
            args.steps = [1] * args.num_inputs

        # Check if pickled data is provided
        # ---------------------------------
        if args.input is None:
            # get Adams and N waters
            # ----------------------
            (B, N, volumes) = read_gcmc(args.num_inputs, args.directories, args.resultsfiles, args.pdbfiles,
                                        args.residue, args.atom, args.skip, args.mode, structures=boxes)
            # Save data to file
            # -----------------
            data = {"Adams": B, "N": N, "volumes": volumes}
            pickle.dump(data, open(args.out, "wb"))
        else:
            data = pickle.load(open(args.input, "rb"))
            B = data["Adams"]
            N = data["N"]
            volumes = data['volumes']

        # Fit the ANN and carry out GCI
        # -----------------------------
        if args.calc in ('ind', 'all'):
            ANNs = gci_individual_sites(num_inputs=args.num_inputs, steps=args.steps, fit_options=args.fit_options,
                                        dg_hydr=args.hydr, mode=args.mode, B=B, N=N, volumes=volumes,
                                        bootstraps=args.bootstraps, centers=None)
            # Make Figure
            # -----------
            make_figure(B=B, N=N, ANNs=ANNs, num_inputs=args.num_inputs,
                        bootstraps=args.bootstraps, outfn='local_gci.pdf')

        if args.calc in ('mult', 'all'):
            print('WARNING: Option to analyse multiple boxes not available; only possible via --clusters')

    # --------
    # CLUSTERS
    # --------
    elif args.mode == "clusters":
        # Parse cluster file
        # ------------------
        sites_centers = read_cluster(args.clusters, args.maxsites)
        args.num_inputs = len(sites_centers)

        # Define the steps
        # ----------------
        if args.steps is not None:
            if args.num_inputs != len(args.steps):
                raise InputError('--steps', 'Number of supplied steps does not equal number of supplied boxes.')
        else:
            print('WARNING: --steps not defined: 1 step for each site is assumed.')
            args.steps = [1] * args.num_inputs

        # Check if B,N data is provided
        # -----------------------------
        if args.input is None:
            # get Adams and N waters
            # ----------------------
            (B, N, volumes) = read_gcmc(args.num_inputs, args.directories, args.resultsfiles, args.pdbfiles,
                                        args.residue, args.atom, args.skip, args.mode,
                                        structures=sites_centers, radius=args.radius)
            # Save data to file
            # -----------------
            data = {"Adams": B, "N": N, "volumes": volumes}
            pickle.dump(data, open(args.out, "wb"))
        else:
            data = pickle.load(open(args.input, "rb"))
            B = data["Adams"]
            N = data["N"]
            volumes = data['volumes']

        #TODO: Either figure out a comprimise with the volume correction or get rid of it
        volumes = [30.0] * args.num_inputs       # Setting to standard state of water so that contribution is 0.

        # Fit the ANN and carry out GCI
        # -----------------------------
        if args.calc in ('ind', 'all'):
            ANNs = gci_individual_sites(num_inputs=args.num_inputs, steps=args.steps, fit_options=args.fit_options,
                                        dg_hydr=args.hydr, B=B, N=N, mode=args.mode, volumes=volumes,
                                        centers=sites_centers, bootstraps=args.bootstraps)
            # Make Figure
            # -----------
            make_figure(B=B, N=N, ANNs=ANNs, num_inputs=args.num_inputs,
                        bootstraps=args.bootstraps, outfn='local_gci.pdf')

        if args.calc in ('mult', 'all'):
            ANN_total, N_total = gci_multiple_sites(num_inputs=args.num_inputs, steps=args.steps, fit_options=args.fit_options,
                               dg_hydr=args.hydr, B=B, N=N, centers=sites_centers, radius=args.radius, bootstraps=args.bootstraps)
            # Make Figure
            # -----------
            # TODO: return ANN from gci_cluster so to use it in make_figure
            make_figure_all_sites(B=B, N=N_total, ANNs=ANN_total, bootstraps=args.bootstraps,
                              outfn='all_sites_local_gci.pdf')

    else:
        raise InputError('--boxes, --clusters', 'Either --boxes or --clusters must be defined')

    print("\nType enter to quit\n>")
    input()


# ============================================================================================================
# Set of functions to analyse free energy for boxes or spheres
# ============================================================================================================
def read_box(gcmcbox):
    """
    Reads the parameters of a ProtoMS formatted box

    Parameters
    ----------
    gcmcbox: str
        the filename of the ProtoMS box, a PDB file

    Returns
    -------
    box: dict
        dictionary containing the dimensions, origin, and length of the box
    """
    gcmcbox = simulationobjects.PDBFile(filename=gcmcbox)
    # Try to find box information in the header
    box = simulationobjects.find_box(gcmcbox)
    if box is None:
        # if that fails, take the extent of the PDB structure
        box = gcmcbox.getBox()
    if "origin" not in box:
        box["origin"] = box["center"] - box["len"] / 2.0
    return box


def number_in_box(pdbfiles, molname, atomname, box, skip):
    """
    Function to count the number of molecules in a PDB file that are within a specified volume.
    """
    box_max = box["origin"] + box["len"]
    box_min = box["origin"]
    residue = molname.lower()
    atom = atomname.lower()
    found = 0.0
    #    for pdb in pdbfiles.pdbs :
    for k in range(skip, len(pdbfiles.pdbs)):
        pdb = pdbfiles.pdbs[k]
        for i, res in pdb.residues.items():
            if res.name.lower() == molname.lower():
                for atom in res.atoms:
                    if atom.name.strip().lower() == atomname.lower():
                        xyz = atom.coords
                        if np.all(xyz < (box_max)) and np.all(xyz > (box_min)):
                            found += 1.0
    return found / len(range(skip, len(pdbfiles.pdbs)))


def number_in_sphere(pdbfiles, molname, atomname, skip, sites, radius):
    """
    Function to count the number of molecules in a PDB file that are 
    found within a distance (radius) from specified sites.
    
    Parameters
    ----------
    pdbfiles : PDBSet object
        Objects containng the PDB trajectories.
    molname : string
        Name of the molecules containing the atoms of which the occurrence 
        will be counted.
    atomname : string
        Name of the atoms of which the occurrence will be counted.
    skip : int
        Number of equilibration frames to skip when analysis the trajectory.
    sites : ndarray
        Array of x,y,z coordinates for the centers if the hydration sites.
    radius : float
        Radii of the spheres that define the hydration sites.
        
    Returns
    -------
    count : float
        Average number of atoms found in the selected volume made up by
        all the spherical hydration sites.
    """
    found = 0.0

    for k in range(skip, len(pdbfiles.pdbs)):
        pdb = pdbfiles.pdbs[k]
        for i, res in pdb.residues.items():
            if res.name.lower() == molname.lower():
                for atom in res.atoms:
                    if atom.name.strip().lower() == atomname.lower():
                        xyz = atom.coords
                        counted = False  # to avoid double counting of waters if distance between sites is < radius*2
                        for site in sites:
                            if np.linalg.norm(site - xyz) <= radius and counted == False:
                                found = found + 1.0
                                counted = True

    count = found / len(range(skip, len(pdbfiles.pdbs)))
    return count


def read_cluster(clusfile, max_sites=9999):
    """
    Function to read the locations of the water oxygen atoms in a PDB file
    containing clustered hydration sites. The sites are grouped by their
    residue numbers.
    
    Parameters
    ----------
    clusfile : string
        Name of PDB file containing the clustered hydration sites.
    
    Returns
    -------
    site_centers : dict
        Dictionary containing arrays with the x,y,z coordinates of 
        oxygen atoms for all groups of hydration sites.
    """
    # made it >7 rather than equals 9 because the convertwater.py pdb file can have 8 items
    # while my clusters pdb file has 10 items
    lines = [l for l in open(clusfile, 'r').readlines() if len(l.split()) > 7 and l.split()[2] == 'O00']
    sites_centers = OrderedDict()
    for i, l in enumerate(lines):
        if i == max_sites:
            break
        tag = str(l.split()[4])  # define groups of sites based on their residue number
        x = float(l.split()[5])
        y = float(l.split()[6])
        z = float(l.split()[7])

        if tag not in sites_centers:
            sites_centers[tag] = []

        sites_centers[tag].append(np.array([x, y, z]))
    # added print statement to let me check if waters are detected early on
    print(sites_centers)
    return sites_centers


def volume(centers, r=1.4):
    """
    Function to calculate the volume of multiple and possibly intersecting 
    spheres given their centers. It works only for intersecting pairs of
    spheres with same radii.
    
    http://mathworld.wolfram.com/Sphere-SphereIntersection.html
    
    Parameters
    ----------
    centers : list of lists
        Array containing a list of x,y,z coordinates.
    r : radii of the spheres in angstrom.

    Returns
    -------
    V : float
        Volume of the (intersecting) spheres in cubic angstroms.
    """

    # Volume of the spheres
    spheres = (4. / 3. * np.pi * (r ** 3.)) * len(centers)

    # Volume of (pairwise) intersections
    intersections = 0.0
    for idx, i in enumerate(centers):
        for jdx, j in enumerate(centers):
            if jdx > idx:
                d = sum((centers[i][0] - centers[j][0]) ** 2) ** 0.5
                # No overlap:
                if d >= (r * 2.):
                    intersections += 0
                # Complete overlap - though this should not happen in our case
                elif d == 0.0:
                    intersections += 4. / 3. * np.pi * (r ** 3.)
                # Partial overlap
                else:
                    intersections += (np.pi * (2. * r - d) ** 2 * (d ** 2 + 4 * d * r)) / (12. * d)
    V = spheres - intersections
    return V


# ============================================================================================================
# Read pdb files and extract N and B
# ============================================================================================================
def read_gcmc(num_inputs, directories, rfilename, afilename, residue, atom, skip, mode, structures, radius=1.4):
    """
    Reads PDB files and extract N and Adams.
    """
    if mode not in ["boxes", "clusters"]:
        print("Error in read_gcmc. Mode must be either 'boxes' or 'clusters'")
        return
    B = []
    N = {}
    volumes = []
    ANNs = {}
    for b in range(num_inputs):
        N[b] = []
        ANNs[b] = []
    print("\nREADING GCMC DATA:")
    for dirs in directories:
        folders = glob.glob(dirs)
        if len(folders) == 0:
            print("\nError. No folder(s) matching '%s'. Exiting program.\n" % directories)
            sys.exit()
        for folder in folders:
            results = ResultsFile()
            resultsfiles = glob.glob(folder + "/" + rfilename + "*")
            pdbfilenames = glob.glob(folder + "/" + afilename + "*")
            if len(resultsfiles) > 1:  # It is assumed the results are from ProtoMS 2.
                results.read([folder, rfilename])
                pdbfiles = simulationobjects.PDBSet()
                for filename in pdbfilenames:
                    pdb = simulationobjects.PDBFile(filename=filename)
                    pdbfiles.pdbs.append(pdb)
            elif len(resultsfiles) == 1:  # It is assumed the results are from ProtoMS 3.
                results.read(folder + "/" + rfilename)
                pdbfiles = simulationobjects.PDBSet()
                pdbfiles.read(pdbfilenames[0])
            else:
                print("\nError. No results file matching %s. Exiting program\n" % folder + "/" + rfilename)
                sys.exit()
            adams = np.array([snap.bvalue for snap in results.snapshots])
            B.append(adams[0])
            for b in range(num_inputs):
                if mode == 'boxes':
                    N[b].append(number_in_box(pdbfiles, residue, atom, structures[b], skip))
                    v = structures[b]['len'][0] * structures[b]['len'][1] * structures[b]['len'][2]
                    volumes.append(v)
                elif mode == 'clusters':
                    # i assume structures[structures.keys()[b]] is trying to access values
                    N[b].append(
                        number_in_sphere(pdbfiles, residue, atom, skip, list(structures.values())[b], radius))
                    volumes.append(4.0 / 3.0 * np.pi * radius ** 3)

            print("...file %s has been read and processed." % pdbfilenames[0])
    B = np.array(B)
    return B, N, volumes


# ============================================================================================================
# ANN fit and GCI functions
# ============================================================================================================
def gci_individual_sites(num_inputs, steps, fit_options, dg_hydr, B, N, mode, volumes, centers=None, bootstraps=None):
    """
    Calculates the binding free energy for all sites/boxes.
    centers is needed if mode='clusters'

    Parameters
    ----------
    num_inputs : int
        Description ...
    steps : array of int
        Description ...
    fit_options : dict
        Description ...
    dg_hydr : float
        Description ...
    B : array
        Description ...
    N : array
        Description ...
    mode : str
        Description ...
    volumes : array
        Description ...
    centers : array, optional
        Description ...
    bootstraps : int
        Description ...

    Returns
    -------
    ANNs : array
        Description ...
    """

    print("\nINDIVIDUAL FREE ENERGIES:")
    print("(kcal/mol)")
    if bootstraps is not None:
        print("Values and error bars calculated using %i bootstrap samples of the titration data." % bootstraps)
        print("'Site' 'Nwat' 'Ideal gas transfer energy' 'Binding energy'   'Standard deviation'")
    else:
        print("'Site' 'Nwat' 'Ideal gas transfer energy' 'Binding energy'")

    ANNs = {}
    for b in range(num_inputs):
        x = np.array(B)
        y = np.array(N[b])
        N_range = np.arange(np.round(y.min()), np.round(y.max()) + 1)
        if mode == 'clusters':
            # centers[centers.keys()[b]] seems to be accessing the coords again?
            # rework for python 3
            Nwat = len(list(centers.values())[b])  # Nwat = N of water sites
        elif mode == 'boxes':
            Nwat = 1  # Nwat not properly defined
        if bootstraps == None:
            ANNs[b], models = calc_gci.fit_ensemble(x=x, y=y, size=steps[b], verbose=False,
                                                    pin_min=fit_options["pin_min"],
                                                    pin_max=fit_options["pin_max"],
                                                    cost=fit_options["cost"],
                                                    c=fit_options["c"],
                                                    randstarts=fit_options["randstarts"],
                                                    repeats=fit_options["repeats"],
                                                    iterations=fit_options["iterations"])
            dG_single = calc_gci.insertion_pmf(N_range, ANNs[b], volumes[b])
            # originally dG_single[1]
            print(" %4.2f   %4.2f %16.2f     %18.2f" % (b + 1, Nwat, dG_single[1], dG_single[1] - dg_hydr * Nwat))
        else:
            indices = range(x.size)
            gci_dGs = np.zeros(bootstraps)
            ANNs[b] = []
            for boot in range(bootstraps):
                sample_inds = np.random.choice(indices, size=x.size)
                x_sample = x[sample_inds]
                y_sample = y[sample_inds]
                ANNs_boot, models = calc_gci.fit_ensemble(x=x_sample, y=y_sample, size=steps[b], verbose=False,
                                                          pin_min=fit_options["pin_min"],
                                                          pin_max=fit_options["pin_max"],
                                                          cost=fit_options["cost"],
                                                          c=fit_options["c"],
                                                          randstarts=fit_options["randstarts"],
                                                          repeats=fit_options["repeats"],
                                                          iterations=fit_options["iterations"])
                ANNs[b].append(ANNs_boot)
                gci_dGs[boot] = calc_gci.insertion_pmf(N_range, ANNs_boot, volume=volumes[b])[-1]
            print(" %4.2f  %4.2f %16.2f     %18.2f  %18.2f" % (
                b + 1, Nwat, gci_dGs.mean(), gci_dGs.mean() - dg_hydr * Nwat, gci_dGs.std()))

    return ANNs

def gci_multiple_sites(num_inputs, steps, fit_options, dg_hydr, B, N, centers, radius=1.4, bootstraps=None):
    """
    The binding free energy of all the sites is calculated.
    The binding free energy calculated is relative to the lowest number of inserted molecules
    For the standard state volume correction, the entire volume encompassed by the clusters is considered

    Parameters
    ----------
    num_inputs : int
        Description ...
    steps : array of int
        Description ...
    fit_options : dict
        Description ...
    dg_hydr : float
        Description ...
    B : array
        Description ...
    N : array
        Description ...
    centers : array, optional
        Description ...
    radius : float
        Description ...
    bootstraps : int
        Description ...

    Returns
    -------

    """

    print('\nWHOLE CLUSTER FREE ENERGY:')
    print("(kcal/mol)")
    print("Relative free energies (in kcal/mol) for all the clusters in the combined volume:")
    print('The free energy is difference between the minimum number (N min) to the maximum number (N max)')
    print('A correction removes the contribution from overlapping volumes\n')
    # Counting the total volume, total number of inserted molecules, and maximum number of steps for the ANN
    total_volume = volume(centers=centers, r=radius)
    total_mols = []
    total_steps = 0

    # Adding up the counts in all the clusters
    for b in range(num_inputs):
        total_mols.append(np.array(N[b]))
        total_steps += steps[b]
    # Defining the vectors and variables of the titration data over all the clusters
    x = np.array(B)
    y = np.sum(np.array(total_mols), axis=0)
    # Taking the minimum number of molecules and maximum to calculate the relative free energy between them
    n_max = np.round(y.max())
    n_min = np.round(y.min())
    N_range = np.array((n_min, n_max))
    # Now calculate the free energy of using the volumes of all the clusters
    if bootstraps is None:
        print("'N min' 'N max' 'Ideal gas transfer energy' 'Binding energy")
        ANN_total, models = calc_gci.fit_ensemble(x=x, y=y, size=total_steps, verbose=False,
                                                  pin_min=fit_options["pin_min"],
                                                  pin_max=fit_options["pin_max"],
                                                  cost=fit_options["cost"],
                                                  c=fit_options["c"],
                                                  randstarts=fit_options["randstarts"],
                                                  repeats=fit_options["repeats"],
                                                  iterations=fit_options["iterations"])
        dG_single = calc_gci.insertion_pmf(N_range, ANN_total, total_volume)
        print(" %5.2f   %5.2f %16.2f     %18.2f" % (
            n_min, n_max, dG_single[1], dG_single[1] - dg_hydr * (n_max - n_min)))

        return ANN_total, y
    else:
        print("'N min' 'N max' 'Ideal gas transfer energy' 'Binding energy 'Standard deviation'")
        indices = range(x.size)
        gci_dGs = np.zeros(bootstraps)
        for boot in range(bootstraps):
            sample_inds = np.random.choice(indices, size=x.size)
            x_sample = x[sample_inds]
            y_sample = y[sample_inds]
            ANNs_boot, models = calc_gci.fit_ensemble(x=x_sample, y=y_sample, size=total_steps, verbose=False,
                                                      pin_min=fit_options["pin_min"],
                                                      pin_max=fit_options["pin_max"],
                                                      cost=fit_options["cost"],
                                                      c=fit_options["c"],
                                                      randstarts=fit_options["randstarts"],
                                                      repeats=fit_options["repeats"],
                                                      iterations=fit_options["iterations"])
            gci_dGs[boot] = calc_gci.insertion_pmf(N_range, ANNs_boot, volume=total_volume)[-1]
        print(" %5.2f  %5.2f %16.2f     %18.2f  %18.2f" % (
            n_min, n_max, gci_dGs.mean(), gci_dGs.mean() - dg_hydr * (n_max - n_min), gci_dGs.std()))

        return ANNs_boot, y


# ============================================================================================================
# Figures/Plotting
# ============================================================================================================
def make_figure(B, N, ANNs, num_inputs, bootstraps, outfn='local_gci.pdf'):
    # TODO: make it usable also for the results of gci_cluster()
    print(len(ANNs))
    fig, axes = plt.subplots(len(ANNs), figsize=(8.0, 5.0*num_inputs))

    for b in range(num_inputs):
        axes[b].scatter(B, N[b], color="black")
        if len(ANNs) != 0:
            if bootstraps is None:
                ANNs[b].x = np.linspace(start=B.min(), stop=B.max(), num=100)
                ANNs[b].forward()
                axes[b].plot(ANNs[b].x, ANNs[b].predicted, color="red", linewidth=3)
            else:
                plot_FitPercentiles2(ANNs[b][0].x, ANNs[b][0].y, ANNs[b], ax=axes[b])
        axes[b].set_title("Site/Box %i" % (b + 1))
        axes[b].set_xlabel("Adams value")
        axes[b].set_ylabel("Occupancy within volume")

    plt.tight_layout()
    fig.savefig(outfn)

def make_figure_all_sites(B, N, ANNs, bootstraps, outfn="all_sites_local_gci.pdf"):
    fig, axes = plt.subplots(1, sharex=True, figsize=(8.0, 5.0))

    axes.scatter(B, N, color="black")
    if bootstraps is None:
        ANNs.x = np.linspace(start=B.min(), stop=B.max(), num=100)
        ANNs.forward()
        axes.plot(ANNs.x, ANNs.predicted, color="red", linewidth=3)
    else:
        plot_FitPercentiles2(ANNs[0].x, ANNs[0].y, ANNs, ax=axes)
    axes.set_title("Titration over all sites")
    axes.set_xlabel("Adams value")
    axes.set_ylabel("Occupancy within all volumes")

    plt.tight_layout()
    fig.savefig(outfn)

def plot_FitPercentiles2(B, N, gcmc_models, resolution=None, level1=None, level1_colour=None, level2=None,
                         level2_colour=None, median_colour=None, smoothness=None, ax=None):
    """
    Tool to plot a collection of fitted ANNs at two confidence limits.

    Parameters
    ----------
    B : numpy array
      vector of the explanatory variable, the B/Adams value
    N : numpy array
      vector of the response variable, the average number of waters for a given B
    gcmc_models : list of Slp objects
      the fitted ANNs that will be plotted
    resolution : float
      the number of data points that will be plotted, i.e. resolution of plotted lines
    level1 : float
      the  confidence level (eg top 50% as a fraction) that will be plotted
    level2 : float
      the  confidence level that will be plotted
    level1_color : string
      the color of the level1 confidence region
    level2_color : string
      the color of the level2 confidence region
    median_colour : string
      colour that the median of all the fits will be plotted in
    smoothness : float
      the standard deviation over x that will be used to smooth over y
    ax : matplotlib Axes, optional
      Axes object to draw the plot onto, otherwise uses the current Axes.

    Return
    ------
    matplotlib.pyplot object
      plot object
    """
    # Setting the defaults:
    if resolution == None: resolution = 50  # Number of points with which the lines will be drawn.
    if level1 == None: level1 = 0.50  # Default is also shade the region that contains 50% of the data
    if level2 == None: level2 = 0.90  # Default is NOT to shade a region at a lower percentile level.
    if level1_colour == None: level1_colour = "orange"  # Default is also shade the region that contains 50% of the data
    if level2_colour == None: level2_colour = "gray"  # Default is to also shade the region that contains 95% of the data
    if median_colour == None: median_colour = "red"  # Colour of the median
    if smoothness == None: smoothness = 0.05  # Degree to which the lines will be smoothed (for aesthetic purposes only).
    if ax is None:
        ax = plt.gca()
    # Generating the predictions of the different gcmc_models:
    yvals = np.zeros((resolution, len(gcmc_models)))  # Will hold all the predictions from the different models
    x = np.linspace(start=B.min(), stop=B.max(), num=resolution)  # The x values that will be plotted.
    for i in range(len(gcmc_models)):
        gcmc_models[i].x = x
        gcmc_models[i].forward()
        yvals[:, i] = gcmc_models[i].predicted
    # Generating and smoothing the confidence intervals.
    y_median, y_low, y_high = calc_gci.percentile_intervals(yvals, level1)
    y_median, y_lowest, y_highest = calc_gci.percentile_intervals(yvals, level2)
    smooth_median = calc_gci.gaussian_smooth(x, y_median,
                                             sigma=smoothness * 3)  # The median is made smoother than the error bars.
    smooth_low = calc_gci.gaussian_smooth(x, y_low, sigma=smoothness)
    smooth_high = calc_gci.gaussian_smooth(x, y_high, sigma=smoothness)
    smooth_lowest = calc_gci.gaussian_smooth(x, y_lowest, sigma=smoothness)
    smooth_highest = calc_gci.gaussian_smooth(x, y_highest, sigma=smoothness)
    # Plotting:
    space = 0.07  # The amount of white space to the side of the x and y axis as a fraction of the ranges.
    ax.fill_between(x, smooth_lowest, smooth_highest, facecolor=level2_colour, linewidth=0, alpha=0.3,
                    interpolate=True)
    ax.fill_between(x, smooth_low, smooth_high, facecolor="white", linewidth=0, interpolate=True)
    ax.fill_between(x, smooth_low, smooth_high, facecolor=level1_colour, linewidth=0, alpha=0.4, interpolate=True)
    ax.set_xlim(B.min() - space * np.ptp(B), B.max() + space * np.ptp(B))
    ax.set_ylim(N.min() - space * np.ptp(N), N.max() + space * np.ptp(N))
    ax.scatter(B, N, color="black", s=40)
    ax.plot(x, smooth_median, color=median_colour, linewidth=3)
    return ax


# ============================================================================================================
# Classes
# ============================================================================================================

class Error(Exception):
    """Base class for exceptions."""
    pass


class InputError(Error):
    """Exception raised for errors in the input
    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class ParseFitDict(argparse.Action):
    """Specifies the default options for artificial neural network, and reads in values from the command line"""

    default = {"monotonic": True, "repeats": 10, "randstarts": 1000, "iterations": 100, "grad_tol_low": -3,
               "grad_tol_high": 1, "pin_min": 0.0, "pin_max": None, "cost": "msd", "c": 2.0, "verbose": False}

    def __init__(self, option_strings, *args, **kwargs):
        self.fit_dict = ParseFitDict.default
        super(ParseFitDict, self).__init__(option_strings=option_strings, *args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values is not None:
            options = values.split()
            for i in range(len(options)):
                if options[i] in self.fit_dict:
                    if type(self.fit_dict[options[i]]) == str:
                        self.fit_dict[options[i]] = str(options[i + 1])
                    if type(self.fit_dict[options[i]]) == float or self.fit_dict[options[i]] is None:
                        self.fit_dict[options[i]] = float(options[i + 1])
                    if type(self.fit_dict[options[i]]) == int:
                        self.fit_dict[options[i]] = int(options[i + 1])
                    if type(self.fit_dict[options[i]]) == str:
                        self.fit_dict[options[i]] = str(options[i + 1])
        setattr(namespace, self.dest, self.fit_dict)


if __name__ == "__main__":
    args = ParseOptions()
    main(args)
