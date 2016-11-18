# Written by Gregory A. Ross
# Contributed to by Matteo Aldeghi

import numpy as np
import glob
import pickle
import sys, os, re
from collections import OrderedDict
import matplotlib.pyplot as plt
protomshome = os.environ["PROTOMSHOME"]
sys.path.append(protomshome +"/tools")
import simulationobjects
import calc_gci
from simulationobjects import ResultsFile
#============================================================================================================
# Parse options
#============================================================================================================
def ParseOptions():
    
    import argparse

    parser = argparse.ArgumentParser(description="Program to plot GCMC titration within discrete sites.")
    mode = parser.add_mutually_exclusive_group(required=True)
    
    parser.add_argument('-d','--directories',nargs="+",help="the directories containing the GCMC simulation data, default=None", default=None)
    parser.add_argument('-f','--pdbfiles',help="the input PDB-files, default=all.pdb",default="all.pdb")
    mode.add_argument('-b','--boxes',nargs="+",help="the pdb files of the box(es) within which you wish to analysis the water content,default=None",default=None)
    mode.add_argument('-c','--clusters',help="a pdb file of clusters file that will be used to determine the center of the hydration spheres, default=None", default=None)
    parser.add_argument('-re','--resultsfiles',nargs="+",help="the input results, default=results",default="results")
    parser.add_argument('-r','--residue',help="the name of the residue to extract, default='wat'",default="wa1")
    parser.add_argument('-a','--atom',help="the name of the atom to extract, default='o00'",default="o00")
    parser.add_argument('-s','--skip',help="the number of initial snapshots that will be discarded when fitting",type=int,default=0)
    parser.add_argument('-o','--out',help="the name of the pickle data package that will save the processed PDB data and ANNs, default='data.pickle'",default="data.pickle")
    parser.add_argument('-i','--input',help="the name of the pickle data package that contains the pre-processed and saved ANNss, default= None",default=None)
    parser.add_argument('--radius',help="Radius of the hydration sites in Angstroms, default=1.4",type=float,default=1.4)
    parser.add_argument('--maxsites',type=int,help="Maximum number of hydration sites to consider; the highest occupied sites will be considered first. Default is all.",default=9999)
    parser.add_argument('--bootstraps',help="The number of bootstrap samples performed on the free energy estimates, default=None",type=int,default=None)
    parser.add_argument('--steps',nargs="+", help="the number of steps to be fitted for each titration box",default=None)
    parser.add_argument('--fit_options',help="additional options to be passed to the artificial neural network",default=None)
    parser.add_argument('--pdf',action='store_true',help="whether to save figures as PDFs",default=False)
    
    args = parser.parse_args()
    
    return args
    
#============================================================================================================
# Set of functions to analyse free energy for boxes or spheres
#============================================================================================================
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
    if box is None :
    # if that fails, take the extent of the PDB structure
      box = gcmcbox.getBox()  
    if "origin" not in box :
      box["origin"] = box["center"] - box["len"]/2.0
    return box

def number_in_box(pdbfiles, molname, atomname, box, skip):
    """
    Function to count the number of molecules in a PDB file that are within a specified volume.
    """
    box_max = box["origin"] + box["len"]
    box_min = box["origin"]
    residue = molname.lower()
    atom    = atomname.lower()
    found = 0.0
#    for pdb in pdbfiles.pdbs :
    for k in range(skip,len(pdbfiles.pdbs)):
      pdb = pdbfiles.pdbs[k]
      for i,res in pdb.residues.iteritems() :
          if res.name.lower() == molname.lower():
              for atom in res.atoms :
                  if atom.name.strip().lower() == atomname.lower() :
                      xyz = atom.coords
                      if np.all(xyz < (box_max)) and np.all(xyz > (box_min)) :
                          found = found + 1.0
    return found/len(range(skip,len(pdbfiles.pdbs)))

def read_cluster(clusfile,max_sites=9999):
    """
    Function to read the locations of the water oxygen atoms in a PDB file.
    """
    lines = [l for l in open(clusfile,'r').readlines() if len(l.split()) == 9 and l.split()[2]=='O00']
    sites_centers = OrderedDict()
    for i,l in enumerate(lines):
        if i == max_sites: break
        tag = str(l.split()[4])   # define groups of sites based on their residue number
        x = float(l.split()[5])
        y = float(l.split()[6])
        z = float(l.split()[7])

        if tag not in sites_centers:
            sites_centers[tag] = []

        sites_centers[tag].append(np.array([x,y,z]))
    return sites_centers

def volume(centers, r=1.4):
    """
    Works only for intersecting pairs of spheres with same radii.
    http://mathworld.wolfram.com/Sphere-SphereIntersection.html

    centers is an list of lists
    """

    # Volume of the spheres
    spheres = (4./3. * np.pi * (r**3.)) * len(centers)

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
                    intersections += 4./3. * np.pi * (r**3.)
                # Partial overlap
                else:
                    intersections += (np.pi * (2.*r - d)**2 * (d**2 + 4*d*r)) / (12.*d)
    V = spheres-intersections
    return V
#============================================================================================================
# Set of functions to analyse free energy for supplied PDB coordinates of water molecules
#============================================================================================================
def read_cluster(clusfile,max_sites=9999):
    lines = [l for l in open(clusfile,'r').readlines() if len(l.split()) == 9 and l.split()[2]=='O00']
    sites_centers = OrderedDict()
    for i,l in enumerate(lines):
        if i == max_sites: break
        tag = str(l.split()[4])   # define groups of sites based on their residue number
        x = float(l.split()[5])
        y = float(l.split()[6])
        z = float(l.split()[7])
        
        if tag not in sites_centers:
            sites_centers[tag] = []
        
        sites_centers[tag].append(np.array([x,y,z]))
    return sites_centers

def number_in_sphere(pdbfiles,molname,atomname,skip,sites,radius):
    found = 0.0

    for k in range(skip,len(pdbfiles.pdbs)):
        pdb = pdbfiles.pdbs[k]
        for i,res in pdb.residues.iteritems():
            if res.name.lower() == molname.lower():
                for atom in res.atoms:
                    if atom.name.strip().lower() == atomname.lower():
                        xyz = atom.coords
                        counted = False   # to avoid double counting of waters if distance between sites is < radius*2
                        for site in sites:
                            if np.linalg.norm(site-xyz) <= radius and counted == False:
                                found = found + 1.0
                                counted = True

    return found/len(range(skip,len(pdbfiles.pdbs)))

#============================================================================================================
# Read all pdb files and extract N and B
#============================================================================================================
def read_gcmc(num_inputs, directories, rfilename, afilename, residue,atom, skip, mode, structures, radius=1.4):
    if mode not in ["boxes","clusters"]:
        print "Error in read_gcmc. Mode must be either 'boxes' or 'clusters'"
        return
    B = []
    N = {}
    volumes = []
    ANNs = {}
    for b in range(num_inputs):
        N[b] = []
        ANNs[b] = []
    print "\nREADING GCMC DATA:"
    for dirs in directories:
        folders =  glob.glob(dirs)
        if len(folders)==0:
            print "\nError. No folder(s) matching '%s'. Exiting program.\n" % directories
            sys.exit()
        for folder in folders:
            results = ResultsFile()
            resultsfiles = glob.glob(folder+ "/"+rfilename+"*")
            pdbfilenames = glob.glob(folder+ "/"+afilename+"*")
            if len(resultsfiles) > 1:				# It is assumed the results are from ProtoMS 2.
                results.read([folder,rfilename])
                pdbfiles = simulationobjects.PDBSet()
                for filename in pdbfilenames :
                    pdb = simulationobjects.PDBFile(filename=filename)
                    pdbfiles.pdbs.append(pdb)
            elif len(resultsfiles)==1:				# It is assumed the results are from ProtoMS 3.
                results.read(folder+ "/"+rfilename)
                pdbfiles = simulationobjects.PDBSet()
                pdbfiles.read(pdbfilenames[0])
            else:
                print "\nError. No results file matching %s. Exiting program\n" % folder+"/"+rfilename
                sys.exit()
            adams = np.array([snap.bvalue for snap in results.snapshots])
            B.append(adams[0])
            for b in range(num_inputs):
                if mode == 'boxes':
                    N[b].append(number_in_box(pdbfiles, residue,atom, structures[b], skip))
                    v = structures[b]['len'][0] * structures[b]['len'][1] * structures[b]['len'][2]
                    volumes.append(v)
                elif mode == 'clusters':
                    N[b].append(number_in_sphere(pdbfiles, residue, atom, skip, structures[structures.keys()[b]], radius))
                    volumes.append(4.0/3.0 * np.pi * radius**3)
                    
            print "...file %s has been read and processed." % pdbfilenames[0]
    B = np.array(B)
    return B, N, volumes


#============================================================================================================
# Miscellaneous
#============================================================================================================
def MergePDFs(filenames,outfname):
    from natsort import natsorted
    
    merger = PdfFileMerger()
    for filename in natsorted(filenames):
        merger.append(PdfFileReader(file(filename, 'rb')))
    
    merger.write(outfname)

def purge(dire, pattern):
    # Remove files in dir if they match a pattern
    for f in os.listdir(dire):
        if re.search(pattern, f):
            os.remove(os.path.join(dire, f))


#============================================================================================================
# MAIN: Read in the GCMC results data from multiple GCMC output folders and calculating the mean number of 
# "on" waters, after skipping a specified number of frames.
#============================================================================================================
if __name__ == "__main__":
    args = ParseOptions()
    dG_hyd = -6.3

    # Parse boxes/clusters
    #TODO: Fix potential conflict if boxes and clusters are supplied
    #TODO: If no boxes or clusters are supplied, what is the default behaviour?
    if args.boxes is not None:
        mode= "boxes"
        boxes = [read_box(b) for b in args.boxes]
        num_inputs = len(boxes)
    elif args.clusters is not None:
        mode = "clusters"
        sites_centers = read_cluster(args.clusters, args.maxsites)
        num_inputs = len(sites_centers)

    if args.steps != None:
        steps =  [int(s) for s in args.steps] 
        if num_inputs != len(steps):
            print "\nInput error: Number of supplied steps does not equal number of supplied boxes."
            sys.exit()
    else:
        steps = [1]*num_inputs

    # get Adams and N waters
    if args.input == None:
        if args.clusters != None:
            (B, N, volumes) = read_gcmc(num_inputs,args.directories,args.resultsfiles,args.pdbfiles,args.residue,args.atom,args.skip,mode,structures=sites_centers,radius=args.radius)
        elif args.boxes != None and args.clusters == None:
            (B, N, volumes) = read_gcmc(num_inputs,args.directories,args.resultsfiles,args.pdbfiles,args.residue,args.atom,args.skip,mode,structures=boxes)
        data = {"Adams":B,"N":N}
        pickle.dump( data, open(args.out, "wb" ) )
    else:
        data = pickle.load( open( args.input, "rb" ) )
        B = data["Adams"]
        N = data["N"]
    
  # Specifying the default options for artificial neural network, and reading in values from the command line.
    fit_dict = {"monotonic":True, "repeats":10, "randstarts":1000, "iterations":100, "grad_tol_low":-3, "grad_tol_high":1, "pin_min":0.0, "pin_max":None, "cost":"msd", "c":2.0, "verbose":False}
    if args.fit_options is not None:
        options = args.fit_options.split()
        for i in range(len(options)):
            if options[i] in fit_dict:
                if type(fit_dict[options[i]]) == str:
                    fit_dict[options[i]]= str(options[i+1])
                if type(fit_dict[options[i]]) == float or fit_dict[options[i]] == None:
                    fit_dict[options[i]]= float(options[i+1])
                if type(fit_dict[options[i]]) == int:
                    fit_dict[options[i]]= int(options[i+1])
                if type(fit_dict[options[i]]) == str:
                    fit_dict[options[i]]= str(options[i+1])

    #-------------------------------------------------------------------
    # Fit the ANN
    #-------------------------------------------------------------------
    print "\nINDIVIDUAL FREE ENERGIES:"
    print "(kcal/mol)"
    if args.bootstraps != None:
        print "Values and error bars calculated using %i bootstrap samples of the titration data." % args.bootstraps
        print "'Site' 'Nwat' 'Ideal gas transfer energy' 'Binding energy'   'Standard deviation'"
    else:
        print "'Site' 'Nwat' 'Ideal gas transfer energy' 'Binding energy'"

    ANNs = {}
    for b in range(num_inputs):
        x = np.array(B)
        y = np.array(N[b])
        N_range = np.arange(np.round(y.min()),np.round(y.max())+1)
        if mode == 'clusters':
            Nwat = len(sites_centers[sites_centers.keys()[b]]) # Nwat = N of water sites??
        elif mode == 'boxes':
            Nwat = 1 # Nwat??    
        if args.bootstraps == None:
            ANNs[b], models = calc_gci.fit_ensemble(x=x,y=y,size=steps[b],verbose=False,pin_min=fit_dict["pin_min"],pin_max=fit_dict["pin_max"],cost=fit_dict["cost"],c=fit_dict["c"],randstarts=fit_dict["randstarts"],repeats=fit_dict["repeats"],iterations=fit_dict["iterations"])
            dG_single = calc_gci.insertion_pmf(N_range, ANNs[b], volumes[b])
            print " %4.2f   %4.2f %16.2f     %18.2f" %(b+1, Nwat, dG_single[1], dG_single[1] - dG_hyd*Nwat)
        else:
            indices = range(x.size)
            gci_dGs = np.zeros(args.bootstraps)
            inflection_dGs = np.zeros(args.bootstraps)
            ANNs[b]=[]
            for boot in range(args.bootstraps):
                sample_inds = np.random.choice(indices,size=x.size)
                x_sample = x[sample_inds]
                y_sample = y[sample_inds]
                ANNs_boot, models = calc_gci.fit_ensemble(x=x_sample,y=y_sample,size=steps[b],verbose=False,pin_min=fit_dict["pin_min"],pin_max=fit_dict["pin_max"],cost=fit_dict["cost"],c=fit_dict["c"],randstarts=fit_dict["randstarts"],repeats=fit_dict["repeats"],iterations=fit_dict["iterations"])
                ANNs[b].append(ANNs_boot)
                gci_dGs[boot] = calc_gci.insertion_pmf(N_range, ANNs_boot, volume=volumes[b])[-1]
                #inflection_dGs[boot] = -ANNs_boot.weights[0][0]/ANNs[b].weights[0][1]*0.592        # To estimate the free energy from the point of inflection.
            print " %4.2f  %4.2f %16.2f     %18.2f  %18.2f" %(b+1, Nwat, gci_dGs.mean() , gci_dGs.mean()- dG_hyd*Nwat, gci_dGs.std() )

    # If using clusters, then the binding free energy of the all the clusters is calculated.
    # The binding free energy calculated is relative to the lowest number of inserted molecules
    # For the standard state volume correction, the entire volume encompassed by the clusters is considered
    if args.clusters is not None:
        print '\nWHOLE CLUSTER FREE ENERGY:'
        print "(kcal/mol)"
        print "Relative free energies (in kcal/mol) for all the clusters in the combined volume:"
        print 'The free energy is difference between the minimum number (N min) to the maximum number (N max)'
        print "A correction removes the contribution from overlapping volumes\n"

        # Counting the total volume, total number of inserted molecules, and maximum number of steps for the ANN
        total_volume = volume(centers=sites_centers, r=args.radius)
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
        if args.bootstraps is None:
            print "'N min' 'N max' 'Ideal gas transfer energy' 'Binding energy"
            ANN_total, models = calc_gci.fit_ensemble(x=x,y=y,size=total_steps,verbose=False,pin_min=fit_dict["pin_min"],pin_max=fit_dict["pin_max"],cost=fit_dict["cost"],c=fit_dict["c"],randstarts=fit_dict["randstarts"],repeats=fit_dict["repeats"],iterations=fit_dict["iterations"])
            dG_single = calc_gci.insertion_pmf(N_range, ANN_total, total_volume)
            print " %5.2f   %5.2f %16.2f     %18.2f" %(n_min, n_max, dG_single[1], dG_single[1] - dG_hyd*(n_max - n_min))
        else:
            print "'N min' 'N max' 'Ideal gas transfer energy' 'Binding energy 'Standard deviation'"
            indices = range(x.size)
            gci_dGs = np.zeros(args.bootstraps)
            inflection_dGs = np.zeros(args.bootstraps)
            for boot in range(args.bootstraps):
                sample_inds = np.random.choice(indices,size=x.size)
                x_sample = x[sample_inds]
                y_sample = y[sample_inds]
                ANNs_boot, models = calc_gci.fit_ensemble(x=x_sample,y=y_sample,size=total_steps,verbose=False,pin_min=fit_dict["pin_min"],pin_max=fit_dict["pin_max"],cost=fit_dict["cost"],c=fit_dict["c"],randstarts=fit_dict["randstarts"],repeats=fit_dict["repeats"],iterations=fit_dict["iterations"])
                gci_dGs[boot] = calc_gci.insertion_pmf(N_range, ANNs_boot, volume=total_volume)[-1]
            print " %5.2f  %5.2f %16.2f     %18.2f  %18.2f" %(n_min, n_max, gci_dGs.mean() , gci_dGs.mean()- dG_hyd*(n_max - n_min), gci_dGs.std() )

    #-------------------------------------------------------------------
    # Make Figures
    #-------------------------------------------------------------------
    print "\n"
    
    for b in range(num_inputs):
        plt.figure("GCMC Titration of Box "+str(b+1))
        currfig = plt
        currfig.scatter(B, N[b],color="black")
        if len(ANNs) != 0:
            if args.bootstraps == None:
                ANNs[b].x = np.linspace(start=B.min(),stop=B.max(),num=100)
                ANNs[b].forward()
                currfig.plot(ANNs[b].x,ANNs[b].predicted,color="red",linewidth=3)
            else:
                currfig = calc_gci.plot_FitPercentiles(ANNs[b][0].x,ANNs[b][0].y,ANNs[b])
        currfig.title("Site %i" %(b+1))
        currfig.xlabel("Adams value")
        currfig.ylabel("Occupancy within volume" )
        if args.pdf == True:
            currfig.savefig("Local_gci_site_%s.pdf" %(b+1))
        currfig.show(block=False)
    

    if args.pdf == True:
        try:
            from PyPDF2 import PdfFileMerger, PdfFileReader
            filenames = glob.glob('Local_gci_site*pdf')
            MergePDFs(filenames,'local_gci.pdf')
            purge('.', 'Local_gci_site*')
        except ImportError:
            pass

    print "\nType enter to quit\n>"
    raw_input()
