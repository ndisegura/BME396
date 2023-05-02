# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 09:07:07 2023

@author: asegura
"""

# Imports
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import matplotlib.colors as clrs
import cometspy as cp
import cobra as cb
import os

## Part I: Visualize Enzymes from AlphaFold
from Bio.PDB import *
import nglview as nv
import ipywidgets



plt.close('all');

#Visualize Enzymes related to Glycolisis Pathway
pdb_parser = PDBParser()
structure = pdb_parser.get_structure("P", "Hexokinase.pdb")
view = nv.show_biopython(structure)
view

structure = pdb_parser.get_structure("P", "phosphoglucoseisomerase.pdb")
view = nv.show_biopython(structure)
view

structure = pdb_parser.get_structure("P", "phosphofructoinase.pdb")
view = nv.show_biopython(structure)
view

structure = pdb_parser.get_structure("P", "fructose biphosphate aldolase.pdb")
view = nv.show_biopython(structure)
view

structure = pdb_parser.get_structure("P", "triosephosphate isomerase.pdb")
view = nv.show_biopython(structure)
view

structure = pdb_parser.get_structure("P", "glyceraldehyde-3-phosphate dehydrogenase.pdb")
view = nv.show_biopython(structure)
view

structure = pdb_parser.get_structure("P", "Phosphoglycerate kinase.pdb")
view = nv.show_biopython(structure)
view

structure = pdb_parser.get_structure("P", "Phosphoglycerate mutase.pdb")
view = nv.show_biopython(structure)
view

structure = pdb_parser.get_structure("P", "enolase.pdb")
view = nv.show_biopython(structure)
view

structure = pdb_parser.get_structure("P", "Pyruvate_kinase.pdb")
view = nv.show_biopython(structure)
view

#%% Part II:
#Michaeles-Menten Kinetic Reactions
import cobra
from cobra import Model, Reaction, Metabolite

# Define Dynamics
# ODE
def MM_dynamics_pair(t,y,K_cat,K_M,E_0):

    # y[1] = B product (Glucose-6-phospate)
    # y[0] = A reactant (Glucose)
    if y[0]<0:
        y[0]=0
    dydt = np.zeros(2)
    
    v = K_cat*E_0*(y[0]/(K_M+y[0]))
        
    dydt[0] = -v
    dydt[1] = v
      
    return dydt
    
def MM_dynamics_few(t,y,K_cat,K_M,E_0):   
    # y[0] = A reactant (Glucose)
    # y[1] = B product (Glucose-6-phospate) ->
    # y[2] = C product (Fructose-6-phosphate) ->w
    # y[3] = D product (Fructose-1,6-biphosphate)
    if y[1]<0:
        y[1]=0
    dydt = np.zeros(5)
    a = K_cat*E_0*(y[0]/(K_M+y[0]))
    b = K_cat*E_0*(y[1]/(K_M+y[1]))
    c = K_cat*E_0*(y[2]/(K_M+y[2]))
    d = K_cat*E_0*(y[3]/(K_M+y[3]))
    dydt[0] = -a
    dydt[1] = a-b
    dydt[2] = b-c
    dydt[3] = c-d
    dydt[4] = d
        
    return dydt

def MM_dynamics_full(t,y,K_cat,K_M,E_0):   
    # y[0] = A reactant (Glucose)
    # y[1] = B product (Glucose-6-phospate)
    # y[2] = C product (Fructose-6-phosphate) 
    # y[3] = D product (Fructose-1,6-biphosphate)
    if y[1]<0:
        y[1]=0
    dydt = np.zeros(10)
    a = K_cat[0]*E_0*(y[0]/(K_M[0]+y[0]))
    b = K_cat[1]*E_0*(y[1]/(K_M[1]+y[1]))
    c = K_cat[2]*E_0*(y[2]/(K_M[2]+y[2]))
    d = K_cat[3]*E_0*(y[3]/(K_M[3]+y[3]))
    e = K_cat[4]*E_0*(y[4]/(K_M[4]+y[4]))
    f = K_cat[5]*E_0*(y[5]/(K_M[5]+y[5]))
    g = K_cat[6]*E_0*(y[6]/(K_M[6]+y[6]))
    h = K_cat[7]*E_0*(y[7]/(K_M[7]+y[7]))
    i = K_cat[8]*E_0*(y[8]/(K_M[8]+y[8]))
    
    dydt[0] = -a
    dydt[1] = a-b
    dydt[2] = b-c
    dydt[3] = c-d
    dydt[4] = d-e
    dydt[5] = e-f
    dydt[6] = f-g
    dydt[7] = g-h
    dydt[8] = h-i
    dydt[9] = i
    
    return dydt

# Define some Generic Constants
K_cat0 = 3430 #3430
K_M0 = 38 #38
# Variables
E_0 = 10e-3

#Define some specific constants from BRENDA
Km_hexokinase = 0.1 # mM
Kcat_hexokinase = 80 # mM/s

Km_glucose6phosphate_isomerase = 0.01 # mM
Kcat_glucose6phosphate_isomerase = 174 # mM/s

Km_phosphofructokinase = 0.05 # mM
Kcat_phosphofructokinase = 47 # mM/s

Km_aldoase = 0.3 # mM
Kcat_aldoase = 4 # mM/s

Km_triose_phosphate_isomerase = 0.12 # mM
Kcat_triose_phosphate_isomerase = 240 # mM/s

Km_glyceraldehyde3phosphate_dehydrogenase = 0.13 # mM
Kcat_glyceraldehyde3phosphate_dehydrogenase = 100 # mM/s

Km_phosphoglycerate_mutase = 1.5 # mM
Kcat_phosphoglycerate_mutase = 70 # mM/s

Km_enolase = 0.4 # mM
Kcat_enolase = 80 # mM/s

Km_pyruvate_kinase = 0.05 # mM
Kcat_pyruvate_kinase = 190 # mM/s

#Group Constants Together
K_M=[Km_hexokinase,Km_glucose6phosphate_isomerase,Km_phosphofructokinase,Km_aldoase,Km_triose_phosphate_isomerase,
    Km_glyceraldehyde3phosphate_dehydrogenase,Km_phosphoglycerate_mutase,Km_enolase,Km_pyruvate_kinase]
K_cat=[Kcat_hexokinase,Kcat_glucose6phosphate_isomerase,Kcat_phosphofructokinase,Kcat_aldoase,Kcat_triose_phosphate_isomerase,
        Kcat_glyceraldehyde3phosphate_dehydrogenase,Kcat_phosphoglycerate_mutase,Kcat_enolase,Kcat_pyruvate_kinase]


reactions_labels=[[['Glucose','Glucose-6-Phosphate'],['Glucose-6-Phosphate','Fructose-6-Phosphate'],['Fructose-6-Phosphate','Fructose-1,6-biposphate']],
        [['Fructose-1,6-biposphate','GADP+DHAP'],['Dihydroxyacetone phosphate','Glyceraldehyde-3-Phosphate'],['Glyceraldehyde-3-Phosphate','1,3-biphosphoglycerate']],
        [['1,3-biphosphoglycerate','3-phosphoglycerate'],['3-phosphoglycerate','2-phosphoglycerate'],['2-phosphoglycerate','Phosphoenolpyruvate']],
        [['Phosphoenolpyruvate','Pyruvate']]]

# Initial Conditions
y0 = [5,0]
# time span
t = np.linspace(0,200,100000)
tspan = [t[0],t[-1]]

#Plot ractions pathway independently
[fig,axs] = plt.subplots(4,3,figsize=[9,4])
for x in range(4):
    for y in range(3):
        # Plot g vs r
        if (x==3 and y>0):
            pass
        else:  
            
            index=int(x+y) 
            ode_sol = solve_ivp(lambda t,y:MM_dynamics_pair(t,y,K_cat[index],K_M[index],E_0),tspan,y0,t_eval=t)
            axs[x][y].plot(t,ode_sol.y[0],'k-')
            axs[x][y].plot(t,ode_sol.y[1],'r--')
            axs[x][y].set_xlabel('time [s]')
            axs[x][y].set_ylabel('concentration [mM]')
            axs[x][y].legend(reactions_labels[x][y])
        
        fig.suptitle('Michaelis-Menten Glycolysis Pairwise Reactions', fontsize=20)
        
#Plot some of cascaded reactions
#Asume same initial Enzyme concentrations
K_cat0 = [1430] * 9
K_M0 = [38] * 9 
# Variables
E_0 = 10e-3
fig=plt.figure(figsize=[9,4])
# Initial Conditions
y0 = [0.1,0,0,0,0,0,0,0,0,0]
# time span
t = np.linspace(0,70,10000)
tspan = [t[0],t[-1]]
#ode_sol = solve_ivp(lambda t,y:MM_dynamics_few(t,y,K_cat0,K_M0,E_0),tspan,y0,t_eval=t)
ode_sol = solve_ivp(lambda t,y:MM_dynamics_full(t,y,K_cat0,K_M0,E_0),tspan,y0,t_eval=t)
plt.plot(t,ode_sol.y[0],'k-')
plt.plot(t,ode_sol.y[1],'r--')
plt.plot(t,ode_sol.y[2],'b--')
plt.plot(t,ode_sol.y[3],'g--')
plt.plot(t,ode_sol.y[4],'m--')
plt.plot(t,ode_sol.y[5],'y--')
plt.plot(t,ode_sol.y[6],'c--')
plt.plot(t,ode_sol.y[7],'m--')
plt.plot(t,ode_sol.y[8],'m--')
plt.plot(t,ode_sol.y[9],'m--')
plt.xlabel('time [s]')
plt.ylabel('concentration [mM]')
plt.legend(['Reactant','Intermediate1','Intermediate2','Intermediate3','Intermediate4','Intermediate5','Intermediate6','Intermediate7','Intermediate8','Product'])
fig.suptitle('Michaelis-Menten Cascaded Reactions', fontsize=20)

#Plot MM cascaded reactions for E. Coli
fig=plt.figure(figsize=[9,4])
# Initial Conditions
y0 = [0.1,0,0,0,0,0,0,0,0,0]
# time span
t = np.linspace(0,70,10000)
tspan = [t[0],t[-1]]
ode_sol = solve_ivp(lambda t,y:MM_dynamics_full(t,y,K_cat,K_M,E_0),tspan,y0,t_eval=t)

plt.plot(t,ode_sol.y[0],'k-')
plt.plot(t,ode_sol.y[1],'r--')
plt.plot(t,ode_sol.y[2],'b--')
plt.plot(t,ode_sol.y[3],'g--')
plt.plot(t,ode_sol.y[4],'m--')
plt.plot(t,ode_sol.y[5],'y--')
plt.plot(t,ode_sol.y[6],'c--')
plt.plot(t,ode_sol.y[7],'m--')
plt.plot(t,ode_sol.y[8],'m--')
plt.plot(t,ode_sol.y[9],'m--')
plt.xlabel('time [s]')
plt.ylabel('concentration [mM]')
plt.legend(['Glucose','Glucose-6-Phosphate','Fructose-6-Phosphate','Fructose-1,6-biposphate','GADP+DHAP',
            '1,3-biphosphoglycerate','3-phosphoglycerate','2-phosphoglycerate','Phosphoenolpyruvate','Pyruvate'])
fig.suptitle('MM cascaded reactions for E.Coli strains parameters', fontsize=20)


#%%  Part III:
#Flux Balance Analysis
import cobra
from cobra import Model, Reaction, Metabolite

#Create Model
model = Model('GlycolisisFBA')

# Metabolites definitions
A=Metabolite('GLC', name='Glucose', formula='C6H12O6',compartment='c' )
B=Metabolite('G6P',name='Glucose-6-phosphate', formula='C6H11O9P',compartment='c' )
C=Metabolite('F6P', name='Fructose-6-phosphate', formula='C6H11O9P', compartment='c')
D=Metabolite('F-1,6-BP', name='Fructose-1,6-bisphosphate', formula='C6H10O12P2', compartment='c')
E=Metabolite('GA3P', name='Glyceraldehyde-3-phosphate', formula='C3H5O3P', compartment='c')
F=Metabolite('1,3BPG',name='1,3-bisphosphoglycerate', formula='C3H7O10P2', compartment='c')
G=Metabolite('3-PG',name='3-phosphoglycerate', formula='C3H7O7P', compartment='c')
H=Metabolite('2-PG',name='2-phosphoglycerate', formula='C3H7O7P', compartment='c')
I=Metabolite('PEP',name='Phosphoenolpyruvate', formula='C3H3O3P', compartment='c')
J=Metabolite('PYR',name='Pyruvate', formula='C3H3O3', compartment='c')

#Cofactors and other products
ATP=Metabolite('ATP',name='ATP', formula='C10H16N5O13P3', compartment='c')
ADP=Metabolite('ADP',name='ADP', formula='C10H15N5O10P2', compartment='c')
NAD=Metabolite('NAD',name='NAD', formula='C21H27N7O14P2', compartment='c')
NADH=Metabolite('NADH',name='NADH', formula='C21H28N7O14P2', compartment='c')
H2O=Metabolite('H2O',name='H2O', formula='H2O',compartment='c')
HYDROGEN=Metabolite('H+',name='H+', formula='H+',compartment='c')
Pi=Metabolite('Pi',name='Pi', formula='PO4',compartment='c')

#Define Exchange reactions

#Set excess Glucose medium
reaction = Reaction('Ex_GLC')
reaction.name='Ex_GLC'
reaction.lower_bound=0.
reaction.upper_bound=1000.
reaction.add_metabolites({A:20.0})
model.add_reactions([reaction])

reaction = Reaction('Ex_NAD')
reaction.name='Ex_NAD'
reaction.lower_bound=0.
reaction.upper_bound=1000.
reaction.add_metabolites({NAD:2.0})
model.add_reactions([reaction])

reaction = Reaction('Ex_Pi')
reaction.name='Ex_Pi'
reaction.lower_bound=0.
reaction.upper_bound=1000.
reaction.add_metabolites({Pi:2.0})
model.add_reactions([reaction])

reaction = Reaction('Ex_ADP')
reaction.name='Ex_ADP'
reaction.lower_bound=0.
reaction.upper_bound=1000.
reaction.add_metabolites({ADP:2.0})
model.add_reactions([reaction])

reaction = Reaction('Ex_Pyr')
reaction.name='Ex_Pyr'
reaction.lower_bound=0.
reaction.upper_bound=1000.
reaction.add_metabolites({J:-2.0})
model.add_reactions([reaction])

reaction = Reaction('Ex_NADH')
reaction.name='Ex_NADH'
reaction.lower_bound=0.
reaction.upper_bound=1000.
reaction.add_metabolites({NADH:-2.0})
model.add_reactions([reaction])

reaction = Reaction('Ex_HYDROGEN')
reaction.name='Ex_HYDROGEN'
reaction.lower_bound=0.
reaction.upper_bound=1000.
reaction.add_metabolites({HYDROGEN:-2.0})
model.add_reactions([reaction])

reaction = Reaction('Ex_H2O')
reaction.name='Ex_H2O'
reaction.lower_bound=0.
reaction.upper_bound=1000.
reaction.add_metabolites({H2O:-2.0})
model.add_reactions([reaction])

reaction = Reaction('Ex_ATP')
reaction.name='Ex_ATP'
reaction.lower_bound=0.
reaction.upper_bound=1000.
reaction.add_metabolites({ATP:-2.0})
model.add_reactions([reaction])

#Define Model Reactions
#v_1:Glucose + ATP--->Glucose-6 phosphate + ADP
reaction=Reaction('v_1')
reaction.name='Glucose_phosphorylation'
reaction.lower_bound=0;
reaction.upper_bound=1000;
reaction.add_metabolites({A:-1.0, ATP: -1.0, B: 1.0 , ADP: 1.0})
model.add_reactions([reaction])

  #v_2:Glucose-6 phosphate ---> Fructose-6 phosphate
reaction=Reaction('v_2')
reaction.name='Isomeration'
reaction.lower_bound=0;
reaction.upper_bound=1000;
reaction.add_metabolites({B:-1.0, C: 1.0})
model.add_reactions([reaction])

#v_3:Fructose-6 phosphate + ATP ---> Fructose-1,6-Biphosphate + ADP
reaction=Reaction('v_3')
reaction.name='Secondary_Phosphorylation'
reaction.lower_bound=0;
reaction.upper_bound=1000;
reaction.add_metabolites({C:-1.0, ATP: -1.0, D:1.0, ADP: 1.0})
model.add_reactions([reaction])

#v_4:Fructose-1,6-Biphosphate ---> Glyceraldehyde-3-Phosphate (and DHAP=GADP)
reaction=Reaction('v_4')
reaction.name='Cleavage'
reaction.lower_bound=0;
reaction.upper_bound=1000;
reaction.add_metabolites({D:-1.0, E: 2.0})
model.add_reactions([reaction])

#v_5:2 Glyceraldehyde-3-phosphate + 2 NAD+ + 2 Pi + 2 ADP ---> 2 1,3-bisphosphoglycerate + 2 NADH + 2 H+ + 2 ATP
reaction=Reaction('v_5')
reaction.name='Oxidation_and_Phosphorilation'
reaction.lower_bound=0;
reaction.upper_bound=1000;
reaction.add_metabolites({E:-2.0, NAD:-2.0, Pi:-2.0, ADP:-2.0, F:2.0, NADH:2.0, HYDROGEN:2.0})
model.add_reactions([reaction])

#v_6:2 1,3-bisphosphoglycerate ---> 2 3-phosphoglycerate
reaction=Reaction('v_6')
reaction.name='Isomeration'
reaction.lower_bound=0;
reaction.upper_bound=1000;
reaction.add_metabolites({F:-2.0, G:2.0, ATP:2.0})
model.add_reactions([reaction])

#v_7: 2 3-phosphoglycerate ---> 2 2-phosphoglycerate 
reaction=Reaction('v_7')
reaction.name='Dehydration'
reaction.lower_bound=0;
reaction.upper_bound=1000;
reaction.add_metabolites({G:-2.0, H:2.0})
model.add_reactions([reaction])

#v_8: 2 2-phosphoglycerate + 2 ADP ---> 2 Phosphoenolpyruvate + 2 H2O
reaction=Reaction('v_8')
reaction.name='Phosphorylation'
reaction.lower_bound=0;
reaction.upper_bound=1000;
reaction.add_metabolites({H:-2.0, ADP:-2.0, I:2.0, H2O:2.0})
model.add_reactions([reaction])

#v_9: 2 Phosphoenolpyruvate → 2 Pyruvate + 2 ATP
reaction=Reaction('v_9')
reaction.name='Phosphorylation'
reaction.lower_bound=0;
reaction.upper_bound=1000;
reaction.add_metabolites({I:-2.0, J:2.0, ATP:2.0})
model.add_reactions([reaction])

##Flux Balance Ananlysis!!

model.objective = 'v_9'
solution = model.optimize()
print('Solution FLuxes')
print(solution.fluxes)

print('\nGlucose FLux')
print(model.metabolites.GLC.summary())

print('\nPyruvate flux')
print(model.metabolites.PYR.summary())

#%% PART IV: COMETS

#ecoli = cp.model(cb.io.read_sbml_model('e_coli_core.xml'))
#ecoli = cp.model(cb.io.read_sbml_model('iML1515.xml')) #to use the iML1515 model
model=cb.io.read_sbml_model('e_coli_core.xml')

##Knock out genes
#model.genes.b1854.knock_out() 
#model.genes.b1676.knock_out() 
#crete comet model from cobra model
ecoli = cp.model(model)

ecoli.open_exchanges()

# Biomass propogation uses a convective flow model here, where biomass propogation is driven by pressure from growth
ecoli.add_convection_parameters(packedDensity = 0.022,
                                    elasticModulus = 1.e-10,
                                    frictionConstant = 1.0,
                                    convDiffConstant = 0.0)

# add small random fluctuations to the growth rates
ecoli.add_noise_variance_parameter(10.)

# Seed 4 colonies. Larger initial amounts along diagonal, smaller on off-diagonal.
#ecoli.initial_pop = [[10,10,1.e-5],
#                     [40,10,1.e-6],
#                     [10,40,1.e-6],
#                     [40,40,1.e-5]]

ecoli.initial_pop = [25,25,1.e-5]
                     

# Set width of the petri dish (width x width square)
width = 51

# Define the petri dish grid, the grid is split into two sections that can have separate diffusion constants
grid_size = [width, width] # width boxes in each direction
region_map = np.zeros(grid_size, dtype = int) # an integer array filled with zeros
region_map[:] = 1 # first fill the whole map with 1s
region_map[int(width/2):width,:] = 2 # next fill the bottom half with 2s

# Define Layout
ly = cp.layout([ecoli])
ly.grid = grid_size
ly.set_region_map(region_map)
num_mets = ly.media.shape[0]
# Define diffusion constants for each metabolite in the two grid sections (they are defined to be the same here)
diffusion_constant_region1 = [5.e-6] * num_mets
diffusion_constant_region2 = [5.e-6] * num_mets
friction_constant = 1.0
ly.set_region_parameters(1, diffusion_constant_region1, friction_constant)
ly.set_region_parameters(2, diffusion_constant_region2, friction_constant)
ly.set_specific_metabolite("glc__D_e", 5.e-5)
ly.set_specific_metabolite("h2o_e", 1000.)
ly.set_specific_metabolite("nh4_e", 1000.)
ly.set_specific_metabolite("h_e", 1000.)
ly.set_specific_metabolite("pi_e", 1000.)
#ly.set_specific_metabolite("pyr_e", 1000.)

# COMETS parameters
p = cp.params()
p.all_params["maxCycles"] = 2000
p.all_params["biomassMotionStyle"] = "Convection 2D"
p.all_params["writeBiomassLog"] = True
p.all_params["BiomassLogRate"] = p.all_params["maxCycles"]
p.all_params["defaultKm"] = 0.01
p.all_params["defaultVmax"] = 10
p.all_params["timeStep"] = 0.01
p.all_params["spaceWidth"] = 0.02
p.all_params["maxSpaceBiomass"] = 10
p.all_params["minSpaceBiomass"] = 1.e-10
p.all_params["allowCellOverlap"] = True
p.all_params["writeFluxLog"] = True
p.all_params["FluxLogRate"] = p.all_params["maxCycles"]
p.all_params["writeMediaLog"] = True
p.all_params["MediaLogRate"] = p.all_params["maxCycles"]

# Run COMETS
sim = cp.comets(ly, p)
sim.run()


# Plot
[fig,axs]=plt.subplots(1,4,figsize=[9,4])

# Biomass
my_cmap = cmap.get_cmap("copper")
my_cmap.set_bad((0,0,0))
im = sim.get_biomass_image('e_coli_core', p.all_params["maxCycles"])
axs[0].imshow(im, norm = clrs.LogNorm(), cmap = my_cmap)
axs[0].set_title('biomass')

# glucose
axs[1].imshow(sim.get_metabolite_image("glc__D_e", p.all_params['maxCycles']))
axs[1].set_title('Glucose')

# glucose
axs[2].imshow(sim.get_metabolite_image("pyr_e", p.all_params['maxCycles']))
axs[2].set_title('Pyruvate')

# growth rate
im = sim.get_flux_image("e_coli_core", "BIOMASS_Ecoli_core_w_GAM", p.all_params["maxCycles"])
axs[3].imshow(im);
axs[3].set_title('growth rate')
    

fig.tight_layout()