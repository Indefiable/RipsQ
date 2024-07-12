from MILP import MILP
import numpy as np


class NoiseChannel:
	def __init__(self,KOs,cliffords_path,):
		'''
		KOs: A list of (4x1) Krauss Operators in the Noise Channel 
		'''
		self.KOs = KOs
		self.solver = MILP(cliffords_path)

	def solve_approx(self,lambda_weight,scaling_factor,verbose=True):
		'''
		Get the Approximated Noise Channel using MILP class
		'''
		self.KOs_approx = []
		for K in self.KOs:
			print(K)
			self.solver.solve(K,lambda_weight,scaling_factor)
			if verbose:
				self.solver.view_solution_summary()
			self.KOs_approx.append(self.solver.get_approx().reshape(2,2))

		return None 

	def get_approx(self,):
		'''
		returns 2x2 approx KOs
		'''
		return self.KOs_approx

	def get_exact(self,):
		'''
		returns 2x2 exact KOs
		'''
		return [K.reshape(2,2) for K in self.KOs]


class Simulation:
	def __init__(self,NCs,lambda_weight,scaling_factor,verbose=True):
		'''
		NCs = List of Noise Channels to be applied in order

		#MILP Level Parameters
		lambda weight: same lambda weight defined in MILP
		scaling_factor: same scaling factor defined in MILP
		verbose: verbose for MILP approximations
		'''
		self.NCs = NCs 
		self.lambda_weight = lambda_weight
		self.scaling_factor = scaling_factor
		self.verbose = verbose
		self.final_rho_approx = np.zeros((2,2),dtype=complex)
		self.final_rho_exact = np.zeros((2,2),dtype=complex)


	def run_density_exact(self,initial_rho):
		for NC in self.NCs:
			NC_approx = NC.get_exact()
			for K in NC_approx:
				self.final_rho_exact += K@rho@np.conjugate(K.T)

		return self.final_rho_exact

	def run_density_approx(self,initial_rho):
		for NC in self.NCs:
			NC.solve_approx(self.lambda_weight,self.scaling_factor,self.verbose)
			NC_approx = NC.get_approx()
			for K in NC_approx:
				self.final_rho_approx += K@rho@np.conjugate(K.T)
		return final_rho

	def run_ket_simulation(self,initial_ket):
		...


	def get_summary(self,result_type):
		'''
		type: density, ket, or both
		'''

		print('*'*40)
		if result_type=='density' or result_type=='both':
			L2_error = np.linalg.norm(self.final_rho_approx.flatten()-self.final_rho_exact.flatten())
			print('Approx Rho prime: \n',self.final_rho_approx)
			print('Exact Rho prime: \n',self.final_rho_exact)
			print('L2_error',L2_error)

		if result_type=='ket' or result_type=='both':
			...

		print('*'*40)



if __name__ == '__main__':
	#Define Noise OSRs 
	I = np.eye(2,dtype=complex)
	X = np.array([[0, 1], [1, 0]], dtype=complex)
	Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
	Z = np.array([[1, 0], [0, -1]], dtype=complex)

	amplitude_dampening = lambda p: [np.array([1+0j, 0+0j, 0+0j, np.sqrt(1-p)+0j]),
                                  np.array([0+0j, 0+0j,np.sqrt(p)+0j, 0+0j])]
    
	phase_dampening = lambda p: [np.sqrt(p)*I,
                              np.sqrt(1-p)*Z]
    
	partial_dampening = lambda p: [np.sqrt(1-(3*p)/4)*I,np.sqrt(p/4)*X,np.sqrt(p/4)*Y,np.sqrt(p/4)*Z]

	PartD = NoiseChannel(partial_dampening(1/2),'cliffords.txt')
	PhaseD = NoiseChannel(phase_dampening(1/3),'cliffords.txt')
	AmpD_1 = NoiseChannel(phase_dampening(1/4),'cliffords.txt')
	AmpD_2 = NoiseChannel(phase_dampening(3/4),'cliffords.txt')


	starting_state = np.sqrt(np.array([1/2, 1/2],dtype=complex))
	rho = np.outer(starting_state, np.conjugate(starting_state))

	Sim = Simulation([PartD,AmpD_1,PhaseD,AmpD_2],lambda_weight=1/3,scaling_factor=50,verbose=True)
	Sim.run_density_exact(rho)
	Sim.run_density_approx(rho)
	Sim.get_summary()









