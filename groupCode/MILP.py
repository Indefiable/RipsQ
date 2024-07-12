import pulp
import numpy as np

class MILP:
    def __init__(self,cliffords_path):
        '''
        cliffords_path: File Name For String Representations of 1 qubit Cliffords 
            ex. PHP (Where H=Hadamard, P=Phase gate)
        '''

        #Statics 
        self.H = np.sqrt(1/2)*np.array([[1, 1],
                             [1,-1]])
        self.P = np.array([[1,0],
                      [0,1j]])
        self.I = np.eye(2)
        with open(cliffords_path,'r') as f:
            cliffords = f.readlines()
            R = np.array([self.compute(c[:-1]).flatten() for c in cliffords]).T
        self.cliffords = cliffords #String Rep of Cliffords
        self.R = R #Matrix Rep of Cliffords 24x4


    ##HELPERS##

    @staticmethod
    def simplify(C):
        C = C.replace('I','')
        C = C.replace('P'*4,'')
        C = C.replace('H'*2,'')
        C = C.replace('PH'*24,'')
        C = C.replace('HP'*24,'')

        if C=='':
            return 'I'
        return C 

    def compute(self,C):
        Cl = self.I
        for o in C:
            if o=='H':
                Cl = Cl@self.H
            elif o=='P':
                Cl = Cl@self.P
        return Cl

    ## SOLVER ##
    def solve(self,K,lambda_weight,scaling_factor,M=0):
        '''
        K: Krauss Operator (4x1) Complex vector (flattented 2x2)
        lambda_weight: regularization weight, a larger lambda will 
                       place a larger weight on reducing rank, while
                       a lower lambda will place a larger weight on error
                       reduction

        scaling_factor: scalling the Krauss operator since MILP does not seem to 
                         like approximating vectors with small norm. We will 
                         Scale up then back down
                         *a hacky solution that will need a real solution soon*

        M: Linking coefficient. Essentially links the imaginary and real compents of each vector
           such that if the real componenet is not used from one vector the imaginary component 
           wont be used either. You want M to be fairly large to give flexibility in coeff value 
           but a M that is too large may introduce numerical instablilty. For this we use a heuristic
           to estimate M, but leave some flexibility to the user to select M
        '''
        self.scaling_factor = scaling_factor
        self.x = self.scaling_factor*K #set Krauss opperator

        #estimating M
        if M==0:
            x_norm = np.linalg.norm(self.x)
            R_norms = np.linalg.norm(self.R, axis=0)
            min_R_norm = np.min(R_norms)
            M = x_norm / min_R_norm + 10
        self.M = M #Make M an accessible attribute for debugging

        # Separate real and imaginary parts
        x_real = self.x.real
        x_imag = self.x.imag

        R_real = self.R.real
        R_imag = self.R.imag

        # Number of vectors in R
        m = self.R.shape[1]
        n = len(self.x)

        # Define the problem
        prob = pulp.LpProblem("Approximate_Vector", pulp.LpMinimize)

        # Define decision variables for real and imaginary parts of coefficients
        c_real = [pulp.LpVariable(f'c_real_{i}', lowBound=-1000, upBound=1000) for i in range(m)]
        c_imag = [pulp.LpVariable(f'c_imag_{i}', lowBound=-1000, upBound=1000) for i in range(m)]
        y = [pulp.LpVariable(f'y_{i}', cat='Binary') for i in range(m)]
        e_real = [pulp.LpVariable(f'e_real_{i}', lowBound=0) for i in range(n)]
        e_imag = [pulp.LpVariable(f'e_imag_{i}', lowBound=0) for i in range(n)]

        # Objective function: minimize sum of absolute errors and number of vectors used
        prob += pulp.lpSum(e_real) + pulp.lpSum(e_imag) + lambda_weight * pulp.lpSum(y), "Minimize total error and number of vectors used"

        # Constraints for real part
        for i in range(n):
            prob += pulp.lpSum(R_real[i, j] * c_real[j] - R_imag[i, j] * c_imag[j] for j in range(m)) - x_real[i] <= e_real[i], f"Constraint_real_pos_{i}"
            prob += x_real[i] - pulp.lpSum(R_real[i, j] * c_real[j] - R_imag[i, j] * c_imag[j] for j in range(m)) <= e_real[i], f"Constraint_real_neg_{i}"

        # Constraints for imaginary part
        for i in range(n):
            prob += pulp.lpSum(R_real[i, j] * c_imag[j] + R_imag[i, j] * c_real[j] for j in range(m)) - x_imag[i] <= e_imag[i], f"Constraint_imag_pos_{i}"
            prob += x_imag[i] - pulp.lpSum(R_real[i, j] * c_imag[j] + R_imag[i, j] * c_real[j] for j in range(m)) <= e_imag[i], f"Constraint_imag_neg_{i}"

        # Linking constraints to ensure c[j] components are zero if y[j] is zero
        for j in range(m):
            prob += c_real[j] <= self.M * y[j], f"Linking_constraint_real_pos_{j}"
            prob += c_real[j] >= -self.M * y[j], f"Linking_constraint_real_neg_{j}"
            prob += c_imag[j] <= self.M * y[j], f"Linking_constraint_imag_pos_{j}"
            prob += c_imag[j] >= -self.M * y[j], f"Linking_constraint_imag_neg_{j}"

        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        # Reconstruct x from the optimal coefficients
        c_real_values = np.array([var.varValue for var in c_real])
        c_imag_values = np.array([var.varValue for var in c_imag])
        c_values = c_real_values + 1j * c_imag_values
        self.c_values = c_values/self.scaling_factor  #Scale back down
        self.optimal = pulp.LpStatus[prob.status] #Optimal Solution Status

        return None

    ## DISPLAY ##

    def view_solution_summary(self,):
        #get approx sol
        x_approx = np.dot(self.R, self.c_values)

        #HP Print
        hp_print = ""
        non_zeros = np.nonzero(self.c_values)[0]
        for non_zero in non_zeros:
            hp_print += str(np.round(self.c_values[non_zero],2))+'*'+self.simplify(self.cliffords[int(non_zero)][:-1])+' + '
        hp_print = hp_print[:-2]

        #Mat Print 
        mat_print = ""
        non_zeros = np.nonzero(self.c_values)[0]
        for non_zero in non_zeros:
            mat_print += str(np.round(self.c_values[non_zero],2))+'*'+str(self.compute(self.cliffords[int(non_zero)][:-1]))+' + '
        mat_print = mat_print[:-2]

        '''
        #Cosine Similarity
        dot_product = np.dot(x_approx, self.x)
        norm_x_approx = np.linalg.norm(x_approx)
        norm_x = np.linalg.norm(self.x)
        cosine_sim = dot_product / (norm_x_approx * norm_x)
        '''

        #L2 Error 
        L2_error = np.linalg.norm(x_approx-self.x/self.scaling_factor)

        print('*'*40)
        print('Krauss Approx: ',x_approx,'\n')
        print('HP Form Approx: ',hp_print,'\n')
        print('Matrix Form Approx: ',mat_print,'\n')
        print('L2 Error: ', L2_error)
        print('M Value:' , self.M)
        print('*'*40)
        #print('Cosine Similarity: ',cosine_sim,'\n')
        #print('Optimal Status: ', self.optimal)


    ### RETREIVAL ###

    def get_approx(self,):
        return np.dot(self.R, self.c_values).reshape(2,2)

    def get_coeff_of_approx(self,):
        non_zeros = np.nonzero(self.c_values)[0]
        return self.c_values[non_zeros]

    def get_cliffords_of_approx(self,):
        cliffords_of_approx = []
        non_zeros = np.nonzero(self.c_values)[0]
        for non_zero in non_zeros:
            cliffords_of_approx.append(self.compute(self.cliffords[int(non_zero)][:-1]))
        return cliffords_of_approx



if __name__ == '__main__':
        cliffords_path = 'cliffords.txt'
        solver = MILP(cliffords_path)

        p = 1/3
        K = np.array([1+0j, 0+0j, 0+0j, np.sqrt(1-p)+0j])
        solver.solve(K,lambda_weight=1/3,scaling_factor=50)
        
        solver.view_solution_summary()
        print(solver.get_approx())
        print(solver.get_coeff_of_approx())
        print(solver.get_cliffords_of_approx())

