import gurobipy as gp
from gurobipy import GRB
import numpy as np

def find_binary_rank_gurobi(M):
    """
    Calculates the binary rank of a 0/1 matrix M using Gurobi.
    Returns the binary rank k, along with the decomposition matrices A and B.
    """
    M = np.array(M)
    n, m = M.shape
    
    # The absolute maximum rank is the number of 1s in the matrix
    max_k = int(np.sum(M))
    
    if max_k == 0:
        print("Matrix is all zeros. Binary rank is 0.")
        return 0, None, None
        
    for k in range(1, max_k + 1):
        print(f"Checking if binary rank is k = {k}...")
        
        # Create an environment and suppress Gurobi's standard console output
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        
        # Create a new model
        model = gp.Model(f"BinaryRank_k_{k}", env=env)
        
        # Define Binary Variables using Gurobi's optimized addVars
        A = model.addVars(n, k, vtype=GRB.BINARY, name="A")
        B = model.addVars(k, m, vtype=GRB.BINARY, name="B")
        Z = model.addVars(n, m, k, vtype=GRB.BINARY, name="Z")
        
        # Apply Constraints
        for i in range(n):
            for j in range(m):
                # 1. Exact partition constraint: The sum of overlaps across k must exactly equal M[i,j]
                model.addConstr(gp.quicksum(Z[i, j, r] for r in range(k)) == M[i, j], name=f"Sum_M_{i}_{j}")
                
                # 2. Linearization constraints for Z[i,j,r] = A[i,r] AND B[r,j]
                for r in range(k):
                    model.addConstr(Z[i, j, r] <= A[i, r], name=f"Z_le_A_{i}_{j}_{r}")
                    model.addConstr(Z[i, j, r] <= B[r, j], name=f"Z_le_B_{i}_{j}_{r}")
                    model.addConstr(Z[i, j, r] >= A[i, r] + B[r, j] - 1, name=f"Z_ge_AB_{i}_{j}_{r}")
                    
        # Solve the model
        model.optimize()
        
        # Check if a feasible exact cover was found
        if model.Status == GRB.OPTIMAL:
            print(f"✓ Match found! The minimal k is {k}.\n")
            
            # Extract the actual values of the A and B matrices
            A_result = np.zeros((n, k), dtype=int)
            B_result = np.zeros((k, m), dtype=int)
            
            for i in range(n):
                for r in range(k):
                    A_result[i, r] = int(round(A[i, r].X))
                    
            for r in range(k):
                for j in range(m):
                    B_result[r, j] = int(round(B[r, j].X))
                    
            return k, A_result, B_result
            
    return max_k, None, None

# --- Example Usage ---
if __name__ == "__main__":
    matrix_M = [
        [0,1,0,0,1,0,0,1,1],
        [0,0,1,0,0,1,1,0,1],
        [1,0,0,1,0,0,1,1,0],
        [1,1,0,0,1,0,1,1,1],
        [0,1,1,0,0,1,1,1,1],
        [1,0,1,1,0,0,1,1,1],
        [1,0,0,0,0,0,0,1,0],
        [0,1,0,0,0,0,0,0,1],
        [0,0,1,0,0,0,1,0,0]
        ] 
        
    """matrix_M=[
    [0,1,1,1,1,1],
    [0,0,1,1,1,1],
    [0,0,0,1,1,1],
    [0,0,0,0,1,0],
    [0,0,0,0,0,1],
    [0,0,0,1,0,0]
    ]"""
    print("Matrix M:")
    print(np.array(matrix_M))
    print("-" * 30)
    
    rank, matrix_A, matrix_B = find_binary_rank_gurobi(matrix_M)
    
    if rank > 0:
        print("Matrix A:")
        print(matrix_A)
        print("\nMatrix B:")
        print(matrix_B)
        
        # Verify the result by multiplying A and B
        print("\nVerification (A * B == M):")
        print(np.dot(matrix_A, matrix_B))