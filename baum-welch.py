import numpy as np
import copy

np.set_printoptions(suppress=True)

# MAIN FUNCTION

# Inputs:
# m = number of states
# seq = input sequence (as string) e.g. 'abbcbabcb'
# alphabet = string of unique observable values e.g. 'abc'

# Outputs:
# A = [mxm] transition matrix (matrix A)
# B = [mxk] emmision matrix (matrix B)
# pi0 = [m] list of starting probabilities for each state
def baum_welch(m, seq, alphabet, max_it = 2000):

    # INITIALISE RANDOM VALUES FOR A, B and pi0
    # Inputs:
    # m = number of states
    # k = number of possible observable values
    
    def init_params(m, k):

        # randomly initialise transition matrix (A)
        A = np.random.rand(m, m)



        # normailise rows so that the transition probabilities for each state sum to 0
        for i in range(m):
            A[i] = A[i] / A[i].sum(axis=0, keepdims=1)

        # randomly initialise emmission matrix (B)
        B = np.random.rand(m, k)

        # normailise rows so that the emmission probabilities for each state sum to 0
        for i in range(m):
            B[i] = B[i] / B[i].sum(axis=0, keepdims=1)

        # randomly initialise starting probabilities
        pi0 = np.random.rand(m)

        # normalise starting probabilities so that the sum to 0
        pi0 = pi0 / pi0.sum(axis=0, keepdims=1)

        return A, B, pi0

    # CALCULATE FORWARDS PROBABILITIES
    def forwards(A, B, pi0, m, seq, alphabet):

        alpha = np.zeros((m, len(seq)))

        G = np.zeros((len(seq)))

        for t in range(len(seq)):
            for i in range(m):

                if t == 0:

                    # BASE CASE: t = 0
                    # alpha_0(i) = p(Q_0 = i) . B_i(t)
                    alpha[i, t] = pi0[i] *  B[i, alphabet.find(seq[t])]


                else:

                    # RECURSIVE CASE: t > 0
                    # alpha_t(i) = SUM(alpha_t-1(j) . A_ij . B_iobs) for all possible j
                    # where j is the previously visited state and obs is the observed value
                    alpha[i, t] = np.sum(
                        [
                            alpha[j, t - 1] * A[j, i] * B[i, alphabet.find(seq[t])]
                            for j in range(m)
                        ]
                    )

            G[t] = np.sum(alpha[:,t])

            alpha[:,t] = (1/G[t]) * alpha[:,t]


        return alpha, G

    # CALCULATE BACKWARDS PROBABILITIES
    def backwards(A, B, m, seq, alphabet, G):

        beta = np.zeros((m, len(seq)))

        for t in range(len(seq) - 1, -1, -1):
            for i in range(m):

                if t == len(seq) - 1:

                    # BASE CASE: t = T
                    # beta_T(i) = 1
                    beta[i, t] = 1

                else:

                    # RECURSIVE CASE: t < 0
                    # beta_t(i) = SUM(beta_t+1(j) . A_ij . B_jobs) for all possible j
                    # where j is the next visited state and obs is the observed value
                    beta[i, t] = np.sum(
                        [
                            beta[j, t + 1] * A[i, j] * B[j, alphabet.find(seq[t + 1])]
                            for j in range(m)
                        ]
                    )

            if t != len(seq) - 1:

                beta[:,t] = (1/G[t]) * beta[:,t]

        return beta

    # CALCULATE 'ETA'
    def get_eta(m, alpha, beta, seq):

        eta = np.zeros((m, len(seq)))

        for t in range(len(seq)):

            # compute denomenator for all possible states j at current time
            # this saves on computation
            denomenator = np.sum([alpha[j, t] * beta[j, t] for j in range(m)])

            for i in range(m):
                if denomenator != 0 and not np.isnan(denomenator):
                    eta[i, t] = alpha[i, t] * beta[i, t] / denomenator
                else:
                    eta[i, t] = 0

        return eta


    
    # CALCULATE 'XI'
    def get_xi(m, alpha, beta, seq, alphabet):

        xi = np.zeros((len(seq) - 1, m, m))

        for t in range(len(seq) - 1):

            # compute denomenator for all possible consecutive states k and l at current time
            # this saves on computation
            denomenator = np.sum(
                [
                    alpha[k, t]
                    * beta[l, t + 1]
                    * A[k, l]
                    * B[l, alphabet.find(seq[t + 1])]
                    for k in range(m)
                    for l in range(m)
                ]
            )

            for i in range(m):

                for j in range(m):

                    if denomenator != 0 and not np.isnan(denomenator):
                        xi[t, i, j] = (
                            alpha[i, t]
                            * beta[j, t + 1]
                            * A[i, j]
                            * B[j, alphabet.find(seq[t + 1])]
                        ) / denomenator

                    else:

                        xi[t, i, j] = 0

        return xi


    # UPDATE 'A'
    def updateA(A, m, xi, seq):

        for i in range(m):

            denomenator = sum(
                [xi[t, i, j] for t in range(len(seq) - 1) for j in range(m)]
            )

            for j in range(m):

                A[i, j] = 0
                for t in range(len(seq) - 1):

                    A[i, j] = A[i, j] + xi[t, i, j]

                if denomenator != 0 and not np.isnan(denomenator):

                    A[i, j] = A[i, j] / denomenator

                else:

                    A[i, j] = 0

        return A

    # UPDATE 'B'
    def updateB(B, m, eta, seq, alphabet):

        for i in range(m):

            denomenator = sum([eta[i, t] for t in range(len(seq))])

            for j in range(len(alphabet)):

                if denomenator != 0 and not np.isnan(denomenator):
                        B[i, j] = (
                            sum(
                                [
                                    eta[i, t]
                                    for t in range(len(seq))
                                    if seq[t] == alphabet[j]
                                ]
                            )
                            / denomenator
                        )
                else:

                    B[i, j] = 0
        return B

    # UPDATE 'pi0'
    def update_pi0(pi0, eta):

        for i in range(len(pi0)):

            pi0[i] = eta[i,0]

        return pi0

    # evaluates the change in A and B over previous iteration to decide whether the HMM can be considered as converged
    def check_for_change(new_A, new_B, A, B, thresh):

        max_A_change = np.max(np.abs(A - new_A))
        max_B_change = np.max(np.abs(B - new_B))

        return not(max_A_change < thresh and max_B_change < thresh)
        

    A, B, pi0 = init_params(m, len(alphabet))

    last_A = A[:]
    last_B = B[:]

    keep_training = True

    i = 0

    while keep_training == True and i < max_it:

        last_A = copy.deepcopy(A)
        last_B = copy.deepcopy(B)

        alpha, G = forwards(A, B, pi0, m, seq, alphabet)
        beta = backwards(A, B, m, seq, alphabet, G)
        eta = get_eta(m, alpha, beta, seq)
        xi = get_xi(m, alpha, beta, seq, alphabet)

        pi0 = update_pi0(pi0, eta)
        A = updateA(A, m, xi, seq)
        B = updateB(B, m, eta, seq, alphabet)

        keep_training = check_for_change(A, B, last_A, last_B, 0.00001)

        i += 1

    return A, B, pi0



# testing function designed to use generated params to sample a string
# can be used to compare against input to validate params
def sample_string(A,B,pi0,alphabet,T):

    seq = '' 
    state = np.argmax(pi0)
    obs = np.argmax(B[state])
    seq += (alphabet[obs])

    for _ in range(T - 1):

        state = np.random.choice(range(len(A[state])), p=A[state])
        obs = np.random.choice(range(len(B[state])), p=B[state])
        seq += (alphabet[obs])

    return seq

def generate_string(T, alphabet):

    seq = ''.join([np.random.choice([c for c in alphabet]) for _ in range(T)])

    return seq




# generate a sample sequence of length 100 from the alphabet 'AGCT-'
seq = generate_string(100, 'AGCT-')



# run the bw algorithm, with 20 states, on the generated sequence 
A, B, pi0 = baum_welch(20, seq, 'AGCT-')


# outputs the HMM parameters
print("INITIAL PROBABILITIES: ")
print()
print(pi0)

print("TRANSITION MATRIX: ")
print()
print(A)

print("EMMISSION MATRIX: ")
print()
print(B)


# print(seq)
# print(sample_string(A,B,pi0,"AGCT-",100))
