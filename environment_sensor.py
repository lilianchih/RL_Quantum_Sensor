
import numpy as np
from scipy.linalg import expm, eig
from scipy.special import comb

N = 8
S = float(N)/2
Omega = np.pi*0.25
g = np.pi*0.25
mJ = np.arange(-S, S+1, dtype=float)
Jz2 = np.diag(np.square(mJ)).astype(complex)
Jz = np.diag(mJ).astype(complex)
Jp = np.diag(np.sqrt(S*(S+1) - mJ[:-1]*mJ[1:]), -1).astype(complex)
Jm = np.diag(np.sqrt(S*(S+1) - mJ[1:]*mJ[:-1]), 1).astype(complex)
Rx = expm(-1j*Omega*(Jp + Jm)/2)
Ry = expm(-Omega*(Jp - Jm)/2)
Sq = expm(-1j*g*Jz2)
psi0 = np.zeros(N+1, dtype=complex)
psi0[0] = 1.


n_theta = 9
n_phi = 8
CSS = [[np.zeros(N+1, dtype=complex)]*n_phi for i in range(n_theta)]
for i in range(n_theta):
    for j in range(n_phi):
         theta = i * np.pi / (n_theta - 1)
         phi = j * 2*np.pi / n_phi - np.pi
         css = np.zeros(N+1, dtype=complex)
         for k in range(N+1):
             css[k] = np.sqrt(comb(N, k)) * np.cos(theta/2)**(N-k) * np.sin(theta/2)**k
             css[k] *= np.exp(-1j*k*phi);
         CSS[i][j] = css

#print(CSS)
class Env(object):
    def __init__(self, noise=0.00):
        super(Env, self).__init__()
        self.n_actions = 3
        self.actions = [Rx, Ry, Sq]
        self.n_states = n_theta * n_phi
        self.psi = psi0
        self.state = self.husimiQ()
        self.n_steps = 20
        self.t = 0
        self.threshold = 0.95
        #self.noise = 0
        #self.var = noise
        
    def reset(self):
        self.psi = psi0
        self.state = self.husimiQ()
        self.t = 0

        return self.state

    def step(self, action):
        U = self.actions[action]
        self.psi = U.dot(self.psi)
        QFI = np.einsum('i,ij,j->', np.conjugate(self.psi), Jz2, self.psi) - np.abs(np.einsum('i,ij,j->', np.conjugate(self.psi), Jz, self.psi))**2
        QFI = np.real(QFI)
        QFI *= 4
        QFI /= N**2
        #reward
        done =( QFI > self.threshold or self.t >= self.n_steps-1)
        rwd = done * QFI
        if QFI > self.threshold:
            print(self.psi)
        self.t += 1
        self.state = self.husimiQ()
        return self.state, rwd, done, QFI
        
    def husimiQ(self):
        Q = np.zeros([n_theta, n_phi])
        for i in range(n_theta):
            for j in range(n_phi):
                Q[i][j] = abs(np.conjugate(CSS[i][j]).dot(self.psi)) ** 2
        return Q.reshape(n_theta * n_phi)
        




