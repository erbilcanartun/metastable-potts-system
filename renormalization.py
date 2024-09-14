import numpy as np
from scipy.misc import derivative

class PottsRenormalizationGroup:

    def __init__(self, state_q):

        self.q = state_q
        self.b = 2 # Rescaling factor
        self.d = 3 # Dimension
        
        self.m = self.b**(self.d - 1) # Bond-moving multiplier
        
        self.eigen = self.b ** self.d
        self.neigen = self.b ** (-self.d)

        def J(interaction):
            
            j = self.m * interaction
                
            e1 = 2 * j
            e2 = 0
            e3 = j
            e4 = 0
            
            emax1 = np.amax([e1, e2])
            emax2 = np.amax([e3, e4])
                
            return emax1 - emax2 + np.log(np.exp(e1 - emax1) + 
                                          (state_q - 1) * 
                                          np.exp(e2 - emax1)) - np.log(2 * np.exp(e3 - emax2) + 
                                                                       (state_q - 2) * np.exp(e4 - emax2))

        def critical_point(state_q, decimal_precision=5):
                
            p = 0.1
            jc = 0.2
            
            for n in range(decimal_precision+1):
                jc = jc - p
                p = 1 / 10 ** (n + 1)
                
                while True:
                    jx = jc
                    jx = J(jx)
                    if jx < jc:
                        jc = jc + p
                    else:
                        break
                        
            return round(jc, decimal_precision)

        self.Jc = critical_point(state_q, 12)


    def _pderivative(self, function, variable=0, point=[]):
        arguments = point[:]
        def wraps(x):
            arguments[variable] = x
            return function(*arguments)
        return derivative(wraps, point[variable], dx=np.sqrt(np.finfo(float).eps))

    
    # 1st RG transformation
    # Hamiltonian: J.(delta(Si,Sj)) + H.(delta(Si.a))
    
    def _additive_tilda_1(self, j, h):

        e1 = 8*j + h
        e2 = 0
        
        emax = np.amax([e1, e2])
        return emax + np.log(np.exp(e1 - emax) + (self.q - 1) * np.exp(e2 - emax))
    
    def _energy_ab_1(j, h):
        
        e1 = 4*j + h
        e2 = 4*j
        e3 = 0
        
        emax = np.amax([e1, e2, e3])
        return emax + np.log(np.exp(e1 - emax) + 
                             np.exp(e2 - emax) + 
                             (self.q - 1) * np.exp(e3 - emax)) - self._additive_tilda_1(j, h)
    
    def _energy_bb_1(j, h):
        
        e1 = 8 * j
        e2 = h
        e3 = 0
        
        emax = np.amax([e1, e2, e3])
        return emax + np.log(np.exp(e1 - emax) + 
                             np.exp(e2 - emax) + 
                             (self.q - 2) * np.exp(e3 - emax)) - self._additive_tilda_1(j, h)
    
    def _energy_bc_1(j, h):
        
        e1 = 4 * j
        e2 = h
        e3 = 0
        
        emax = np.amax([e1, e2, e3])
        return emax + np.log(2 * np.exp(e1 - emax) + 
                             np.exp(e2 - emax) + 
                             (self.q - 3) * np.exp(e3 - emax)) - self._additive_tilda_1(j, h)
    
    
    def _recursion_matrix_1(j, h):
    
        T12 = self._pderivative(self._additive_tilda_1, 0, [j, h])
        T22 = self._pderivative(self._energy_ab_1,      0, [j, h])
        T32 = self._pderivative(self._energy_bb_1,      0, [j, h])
        T42 = self._pderivative(self._energy_bc_1,      0, [j, h])
        
        T13 = self._pderivative(self._additive_tilda_1, 1, [j, h])
        T23 = self._pderivative(self._energy_ab_1,      1, [j, h])
        T33 = self._pderivative(self._energy_bb_1,      1, [j, h])
        T43 = self._pderivative(self._energy_bc_1,      1, [j, h])
        
        return [[self.eigen,  T12,  self.eigen * T13],
                [0,           T22,  self.eigen * T23],
                [0,           T32,  self.eigen * T33],
                [0,           T42,  self.eigen * T43]]
    
    
    def _recursion_matrix_11(j, h):
    
        T12 = (8 * np.exp(8*j + h)) / (np.exp(8*j + h) + self.q - 1)
        T22 = (4 * np.exp(4*j + h)) / (np.exp(4*j + h) + self.q - 1) - T12
        T32 = (8 * np.exp(8 * j)) / (np.exp(8 * j) + np.exp(h) + self.q - 2) - T12
        T42 = (8 * np.exp(4 * j)) / (2 * np.exp(4 * j) + np.exp(h) + self.q - 3) - T12
        
        T13 = (np.exp(8*j + h)) / (np.exp(8*j + h) + self.q - 1)
        T23 = (np.exp(4*j + h)) / (np.exp(4*j + h) + self.q - 1) - T13
        T33 = (np.exp(h)) / (np.exp(8 * j) + np.exp(h) + self.q - 2) - T13
        T43 = (np.exp(h)) / (2 * np.exp(4 * j) + np.exp(h) + self.q - 3) - T13
        
        return [[self.eigen, T12,  self.eigen * T13],
                [0,          T22,  self.eigen * T23],
                [0,          T32,  self.eigen * T33],
                [0,          T42,  self.eigen * T43]]
    
    
    # Other RG transformations
    # Hamiltonian: J.(delta(Si,Sj)) + H.(delta(Si.a)+delta(Sj.a))
    
    def _additive_tilda(energy_ab, energy_bb, energy_bc):
        Eab = self.m * energy_ab
        
        e1 = 0
        e2 = 2 * Eab
        
        emax = np.amax([e1, e2])
        return emax + np.log(np.exp(e1 - emax) + 
                             (self.q - 1) * np.exp(e2 - emax))
        
    def _energy_ab(energy_ab, energy_bb, energy_bc):
        Eab, Ebb, Ebc = self.m * energy_ab, self.m * energy_bb, self.m * energy_bc
        
        e1 = Eab
        e2 = Ebb + Eab
        e3 = Ebc + Eab
        
        emax = np.amax([e1, e2, e3])
        return emax + np.log(np.exp(e1 - emax) + 
                             np.exp(e2 - emax) + 
                             (self.q - 2) * np.exp(e3 - emax)) - self._additive_tilda(energy_ab, energy_bb, energy_bc)
    
    def _energy_bb(energy_ab, energy_bb, energy_bc):
        Eab, Ebb, Ebc = self.m * energy_ab, self.m * energy_bb, self.m * energy_bc
        
        e1 = 2 * Eab
        e2 = 2 * Ebb
        e3 = 2 * Ebc
        
        emax = np.amax([e1, e2, e3])
        return emax + np.log(np.exp(e1 - emax) + 
                             np.exp(e2 - emax) + 
                             (self.q - 2) * np.exp(e3 - emax)) - self._additive_tilda(energy_ab, energy_bb, energy_bc)
    
    def _energy_bc(energy_ab, energy_bb, energy_bc):
        Eab, Ebb, Ebc = self.m * energy_ab, self.m * energy_bb, self.m * energy_bc
        
        e1 = 2 * Eab
        e2 = Ebb + Ebc
        e3 = 2 * Ebc
        
        emax = np.amax([e1, e2, e3])
        return emax + np.log(np.exp(e1 - emax) + 
                             2 * np.exp(e2 - emax) + 
                             (self.q - 3) * np.exp(e3 - emax)) - self._additive_tilda(energy_ab, energy_bb, energy_bc)
    
    def _recursion_matrix(energy_ab, energy_bb, energy_bc):

        Eab, Ebb, Ebc = energy_ab, energy_bb, energy_bc
    
        T12 = self._pderivative(self._additive_tilda, 0, [Eab, Ebb, Ebc])
        T22 = self._pderivative(self._energy_ab,      0, [Eab, Ebb, Ebc])
        T32 = self._pderivative(self._energy_bb,      0, [Eab, Ebb, Ebc])
        T42 = self._pderivative(self._energy_bc,      0, [Eab, Ebb, Ebc])
        
        T13 = self._pderivative(self._additive_tilda, 1, [Eab, Ebb, Ebc])
        T23 = self._pderivative(self._energy_ab,      1, [Eab, Ebb, Ebc])
        T33 = self._pderivative(self._energy_bb,      1, [Eab, Ebb, Ebc])
        T43 = self._pderivative(self._energy_bc,      1, [Eab, Ebb, Ebc])
      
        T14 = self._pderivative(self._additive_tilda, 2, [Eab, Ebb, Ebc])
        T24 = self._pderivative(self._energy_ab,      2, [Eab, Ebb, Ebc])
        T34 = self._pderivative(self._energy_bb,      2, [Eab, Ebb, Ebc])
        T44 = self._pderivative(self._energy_bc,      2, [Eab, Ebb, Ebc])
        
        return [[self.eigen, T12,  T13,  T14],
                [0,          T22,  T23,  T24],
                [0,          T32,  T33,  T34],
                [0,          T42,  T43,  T44]]