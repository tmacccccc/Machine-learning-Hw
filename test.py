import numpy as np
from multiprocessing import Pool, Manager
import pickle
from time import time

np.seterr(divide='ignore', invalid='ignore')

class GMM_MLE(object):
    def __init__(self, m0s, c0s, weights):
        self.init_cache = (m0s, c0s, weights)
        
    @classmethod
    def get_gaussian(cls, data, mu, c):
        n = mu.shape[0]
        inv_c = np.linalg.inv(c)
        constant = 1 / ((2*np.pi)**(n/2) * np.linalg.det(c) ** (1/2))
        
        part1 = np.einsum('nx, xy -> ny', data - mu.T, inv_c)
        power = np.einsum('ny, ny -> n', part1, data - mu.T)
        return constant * np.exp(-0.5 * power)
    
    @classmethod
    def get_log_gaussian(cls, data, mu, c):
        n = mu.shape[0]
        inv_c = np.linalg.inv(c)
        constant = 1 / ((2*np.pi)**(n/2) * np.linalg.det(c) ** (1/2))
        
        part1 = np.einsum('nx, xy -> ny', data - mu.T, inv_c)
        power = np.einsum('ny, ny -> n', part1, data - mu.T)
        return np.log(constant) - 0.5 * power
    
    @classmethod
    def calc_expectation_cache(cls, data, current_cache):
        m0s, c0s, weights = current_cache
        
        # Li0: shape [n,]
        gmm_Ls = [w * cls.get_gaussian(data, m, c) for w,m,c in zip(weights, m0s, c0s)]
        Li0 = np.sum(gmm_Ls, axis=0)
        
        # wik: shape [n,k]
        wik = (gmm_Ls / Li0).T
        
        return wik
    
    @classmethod
    def update_c(cls, data, m, weights):
        part1 = np.einsum('nx,ny->nxy', data - m, data - m)
        return np.einsum('n, nxy->xy', weights, part1) / weights.sum() 
    
    @classmethod
    def update(cls, data, current_cache, exp_cache):
        m0s, c0s, weights = current_cache
        wik = exp_cache
         
        pk = wik.sum(axis=0) / wik.shape[0] 
        m_weight = wik.T # [k,n]
        m0s_new = np.einsum('nk,nd -> kd', wik, data) / np.sum(wik,axis=0)[...,np.newaxis] #[k,d]
        c0s_new = [cls.update_c(data, mk, wk) for mk,wk in zip(m0s_new, m_weight)] 
        return m0s_new, c0s_new, pk
    
    @classmethod
    def calc_Q(cls, data, current_cache, exp_cache):
        m0s, c0s, weights = current_cache
        wik = exp_cache
        
        #part_temp = [cls.get_gaussian(data, mk, ck) * wk 
        #             for mk,ck,wk in zip(m0s,c0s,weights)]
        #print(part_temp)
        log_part = [cls.get_log_gaussian(data, mk, ck) * wk 
                    for mk,ck,wk in zip(m0s,c0s,weights)]
        m_weight = wik.T #[k,n]
        part_0 = (log_part * m_weight).sum()
        return part_0 
        
    def __call__(self, data, criteria=0.25, max_iter=1000):
        Q_list = list()
        init_exp_cache = self.calc_expectation_cache(data, self.init_cache)
        current_cache = self.update(data, self.init_cache, init_exp_cache)
        
        init_Q = self.calc_Q(data, self.init_cache, init_exp_cache)
        Q_list.append(init_Q)
        i = 0
        while True:
            exp_cache = self.calc_expectation_cache(data, current_cache)
            new_cache = self.update(data, current_cache, exp_cache)
            Q = self.calc_Q(data, new_cache, exp_cache)
            Q_list.append(Q)
            current_cache = new_cache
            i += 1
            
            if np.abs(Q - Q_list[-2]) < criteria:
                break
            if i > max_iter:
                break

        return Q_list, new_cache
    
    
class GMM_EM_M(GMM_MLE):
    def __init__(self, m):
        def rdm():
            c = np.random.rand(1)[0] * 0.25 + 0.85
            # print(c)
            return c
        
        mus = [np.array([3*x*rdm() ,5*y*rdm() ])
               for x,y in zip(np.arange(-4,4,8/m), np.arange(-4,4,8/m))]
        sigmas = [np.eye(2) * r * rdm() for r in np.arange(1,m+1)/4]
        weights = [1/m] * m
        super(GMM_EM_M, self).__init__(mus, sigmas, weights)
        
    def train(self, data, criteria=0.25, max_iter=1000):
        return super(GMM_EM_M, self).__call__(data, criteria=criteria, max_iter=max_iter)
    
    def __call__(self, data, current_cache, bic=True):
        m0s, c0s, weights = current_cache
        
        # Li0: shape [n,]
        gmm_Ls = [w * self.get_gaussian(data, m, c) for w,m,c in zip(weights, m0s, c0s)]
        Li0 = np.sum(gmm_Ls, axis=0)
        likelihood = np.log(Li0).sum()
        
        if bic:
            k = np.prod(m0s[0].shape) + np.prod(c0s[0].shape) + 1
            k *= len(m0s)
            result = -2 * likelihood + k * np.log(data.shape[0])
        else:
            result = likelihood
        
        return result
     
        
def get_gaussian_MLE_parameters(data):
    n = data.shape[0]
    m_new = np.mean(data, axis = 0)
    part1 = data - m_new #[n,2]
    c_new = np.einsum('nx,yn -> xy', part1, part1.T)
    return m_new, c_new
    
    
def generate_gmm(mus: list, sigmas: list, weights: list, 
                 sample_size = 1000, seed=233):
    """
    Generate samples for Multivariate-GMM
    """
    rs = np.random.RandomState(seed)
    assert len(mus) == len(sigmas)
    assert len(sigmas) == len(weights)
    
    components = np.stack([rs.multivariate_normal(mu, sigma, sample_size) 
                           for mu, sigma in zip(mus, sigmas)])
    comp_label = np.random.multinomial(1, weights, sample_size)
    return np.einsum('cnd,nc->nd', components, comp_label)
    
    
def one_epoch_bic_test(data, value_dict, iter_idx):
    biclist = []
    em_100 = GMM_EM_M(1)
    # m = 1
    m, c = get_gaussian_MLE_parameters(data)
    biclist.append(em_100(data, [[m],[c],[1]]))
    # m = 2:20
    for m in range(2,21):
        """
        try:
            em_100 = GMM_EM_M(m)
            a,b = em_100.train(data)
            biclist.append(em_100(data,b))
        except:
            biclist.append(np.nan)
        """
        em_100 = GMM_EM_M(m)
        a,b = em_100.train(data)
        biclist.append(em_100(data,b))
    value_dict[iter_idx] = biclist


if __name__ == '__main__':
    # initialize mus and sigmas
    Mus_list = [np.array([3*x,5*y]) 
                for x,y in zip(np.arange(-4,4,0.5), np.arange(-4,4,0.5))]
    Sigma_list = [np.eye(2) * r for r in np.arange(1,len(Mus_list)+1)/4]
    np.random.shuffle(Sigma_list)
    weights_list = np.array([1/len(Mus_list)]*len(Mus_list))
    
    p = Pool(6)
    
    file_name = {100: '100', 1000: '1000', 1e4: '1e4', 1e5: '1e5', 1e6: '1e6'}
    for sample_size in [100, 1000, 1e4, 1e5, 1e6]:
        b = time()
        print('Size {} begin training ...... '.format(int(sample_size)), end='')
        data = generate_gmm(Mus_list, Sigma_list, weights_list, 
                                    sample_size = int(sample_size), 
                                    seed = 100)
        d = Manager().dict()
        tasks = [(data, d, i) for i in range(10)]
        p.starmap(one_epoch_bic_test, tasks)

        d = {k: v for k,v in d.items()}
        e = time()
        print('DONE! ({} mins)'.format(round((e - b)/60, 2)))
        
        print('Size {} begin stroing ...... '.format(int(sample_size)), end='')
        with open('test_result/{}_result.pkl'.format(int(sample_size)),'wb') as f:
            pickle.dump(d, f)
        print('DONE!')
            
    p.terminate()
    p.join()
    