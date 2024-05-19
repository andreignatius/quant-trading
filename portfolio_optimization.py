class optimPort:
    def __init__(self, raw_rets):
        self.raw_rets = raw_rets

    def calc_port_rets (self,w, freq=252):
        w= np.array(w)
        return (self.raw_rets@w).mean()
    
    def calc_port_sharpe (self,w, freq=252):
        rets = self.raw_rets@w
        return ((rets.mean()/rets.std())*(freq**0.5))
    
    def calc_port_std(self,w,cov=None, freq =252): 
        w= np.array(w)
        if type(cov)!=pd.DataFrame:
            cov =self.raw_rets.cov()
        port_std= np.sqrt((w.T@cov@w).values[0]) 
        return port_std


    def get_msr_wts(self, cov, riskfree_rate=0,exposure_constraint=1):
        n = self.raw_rets.shape[1]
        initial_guess = np.repeat (1/n,n) 

        if exposure_constraint ==0:
            low_bnd, upp_bnd = -0.5,0.5 
        elif exposure_constraint ==1:
            low_bnd, upp_bnd = 0,1
        elif exposure_constraint ==-1:
            low_bnd, upp_bnd -1,0
        else:
            low_bnd, upp_bnd = 0,1

        bounds = ((low_bnd, upp_bnd), )*n 
        weights_constraint = {'type':"eq",
                            'fun': lambda w: np. sum(w) - exposure_constraint
                            }
        def neg_sharpe (w, raw_rets, cov, riskfree_rate):
            r = self.calc_port_rets (w)
            vol =self.calc_port_std (w,cov = cov)
            return -(r-riskfree_rate)/vol
        
        weights = minimize (neg_sharpe, initial guess, 
                            args = (self.raw_rets, cov, riskfree_rate),
                            method= "SLSQP", 
                            constraints= (weights_constraint),
                            ounds = bounds)
        return weights.x


