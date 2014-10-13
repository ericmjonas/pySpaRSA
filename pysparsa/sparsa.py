import numpy as np
import sys
import util
import time

default_phi_function = lambda x: np.sum(np.abs(x))
default_psi_function = util.soft


def sparsa(y, Aop, tau,
           stopCriterion = 2, tolA = 0.01, tolD = 0.001, debias=0, maxiter = 10000,
           maxiter_debias = 200, miniter = 5, miniter_debias=0,
           init = 'zeros', bbVariant = 1, bbCycle = 1,
           enforceMonotone = 0, enforceSafeguard = 0,
           M = 5, sigma = 0.01, cont_steps = -1,
           verbose=True, 
           alphamin = 1e-30, alphamax = 1e30, compute_mse = 0):
    """
    Initializations:
       'zeros': 
    

    """
    # FIXME add lots of sanity checks

    # precompute A^T(y)
    ATy = Aop.compute_ATx(y)

    # sanity check phi function
    psi_function = default_psi_function # FIXME use arbitary 
    phi_function = default_phi_function # FIXME use arbitary 

    if init == 'zeros':
        x = Aop.compute_ATx(np.zeros(len(y)))
        
    else:
        raise NotImplementedError("unknown initialization")

    # FIXME something to handle large tau? 

    # FIXME do something with true x?

    nz_x = x == 0.0 # FIXME should this be a tolerance?
    num_nz_x = np.sum(nz_x)

    final_tau = tau
    
    #  FIXME some continuation stuff

    final_stopCriterion = stopCriterion
    final_tolA = tolA
    
    if cont_steps == -1:
        tau = sys.float_info.max
        
    keep_continuation = True
    cont_loop = 1
    iter = 1
    taus = []
    mses = []
    
    objective = []
    times = []
    t0 = time.time()

    debias_start = 0
    x_debias = []

    
    while keep_continuation:
        iterThisCycle = 0

        # compute initial resideu and gradient
        resid = Aop.compute_Ax(x) - y
        gradq = Aop.compute_ATx(resid)

        if cont_steps == -1:
            temp_tau = max(final_tau, 0.2 * np.max(np.abs(gradq)))
            if temp_tau > tau:
                tau = final_tau
            else:
                tau = temp_tau;

                if tau == final_tau:
                    stopCriterion = final_stopCriterion
                    tolA = final_tolA
                    keep_continuation = 0
                else:
                    stopCriterion = 1
                    tolA = 1e-5

        else:
            tau = final_tau * cont_factors(cont_loop)
            if cont_loop == cont_steps:
                pass # FIXME don't handle this now
            else:
                pass # FIXME don't handle this now
        taus.append(tau)
        
        
        # compute and store initial value of the objective function for this tau
        alpha = 1.0
        f = 0.5 * np.dot(resid, resid) + tau * phi_function(x)
    
        if enforceSafeguard:
            f_lastM = f

        # at the very start of the process, store the initial mses and objective in
        # plotting arrays FIXME DO

        keep_going = 1

        while keep_going:
            gradq = Aop.compute_ATx(resid)

            # save current values
            prev_x = x
            prev_f = f
            prev_resid = resid

            # computation of step 
            cont_inner = True
            while cont_inner:
                x = psi_function(prev_x - gradq*1.0/alpha, tau/alpha)
                dx = x - prev_x
                Adx = Aop.compute_Ax(dx)
                resid = prev_resid + Adx
                f = 0.5 * np.dot(resid, resid) + tau * phi_function(x)
                if enforceMonotone:
                    f_threshold = prev_f
                elif enforceSafeguard:
                    f_threshold = max(f_lastM) - 0.5 * sigma * alpha * np.dot(dx, dx)
                else:
                    f_threshold = np.inf

                if f < f_threshold:
                    cont_inner = False
                else:
                    # not good enough, increase alpha and try again
                    alpha = eta * alpha
            if enforceSafeguard:
                if len(f_lastM) > M:
                    f_lastm.pop(0)

                f_lastM.append(f)
                    
            if verbose:
                print "t=%4d, obj=%10.6f, alpha=%f" % (iter, f, alpha)

            if bbVariant == 1: # Fixme pick a better name
                # standard BB Choice of init alpha for next step
                if iterThisCycle == 0 or enforceMonotone:
                    dd = np.dot(dx, dx)
                    dGd = np.dot(Adx, Adx)
                    alpha = min(alphamax, max(alphamin, dGd / (sys.float_info.min + dd)))
            elif bbVariant == 2:
                raise NotImplementedError("Unkown bbvariant")
            else:
                alpha = alpha * alphaFactor

            # update counts
            iter +=1
            iterThisCycle = (iterThisCycle + 1) % bbCycle
            objective.append(f)
            times.append(time.time() - t0)

            if compute_mse:
                err = true - x
                mses.append(np.dot(err, err))

            if stopCriterion == 0: # FIXME better name
                # compute stoping criterion based on the change of the number
                # of non-zero components of the estimate
                nz_x_prev = nz_x
                nz_x = np.abs(x) != 0.0
                num_nz_x = np.sum(nz_x)
                if num_nz_x > 1:
                    criterionActiveSet = num_changes_Active / num_nz_x
                    keep_going = criterionActiveSet > tolA
                if verbose:
                    print "Delta nz= %d (target = %f)" % (criterionActiveSet, tolA)
            elif stopCriterion == 1:
                criterionObjective = np.abs(f - prev_f) / prev_f
                keep_going = criterionObjective > tolA
                
                    
            elif stopCriterion == 2:
                # compute the "duality" stoping criterion
                scaleFactor = np.linalg.norm(gradq, np.inf)
                w = tau * prev_resid / scaleFactor
                criterionDuality = 0.5 * np.dot(prev_resid, prev_resid) + tau * phi_function(prev_x) + 0.5 * np.dot(w, w) + np.dot(y, w)
                criterionDuality /= prev_f
                keep_going = criterionDuality > tolA
                
                    
            else:
                raise NotImplementedError("Unknown Stopping Criterion %d" % stopCriterion) 
            
                    
            if iter < miniter:
                keep_going = True
            elif iter > maxiter:
                keep_going = False

        cont_loop += 1
        
        if verbose:
            # print some stuf
            pass

        # FIXME add debias
        
                
                    
                            
        # fixme MSEs
        
    return {'x' : x,
            'x_debias' : x_debias,
            'objective' : objective,
            'times' : times,
            'debias_start' : debias_start,
            'mses' : mses,
            'taus' : taus}
