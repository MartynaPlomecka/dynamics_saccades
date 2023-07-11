import numpy as np
from itertools import compress

def generateHankelMatrix(time_series, delay_dimension):
    #detect trial transitions
    num_segments = len(time_series)#get number of trials
    segment_lengths = [len(segment) for segment in time_series]#get list of trial time step lengths
    min_segment_length = min(segment_lengths)#what is the minimum trial step length
    max_segment_length = max(segment_lengths)#what is the maximum trial step length
    assert min_segment_length > delay_dimension#check that the shortest trial length is still greater than the desired embedding dimension
    num_rows = delay_dimension#the number of rows in the Hankel matrix is the minimum trial step length minus the delay dimension plus 1.
    #get the number of columns of the Hankel matrix
    #this is slightly more complicated, longer trials will have more columns
    #as the number of time steps in each column is dictated by the shortest trial and thus
    #to span the entire length of the dynamic of each trial, more columns may be needed.
    num_cols = sum([segment_lengths[i] - delay_dimension + 1 for i in range(len(segment_lengths))])

    hankel_matrix = np.zeros((num_rows, num_cols))#create empty hankel matrix
    hankel_ids = []
    start = 0
    for i in range(num_segments):#iterate over each trial
        segment = time_series[i]#get the trial data
        embeddings = []
        num_embd_vec = len(segment) - delay_dimension + 1
        for j in range(num_embd_vec):
            embd_tmp = segment[j:j+delay_dimension]
            embeddings.append(embd_tmp)
            hankel_ids.append(i)
        hankel_matrix[:, start:start+len(embeddings)] = np.array(embeddings).T
        start += len(embeddings)

    return hankel_matrix, hankel_ids


def nearestNeighbors(trajectories, delay_dimension, gamma):
    #gamma needs to be a list or array
    H0_full, H0_ids_full = generateHankelMatrix(trajectories, delay_dimension)#lower, or reference, embedding dimension hankel matrix
    H1, _ = generateHankelMatrix(trajectories, delay_dimension+1)#higher embedding dimension hankel matrix
    perc_errors = []
    H0 = np.zeros_like(H1)
    H0 = np.delete(H0, obj=-1, axis=0)
    #remove final samples in trajectories
    r=0
    for r_full in range(H0_full.shape[0]):
        c=0
        for c_full in range(H0_full.shape[1]-1):
            if H0_ids_full[c_full] == H0_ids_full[c_full+1]: #when on any sample that isnt the last of trial we add it to the new hankel matrix
                H0[r][c] = H0_full[r_full][c_full]
                c+=1
        r+=1
    #iterate over each sample, i.e. column space of hankel matrix
    for i in range(H0.shape[1]):
        hi = H0[:,i]#get the embedding vector for this sample index
        #calculate the distances between the embedding vector of interest and all other embedding vectors
        #in the lower embedding dimension H0
        dists=np.sqrt(np.sum(np.square(H0-hi[:,None]),axis=0))
        dists[i] = np.inf#set the distance of the current sample to itself to be infinity
        j = np.argmin(dists)#get the minimum distance index
        Rk=dists[j]**2#square the distance within the lower embedding dimension hankel matrix
        Rkp1=np.sum(np.square(H1[:,i]-H1[:,j]))#same for higher embedding dimension, i.e. the same indices as the lower dimension hankel matrix
        #calculate the error as a percentage of the distance between the two neighbors in the lower dimension space
        if(Rk!=0): #if there is no error in Rk
            perc_error=np.sqrt(np.abs(Rkp1-Rk)/Rk)#see Kennel et al 1992 DETERMINING EMBEDDING DIMENSION FOR PHASE-SPACE...
        elif(Rkp1==0): #when the higher dimensional error is also 0
            perc_error=0
        else:
            perc_error=np.inf
        perc_errors.append(perc_error)#collect the percent errors
    perc_errors=np.array(perc_errors)#change to a numpy array
    #need to calculate the number of perc errors that exceed given threshold
    ratios = []
    for g in gamma:#iterate over passed threshold values
        ratio = sum(perc_errors > g)/len(perc_errors)#compare all perc_errors against the threshold
        ratios.append(ratio)#append the ratio of perc errors that exceed g
    #return array of ratios for each gamma
    return ratios

def nearestNeighbors_parallel(trajectories, delay_dimension, g):
    #gamma is a single value, for parallelization
    H0_full, H0_ids_full = generateHankelMatrix(trajectories, delay_dimension)#lower, or reference, embedding dimension hankel matrix
    H1, _ = generateHankelMatrix(trajectories, delay_dimension+1)#higher embedding dimension hankel matrix
    perc_errors = []
    H0 = np.zeros_like(H1)
    H0 = np.delete(H0, obj=-1, axis=0)
    #remove final samples in trajectories
    r=0
    for r_full in range(H0_full.shape[0]):
        c=0
        for c_full in range(H0_full.shape[1]-1):
            if H0_ids_full[c_full] == H0_ids_full[c_full+1]: #when on any sample that isnt the last of trial we add it to the new hankel matrix
                H0[r][c] = H0_full[r_full][c_full]
                c+=1
        r+=1
    #iterate over each sample, i.e. column space of hankel matrix
    for i in range(H0.shape[1]):
        hi = H0[:,i]#get the embedding vector for this sample index
        #calculate the distances between the embedding vector of interest and all other embedding vectors
        #in the lower embedding dimension H0
        dists=np.sqrt(np.sum(np.square(H0-hi[:,None]),axis=0))
        dists[i] = np.inf#set the distance of the current sample to itself to be infinity
        j = np.argmin(dists)#get the minimum distance index
        Rk=dists[j]**2#square the distance within the lower embedding dimension hankel matrix
        Rkp1=np.sum(np.square(H1[:,i]-H1[:,j]))#same for higher embedding dimension, i.e. the same indices as the lower dimension hankel matrix
        #calculate the error as a percentage of the distance between the two neighbors in the lower dimension space
        if(Rk!=0): #if there is no error in Rk
            perc_error=np.sqrt(np.abs(Rkp1-Rk)/Rk)#see Kennel et al 1992 DETERMINING EMBEDDING DIMENSION FOR PHASE-SPACE...
        elif(Rkp1==0): #when the higher dimensional error is also 0
            perc_error=0
        else:
            perc_error=np.inf
        perc_errors.append(perc_error)#collect the percent errors
    perc_errors=np.array(perc_errors)#change to a numpy array
    #need to calculate the number of perc errors that exceed given threshold
    ratio = sum(perc_errors > g)/len(perc_errors)#compare all perc_errors against the threshold
    return ratio

def find_last_indices(lst):
    last_indices = {}
    for index, num in enumerate(lst):
        last_indices[num] = index
    return last_indices

def find_first_indices(lst):
    first_indices = {}
    for index, num in enumerate(lst):
        if num not in first_indices:
            first_indices[num] = index
    return first_indices

def compute_jacobians(trajectories, delay_dimension=3, num_singular_vectors=3, scale=[2], residuals=False):
    #given our equation for the system dynamics, we are interested in approximating
    #the dynamics around each ith sample as follows:
    #xn+1 = f(xn)
    #xn+1 = f(x0) + J(x0)(xn - x0) + higher order terms ...
    #rearranging...
    # xn+1 - f(x0) = J(x0)(xn - x0)
    # (xn+1 - x1) = J(x0)(xn - x0)
    #where x0 represents the point at which we are approximating the dynamics, i.e. the ith data point

    H, H_ids = generateHankelMatrix(trajectories, delay_dimension)#lower, or reference, embedding dimension hankel matrix
    u,s,vt=np.linalg.svd(H,full_matrices=True)#take SVD to get the right singular vectors vt
    all_v_np=vt[:num_singular_vectors,:]#get the first k number of right singular vectors (rows of vt)
    last_indices = find_last_indices(H_ids)#get the indices of the Hankel matrix corresponding to the last time delay vectors for each trial
    first_indices = find_first_indices(H_ids)#get the indices of the Hankel matrix corresponding to the first time delay vectors for each trial
    #get single time step shifted delay embedded data coordinates:
    X0 = np.delete(all_v_np, list(last_indices.values()),axis=1)#remove the last sample indices for each trial from the collection of right singular vectors
    X1 = np.delete(all_v_np, list(first_indices.values()),axis=1)#remove the first sample indices for each trial from the collection of right singular vectors
    trial_ids = np.delete(H_ids, list(last_indices.values()))#remove the last sample indices for each trial from the list of sample indices
    #iterate over all the samples
    if residuals:
        Rs = []
    Js = []
    for s in scale:
        J = []
        if residuals:
            R = []
        for i in range(X0.shape[1]):
            xi=np.array([X0[:,i]]).T#X0 has sample along the columns, and vs along the rows, grab the ith sample point
            X0_tilde = X0 - xi#(xn - x0), subtract the ith sample point from all other sample points
            x1i=np.array([X1[:,i]]).T#X1 has sample along the columns, and vs along the rows, grab the ith sample point
            X1_tilde = X1 - x1i#(xn+1 - x1), subtract the ith sample point from all other sample points.
            #calculate the distances between all data points
            #distance array has length equal to number of time points across all trials
            dist0=np.squeeze(np.sqrt(np.sum(np.square(X0_tilde),axis=0)))#each entry in dist0 corresponds to the distance to the ith sample point
            #get exponential decaying weights with magnitude of v. this is how much each sample point should
            #contribute to the linear regression used to approximate the dynamics
            weights0=np.exp(-np.square(dist0)/(2*s**2))*(1/(s*np.sqrt(2*np.pi)))

            W_squared = np.diag(weights0**2)#place weights along the diagonal
            #weight the vs 
            #J^* = min_J ||YW-JXW||_F^2 = \tilde{X}_1 W^2 \tilde{X}_0^T (\tilde{X}_0 W^2 \tilde{X}^T)^{-1} ... trust us, the math is good
            Eps = np.linalg.norm(X0_tilde @ W_squared @ X0_tilde.T)*(10E-9)*np.eye(X0_tilde.shape[0])
            Ji = X1_tilde @ W_squared @ X0_tilde.T @ np.linalg.inv(X0_tilde @ W_squared @ X0_tilde.T + Eps)
            J.append(Ji)

            #calculate residuals
            if residuals:
                W = np.diag(weights0)
                Ri = X1_tilde @ W - Ji @ X0_tilde @ W
                R.append(np.linalg.norm(Ri, axis=0))
        if residuals:
            Rs.append(R)
        Js.append(J)
    if residuals:
        return Js, Rs, trial_ids, X0
    else:
        return Js, trial_ids, X0


def compute_jacobians_parallel(trajectories, delay_dimension=3, num_singular_vectors=3, s=2, residuals=False):
    #given our equation for the system dynamics, we are interested in approximating
    #the dynamics around each ith sample as follows:
    #xn+1 = f(xn)
    #xn+1 = f(x0) + J(x0)(xn - x0) + higher order terms ...
    #rearranging...
    # xn+1 - f(x0) = J(x0)(xn - x0)
    # (xn+1 - x1) = J(x0)(xn - x0)
    #where x0 represents the point at which we are approximating the dynamics, i.e. the ith data point

    H, H_ids = generateHankelMatrix(trajectories, delay_dimension)#lower, or reference, embedding dimension hankel matrix
    u,s_,vt=np.linalg.svd(H,full_matrices=True)#take SVD to get the right singular vectors vt
    all_v_np=vt[:num_singular_vectors,:]#get the first k number of right singular vectors (rows of vt)
    last_indices = find_last_indices(H_ids)#get the indices of the Hankel matrix corresponding to the last time delay vectors for each trial
    first_indices = find_first_indices(H_ids)#get the indices of the Hankel matrix corresponding to the first time delay vectors for each trial
    #get single time step shifted delay embedded data coordinates:
    X0 = np.delete(all_v_np, list(last_indices.values()),axis=1)#remove the last sample indices for each trial from the collection of right singular vectors
    X1 = np.delete(all_v_np, list(first_indices.values()),axis=1)#remove the first sample indices for each trial from the collection of right singular vectors
    trial_ids = np.delete(H_ids, list(last_indices.values()))#remove the last sample indices for each trial from the list of sample indices

    J = []
    if residuals:
        R = []
    for i in range(X0.shape[1]):
        xi=np.array([X0[:,i]]).T#X0 has sample along the columns, and vs along the rows, grab the ith sample point
        X0_tilde = X0 - xi#(xn - x0), subtract the ith sample point from all other sample points
        x1i=np.array([X1[:,i]]).T#X1 has sample along the columns, and vs along the rows, grab the ith sample point
        X1_tilde = X1 - x1i#(xn+1 - x1), subtract the ith sample point from all other sample points.
        #calculate the distances between all data points
        #distance array has length equal to number of time points across all trials
        dist0=np.squeeze(np.sqrt(np.sum(np.square(X0_tilde),axis=0)))#each entry in dist0 corresponds to the distance to the ith sample point
        #get exponential decaying weights with magnitude of v. this is how much each sample point should
        #contribute to the linear regression used to approximate the dynamics
        weights0=np.exp(-np.square(dist0)/(2*s**2))*(1/(s*np.sqrt(2*np.pi)))

        W_squared = np.diag(weights0**2)#place weights along the diagonal
        #weight the vs 
        #J^* = min_J ||YW-JXW||_F^2 = \tilde{X}_1 W^2 \tilde{X}_0^T (\tilde{X}_0 W^2 \tilde{X}^T)^{-1} ... trust us, the math is good
        Eps = np.linalg.norm(X0_tilde @ W_squared @ X0_tilde.T)*(10E-9)*np.eye(X0_tilde.shape[0])
        Ji = X1_tilde @ W_squared @ X0_tilde.T @ np.linalg.inv(X0_tilde @ W_squared @ X0_tilde.T + Eps)
        J.append(Ji)

        #calculate residuals
        if residuals:
            W = np.diag(weights0)
            Ri = X1_tilde @ W - Ji @ X0_tilde @ W
            R.append(np.linalg.norm(Ri, axis=0))

    if residuals:
        return J, R, trial_ids, X0
    else:
        return J, trial_ids, X0
    
def compute_FTLEs_parallel(trajectories, scale, T, delay_dimension=5, num_singular_vectors=3):
    Js, trial_ids, X0 = compute_jacobians_parallel(trajectories, delay_dimension=delay_dimension, num_singular_vectors=num_singular_vectors, s=scale, residuals=False)
    vc = []
    lyv_vc = []
    les_vc = []
    ly_ids = []
    Ms_all = Js
    for j in np.unique(trial_ids):
        Ms_v = list(compress(Ms_all, trial_ids == j))
        X0_seg = X0[:, trial_ids == j][:, :-T+1]

        for i in range(len(Ms_v)-T+1):
            Ms_iT = Ms_v[i:i+T]

            Q, les = getFTL(Ms_iT)

            lyv_vc.append(Q[:, 0])
            les_vc.append(les)
            vc.append(X0_seg[:, i])
            ly_ids.append(j)

    lyv_vc = np.array(lyv_vc).T
    les_vc = np.array(les_vc)
    vc = np.stack(vc, axis=0).T
    ly_ids = np.array(ly_ids)

    return {'lyv_vc':lyv_vc,'les_vc':les_vc,'vc':vc,'ly_ids':ly_ids}

def getFTL(Ms_iT):
    T=len(Ms_iT)
    d=np.shape(Ms_iT[0])[0]
    les=np.zeros(d)
    Q=np.eye(d)
    for M in Ms_iT:
        P=np.matmul(M,Q)
        Q,R=np.linalg.qr(P)#,mode='complete')
        #rsorted=np.sort(np.diag(R))[::-1]
        rdiag=np.diag(R)
        Q[:,rdiag<0]=-Q[:,rdiag<0]
        #if np.sum(rdiag<0)>0:
        #    print(rdiag)
        les=np.copy(les)+np.log(np.abs(rdiag))
    les=les/T
    sortID=np.argsort(-les)
    return Q[:,sortID],les[sortID]

def getFTL2(Ms_iT):
    T=len(Ms_iT)
    d=np.shape(Ms_iT[0])[0]
    #les=np.zeros(d)
    MF=np.eye(d)
    for M in Ms_iT:
        MF=np.matmul(M,MF)
    Q,les,Q2=np.linalg.svd(MF)
    return Q2.T,np.log(les)/T


import contextlib
import joblib
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()