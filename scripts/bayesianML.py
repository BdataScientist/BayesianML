import numpy as np
def gaussian(mean,var,xi):
    val1 = 1.0 / (np.sqrt(2*np.pi) * var)
    val2 = np.exp( (-1.0 / (2 * var**2) )  * (xi-mean)**2)
    return val1*val2

def estimate_param_for_feature(inputs,findex,targets,clabel):
    sample_indicies = np.where(targets==clabel)
    f_list = inputs[sample_indicies,findex]
    mean = np.mean(f_list)
    var = np.var(f_list)
    return mean,var
    
"""
 this method  returns M*N matrix.
 Here N represents the number of feature
 M represents the number of unique class label
"""
def fit_gaussian(x,inputs,targets):
    if(len(x) != np.shape(inputs)[1]):
        print("Invalid vector")
        return None
    x_prob_list = []
    unique_class = np.unique(targets)
    for c in unique_class:
        temp_prob_list = []
        for findex in range(np.shape(x)[0]):
            mean,var = estimate_param_for_feature(inputs,findex,targets,c)
            p_xi_ck = gaussian(mean,var,x[findex])
            temp_prob_list.append(p_xi_ck)
        x_prob_list.append(temp_prob_list)
    return x_prob_list
    


def estimate_priori_ck(targets):
    ck_list = []
    unique_class = np.unique(targets)
    for clabel in unique_class:
        ssk = len(np.where(targets==clabel)[0])
        tot = len(targets) + len(unique_class)
        ck = ((ssk + 1.0) / tot)
        ck_list.append(ck)
    return ck_list
    

def compute_class_conditional_likelihood(x,inputs,targets):
    ps = fit_gaussian(x,inputs,targets)
    ck = estimate_priori_ck(targets)
    p_x_c_p_c = []
    for i in range(len(ck)):
        elem_p_x_ck = ps[i]
        elem_c_k = ck[i]
        p_x_ck = np.product(elem_p_x_ck)
        p_x_c_p_c.append(p_x_ck * elem_c_k)
    return p_x_c_p_c

def perform_leave_one_out(inputs,targets):
    ss = np.shape(inputs)[0]
    predict_labels = []
    for i in range(ss):
        trainInputs = [x for j,x in enumerate(inputs) if j!=i]
        trainTargets = [x for j,x in enumerate(targets) if j!=i]
        testInputs = inputs[i,:]
        trainInputs = np.array(trainInputs)
        testInputs = np.array(testInputs)
        trainTargets = np.array(trainTargets)
        ps = compute_class_conditional_likelihood(testInputs,trainInputs,trainTargets)
        clabel = np.where(ps == max(ps))
        clabel = clabel[0][0]
        predict_labels.append(clabel)
    return predict_labels

def estimate_params_for_bayes(inputs,targets,clabel):
    fs = np.shape(inputs)[1]
    sample_indicies = np.where(targets==clabel)
    inp_belong_c = inputs[sample_indicies]
    tot = np.sum(inp_belong_c,axis=0)
    mean = tot / len(sample_indicies[0])
    sigma = np.zeros([fs,fs])
    for elem in inp_belong_c:
        sigma = sigma + (elem-mean) * np.transpose([elem-mean])
    return mean,sigma/len(sample_indicies[0])

def calc_gauss_for_bayes(x,mean,sigma,d):
    val1 = ((2*np.pi) ** (d / 2.0)) * np.sqrt(np.linalg.det(sigma))   
    differ = x-mean
    inv_sigma = np.linalg.inv(sigma)
    val2 = np.exp(-0.5 * (differ @ inv_sigma @ np.transpose(differ)))
    return val1*val2    

def compute_p_x_ck(x,inputs,targets):
    unique_class = np.unique(targets)
    fs = np.shape(inputs)[1]    
    p_x_ck = []
    for clabel in unique_class:
        mean,sigma = estimate_params_for_bayes(inputs,targets,clabel) 
        p_x_ck.append(calc_gauss_for_bayes(x,mean,sigma,fs))
    return p_x_ck
    
def perform_leave_one_out_for_bayes(inputs,targets):
    ss = np.shape(inputs)[0]
    predict_labels = []
    for i in range(ss):
        trainInputs = [x for j,x in enumerate(inputs) if j!=i]
        trainTargets = [x for j,x in enumerate(targets) if j!=i]
        testInputs = inputs[i,:]
        trainInputs = np.array(trainInputs)
        testInputs = np.array(testInputs)
        trainTargets = np.array(trainTargets)
        
        p_x_ck = compute_p_x_ck(testInputs,trainInputs,trainTargets)
        p_ck = estimate_priori_ck(trainTargets)     
        ps = np.multiply(p_x_ck,p_ck)
        clabel = np.where(ps == max(ps))
        clabel = clabel[0][0]
        predict_labels.append(clabel)
    return predict_labels
