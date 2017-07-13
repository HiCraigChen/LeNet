from numpy import *
from FullyConLayer import *
import math 

class OutputLayer(FcLayer) :
    def __init__(self, lay_len, pre_nodesnum) :
        FcLayer.__init__(self, lay_len, pre_nodesnum)
        # We have to assign float64 to ensure the weight is float number. 
        self.weight = float64(random.choice([-1,1], [lay_len, pre_nodesnum])) # 10x84

    def rbf(self, pre_maps, node_index = -1) :
        pre_nodes = pre_maps.reshape([pre_maps.shape[0] * pre_maps.shape[1] * pre_maps.shape[2]]) #84

        if node_index != -1 :
            self.maps[0][0][node_index] = - 0.5 * sum((pre_nodes - self.weight[node_index])**2)

        else :
            for i in range(len(self.maps[0][0])) :
                self.maps[0][0][i] = - 0.5 * sum((pre_nodes - self.weight[i])**2)

    def back_propa(self, pre_mapset, current_error, learn_rate, isweight_update) :
        self.current_error = current_error
        current_error_matrix = array(matrix(list(current_error[0]) * self.weight.shape[1]).T)  #84x10 > 10x84
        if isweight_update :
            # pre_mapset : 1x1x84
            weight_update = (self.weight - array(list(pre_mapset[0]) * self.weight.shape[0]))
            self.weight -= learn_rate * weight_update
        pre_error = ((array(list(pre_mapset[0]) * self.weight.shape[0]) - self.weight) * current_error_matrix).sum(axis = 0)
        return pre_error.reshape(pre_mapset.shape) 



    def rbf_softmax(self, pre_maps, node_index = -1) :
        self.pre_nodes = pre_maps.reshape([pre_maps.shape[0] * pre_maps.shape[1] * pre_maps.shape[2]]) #84
        output = matmul(self.weight,self.pre_nodes)   # 10 x 1
        max_ineach = max(output)
        output = exp(output - max_ineach)
        output = output/sum(output)        
        self.maps[0][0] = output

    def back_propa_softmax(self, pre_mapset, current_error, learn_rate, isweight_update) :
        
        self.current_error = current_error
        current_error_matrix = array(matrix(list(current_error[0]) * self.weight.shape[1]).T)  #84x10 > 10x84
        if isweight_update :
            # pre_mapset : 1x1x84
            weight_update = (self.weight - array(list(pre_mapset[0]) * self.weight.shape[0])) * current_error_matrix   
            for i in range(10):
                weight_update[i] = (self.current_error[0][0][i] - self.maps[0][0][i])* self.pre_nodes
            self.weight -= learn_rate * weight_update
        pre_error = ((array(list(pre_mapset[0]) * self.weight.shape[0]) - self.weight) * current_error_matrix).sum(axis = 0)
        return pre_error.reshape(pre_mapset.shape) 







