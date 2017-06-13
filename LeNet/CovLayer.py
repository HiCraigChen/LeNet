from numpy import *
from Layer import *


class CovLayer(Layer) :
        def __init__(self, lay_size = [], cov_core_sizes = [], mapcombindex = []) :
                Layer.__init__(self, lay_size)
                self.covcores = []
                self.covbias = []
                self.mapcombindex = mapcombindex

                # Initialize the parameters. 
                # The number -2.4/Fi and 2.4/Fi comes from the paper in Appendices A.
                for cov_core_size in cov_core_sizes :
                        # Fi is from the definition in paper
                        Fi = cov_core_size[0] * cov_core_size[1] + 1
                        # Make random filters
                        self.covcores.append(random.uniform(-2.4/Fi, 2.4/Fi, cov_core_size)) 
                        # Make random biases
                        self.covbias.append(random.uniform(-2.4/Fi, 2.4/Fi))

                self.covcores = array(self.covcores)


        def cov_op(self, pre_maps, covcore_index) :
                pre_map_shape = pre_maps.shape
                covcore_shape = self.covcores[covcore_index].shape
                map_shape = self.maps[covcore_index].shape

                # Check the input size, 
                # If the result from input size match the output
                # Calculate the Convolution Layer
                if not (map_shape[-2] == pre_map_shape[-2] - covcore_shape[-2] + 1 \
                    and map_shape[-1] == pre_map_shape[-1] - covcore_shape[-1] + 1) :
                    
                    return None

                for i in range(map_shape[-2]) :
                        for j in range(map_shape[-1]) :

                                # Filter caculation in HW4 
                                localrecept = pre_maps[ : , i : i + covcore_shape[-2], j : j + covcore_shape[-1]]
                                val = sum(localrecept * self.covcores[covcore_index]) + self.covbias[covcore_index]

                                # Use tanh(x) as activation function.
                                # Remark that tanh(x) = ((e^2x) -1)/((e^2x)+1)
                                # We use the parameters in the paper
                                # f(a) = Atanh(Sa), where A = 1.7159, S = 2/3

                                val = exp((4.0/3)*val) 
                                self.maps[covcore_index][i][j] = 1.7159 * (val - 1) / (val + 1) 


        # The mapcombflag parameter is for deciding which method should use.
        # It's because the Conv Layers will use different way to filter the input.
        def calc_maps(self, pre_mapset, mapcombflag = False) :
                
                # mapcombflag = False, the first Conv Layer
                if not mapcombflag :
                    for i in range(len(self.maps)) :
                        self.cov_op(pre_mapset, i)
                
                # Mapcombflag = True, the second Conv Layer
                else :
                    for i in range(len(self.maps)) :
                        self.cov_op(pre_mapset[self.mapcombindex[i]], i)


        def back_propa(self, pre_mapset, current_error, learn_rate, isweight_update) :
                self.current_error = current_error
                
                # Flatten the array into one dim
                selfmap_line = self.maps.reshape([self.maps.shape[0] * self.maps.shape[1] * self.maps.shape[2]])
                currenterror_line = current_error.reshape([current_error.shape[0] * current_error.shape[1] * current_error.shape[2]])
                
                # Backward : df/da = 2/3 * 1.7159 * (1-(tanh(2/3*a)^2))
                # Where tanh(2/3*a) = selfmap_line[i]/1.7159
                pcurrent_error = array([((2.0/3)*(1.7159 - (1/1.7159) * selfmap_line[i]**2))*currenterror_line[i]\
                                for i in range(len(selfmap_line))]).reshape(self.maps.shape)
                
                # Reset update data
                weight_update = self.covcores * 0
                bias_update = zeros([len(self.covbias)])
                pre_error = zeros(pre_mapset.shape)
                
                
                for i in range(self.maps.shape[0]) :
                        # If Conv2
                        if self.mapcombindex != [] :
                                pre_maps = pre_mapset[self.mapcombindex[i]]
                                select_pre_error = pre_error[self.mapcombindex[i]]
                        # If Conv1
                        else :
                                pre_maps = pre_mapset
                                select_pre_error = pre_error 
                        # Calculate Gradience.
                        for mi in range(self.maps.shape[1]) :
                                for mj in range(self.maps.shape[2]) :
                                    cov_maps = pre_maps[:, mi:mi+self.covcores[i].shape[1], mj:mj+self.covcores[i].shape[2]]
                                    weight_update[i] += cov_maps * pcurrent_error[i][mi][mj]
                                    bias_update[i] += pcurrent_error[i][mi][mj]
                                    select_pre_error[:, mi:mi+self.covcores[i].shape[1], mj:mj+self.covcores[i].shape[2]]\
                                                    += self.covcores[i] * pcurrent_error[i][mi][mj]
                # Update weights and biases
                if isweight_update :
                        self.covcores -= learn_rate * weight_update
                        self.covbias -= learn_rate * bias_update
                return pre_error