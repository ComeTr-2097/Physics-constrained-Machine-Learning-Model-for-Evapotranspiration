# About Physics-contrained ET Hybrid Model

The hybrid model in this study consists of two main components: a machine learning module for simulating rs or ra, and a physical model for predicting LE. The introduction of the machine learning component adds computational cost compared to the physical model. The training time of the RF and LGBM models is relatively fast (within 10 minutes). The fitting process of the ANN requires a relatively longer training time (about 30 minutes), mainly due to the adjustment of synaptic weights through backpropagation. However, we implemented an early stopping strategy to minimize computational resource waste and prevent overfitting. As machine learning is less dependent on specific hardware, the hybrid model remains competitive, offering high prediction accuracy with only a slight increase in computational cost, thus providing excellent cost-effectiveness. The hardware platform used in the experiments consists of a 32-core CPU and 64GB system memory. During training, the average CPU and memory utilization were 30% and 50%, respectively.

We developed our hybrid models using Python, with the implementation details available on GitHub (). The key hyperparameters before model fitting include hidden_layers, neurons_per_layer, learning_rate, batch_size, n_estimators, max_depth, etc. These hyperparameters were optimized through cross-validation to achieve the best performance for the specific datasets. Considering previous research and computational efficiency (Zhao et al., 2019), we adopted 6, 64, 0.005, 32, 500, and 50 as the above hyperparameters after careful tuning. Similar to machine learning models, the dataset was partitioned into training (70%) and validation (20%) sets to dynamically update the internal model parameters during the fitting process. The test set (10%) was then used to assess the performance of the physics-constrained hybrid models. 
