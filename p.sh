The system is characterized by a set of Cartesian coordinates q(t) that represent the configuration of a rigid body at time t. The Lagrangian function of the system, defined in Equation (3), is a fundamental component of our proposed methodology:
		(3)
where T represents the kinetic energy of the system, and V represents its potential energy. To efficiently learn Lagrangian quantities of physical systems, we parameterize the kinetic and potential energy using two separate neural networks. We leverage Lagrangian quantities as prior knowledge in this process.
In the context of mechanical systems, the kinetic energy T of the system is determined by Equation (4):
		(4)
where the generalized mass matrix  which is a constant matrix in Cartesian coordinates and does not depend on the state q(t). Therefore, learning the constant values of this matrix using a neural network can simplify the form of the Lagrangian function. The dynamics of a multi-body system can be expressed using the DAE as shown in Equation (5):
		(5)
The above equation describes the system using the generalized velocity , the Lagrange multiplier λ, the constraint equation  for the position coordinate array q, the generalized force matrix F, and the Jacobi matrix of the constraint equation, . To obtain the velocity constraint equation  and the acceleration constraint equation  for the system, we need to solve the constraint equation  for the first and second-order derivatives with respect to time t, respectively. The velocity constraint equation, as shown in Equation (6), is given by:
		(6)
Similarly, the acceleration constraint equation, as shown in Equation (7), is given by:
		(7)
These equations result in the system of index-1 DAE, given by Equation (8):
		(8)
where the generalized force array . Equation (8) can be reformulated as Equation (9), which explicitly expresses the acceleration of the generalized coordinates q(t) and the Lagrange multipliers λ as a function of the other variables. This reformulation can be particularly helpful in numerical integration of the DAE.
		(9)
3.2  |Multiscale Module
Dynamical systems often exhibit multi-frequency phenomena, particularly when their components vary significantly in parameters or involve a combination of slow variables with a wide range of motion and fast variables with elastic deformation. Consequently, the solutions of such systems comprise multiple frequency components that are superimposed on each other. Deep neural networks (DNNs) excel at processing data with low-frequency content, as supported by the frequency principle (F-principle) 38. DNNs can rapidly learn the low-frequency content of data and achieve commendable generalization accuracy. However, neural networks often struggle when confronted with high-frequency data, leading to reduced convergence or even non-convergence of the learning method. In the domain of multi-body dynamics, the learning of dynamical systems poses challenges due to the frequency disparities between the motions of objects within the system.
To tackle this challenge, we employ a multiscale structure to preprocess the input data and extract frequency features that are better suited for learning. Specifically, we adopt the multiscale structure proposed by Liu et al. 39. This approach has demonstrated its efficacy in facilitating the rapid learning of high-frequency components and expediting the solution of partial differential equations in comparison to traditional fully connected network structures. The multiscale module employs radial scaling to convert solution content from higher frequency ranges to lower frequency ranges, thereby rendering the solution content easier to learn. The module takes the state variable q as input and produces the potential energy V of the system as output. A schematic diagram representation of the module is depicted in Figure 2.


Figure 2 Schematic diagram of multiscale module.
In our multiscale module, we utilize the linear combination property of energy to establish the potential energy of the system 35. The potential energy comprises two main components: the potential energy, denoted as Vi, between the elements and the environment, and the potential energy, denoted as Vij, between the elements themselves. The expression for the potential energy is presented in Equation (10).
		(10)
where the weight parameters of Vi and Vij are denoted by ci and cij, respectively. Additionally, Vij is symmetric, i.e., Vij=Vji.
To obtain consistent frequency accuracy solutions, the position coordinates q are segmented into different frequency ranges using radial scaling. This segmentation allows for the utilization of m parallel subneural networks, denoted as Vθ, which are responsible for calculating the potential energy within each frequency range. Finally, we calculate the potential energy using a weighted summation, as shown in Equation (11).
		(11)
where αk and θk denote the scaling factor and network parameters of the k-th sub-network, respectively. Additionally, we use residual connections to address the gradient vanishing issue.
During the training process, our model takes the system's states  as inputs and predicts the corresponding states . To update the network parameters, we utilize the L2 loss function, which is expressed as follows in Equation (12):
		(12)
where  and  represent the true and predicted system states, respectively. N represents the number of samples in the dataset. This loss function quantifies the discrepancy between the predicted and actual states, guiding the adjustment of network parameters during the training process.
