# Hyperbolic-Neural-Networks-with-Poincare-embeddings
Exploring Adaptive Curvature Techniques on Hyperbolic Embeddings for Hierarchical Text Classification

Link to paper: https://ieeexplore.ieee.org/abstract/document/10799279

Non-Euclidean geometries in machine learning models are becoming a common tool to use nowadays. This has caused more researchers to explore this domain and the many possibilities it opens up. The focus for this repository is on hyperbolic spaces which have capabilities to represent hierarchical and complex data structures. Specifically, it probes into the realm of hyperbolic machine learning models, specifically focusing on adaptive curvature hyperbolic neural networks (HNNs) using Poincare´ embeddings.

For this task, four datasets were utilized to assess our proposed methods: a real-world dataset from Kaggle and three synthetic datasets. The real-world dataset, called Animal Taxonomy and the three synthetic datasets are: Reuters5, a five-leveled categorical dataset, and Reuters7, a seven-leveled categorical dataset, and a heterogenous dataset. Poincare Embeddings learn hierarchical representations of symbolic data by embedding them into hyperbolic space– or more precisely into an n-dimensional Poincare ball (shown below) t-SNE visualisation of poincare embeddings image

Now, coming to the working, I used the poincare embeddings as data for the Hyperbolic Neural Networks, which were consturcted using Mobius arithmetic operations. Adaptive curvature and cross entropy loss functions were used to enhance the HNN. The details of this can be found in the paper linked above.

<img width="684" height="590" alt="image" src="https://github.com/user-attachments/assets/e3629b0d-dd21-4ff6-95cc-d923aa1b2b01" />

The results for this are as follows:

1) Taxonomy dataset:
   
   <img width="838" height="1008" alt="image" src="https://github.com/user-attachments/assets/45d75353-c60a-45fe-9a0f-2c1db64262f2" />

   Hierarchical Precision: 0.9448

   Hierarchical Recall: 0.4997

   Hierarchical F1 Score: 0.6537

   Predicted Hierarchy for 'Canis Lupus': Canis Hierarchy for 'Xenopus laevis': ['Xenopus laevis', 'Xenopus', 'Pipidae', 'Anura', 'Amphibia', 'Chordata', 'Animalia']

   Average LCA Metrics: {'P_LCA': 0.998263888888889, 'R_LCA': 0.998263888888889, 'F_LCA': 0.998263888888889}

   Average Path-Based Loss: 0.0625

   Average Level-Based Loss: 0.0396

   Weighted Total Error (WTIE): 1.0208

2) Heterogenous dataset:

   <img width="833" height="714" alt="image" src="https://github.com/user-attachments/assets/234a514c-3941-4e30-b7dd-7356c1b7166a" />

   Hierarchical Precision: 0.9900, Hierarchical Recall: 0.9999, Hierarchical F-Score: 0.9949

   Inferred category for "Convolutional": Convolutional Hierarchy for "Convolutional": ['Technology', 'AI', 'Deep Learning', 'Neural Networks', 'Convolutional']

   Parent Classes for 'Convolutional': {'Level_1': 'Technology', 'Level_2': 'AI', 'Level_3': 'Deep Learning', 'Level_4': 'Neural Networks', 'Level_5': 'Convolutional'}

   Average LCA Metrics: {'P_LCA': 0.9779999999999996, 'R_LCA': 0.5, 'F_LCA': 0.9960000000000001}

   Average Path-Based Loss: 0.8000

   Average Level-Based Loss: 0.3670 Weighted Total Error (WTIE): 1.1500

3) Reuters7 dataset:

   <img width="844" height="714" alt="image" src="https://github.com/user-attachments/assets/61ca087e-1686-41c0-a3a7-708258393c9a" />

   Hierarchical Precision: 0.8300

   Hierarchical Recall: 0.4970

   Hierarchical F1 Score: 0.6217

   Predicted Hierarchy for 'Credit Analysis': Credit Analysis Hierarchy for 'Credit Analysis': ['Business', 'Finance', 'Banking', 'Retail Banking', 'Business Accounts', 'Account Management', 'Customer Service']

   Parent Classes for 'Customer Service': {'Level1': 'Business', 'Level2': 'Finance', 'Level3': 'Banking', 'Level4': 'Retail Banking', 'Level5': 'Business Accounts', 'Level6': 'Account Management', 'Level7': 'Customer Service'}

   Average LCA Metrics: {'P_LCA': 0.9191049913941479, 'R_LCA': 0.9191049913941479, 'F_LCA': 0.9191049913941479}

   Average Path-Based Loss: 1.3494

   Average Level-Based Loss: 1.9639

   Weighted Total Error (WTIE): 2.6000

4) Reuters5 dataset:

   <img width="844" height="714" alt="image" src="https://github.com/user-attachments/assets/783b4f28-11e9-4899-a307-4bb7b04f6fcb" />

   Hierarchical Precision: 0.9375

   Hierarchical Recall: 0.9375

   Hierarchical F1 Score: 0.9375

   Predicted hierarchical category for Business Accounts: Business Accounts ['Business', 'Finance', 'Banking', 'Retail Banking', 'Personal Accounts']

   Parent Classes for 'Android': {'Level_1': 'Technology', 'Level_2': 'Hardware', 'Level_3': 'Devices', 'Level_4': 'Smartphones', 'Level_5': 'Android'} Average LCA Metrics: {'P_LCA': 0.8600000000000001, 'R_LCA': 0.9600000000000001, 'F_LCA': 0.9699999999999999}

   Average Path-Based Loss: 2.2000

   Average Level-Based Loss: 0.6730

   Weighted Total Error (WTIE): 1.6200
