# SOARN
Self Organising Associative Learning Recurrent Network 


A deep learning model that integrates convolutional neural networks with classical associative learning theory to study visual stimulus representation. This implementation demonstrates how complex visual representations form through associative mechanisms.

🎯 Overview
SOARN (Self-Organizing Associative Recurrent Network) is a computational model that:

Integrates CNN-based visual processing with associative learning mechanisms
Processes actual visual stimuli (256×256 RGB images) rather than abstract representations
Reproduces classical learning phenomena (blocking, overshadowing, negative patterning)
Extracts and visualizes learned receptive fields showing how associations modify perceptual representations

This model bridges the gap between traditional associative learning theories and modern deep learning approaches, providing insights into how perceptual and associative processes interact during learning.

📋 Table of Contents

Features
Architecture
Installation
Quick Start
Usage Examples
Experimental Phenomena
Model Components
Results
Citation
License

✨ Features

Biologically-Inspired Architecture: 5-layer CNN mimicking hierarchical visual processing
Real-Time Learning: Processes visual stimuli in real-time with temporal dynamics
Elemental Framework: 12,544 elements per stimulus enabling complex representations
Dynamic Learning: Activity-dependent learning rate based on DDA framework
Receptive Field Visualization: Novel capability to visualize learned representations
Classical Phenomena: Reproduces acquisition, extinction, blocking, conditioned inhibition, and negative patterning


Python 3.8 or higher
NumPy, Matplotlib, Jupyter
8GB RAM minimum
GPU optional but recommended

Step 1: Clone the repository
bashgit clone https://github.com/ESSYCHAN/SOARN.git
cd SOARN

Step 2: Set up virtual environment
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Step 3: Install dependencies
bashpip install -r requirements.txt


# Quick Start

Basic Acquisition Experiment
pythonfrom SOARN import Network, Group
import matplotlib.pyplot as plt

# Initialize network
net = Network()

# Create experimental group
group = Group(name="acquisition", phases=1, model=net)
group.add_phase("100A+")  # 100 trials of A paired with outcome

# Run experiment
results = net.run(group)

# Plot learning curve
plt.plot(results['V_values'])
plt.xlabel('Trials')
plt.ylabel('Associative Strength (V)')
plt.title('Acquisition Learning Curve')
plt.show()

# Visualize receptive fields
net.visualize_receptive_fields('A'

🔧 Model Components
Core Classes

Network: Main engine coordinating all computations
Group: Manages experimental groups and phases
Element: Computational unit (12,544 per stimulus)
ConvLayer: CNN layer implementation
Phase: Handles trial sequences and timing


# 📝 Citation

If you use SOARN in your research, please cite:
bibtex@phdthesis{mulwa2024soarn, 
  title={Self Organising Associative Representation Learning Model}, 
  author={Esther Mulwa}, 
  year={2025},
  school={[City st Georges University]}
}


@article{mondragon2017associative,
  title={Associative Learning Should Go Deep},
  author={Mondragón, Esther and Alonso, Eduardo and Kokkola, Niklas},
  journal={Trends in Cognitive Sciences},
  volume={21},
  number={11},
  pages={822--825},
  year={2017},
  publisher={Elsevier}
}


# 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.


# 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

# 🐛 Known Issues

Performance scales with image size (optimized for 256×256)
Memory intensive for >3 stimuli experiments
Requires ~30-45 minutes for 200-trial experiment

# 📧 Contact
For questions or collaborations:

GitHub Issues: https://github.com/ESSYCHAN/SOARN/issues
Email: [Esther.Mulwa@city.ac.uk]

# 

Based on the DDA framework (Kokkola et al., 2019)



Note: This is research software. While functional, it may require adjustments for specific use cases.
