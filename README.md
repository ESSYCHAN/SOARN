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
🏃 Quick Start
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
net.visualize_receptive_fields('A')
📚 Usage Examples
1. Blocking Experiment
python# Create blocking design
blocking_group = Group(name="blocking", phases=2, model=net)
blocking_group.add_phase("50A+")    # Phase 1: A predicts outcome
blocking_group.add_phase("50AB+")   # Phase 2: AB predicts outcome

control_group = Group(name="control", phases=2, model=net)
control_group.add_phase("50C+")     # Phase 1: C predicts outcome  
control_group.add_phase("50AB+")    # Phase 2: AB predicts outcome

# Run and compare
blocking_results = net.run(blocking_group)
control_results = net.run(control_group)

🧪 Experimental Phenomena
The model successfully reproduces five fundamental associative learning phenomena:
PhenomenonDesignKey FindingAcquisition100A+Negatively accelerated learning curve, V→0.93Extinction100A+, 100A-Incomplete extinction, V→0.22Blocking50A+, 50AB+B acquires minimal strength, V→0.11Conditioned Inhibition200A+/200AX-X becomes inhibitor, V→-0.08Negative Patterning200A+/200B+/200AB-Non-linear discrimination, VAB→0.05

🔧 Model Components
Core Classes

Network: Main engine coordinating all computations
Group: Manages experimental groups and phases
Element: Computational unit (12,544 per stimulus)
ConvLayer: CNN layer implementation
Phase: Handles trial sequences and timing

Key Parameters
ParameterSymbolDefaultDescriptionLearning rateα0.1Weight update rateSalienceS0.5Stimulus salienceTemporal spreadδ1.0Activation temporal widthAsymptote weightν0.5Predictor/outcome balanceMax activationA_max1.0Activation ceiling
📊 Results
Learning Curves
The model produces characteristic learning patterns matching empirical data:

Acquisition: Negatively accelerated curve
Extinction: Gradual decrease, incomplete
Blocking: Reduced learning for blocked stimulus

Receptive Field Visualizations
Novel capability showing how associations modify perceptual representations:

Before learning: Clear geometric patterns matching input
After learning: Integration of outcome features into CS representation
Extinction: Degraded, noisy representations
Negative patterning: Unique compound representation

📝 Citation
If you use SOARN in your research, please cite:
bibtex@phdthesis{chan2024soarn,
  title={A Deep Learning Approach to Visual Associative Learning},
  author={Esther Chan},
  year={2024},
  school={[City st Georgens University]}
}
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
🐛 Known Issues

Performance scales with image size (optimized for 256×256)
Memory intensive for >3 stimuli experiments
Requires ~30-45 minutes for 200-trial experiment

📧 Contact
For questions or collaborations:

GitHub Issues: https://github.com/ESSYCHAN/SOARN/issues
Email: [Esther.Mulwa@city.ac.uk]

🙏 Acknowledgments

Based on the DDA framework (Kokkola et al., 2019)
Inspired by Rescorla-Wagner model and CNN architectures


Note: This is research software. While functional, it may require adjustments for specific use cases.
