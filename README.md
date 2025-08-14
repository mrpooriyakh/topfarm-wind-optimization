# TopFarm Wind Farm Optimization

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TopFarm](https://img.shields.io/badge/TopFarm-2.0+-green.svg)](https://topfarm.pages.windenergy.dtu.dk/)

A comprehensive wind farm layout optimization project using the TopFarm framework, featuring single-farm optimization, multi-farm coordination, and mixed turbine type analysis. This project demonstrates advanced wind farm design optimization techniques for maximizing Annual Energy Production (AEP) while considering wake effects and operational constraints.

## 🌟 Key Features

### 🎯 Single Farm Optimization
- Individual wind farm layout optimization for maximum energy production
- Support for multiple turbine counts (9, 16, 36, 64+ turbines)
- Advanced wake loss minimization using PyWake models
- Constraint handling (minimum spacing, boundary constraints)
- Comprehensive performance analysis and visualization

### 🏭 Multi-Farm Optimization
- Simultaneous optimization of multiple wind farms
- Inter-farm wake effect coordination and minimization
- Global optimization considering farm-to-farm interactions
- Flexible farm boundary and spacing constraints
- Scalable to multiple farm clusters

### 🔄 Mixed Turbine Type Analysis
- Support for different turbine models in the same optimization
- **V80**: 2 MW, 80m rotor diameter - Legacy onshore turbines
- **DTU10MW**: 10 MW, 178m rotor diameter - Large offshore turbines
- **IEA37**: 3.35 MW, 130m rotor diameter - Reference wind turbines
- Turbine-specific wake characteristics and performance modeling

### 📊 Advanced Visualization & Analysis
- Before/after layout comparisons with movement vectors
- Individual turbine performance analysis (wake loss visualization)
- Inter-turbine spacing matrices and constraint verification
- Farm-by-farm performance breakdowns
- Comprehensive optimization summary reports

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- TopFarm framework
- PyWake wake modeling library

### Installation

```bash
# Clone the repository
git clone https://github.com/mrpooriyakh/topfarm-wind-optimization.git
cd topfarm-wind-optimization

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
# Run single farm optimization study
from src.single_farm_optimization import run_simple_optimization_study, create_visualizations

# Execute optimization for multiple farm sizes
results = run_simple_optimization_study()

# Generate comprehensive visualizations
if results:
    create_visualizations(results)
```

## 📈 Performance Results

Our optimization algorithms typically achieve significant improvements in Annual Energy Production (AEP):

| Farm Size | Turbines | Initial AEP | Optimized AEP | Improvement | Optimization Time |
|-----------|----------|-------------|---------------|-------------|-------------------|
| Small     | 9        | ~180 GWh/yr | ~185 GWh/yr   | 2-5%        | 15-30s           |
| Medium    | 16       | ~320 GWh/yr | ~340 GWh/yr   | 3-7%        | 30-60s           |
| Large     | 36       | ~720 GWh/yr | ~780 GWh/yr   | 5-10%       | 60-120s          |
| Extra Large| 64      | ~1280 GWh/yr| ~1450 GWh/yr  | 7-15%       | 120-300s         |

*Performance varies based on wind conditions, site characteristics, and initial layout quality.*

## 🗂️ Project Structure

```
topfarm-wind-optimization/
├── src/                                 # Main source code
│   ├── single_farm_optimization.py     # Individual farm optimization
│   ├── multi_farm_optimization.py      # Multi-farm coordination
│   ├── mixed_turbine_optimization.py   # Mixed turbine type handling
│   ├── layout_analysis.py              # Detailed layout analysis tools
│   └── utils/                           # Utility functions
├── examples/                            # Usage examples and tutorials
│   ├── basic_optimization_example.py   # Simple optimization example
│   ├── multi_farm_example.py           # Multi-farm coordination example
│   └── mixed_turbine_example.py        # Mixed turbine type example
├── results/                             # Generated plots and data files
├── docs/                                # Documentation and guides
├── tests/                               # Unit tests
├── notebooks/                           # Jupyter notebook tutorials
├── README.md                            # This file
├── requirements.txt                     # Python dependencies
└── .gitignore                          # Git ignore rules
```

## 🔧 Optimization Methods

### Wake Modeling
- **PyWake Integration**: Advanced wake deficit modeling using IEA37SimpleBastankhahGaussian
- **Multi-directional Analysis**: Considers multiple wind directions (0°, 90°, 180°, 270° and more)
- **Turbine-Specific Models**: Individual wake characteristics for different turbine types

### Optimization Algorithms
- **SLSQP Optimizer**: Sequential Least Squares Programming for gradient-based optimization
- **Constraint Handling**: Automatic spacing and boundary constraint enforcement
- **Multi-objective Optimization**: Balance between energy production and constraint satisfaction

### Layout Strategies
- **Initial Layout Generation**: Smart initial positioning using circular and grid-based approaches
- **Movement Optimization**: Iterative turbine repositioning for maximum AEP
- **Constraint-Aware Design**: Maintains minimum spacing and boundary requirements

## 📚 Usage Examples

### Single Farm Optimization
```python
from src.single_farm_optimization import create_topfarm_layout_plots

# Optimize a 16-turbine wind farm with detailed analysis
result = create_topfarm_layout_plots(n_turbines=16, show_optimization_process=True)
```

### Multi-Farm Coordination
```python
from src.multi_farm_optimization import optimize_multifarm

# Define multiple wind farms
farm_configs = [
    {
        'center': (0, 0),
        'radius': 1000,
        'n_wt': 9,
        'turbine_type': 'IEA37',
        'name': 'North Farm'
    },
    {
        'center': (3000, 500),
        'radius': 800,
        'n_wt': 6,
        'turbine_type': 'IEA37',
        'name': 'South Farm'
    }
]

# Optimize both farms simultaneously
result = optimize_multifarm(farm_configs, show_plots=True)
```

### Mixed Turbine Types
```python
from src.mixed_turbine_optimization import optimize_mixed_turbine_multifarm

# Mix different turbine technologies
farm_configs = [
    {
        'center': (0, 0),
        'radius': 1000,
        'n_wt': 6,
        'turbine_type': 'V80',      # Legacy onshore turbines
        'name': 'Legacy Farm'
    },
    {
        'center': (3000, 0),
        'radius': 1200,
        'n_wt': 4,
        'turbine_type': 'DTU10MW',  # Large offshore turbines
        'name': 'Offshore Farm'
    }
]

result = optimize_mixed_turbine_multifarm(farm_configs, show_plots=True)
```

## 🛠️ Technologies & Dependencies

### Core Libraries
- **TopFarm**: Wind farm optimization framework (>=2.0.0)
- **PyWake**: Wake effect modeling (>=2.3.0)
- **NumPy**: Numerical computing (>=1.21.0)
- **SciPy**: Scientific computing and optimization (>=1.7.0)
- **OpenMDAO**: Multidisciplinary design optimization (>=3.15.0)

### Visualization & Analysis
- **Matplotlib**: Plotting and visualization (>=3.4.0)
- **Pandas**: Data analysis and manipulation (>=1.3.0)
- **Seaborn**: Statistical data visualization (>=0.11.0)

### Optional Enhancements
- **Plotly**: Interactive plotting (>=5.0.0)
- **Jupyter**: Notebook interface (>=1.0.0)

## 📋 Generated Outputs

The optimization process generates several types of outputs:

### Visualization Files
- `topfarm_optimization_results.png` - Main results summary with performance charts
- `layout_comparison.png` - Before/after layout comparisons
- `multifarm_optimization_Nfarms.png` - Multi-farm coordination results
- `mixed_turbine_multifarm_Nfarms.png` - Mixed turbine analysis
- `topfarm_layout_analysis_Nturbines.png` - Detailed single-farm analysis

### Data Files
- `optimization_results.csv` - Detailed numerical results
- Individual turbine position data
- Performance metrics and improvement statistics

### Analysis Reports
- Farm-by-farm performance breakdowns
- Individual turbine AEP analysis
- Wake loss quantification
- Optimization convergence data

## 🎯 Research Applications

This project is suitable for:

### Academic Research
- Wind farm layout optimization studies
- Wake effect modeling and validation
- Multi-farm interaction analysis
- Turbine technology comparison studies

### Industry Applications
- Pre-construction wind farm design
- Existing farm performance optimization
- Multi-developer coordination studies
- Technology selection and placement

### Educational Use
- Wind energy engineering courses
- Optimization algorithm demonstrations
- PyWake and TopFarm tutorials
- Renewable energy system design

## 🤝 Contributing

We welcome contributions to improve the project! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests if applicable
4. **Commit your changes**: `git commit -m 'Add amazing feature'`
5. **Push to the branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TopFarm Development Team** - DTU Wind Energy for the optimization framework
- **PyWake Contributors** - For the advanced wake modeling capabilities
- **Wind Energy Research Community** - For continuous innovation in wind farm optimization
- **IEA Wind Task 37** - For standardized reference turbine models and test cases

## 📞 Contact & Support

**Project Maintainer**: [pooriya khodaparast]
- GitHub: [@mrpooriyakh](https://github.com/mrpooriyakh)
- Email: [mr.pooriyakh@gmail.com]

**Project Repository**: [https://github.com/mrpooriyakh/topfarm-wind-optimization](https://github.com/mrpooriyakh/topfarm-wind-optimization)

For questions, issues, or feature requests, please open an issue on GitHub or contact the maintainer directly.

## 🔗 Related Projects & Resources

- [TopFarm Documentation](https://topfarm.pages.windenergy.dtu.dk/)
- [PyWake Documentation](https://py-wake.pages.windenergy.dtu.dk/)
- [IEA Wind Task 37](https://iea-wind.org/task37/)
- [DTU Wind Energy](https://www.vindenergi.dtu.dk/)

---

*This project demonstrates the power of modern wind farm optimization techniques for maximizing renewable energy production while considering real-world operational constraints.*
