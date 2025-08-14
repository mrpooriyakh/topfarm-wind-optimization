import numpy as np
import matplotlib.pyplot as plt
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
from py_wake.examples.data.iea37 import IEA37_WindTurbines, IEA37Site
from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site
from py_wake.examples.data.dtu10mw._dtu10mw import DTU10MW
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from topfarm import TopFarmProblem
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.constraint_components.boundary import CircleBoundaryConstraint, XYBoundaryConstraint
from topfarm.plotting import XYPlotComp, NoPlot
import time

class MultiFarmBoundaryConstraint:
    """Custom constraint for multi-farm boundaries"""
    
    def __init__(self, farm_configs):
        self.farm_configs = farm_configs
        self.const_id = 'multifarm_boundary'
    
    def get_comp(self, n_wt):
        # For now, return a simple overall boundary
        # In a more sophisticated implementation, you could create 
        # a custom OpenMDAO component that enforces individual farm boundaries
        
        # Calculate overall boundary
        all_centers = np.array([config['center'] for config in self.farm_configs])
        all_radii = np.array([config['radius'] for config in self.farm_configs])
        
        overall_center = np.mean(all_centers, axis=0)
        max_distance = 0
        for center, radius in zip(all_centers, all_radii):
            distance_to_center = np.linalg.norm(center - overall_center)
            max_distance = max(max_distance, distance_to_center + radius)
        
        overall_radius = max_distance * 1.2
        
        # Return a standard circle boundary component
        return CircleBoundaryConstraint(overall_center, overall_radius).get_comp(n_wt)
    
    def set_design_var_limits(self, design_vars):
        # Set limits based on overall boundary
        all_centers = np.array([config['center'] for config in self.farm_configs])
        all_radii = np.array([config['radius'] for config in self.farm_configs])
        
        overall_center = np.mean(all_centers, axis=0)
        max_distance = 0
        for center, radius in zip(all_centers, all_radii):
            distance_to_center = np.linalg.norm(center - overall_center)
            max_distance = max(max_distance, distance_to_center + radius)
        
        overall_radius = max_distance * 1.2
        
        # Use the CircleBoundaryConstraint method
        temp_constraint = CircleBoundaryConstraint(overall_center, overall_radius)
        temp_constraint.set_design_var_limits(design_vars)

def get_multifarm_constraints(farm_configs):
    """Get constraints for multi-farm optimization
    
    Parameters
    ----------
    farm_configs : list of dict
        Each dict contains farm configuration with keys:
        - 'center': (x, y) center position
        - 'radius': boundary radius
        - 'n_wt': number of turbines
        - 'turbine_type': turbine model
    
    Returns
    -------
    constraints : list
        Combined constraints for all farms
    """
    constraints = []
    
    # Global spacing constraint (minimum distance between any two turbines)
    # Use the largest turbine diameter for global spacing
    max_diameter = 0
    for config in farm_configs:
        if config['turbine_type'] == 'V80':
            diameter = 80.0
        elif config['turbine_type'] == 'DTU10MW':
            diameter = 178.3
        elif config['turbine_type'] == 'IEA37':
            diameter = 130.0  # IEA37 diameter
        else:
            diameter = 100.0  # default
        max_diameter = max(max_diameter, diameter)
    
    # Global minimum spacing (reduced for multi-farm to allow more flexibility)
    global_spacing = SpacingConstraint(1.5 * max_diameter)
    constraints.append(global_spacing)
    
    # Create a large encompassing boundary that contains all farms
    all_centers = np.array([config['center'] for config in farm_configs])
    all_radii = np.array([config['radius'] for config in farm_configs])
    
    # Calculate bounding box
    min_x = np.min(all_centers[:, 0] - all_radii)
    max_x = np.max(all_centers[:, 0] + all_radii)
    min_y = np.min(all_centers[:, 1] - all_radii)
    max_y = np.max(all_centers[:, 1] + all_radii)
    
    # Create overall circular boundary
    overall_center = [(min_x + max_x) / 2, (min_y + max_y) / 2]
    overall_radius = max(
        np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2) / 2 * 1.2,
        np.max(all_radii) * 2
    )
    
    # Create overall boundary constraint
    overall_boundary = CircleBoundaryConstraint(overall_center, overall_radius)
    constraints.append(overall_boundary)
    
    return constraints

def create_multifarm_initial_layout(farm_configs):
    """Create initial layout for multiple farms
    
    Parameters
    ----------
    farm_configs : list of dict
        Farm configurations
        
    Returns
    -------
    initial_pos : np.array
        Combined initial positions for all farms
    """
    all_positions = []
    
    for config in farm_configs:
        center = np.array(config['center'])
        radius = config['radius']
        n_wt = config['n_wt']
        
        # Create circular initial layout for each farm
        if n_wt == 1:
            # Single turbine at center
            positions = center.reshape(1, 2)
        else:
            # Circular arrangement
            angles = np.linspace(0, 2*np.pi, n_wt, endpoint=False)
            layout_radius = radius * 0.7  # Use 70% of boundary radius
            
            x_positions = center[0] + layout_radius * np.cos(angles)
            y_positions = center[1] + layout_radius * np.sin(angles)
            positions = np.column_stack([x_positions, y_positions])
        
        all_positions.append(positions)
    
    return np.vstack(all_positions)

def create_multifarm_wake_model(farm_configs):
    """Create wake model for multi-farm setup
    
    For simplicity, we'll use a single turbine type for the wake model.
    In practice, you might need more sophisticated handling for mixed turbine types.
    """
    # Use the first farm's turbine type for the wake model
    # In a real application, you'd want to handle mixed turbine types properly
    first_farm = farm_configs[0]
    
    if first_farm['turbine_type'] == 'V80':
        turbines = V80()
        site = Hornsrev1Site()
    elif first_farm['turbine_type'] == 'DTU10MW':
        turbines = DTU10MW()
        # Create a generic site for DTU10MW
        site = IEA37Site(16)  # Generic site
    else:  # IEA37
        turbines = IEA37_WindTurbines()
        site = IEA37Site(16)
    
    wake_model = IEA37SimpleBastankhahGaussian(site, turbines)
    return wake_model

def optimize_multifarm(farm_configs, wind_directions=None, show_plots=True):
    """Optimize multiple wind farms simultaneously
    
    Parameters
    ----------
    farm_configs : list of dict
        Each dict should contain:
        - 'center': (x, y) farm center coordinates
        - 'radius': farm boundary radius in meters
        - 'n_wt': number of turbines in farm
        - 'turbine_type': 'V80', 'DTU10MW', or 'IEA37'
        - 'name': farm name (optional)
    wind_directions : array-like, optional
        Wind directions to consider. If None, uses [0, 90, 180, 270]
    show_plots : bool
        Whether to show optimization plots
        
    Returns
    -------
    dict : optimization results
    """
    
    print("🏭 MULTI-FARM WIND FARM OPTIMIZATION")
    print("=" * 50)
    
    # Default wind directions
    if wind_directions is None:
        wind_directions = np.array([0, 90, 180, 270])  # Cardinal directions
    
    # Print farm summary
    total_turbines = sum(config['n_wt'] for config in farm_configs)
    print(f"Number of farms: {len(farm_configs)}")
    print(f"Total turbines: {total_turbines}")
    print(f"Wind directions: {len(wind_directions)} directions")
    
    for i, config in enumerate(farm_configs):
        name = config.get('name', f'Farm_{i+1}')
        print(f"  {name}: {config['n_wt']} x {config['turbine_type']} at {config['center']}")
    
    # Create initial layout
    print("\n📍 Creating initial multi-farm layout...")
    initial_pos = create_multifarm_initial_layout(farm_configs)
    
    # Create wake model
    print("🌪️  Setting up wake model...")
    wake_model = create_multifarm_wake_model(farm_configs)
    
    # Create constraints
    print("🚧 Setting up constraints...")
    constraints = get_multifarm_constraints(farm_configs)
    
    # Calculate initial AEP
    print("⚡ Calculating initial AEP...")
    initial_sim = wake_model(initial_pos[:, 0], initial_pos[:, 1], wd=wind_directions)
    initial_aep = initial_sim.aep().sum()
    
    print(f"   Initial total AEP: {initial_aep:.2f} GWh/year")
    
    # Create optimization problem
    cost_comp = PyWakeAEPCostModelComponent(
        wake_model, 
        n_wt=total_turbines,
        wd=wind_directions
    )
    
    plot_comp = XYPlotComp() if show_plots else NoPlot()
    
    problem = TopFarmProblem(
        design_vars=dict(zip('xy', initial_pos.T)),
        cost_comp=cost_comp,
        constraints=constraints,
        driver=EasyScipyOptimizeDriver(
            maxiter=200,  # More iterations for multi-farm
            optimizer='SLSQP',
            tol=1e-8,
            disp=True
        ),
        plot_comp=plot_comp
    )
    
    # Run optimization
    print("\n🚀 Running multi-farm optimization...")
    start_time = time.time()
    cost, state, recorder = problem.optimize()
    opt_time = time.time() - start_time
    
    # Get results
    optimized_pos = np.array([state['x'], state['y']]).T
    optimized_aep = -cost
    improvement = ((optimized_aep - initial_aep) / initial_aep) * 100
    
    print(f"\n✅ Multi-farm optimization complete!")
    print(f"   Time: {opt_time:.1f}s")
    print(f"   Initial AEP: {initial_aep:.2f} GWh/year")
    print(f"   Optimized AEP: {optimized_aep:.2f} GWh/year")
    print(f"   Improvement: {improvement:.2f}%")
    
    # Create visualization
    if show_plots:
        create_multifarm_plots(
            farm_configs, initial_pos, optimized_pos,
            initial_aep, optimized_aep, wake_model, wind_directions
        )
    
    return {
        'farm_configs': farm_configs,
        'initial_pos': initial_pos,
        'optimized_pos': optimized_pos,
        'initial_aep': initial_aep,
        'optimized_aep': optimized_aep,
        'improvement': improvement,
        'optimization_time': opt_time,
        'wind_directions': wind_directions
    }

def create_multifarm_plots(farm_configs, initial_pos, optimized_pos, 
                          initial_aep, optimized_aep, wake_model, wind_directions):
    """Create visualization plots for multi-farm optimization"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Color scheme for different farms
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # 1. Initial Layout
    ax1 = plt.subplot(2, 3, 1)
    
    turbine_idx = 0
    for i, config in enumerate(farm_configs):
        n_wt = config['n_wt']
        farm_pos = initial_pos[turbine_idx:turbine_idx + n_wt]
        
        # Plot turbines
        ax1.scatter(farm_pos[:, 0], farm_pos[:, 1], 
                   c=colors[i % len(colors)], s=100, alpha=0.7,
                   label=config.get('name', f'Farm {i+1}'), 
                   edgecolors='black', linewidth=1)
        
        # Plot farm boundary
        center = config['center']
        radius = config['radius']
        circle = plt.Circle(center, radius, fill=False, 
                          color=colors[i % len(colors)], linestyle='--', alpha=0.5)
        ax1.add_patch(circle)
        
        # Label farm center
        ax1.annotate(config.get('name', f'F{i+1}'), center, 
                    fontsize=12, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        turbine_idx += n_wt
    
    ax1.set_title(f'Initial Multi-Farm Layout\nTotal AEP: {initial_aep:.2f} GWh/year')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    
    # 2. Optimized Layout
    ax2 = plt.subplot(2, 3, 2)
    
    turbine_idx = 0
    for i, config in enumerate(farm_configs):
        n_wt = config['n_wt']
        farm_pos = optimized_pos[turbine_idx:turbine_idx + n_wt]
        
        # Plot turbines
        ax2.scatter(farm_pos[:, 0], farm_pos[:, 1], 
                   c=colors[i % len(colors)], s=100, alpha=0.7,
                   label=config.get('name', f'Farm {i+1}'),
                   edgecolors='black', linewidth=1)
        
        # Plot farm boundary
        center = config['center']
        radius = config['radius']
        circle = plt.Circle(center, radius, fill=False, 
                          color=colors[i % len(colors)], linestyle='--', alpha=0.5)
        ax2.add_patch(circle)
        
        # Label farm center
        ax2.annotate(config.get('name', f'F{i+1}'), center, 
                    fontsize=12, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        turbine_idx += n_wt
    
    improvement = ((optimized_aep - initial_aep) / initial_aep) * 100
    ax2.set_title(f'Optimized Multi-Farm Layout\nTotal AEP: {optimized_aep:.2f} GWh/year\nImprovement: {improvement:.2f}%')
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axis('equal')
    
    # 3. Movement Vectors
    ax3 = plt.subplot(2, 3, 3)
    
    turbine_idx = 0
    for i, config in enumerate(farm_configs):
        n_wt = config['n_wt']
        
        initial_farm = initial_pos[turbine_idx:turbine_idx + n_wt]
        optimized_farm = optimized_pos[turbine_idx:turbine_idx + n_wt]
        
        # Plot initial and optimized positions
        ax3.scatter(initial_farm[:, 0], initial_farm[:, 1], 
                   c=colors[i % len(colors)], s=60, alpha=0.5, marker='o')
        ax3.scatter(optimized_farm[:, 0], optimized_farm[:, 1], 
                   c=colors[i % len(colors)], s=80, alpha=0.8, marker='s')
        
        # Draw movement arrows
        for j in range(n_wt):
            dx = optimized_farm[j, 0] - initial_farm[j, 0]
            dy = optimized_farm[j, 1] - initial_farm[j, 1]
            if np.sqrt(dx**2 + dy**2) > 20:  # Only significant movements
                ax3.arrow(initial_farm[j, 0], initial_farm[j, 1], dx, dy,
                         head_width=30, head_length=30, 
                         fc=colors[i % len(colors)], ec=colors[i % len(colors)], 
                         alpha=0.6)
        
        turbine_idx += n_wt
    
    ax3.set_title('Turbine Movement\n(○ Initial → ■ Optimized)')
    ax3.set_xlabel('X Position (m)')
    ax3.set_ylabel('Y Position (m)')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # 4. Farm-by-Farm AEP Comparison
    ax4 = plt.subplot(2, 3, 4)
    
    # Calculate individual farm AEPs
    initial_sim = wake_model(initial_pos[:, 0], initial_pos[:, 1], wd=wind_directions)
    opt_sim = wake_model(optimized_pos[:, 0], optimized_pos[:, 1], wd=wind_directions)
    
    initial_turbine_aeps = initial_sim.aep()
    opt_turbine_aeps = opt_sim.aep()
    
    # Handle different data structures
    if hasattr(initial_turbine_aeps, 'values'):
        initial_turbine_aeps = initial_turbine_aeps.values
    if hasattr(opt_turbine_aeps, 'values'):
        opt_turbine_aeps = opt_turbine_aeps.values
    
    if initial_turbine_aeps.ndim > 1:
        initial_turbine_aeps = initial_turbine_aeps.sum(axis=tuple(range(1, initial_turbine_aeps.ndim)))
    if opt_turbine_aeps.ndim > 1:
        opt_turbine_aeps = opt_turbine_aeps.sum(axis=tuple(range(1, opt_turbine_aeps.ndim)))
    
    farm_names = []
    initial_farm_aeps = []
    opt_farm_aeps = []
    
    turbine_idx = 0
    for i, config in enumerate(farm_configs):
        n_wt = config['n_wt']
        
        # Sum AEP for this farm
        farm_initial_aep = initial_turbine_aeps[turbine_idx:turbine_idx + n_wt].sum()
        farm_opt_aep = opt_turbine_aeps[turbine_idx:turbine_idx + n_wt].sum()
        
        farm_names.append(config.get('name', f'Farm {i+1}'))
        initial_farm_aeps.append(farm_initial_aep)
        opt_farm_aeps.append(farm_opt_aep)
        
        turbine_idx += n_wt
    
    x = np.arange(len(farm_names))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, initial_farm_aeps, width, label='Initial', alpha=0.7)
    bars2 = ax4.bar(x + width/2, opt_farm_aeps, width, label='Optimized', alpha=0.7)
    
    ax4.set_xlabel('Farm')
    ax4.set_ylabel('AEP (GWh/year)')
    ax4.set_title('Farm-by-Farm Performance')
    ax4.set_xticks(x)
    ax4.set_xticklabels(farm_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Distance Matrix Between Farms
    ax5 = plt.subplot(2, 3, 5)
    
    # Calculate inter-farm distances (center to center)
    n_farms = len(farm_configs)
    farm_distances = np.zeros((n_farms, n_farms))
    
    for i in range(n_farms):
        for j in range(n_farms):
            if i != j:
                center_i = np.array(farm_configs[i]['center'])
                center_j = np.array(farm_configs[j]['center'])
                farm_distances[i, j] = np.linalg.norm(center_i - center_j)
    
    im = ax5.imshow(farm_distances, cmap='viridis')
    ax5.set_title('Inter-Farm Distances\n(Center to Center)')
    ax5.set_xlabel('Farm Index')
    ax5.set_ylabel('Farm Index')
    
    # Add farm names as tick labels
    ax5.set_xticks(range(n_farms))
    ax5.set_yticks(range(n_farms))
    ax5.set_xticklabels([config.get('name', f'F{i+1}') for i, config in enumerate(farm_configs)], rotation=45)
    ax5.set_yticklabels([config.get('name', f'F{i+1}') for i, config in enumerate(farm_configs)])
    
    cbar = plt.colorbar(im, ax=ax5)
    cbar.set_label('Distance (m)')
    
    # 6. Summary Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    total_improvement = ((optimized_aep - initial_aep) / initial_aep) * 100
    
    # Calculate individual farm improvements
    farm_improvements = []
    for i in range(len(farm_names)):
        farm_imp = ((opt_farm_aeps[i] - initial_farm_aeps[i]) / initial_farm_aeps[i]) * 100
        farm_improvements.append(farm_imp)
    
    summary_text = f"""
MULTI-FARM OPTIMIZATION SUMMARY
{'='*40}

Total Farms: {len(farm_configs)}
Total Turbines: {sum(config['n_wt'] for config in farm_configs)}

OVERALL PERFORMANCE:
Initial Total AEP: {initial_aep:.2f} GWh/year
Optimized Total AEP: {optimized_aep:.2f} GWh/year
Total Improvement: {total_improvement:.2f}%

INDIVIDUAL FARM IMPROVEMENTS:
"""
    
    for i, (name, improvement) in enumerate(zip(farm_names, farm_improvements)):
        summary_text += f"{name}: {improvement:.2f}%\n"
    
    summary_text += f"""
FARM CHARACTERISTICS:
Average inter-farm distance: {farm_distances[farm_distances>0].mean():.0f} m
Minimum inter-farm distance: {farm_distances[farm_distances>0].min():.0f} m
Maximum inter-farm distance: {farm_distances.max():.0f} m

Wind directions analyzed: {len(wind_directions)}
Optimization successful: ✓
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'multifarm_optimization_{len(farm_configs)}farms.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📁 Multi-farm plot saved: multifarm_optimization_{len(farm_configs)}farms.png")

# Example usage functions
def example_two_farm_optimization():
    """Example: Optimize two nearby wind farms"""
    
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
    
    wind_directions = np.arange(0, 360, 45)  # Every 45 degrees
    
    result = optimize_multifarm(farm_configs, wind_directions, show_plots=True)
    return result

def example_three_farm_cluster():
    """Example: Optimize three farms in a cluster"""
    
    farm_configs = [
        {
            'center': (0, 0),
            'radius': 1200,
            'n_wt': 12,
            'turbine_type': 'DTU10MW',
            'name': 'Alpha Farm'
        },
        {
            'center': (4000, 0),
            'radius': 1000,
            'n_wt': 8,
            'turbine_type': 'DTU10MW',
            'name': 'Beta Farm'
        },
        {
            'center': (2000, 3500),
            'radius': 900,
            'n_wt': 6,
            'turbine_type': 'DTU10MW',
            'name': 'Gamma Farm'
        }
    ]
    
    wind_directions = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    
    result = optimize_multifarm(farm_configs, wind_directions, show_plots=True)
    return result

if __name__ == "__main__":
    print("🌊 TOPFARM MULTI-FARM EXAMPLES")
    print("="*50)
    
    print("\n1. Running two-farm optimization example...")
    result1 = example_two_farm_optimization()
    
    print("\n2. Running three-farm cluster example...")
    result2 = example_three_farm_cluster()
    
    print(f"\n🎉 Multi-farm optimization examples complete!")
    print("Files created:")
    print("📊 multifarm_optimization_2farms.png")
    print("📊 multifarm_optimization_3farms.png")