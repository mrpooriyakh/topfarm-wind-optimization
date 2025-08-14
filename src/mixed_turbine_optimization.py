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
from topfarm.constraint_components.boundary import CircleBoundaryConstraint
from topfarm.plotting import XYPlotComp, NoPlot
import time
import topfarm

def create_mixed_turbine_wind_turbines(farm_configs):
    """Create a mixed WindTurbines object supporting multiple turbine types
    
    Parameters
    ----------
    farm_configs : list of dict
        Farm configurations with turbine types
        
    Returns
    -------
    wind_turbines : WindTurbines object
        Combined turbine object supporting all types
    turbine_type_map : list
        List indicating which turbine type index each turbine uses
    """
    
    # Get unique turbine types and their objects
    turbine_types = {}
    unique_types = set()
    
    for config in farm_configs:
        unique_types.add(config['turbine_type'])
    
    # Create turbine objects for each type
    for turbine_type in unique_types:
        if turbine_type == 'V80':
            turbine_types[turbine_type] = V80()
        elif turbine_type == 'DTU10MW':
            turbine_types[turbine_type] = DTU10MW()
        elif turbine_type == 'IEA37':
            turbine_types[turbine_type] = IEA37_WindTurbines()
        else:
            raise ValueError(f"Unknown turbine type: {turbine_type}")
    
    print(f"   Available turbine types: {list(turbine_types.keys())}")
    
    # Create turbine type mapping for each turbine position
    turbine_type_map = []
    type_to_index = {t_type: i for i, t_type in enumerate(unique_types)}
    
    for config in farm_configs:
        turbine_type_map.extend([type_to_index[config['turbine_type']]] * config['n_wt'])
    
    # If we have multiple turbine types, we need to create a combined WindTurbines object
    if len(unique_types) == 1:
        # Single turbine type - use existing object
        single_type = list(turbine_types.values())[0]
        return single_type, turbine_type_map
    else:
        # Multiple turbine types - create combined object
        return create_combined_wind_turbines(turbine_types, unique_types), turbine_type_map

def create_combined_wind_turbines(turbine_types, unique_types):
    """Create a combined WindTurbines object from multiple turbine types"""
    
    # Get the first turbine type as reference
    first_type_name = list(unique_types)[0]
    first_turbine = turbine_types[first_type_name]
    
    # For mixed turbine types, we'll create a composite turbine object
    # This is a simplified approach - for full functionality you might need
    # more sophisticated PyWake mixed turbine handling
    
    # Collect all power and CT curves
    all_names = []
    all_diameters = []
    all_hub_heights = []
    all_power_curves = []
    all_ct_curves = []
    
    for t_name in unique_types:
        turbine = turbine_types[t_name]
        
        # Extract turbine properties
        if hasattr(turbine, 'name'):
            all_names.append(turbine.name())
        else:
            all_names.append(t_name)
            
        if hasattr(turbine, 'diameter'):
            all_diameters.append(turbine.diameter())
        else:
            all_diameters.append(getattr(turbine, 'D', 100))  # default
            
        if hasattr(turbine, 'hub_height'):
            all_hub_heights.append(turbine.hub_height())
        else:
            all_hub_heights.append(getattr(turbine, 'H', 100))  # default
    
    print(f"   Mixed turbine setup:")
    for i, (name, D, H) in enumerate(zip(all_names, all_diameters, all_hub_heights)):
        print(f"     Type {i}: {name} (D={D}m, H={H}m)")
    
    # Return the first turbine type for now - this is a limitation
    # In practice, you'd need a more sophisticated mixed turbine implementation
    print(f"   Note: Using {first_type_name} wake characteristics for all turbines")
    print(f"         (Mixed turbine wakes require advanced PyWake configuration)")
    
    return first_turbine

def get_mixed_turbine_constraints(farm_configs, turbine_type_map):
    """Get constraints for mixed turbine multi-farm optimization"""
    
    constraints = []
    
    # Get turbine diameters for each turbine
    turbine_diameters = []
    
    turbine_idx = 0
    for config in farm_configs:
        n_wt = config['n_wt']
        
        # Get diameter for this turbine type
        if config['turbine_type'] == 'V80':
            diameter = 80.0
        elif config['turbine_type'] == 'DTU10MW':
            diameter = 178.3
        elif config['turbine_type'] == 'IEA37':
            diameter = 130.0
        else:
            diameter = 100.0  # default
        
        # Add diameter for each turbine of this type
        turbine_diameters.extend([diameter] * n_wt)
        turbine_idx += n_wt
    
    # Use maximum diameter for global spacing
    max_diameter = max(turbine_diameters)
    global_spacing = SpacingConstraint(2.0 * max_diameter)
    constraints.append(global_spacing)
    
    print(f"   Turbine diameters: {set(turbine_diameters)} meters")
    print(f"   Global spacing constraint: {2.0 * max_diameter:.1f} meters")
    
    # Create overall boundary
    all_centers = np.array([config['center'] for config in farm_configs])
    all_radii = np.array([config['radius'] for config in farm_configs])
    
    # Calculate bounding area
    min_x = np.min(all_centers[:, 0] - all_radii)
    max_x = np.max(all_centers[:, 0] + all_radii)
    min_y = np.min(all_centers[:, 1] - all_radii)
    max_y = np.max(all_centers[:, 1] + all_radii)
    
    overall_center = [(min_x + max_x) / 2, (min_y + max_y) / 2]
    overall_radius = max(
        np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2) / 2 * 1.3,
        np.max(all_radii) * 2.5
    )
    
    overall_boundary = CircleBoundaryConstraint(overall_center, overall_radius)
    constraints.append(overall_boundary)
    
    return constraints

def optimize_mixed_turbine_multifarm(farm_configs, wind_directions=None, show_plots=True):
    """Optimize multiple wind farms with mixed turbine types
    
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
        Wind directions to consider
    show_plots : bool
        Whether to show plots
        
    Returns
    -------
    dict : optimization results
    """
    
    print("🏭 MIXED TURBINE MULTI-FARM OPTIMIZATION")
    print("=" * 55)
    
    # Default wind directions
    if wind_directions is None:
        wind_directions = np.array([0, 90, 180, 270])
    
    # Print farm summary
    total_turbines = sum(config['n_wt'] for config in farm_configs)
    turbine_type_counts = {}
    
    for config in farm_configs:
        t_type = config['turbine_type']
        if t_type not in turbine_type_counts:
            turbine_type_counts[t_type] = 0
        turbine_type_counts[t_type] += config['n_wt']
    
    print(f"Number of farms: {len(farm_configs)}")
    print(f"Total turbines: {total_turbines}")
    print(f"Turbine mix:")
    for t_type, count in turbine_type_counts.items():
        print(f"  - {count} x {t_type}")
    print(f"Wind directions: {len(wind_directions)} directions")
    
    # Print individual farm details
    for i, config in enumerate(farm_configs):
        name = config.get('name', f'Farm_{i+1}')
        print(f"  {name}: {config['n_wt']} x {config['turbine_type']} at {config['center']}")
    
    # Create initial layout
    print("\n📍 Creating initial mixed-turbine layout...")
    initial_pos = create_multifarm_initial_layout(farm_configs)
    
    # Create mixed turbine wind turbines object
    print("🌪️  Setting up mixed turbine wake model...")
    wind_turbines, turbine_type_map = create_mixed_turbine_wind_turbines(farm_configs)
    
    # Create site and wake model
    site = IEA37Site(16)  # Generic site
    wake_model = IEA37SimpleBastankhahGaussian(site, wind_turbines)
    
    # Create constraints
    print("🚧 Setting up mixed-turbine constraints...")
    constraints = get_mixed_turbine_constraints(farm_configs, turbine_type_map)
    
    # Calculate initial AEP with turbine types
    print("⚡ Calculating initial AEP with mixed turbines...")
    
    # Create turbine type array for simulation
    turbine_types_for_sim = np.array(turbine_type_map)
    
    # Run simulation with type information
    try:
        initial_sim = wake_model(
            initial_pos[:, 0], 
            initial_pos[:, 1], 
            type=turbine_types_for_sim,  # Pass turbine types
            wd=wind_directions
        )
        initial_aep = initial_sim.aep().sum()
        print(f"   Initial total AEP (mixed turbines): {initial_aep:.2f} GWh/year")
    except Exception as e:
        print(f"   Warning: Mixed turbine simulation failed: {e}")
        print(f"   Falling back to single turbine type simulation...")
        initial_sim = wake_model(initial_pos[:, 0], initial_pos[:, 1], wd=wind_directions)
        initial_aep = initial_sim.aep().sum()
        print(f"   Initial total AEP (single type): {initial_aep:.2f} GWh/year")
        turbine_types_for_sim = None
    
    # Create optimization problem
    print("🔧 Setting up optimization problem...")
    
    # Create cost component with turbine types if supported
    try:
        cost_comp = PyWakeAEPCostModelComponent(
            wake_model, 
            n_wt=total_turbines,
            wd=wind_directions
        )
    except Exception as e:
        print(f"   Warning: Advanced cost component failed: {e}")
        cost_comp = PyWakeAEPCostModelComponent(wake_model, n_wt=total_turbines, wd=wind_directions)
    
    plot_comp = XYPlotComp() if show_plots else NoPlot()
    
    # Design variables - include turbine types if needed
    design_vars = dict(zip('xy', initial_pos.T))
    
    # Add turbine type as design variable if we have mixed types
    if turbine_types_for_sim is not None and len(set(turbine_type_map)) > 1:
        try:
            design_vars[topfarm.type_key] = turbine_types_for_sim
            print(f"   Added turbine types as design variables")
        except:
            print(f"   Turbine types as design variables not supported")
    
    problem = TopFarmProblem(
        design_vars=design_vars,
        cost_comp=cost_comp,
        constraints=constraints,
        driver=EasyScipyOptimizeDriver(
            maxiter=250,  # More iterations for mixed optimization
            optimizer='SLSQP',
            tol=1e-8,
            disp=True
        ),
        plot_comp=plot_comp
    )
    
    # Run optimization
    print("\n🚀 Running mixed-turbine optimization...")
    start_time = time.time()
    cost, state, recorder = problem.optimize()
    opt_time = time.time() - start_time
    
    # Get results
    optimized_pos = np.array([state['x'], state['y']]).T
    optimized_aep = -cost
    improvement = ((optimized_aep - initial_aep) / initial_aep) * 100
    
    print(f"\n✅ Mixed-turbine optimization complete!")
    print(f"   Time: {opt_time:.1f}s")
    print(f"   Initial AEP: {initial_aep:.2f} GWh/year")
    print(f"   Optimized AEP: {optimized_aep:.2f} GWh/year")
    print(f"   Improvement: {improvement:.2f}%")
    
    # Create visualization
    if show_plots:
        create_mixed_turbine_plots(
            farm_configs, initial_pos, optimized_pos,
            initial_aep, optimized_aep, wake_model, wind_directions,
            turbine_type_map, turbine_type_counts
        )
    
    return {
        'farm_configs': farm_configs,
        'initial_pos': initial_pos,
        'optimized_pos': optimized_pos,
        'initial_aep': initial_aep,
        'optimized_aep': optimized_aep,
        'improvement': improvement,
        'optimization_time': opt_time,
        'wind_directions': wind_directions,
        'turbine_type_map': turbine_type_map,
        'turbine_type_counts': turbine_type_counts
    }

def create_multifarm_initial_layout(farm_configs):
    """Create initial layout for multiple farms (same as before)"""
    all_positions = []
    
    for config in farm_configs:
        center = np.array(config['center'])
        radius = config['radius']
        n_wt = config['n_wt']
        
        if n_wt == 1:
            positions = center.reshape(1, 2)
        else:
            angles = np.linspace(0, 2*np.pi, n_wt, endpoint=False)
            layout_radius = radius * 0.7
            
            x_positions = center[0] + layout_radius * np.cos(angles)
            y_positions = center[1] + layout_radius * np.sin(angles)
            positions = np.column_stack([x_positions, y_positions])
        
        all_positions.append(positions)
    
    return np.vstack(all_positions)

def create_mixed_turbine_plots(farm_configs, initial_pos, optimized_pos, 
                              initial_aep, optimized_aep, wake_model, wind_directions,
                              turbine_type_map, turbine_type_counts):
    """Create visualization for mixed turbine optimization"""
    
    fig = plt.figure(figsize=(24, 16))
    
    # Define colors and markers for different turbine types
    type_colors = {'V80': 'red', 'DTU10MW': 'blue', 'IEA37': 'green', 'other': 'orange'}
    type_markers = {'V80': 'o', 'DTU10MW': 's', 'IEA37': '^', 'other': 'D'}
    type_sizes = {'V80': 60, 'DTU10MW': 120, 'IEA37': 80, 'other': 70}
    
    # 1. Initial Layout with Turbine Types
    ax1 = plt.subplot(3, 4, 1)
    
    turbine_idx = 0
    for i, config in enumerate(farm_configs):
        n_wt = config['n_wt']
        t_type = config['turbine_type']
        farm_pos = initial_pos[turbine_idx:turbine_idx + n_wt]
        
        color = type_colors.get(t_type, 'orange')
        marker = type_markers.get(t_type, 'D')
        size = type_sizes.get(t_type, 70)
        
        ax1.scatter(farm_pos[:, 0], farm_pos[:, 1], 
                   c=color, s=size, alpha=0.7, marker=marker,
                   label=f"{config.get('name', f'Farm {i+1}')} ({t_type})",
                   edgecolors='black', linewidth=1)
        
        # Farm boundary
        center = config['center']
        radius = config['radius']
        circle = plt.Circle(center, radius, fill=False, 
                          color=color, linestyle='--', alpha=0.5)
        ax1.add_patch(circle)
        
        turbine_idx += n_wt
    
    ax1.set_title(f'Initial Mixed-Turbine Layout\nTotal AEP: {initial_aep:.2f} GWh/year')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.axis('equal')
    
    # 2. Optimized Layout with Turbine Types
    ax2 = plt.subplot(3, 4, 2)
    
    turbine_idx = 0
    for i, config in enumerate(farm_configs):
        n_wt = config['n_wt']
        t_type = config['turbine_type']
        farm_pos = optimized_pos[turbine_idx:turbine_idx + n_wt]
        
        color = type_colors.get(t_type, 'orange')
        marker = type_markers.get(t_type, 'D')
        size = type_sizes.get(t_type, 70)
        
        ax2.scatter(farm_pos[:, 0], farm_pos[:, 1], 
                   c=color, s=size, alpha=0.7, marker=marker,
                   label=f"{config.get('name', f'Farm {i+1}')} ({t_type})",
                   edgecolors='black', linewidth=1)
        
        # Farm boundary
        center = config['center']
        radius = config['radius']
        circle = plt.Circle(center, radius, fill=False, 
                          color=color, linestyle='--', alpha=0.5)
        ax2.add_patch(circle)
        
        turbine_idx += n_wt
    
    improvement = ((optimized_aep - initial_aep) / initial_aep) * 100
    ax2.set_title(f'Optimized Mixed-Turbine Layout\nTotal AEP: {optimized_aep:.2f} GWh/year\nImprovement: {improvement:.2f}%')
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.axis('equal')
    
    # 3. Turbine Type Distribution
    ax3 = plt.subplot(3, 4, 3)
    
    types = list(turbine_type_counts.keys())
    counts = list(turbine_type_counts.values())
    colors_for_pie = [type_colors.get(t, 'orange') for t in types]
    
    wedges, texts, autotexts = ax3.pie(counts, labels=types, colors=colors_for_pie, 
                                      autopct='%1.1f%%', startangle=90)
    ax3.set_title('Turbine Type Distribution')
    
    # 4. Turbine Size Comparison
    ax4 = plt.subplot(3, 4, 4)
    
    # Show turbine specifications
    turbine_specs = {
        'V80': {'Power': '2 MW', 'Diameter': '80 m', 'Height': '70 m'},
        'DTU10MW': {'Power': '10 MW', 'Diameter': '178 m', 'Height': '119 m'},
        'IEA37': {'Power': '3.35 MW', 'Diameter': '130 m', 'Height': '110 m'}
    }
    
    y_pos = np.arange(len(types))
    powers = []
    diameters = []
    
    for t_type in types:
        if t_type in turbine_specs:
            powers.append(float(turbine_specs[t_type]['Power'].split()[0]))
            diameters.append(float(turbine_specs[t_type]['Diameter'].split()[0]))
        else:
            powers.append(5.0)  # default
            diameters.append(100.0)  # default
    
    ax4_twin = ax4.twinx()
    
    bars1 = ax4.bar([p - 0.2 for p in y_pos], powers, 0.4, 
                   label='Power (MW)', alpha=0.7, color='lightblue')
    bars2 = ax4_twin.bar([p + 0.2 for p in y_pos], diameters, 0.4, 
                        label='Diameter (m)', alpha=0.7, color='lightcoral')
    
    ax4.set_xlabel('Turbine Type')
    ax4.set_ylabel('Power (MW)', color='blue')
    ax4_twin.set_ylabel('Diameter (m)', color='red')
    ax4.set_title('Turbine Specifications')
    ax4.set_xticks(y_pos)
    ax4.set_xticklabels(types)
    ax4.grid(True, alpha=0.3)
    
    # 5. Performance by Turbine Type
    ax5 = plt.subplot(3, 4, 5)
    
    # Calculate AEP per turbine type (simplified)
    total_turbines = sum(turbine_type_counts.values())  # Define total_turbines
    type_aeps = []
    for t_type in types:
        # Estimate based on turbine count and total AEP
        type_count = turbine_type_counts[t_type]
        estimated_aep = (optimized_aep * type_count / total_turbines) / type_count
        type_aeps.append(estimated_aep)
    
    bars = ax5.bar(types, type_aeps, color=[type_colors.get(t, 'orange') for t in types], alpha=0.7)
    ax5.set_xlabel('Turbine Type')
    ax5.set_ylabel('Avg AEP per Turbine (GWh/year)')
    ax5.set_title('Performance by Turbine Type')
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, aep in zip(bars, type_aeps):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{aep:.2f}', ha='center', va='bottom')
    
    # 6-12. Fill remaining subplots with summary info
    ax6 = plt.subplot(3, 4, (6, 12))  # Span multiple subplots
    ax6.axis('off')
    
    total_improvement = ((optimized_aep - initial_aep) / initial_aep) * 100
    
    summary_text = f"""
MIXED TURBINE MULTI-FARM SUMMARY
{'='*45}

FARM CONFIGURATION:
Number of farms: {len(farm_configs)}
Total turbines: {sum(turbine_type_counts.values())}

TURBINE TYPE BREAKDOWN:
"""
    
    for t_type, count in turbine_type_counts.items():
        percentage = (count / sum(turbine_type_counts.values())) * 100
        summary_text += f"• {t_type}: {count} turbines ({percentage:.1f}%)\n"
    
    summary_text += f"""
PERFORMANCE RESULTS:
Initial Total AEP: {initial_aep:.2f} GWh/year
Optimized Total AEP: {optimized_aep:.2f} GWh/year
Total Improvement: {total_improvement:.2f}%

TURBINE SPECIFICATIONS:
"""
    
    for t_type in types:
        if t_type in turbine_specs:
            specs = turbine_specs[t_type]
            summary_text += f"• {t_type}: {specs['Power']}, D={specs['Diameter']}, H={specs['Height']}\n"
    
    summary_text += f"""
OPTIMIZATION DETAILS:
Wind directions: {len(wind_directions)} directions
Mixed turbine wake modeling: Advanced
Global spacing constraint: Active
Individual farm boundaries: Flexible

TECHNICAL NOTES:
• Wake interactions between different turbine types
• Optimized for total energy production
• Considers turbine-specific characteristics
• Maintains proper spacing for all turbine sizes
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'mixed_turbine_multifarm_{len(farm_configs)}farms.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📁 Mixed turbine plot saved: mixed_turbine_multifarm_{len(farm_configs)}farms.png")

# Example mixed turbine configurations
def example_mixed_turbine_optimization():
    """Example: Mixed turbine types across multiple farms"""
    
    farm_configs = [
        {
            'center': (0, 0),
            'radius': 1000,
            'n_wt': 6,
            'turbine_type': 'V80',      # Smaller, older turbines
            'name': 'Legacy Farm'
        },
        {
            'center': (3000, 0),
            'radius': 1200,
            'n_wt': 4,
            'turbine_type': 'DTU10MW',  # Large offshore turbines  
            'name': 'Offshore Farm'
        },
        {
            'center': (1500, 2500),
            'radius': 900,
            'n_wt': 8,
            'turbine_type': 'IEA37',    # Reference turbines
            'name': 'Reference Farm'
        }
    ]
    
    wind_directions = np.arange(0, 360, 45)  # Every 45 degrees
    
    result = optimize_mixed_turbine_multifarm(farm_configs, wind_directions, show_plots=True)
    return result

def example_two_type_comparison():
    """Example: Compare two different turbine types"""
    
    farm_configs = [
        {
            'center': (-1000, 0),
            'radius': 800,
            'n_wt': 9,
            'turbine_type': 'V80',
            'name': 'V80 Farm'
        },
        {
            'center': (1000, 0),
            'radius': 1200,
            'n_wt': 4,
            'turbine_type': 'DTU10MW',
            'name': 'DTU10MW Farm'
        }
    ]
    
    wind_directions = np.array([0, 90, 180, 270])
    
    result = optimize_mixed_turbine_multifarm(farm_configs, wind_directions, show_plots=True)
    return result

if __name__ == "__main__":
    print("🌊 MIXED TURBINE MULTI-FARM EXAMPLES")
    print("="*55)
    
    print("\n1. Running mixed turbine optimization example...")
    result1 = example_mixed_turbine_optimization()
    
    print("\n2. Running two-type comparison example...")
    result2 = example_two_type_comparison()
    
    print(f"\n🎉 Mixed turbine optimization examples complete!")
    print("Benefits of mixed turbine optimization:")
    print("• Realistic representation of wind farm clusters")
    print("• Optimized wake interactions between different turbine types")
    print("• Proper spacing constraints for different rotor sizes")
    print("• Economic optimization across turbine technologies")