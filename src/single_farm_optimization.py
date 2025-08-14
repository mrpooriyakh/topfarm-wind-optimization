import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
from py_wake.examples.data.iea37 import IEA37_WindTurbines, IEA37Site
from py_wake.examples.data.hornsrev1 import V80 , Hornsrev1Site
from py_wake.examples.data.dtu10mw._dtu10mw import DTU10MW, DTU10WM_RWT, ct_curve, power_curve
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from topfarm import TopFarmProblem
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.examples.iea37 import get_iea37_initial, get_iea37_constraints
from topfarm.plotting import NoPlot
import time
def get_v80_constraints(n_wt=9):
    """Constraints for V80 wind farms

    Parameters
    ----------
    n_wt : int, optional
        Number of wind turbines in farm

    Returns
    -------
    constr : list of topfarm constraints
        Spacing constraint and boundary constraint for V80 model
    """
    from topfarm.constraint_components.spacing import SpacingConstraint
    from topfarm.constraint_components.boundary import CircleBoundaryConstraint
    import numpy as np
    
    # V80 rotor diameter is 80m
    diam = 80.0
    
    # Minimum spacing: 2 rotor diameters (160m)
    spac_constr = SpacingConstraint(2 * diam)
    
    # Boundary radius based on number of turbines
    # Simple scaling: more turbines need larger area
    if n_wt <= 9:
        bound_rad = 900
    elif n_wt <= 16:
        bound_rad = 1300
    elif n_wt <= 36:
        bound_rad = 2000
    else:
        bound_rad = 3000
    
    bound_constr = CircleBoundaryConstraint((0, 0), bound_rad)
    
    return [spac_constr, bound_constr]
def get_dtu10mw_constraints(n_wt=9):
    """Constraints for DTU10MW wind farms

    Parameters
    ----------
    n_wt : int, optional
        Number of wind turbines in farm

    Returns
    -------
    constr : list of topfarm constraints
        Spacing constraint and boundary constraint for DTU10MW model
    """
    from topfarm.constraint_components.spacing import SpacingConstraint
    from topfarm.constraint_components.boundary import CircleBoundaryConstraint
    import numpy as np
    
    # DTU10MW rotor diameter is 178.3m
    diam = 178.3
    
    # Minimum spacing: 2 rotor diameters (356.6m)
    spac_constr = SpacingConstraint(2 * diam)
    
    # Boundary radius based on number of turbines
    # Larger turbines need more space
    if n_wt <= 9:
        bound_rad = 900
    elif n_wt <= 16:
        bound_rad = 1300
    elif n_wt <= 36:
        bound_rad = 2000
    else:
        bound_rad = 3000
    
    bound_constr = CircleBoundaryConstraint((0, 0), bound_rad)
    
    return [spac_constr, bound_constr]
def run_simple_optimization_study():
    """Run a focused optimization study and create visualizations"""
    
    print("🚀 TopFarm Optimization Study")
    print("=" * 50)
    
    # Setup basic components
    turbines =  DTU10MW()
    
    # Test different numbers of turbines
    n_turbines_list = [9, 16, 36, 64]
    results = []
    
    for n_turbines in n_turbines_list:
        print(f"\n--- Optimizing {n_turbines} turbines ---")
        
        try:
            # Setup site and wake model
            site = Hornsrev1Site(n_turbines)
            wake_model = IEA37SimpleBastankhahGaussian(site, turbines)
            
            # Get initial layout
            initial_pos = get_iea37_initial(n_turbines)
            constraints = get_dtu10mw_constraints(n_turbines)
            
            # Calculate initial AEP
            initial_sim = wake_model(initial_pos[:, 0], initial_pos[:, 1])
            initial_aep = initial_sim.aep().sum()
            wd = np.arange(0, 360, 30)  # Every 30 degrees: [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

            # Create optimization problem
            cost_comp = PyWakeAEPCostModelComponent(wake_model, n_wt=n_turbines,wd=wd)
            
            problem = TopFarmProblem(
                design_vars=dict(zip('xy', initial_pos.T)),
                cost_comp=cost_comp,
                constraints=constraints,
                driver=EasyScipyOptimizeDriver(
                    maxiter=100,
                    optimizer='SLSQP',
                    tol=1e-8,
                    disp=True
                ),  # 20 iterations for good results
                plot_comp=NoPlot()
            )
            
            # Run optimization
            start_time = time.time()
            cost, state, recorder = problem.optimize()
            opt_time = time.time() - start_time
            
            # Get optimized positions
            optimized_pos = np.array([state['x'], state['y']]).T
            
            # Calculate optimized AEP
            optimized_aep = -cost
            improvement = ((optimized_aep - initial_aep) / initial_aep) * 100
            
            # Store results
            result = {
                'n_turbines': n_turbines,
                'initial_aep': initial_aep,
                'optimized_aep': optimized_aep,
                'improvement_pct': improvement,
                'optimization_time': opt_time,
                'initial_positions': initial_pos,
                'optimized_positions': optimized_pos,
                'wake_model': wake_model
            }
            results.append(result)
            
            print(f"  ✅ Initial: {initial_aep:.1f} GWh/year")
            print(f"  ✅ Optimized: {optimized_aep:.1f} GWh/year") 
            print(f"  ✅ Improvement: {improvement:.2f}%")
            print(f"  ⏱️ Time: {opt_time:.1f}s")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    return results

def create_visualizations(results):
    """Create comprehensive visualizations for the project report"""
    
    if not results:
        print("No results to visualize!")
        return
    
    print("\n📊 Creating Visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # Create a comprehensive figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Extract data for plotting
    n_turbines = [r['n_turbines'] for r in results]
    initial_aep = [r['initial_aep'] for r in results]
    optimized_aep = [r['optimized_aep'] for r in results]
    improvements = [r['improvement_pct'] for r in results]
    opt_times = [r['optimization_time'] for r in results]
    
    # 1. AEP Comparison Chart
    ax1 = plt.subplot(2, 3, 1)
    x_pos = np.arange(len(n_turbines))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, initial_aep, width, label='Initial AEP', alpha=0.7, color='lightblue')
    bars2 = ax1.bar(x_pos + width/2, optimized_aep, width, label='Optimized AEP', alpha=0.7, color='darkblue')
    
    ax1.set_xlabel('Number of Turbines')
    ax1.set_ylabel('Annual Energy Production (GWh/year)')
    ax1.set_title('AEP Comparison: Initial vs Optimized')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(n_turbines)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{height:.0f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{height:.0f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Improvement Percentage Chart
    ax2 = plt.subplot(2, 3, 2)
    bars = ax2.bar(n_turbines, improvements, color='green', alpha=0.7)
    ax2.set_xlabel('Number of Turbines')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Optimization Improvement by Layout Size')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, improvement in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{improvement:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # 3. Optimization Time
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(n_turbines, opt_times, 'ro-', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Turbines')
    ax3.set_ylabel('Optimization Time (seconds)')
    ax3.set_title('Computational Complexity')
    ax3.grid(True, alpha=0.3)
    
    # 4. Layout Visualization (Before/After for largest case)
    if len(results) > 0:
        # Find the case with most turbines for detailed layout view
        largest_case = max(results, key=lambda x: x['n_turbines'])
        
        ax4 = plt.subplot(2, 3, 4)
        initial_pos = largest_case['initial_positions']
        ax4.scatter(initial_pos[:, 0], initial_pos[:, 1], c='lightcoral', s=50, alpha=0.7, label='Initial Layout')
        ax4.set_xlabel('X Position (m)')
        ax4.set_ylabel('Y Position (m)')
        ax4.set_title(f'Initial Layout ({largest_case["n_turbines"]} Turbines)')
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
        ax4.legend()
        
        ax5 = plt.subplot(2, 3, 5)
        optimized_pos = largest_case['optimized_positions']
        ax5.scatter(optimized_pos[:, 0], optimized_pos[:, 1], c='darkgreen', s=50, alpha=0.7, label='Optimized Layout')
        ax5.set_xlabel('X Position (m)')
        ax5.set_ylabel('Y Position (m)')
        ax5.set_title(f'Optimized Layout ({largest_case["n_turbines"]} Turbines)')
        ax5.grid(True, alpha=0.3)
        ax5.axis('equal')
        ax5.legend()
    
    # 5. Summary Statistics Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create summary table
    table_data = []
    for i, result in enumerate(results):
        table_data.append([
            f"{result['n_turbines']}",
            f"{result['initial_aep']:.1f}",
            f"{result['optimized_aep']:.1f}",
            f"{result['improvement_pct']:.2f}%",
            f"{result['optimization_time']:.1f}s"
        ])
    
    columns = ['Turbines', 'Initial AEP\n(GWh/year)', 'Optimized AEP\n(GWh/year)', 'Improvement', 'Time']
    
    table = ax6.table(cellText=table_data, colLabels=columns, 
                     cellLoc='center', loc='center',
                     colWidths=[0.15, 0.25, 0.25, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax6.set_title('Results Summary', pad=20)
    
    plt.tight_layout()
    plt.savefig('topfarm_optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a detailed wind farm layout comparison
    create_layout_comparison_plot(results)
    
    print("✅ Visualizations created!")
    print("📁 Saved: topfarm_optimization_results.png")
    print("📁 Saved: layout_comparison.png")

def create_layout_comparison_plot(results):
    """Create a detailed layout comparison plot"""
    
    fig, axes = plt.subplots(2, len(results), figsize=(4*len(results), 8))
    if len(results) == 1:
        axes = axes.reshape(2, 1)
    
    for i, result in enumerate(results):
        n_turb = result['n_turbines']
        initial_pos = result['initial_positions']
        optimized_pos = result['optimized_positions']
        
        # Initial layout
        ax_init = axes[0, i] if len(results) > 1 else axes[0]
        ax_init.scatter(initial_pos[:, 0], initial_pos[:, 1], c='red', s=30, alpha=0.7)
        ax_init.set_title(f'Initial Layout\n{n_turb} Turbines')
        ax_init.grid(True, alpha=0.3)
        ax_init.axis('equal')
        
        # Optimized layout
        ax_opt = axes[1, i] if len(results) > 1 else axes[1]
        ax_opt.scatter(optimized_pos[:, 0], optimized_pos[:, 1], c='green', s=30, alpha=0.7)
        ax_opt.set_title(f'Optimized Layout\n{result["improvement_pct"]:.2f}% Improvement')
        ax_opt.grid(True, alpha=0.3)
        ax_opt.axis('equal')
        
        # Set same scale for comparison
        all_x = np.concatenate([initial_pos[:, 0], optimized_pos[:, 0]])
        all_y = np.concatenate([initial_pos[:, 1], optimized_pos[:, 1]])
        margin = 200
        xlim = [all_x.min() - margin, all_x.max() + margin]
        ylim = [all_y.min() - margin, all_y.max() + margin]
        
        ax_init.set_xlim(xlim)
        ax_init.set_ylim(ylim)
        ax_opt.set_xlim(xlim)
        ax_opt.set_ylim(ylim)
    
    plt.tight_layout()
    plt.savefig('layout_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results_to_csv(results):
    """Save results to CSV for further analysis"""
    
    data = []
    for result in results:
        data.append({
            'n_turbines': result['n_turbines'],
            'initial_aep_gwh': result['initial_aep'],
            'optimized_aep_gwh': result['optimized_aep'],
            'improvement_percent': result['improvement_pct'],
            'optimization_time_seconds': result['optimization_time'],
            'aep_gain_gwh': result['optimized_aep'] - result['initial_aep']
        })
    
    df = pd.DataFrame(data)
    df.to_csv('optimization_results.csv', index=False)
    print(f"📊 Results saved to: optimization_results.csv")
    
    # Print summary
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"   Average improvement: {df['improvement_percent'].mean():.2f}%")
    print(f"   Best improvement: {df['improvement_percent'].max():.2f}%")
    print(f"   Total AEP gain: {df['aep_gain_gwh'].sum():.1f} GWh/year")
    
    return df

def main():
    """Main function to run the complete analysis"""
    
    print("🎯 SIMPLE TOPFARM ANALYSIS")
    print("=" * 50)
    print("This will:")
    print("1. Run optimizations for 9, 16, 36, 64 turbines")
    print("2. Create comprehensive visualizations") 
    print("3. Save results to CSV")
    print("4. Generate plots for your project report")
    print()
    
    # Run optimizations
    results = run_simple_optimization_study()
    
    if results:
        # Create visualizations
        create_visualizations(results)
        
        # Save results
        df = save_results_to_csv(results)
        
        print("\n🎉 ANALYSIS COMPLETE!")
        print("=" * 50)
        print("Files created:")
        print("📊 optimization_results.csv - Detailed results data")
        print("📈 topfarm_optimization_results.png - Main results charts")
        print("🗺️ layout_comparison.png - Layout before/after comparison")
        print()
        print("Use these files for your project report!")
        
        return results, df
    else:
        print("❌ No successful optimizations!")
        return None, None

if __name__ == "__main__":
    results, df = main()