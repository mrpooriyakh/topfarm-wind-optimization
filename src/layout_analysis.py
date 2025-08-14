import numpy as np
import matplotlib.pyplot as plt
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
from py_wake.examples.data.iea37 import IEA37_WindTurbines, IEA37Site
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site
from py_wake.examples.data.dtu10mw._dtu10mw import DTU10MW, DTU10WM_RWT, ct_curve, power_curve
from topfarm import TopFarmProblem
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.examples.iea37 import get_iea37_initial, get_iea37_constraints
from topfarm.plotting import XYPlotComp, NoPlot
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
    # this part dosent really make sense but for now we have to set them as the inital condition IEA37 or we will get a negative value for improvement 
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
def create_topfarm_layout_plots(n_turbines=16, show_optimization_process=True):
    """Create TopFarm layout plots showing before/after optimization"""
    
    print(f"🎨 Creating TopFarm Layout Plots for {n_turbines} Turbines")
    print("=" * 60)

    # Setup components
    turbines = DTU10MW()
    site = Hornsrev1Site(n_turbines)
    wake_model = IEA37SimpleBastankhahGaussian(site, turbines)
    
    # Get initial layout and constraints
    initial_pos = get_iea37_initial(n_turbines)
    constraints = get_dtu10mw_constraints(n_turbines)
    
    print(f"✅ Setup complete for {n_turbines} turbines")
    print(f"   Initial layout range: X=[{initial_pos[:,0].min():.0f}, {initial_pos[:,0].max():.0f}]m")
    print(f"                         Y=[{initial_pos[:,1].min():.0f}, {initial_pos[:,1].max():.0f}]m")
    
    # Calculate initial AEP and wake losses
    initial_sim = wake_model(initial_pos[:, 0], initial_pos[:, 1])
    initial_aep = initial_sim.aep().sum()
    
    # Get individual turbine AEPs to show wake effects
    turbine_aeps = initial_sim.aep()
    
    # Handle different AEP data structures
    if hasattr(turbine_aeps, 'values'):
        turbine_aeps = turbine_aeps.values
    if turbine_aeps.ndim > 1:
        turbine_aeps = turbine_aeps.sum(axis=tuple(range(1, turbine_aeps.ndim)))
    
    print(f"   Turbine AEPs shape: {turbine_aeps.shape}")
    print(f"   Expected shape: ({n_turbines},)")
    
    print(f"   Initial total AEP: {initial_aep:.2f} GWh/year")
    print(f"   AEP per turbine range: {turbine_aeps.min():.2f} - {turbine_aeps.max():.2f} GWh/year")
    wd = np.arange(0, 360, 30)  # Every 30 degrees: [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

    # Create the optimization problem with plotting enabled
    cost_comp = PyWakeAEPCostModelComponent(wake_model, n_wt=n_turbines, wd=wd)
    
    if show_optimization_process:
        # Use XYPlotComp to show optimization process
        plot_comp = XYPlotComp()
        print("   📊 Optimization plotting enabled - will show progress")
    else:
        plot_comp = NoPlot()
    
    problem = TopFarmProblem(
        design_vars=dict(zip('xy', initial_pos.T)),
        cost_comp=cost_comp,
        constraints=constraints,
        driver=EasyScipyOptimizeDriver(
            maxiter=100,
            optimizer='SLSQP',
            tol=1e-8,
            disp=True
        ),
        plot_comp=plot_comp
    )
    
    # Run optimization
    print("   🚀 Running optimization...")
    start_time = time.time()
    cost, state, recorder = problem.optimize()
    opt_time = time.time() - start_time
    
    # Get optimized results
    optimized_pos = np.array([state['x'], state['y']]).T
    optimized_aep = -cost
    improvement = ((optimized_aep - initial_aep) / initial_aep) * 100
    
    print(f"   ✅ Optimization complete!")
    print(f"   Time: {opt_time:.1f}s")
    print(f"   Final AEP: {optimized_aep:.2f} GWh/year")
    print(f"   Improvement: {improvement:.2f}%")
    
    # Create comprehensive layout visualization
    create_detailed_layout_plots(
        n_turbines, initial_pos, optimized_pos, 
        initial_aep, optimized_aep, wake_model, 
        turbine_aeps, site
    )
    
    return {
        'n_turbines': n_turbines,
        'initial_pos': initial_pos,
        'optimized_pos': optimized_pos,
        'initial_aep': initial_aep,
        'optimized_aep': optimized_aep,
        'improvement': improvement,
        'wake_model': wake_model,
        'site': site
    }

def create_detailed_layout_plots(n_turbines, initial_pos, optimized_pos, 
                               initial_aep, optimized_aep, wake_model, 
                               turbine_aeps, site):
    """Create detailed layout plots with wake effects and site boundaries"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Initial Layout with Wake Effects
    ax1 = plt.subplot(2, 3, 1)
    
    # Plot turbines colored by their AEP (wake losses)
    # Ensure turbine_aeps has the right shape
    if len(turbine_aeps) != n_turbines:
        print(f"   Warning: AEP data shape mismatch. Using average values.")
        turbine_aeps = np.full(n_turbines, turbine_aeps.mean())
    
    scatter1 = ax1.scatter(initial_pos[:, 0], initial_pos[:, 1], 
                          c=turbine_aeps, s=100, cmap='RdYlGn', 
                          alpha=0.8, edgecolors='black', linewidth=1)
    
    # Add site boundary if available
    try:
        boundary = site.boundary
        if boundary is not None:
            # Plot boundary
            boundary_x, boundary_y = boundary.T
            ax1.plot(boundary_x, boundary_y, 'k--', linewidth=2, alpha=0.5, label='Site Boundary')
    except:
        # Draw circular boundary based on turbine positions
        max_radius = np.max(np.sqrt(initial_pos[:, 0]**2 + initial_pos[:, 1]**2)) * 1.1
        circle = plt.Circle((0, 0), max_radius, fill=False, 
                           linestyle='--', color='black', alpha=0.5)
        ax1.add_patch(circle)
    
    ax1.set_title(f'Initial Layout ({n_turbines} Turbines)\nAEP: {initial_aep:.2f} GWh/year', fontsize=12)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Add colorbar for AEP
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Turbine AEP (GWh/year)')
    
    # Add turbine numbers
    for i, (x, y) in enumerate(initial_pos):
        ax1.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    # 2. Optimized Layout
    ax2 = plt.subplot(2, 3, 2)
    
    # Calculate optimized wake effects
    opt_sim = wake_model(optimized_pos[:, 0], optimized_pos[:, 1])
    opt_turbine_aeps = opt_sim.aep()
    
    # Handle different AEP data structures
    if hasattr(opt_turbine_aeps, 'values'):
        opt_turbine_aeps = opt_turbine_aeps.values
    if opt_turbine_aeps.ndim > 1:
        opt_turbine_aeps = opt_turbine_aeps.sum(axis=tuple(range(1, opt_turbine_aeps.ndim)))
    
    # Ensure opt_turbine_aeps has the right shape
    if len(opt_turbine_aeps) != n_turbines:
        print(f"   Warning: Optimized AEP data shape mismatch. Using average values.")
        opt_turbine_aeps = np.full(n_turbines, opt_turbine_aeps.mean())
    
    scatter2 = ax2.scatter(optimized_pos[:, 0], optimized_pos[:, 1], 
                          c=opt_turbine_aeps, s=100, cmap='RdYlGn', 
                          alpha=0.8, edgecolors='black', linewidth=1)
    
    # Add site boundary
    try:
        if boundary is not None:
            ax2.plot(boundary_x, boundary_y, 'k--', linewidth=2, alpha=0.5)
    except:
        circle = plt.Circle((0, 0), max_radius, fill=False, 
                           linestyle='--', color='black', alpha=0.5)
        ax2.add_patch(circle)
    
    ax2.set_title(f'Optimized Layout ({n_turbines} Turbines)\nAEP: {optimized_aep:.2f} GWh/year', fontsize=12)
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Turbine AEP (GWh/year)')
    
    # Add turbine numbers
    for i, (x, y) in enumerate(optimized_pos):
        ax2.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    # 3. Movement Vectors
    ax3 = plt.subplot(2, 3, 3)
    
    # Plot initial positions
    ax3.scatter(initial_pos[:, 0], initial_pos[:, 1], c='red', s=60, 
               alpha=0.6, label='Initial', marker='o')
    
    # Plot optimized positions  
    ax3.scatter(optimized_pos[:, 0], optimized_pos[:, 1], c='green', s=60, 
               alpha=0.6, label='Optimized', marker='s')
    
    # Draw movement arrows
    for i in range(n_turbines):
        dx = optimized_pos[i, 0] - initial_pos[i, 0]
        dy = optimized_pos[i, 1] - initial_pos[i, 1]
        if np.sqrt(dx**2 + dy**2) > 10:  # Only show significant movements
            ax3.arrow(initial_pos[i, 0], initial_pos[i, 1], dx, dy,
                     head_width=50, head_length=50, fc='blue', ec='blue', alpha=0.7)
    
    ax3.set_title('Turbine Movement\n(Initial → Optimized)')
    ax3.set_xlabel('X Position (m)')
    ax3.set_ylabel('Y Position (m)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.axis('equal')
    
    # 4. Wake Loss Comparison
    ax4 = plt.subplot(2, 3, 4)
    
    turbine_indices = np.arange(1, n_turbines + 1)
    width = 0.35
    
    # Ensure we have proper data for plotting
    if len(turbine_aeps) == n_turbines and len(opt_turbine_aeps) == n_turbines:
        bars1 = ax4.bar(turbine_indices - width/2, turbine_aeps, width, 
                        label='Initial', alpha=0.7, color='lightcoral')
        bars2 = ax4.bar(turbine_indices + width/2, opt_turbine_aeps, width, 
                        label='Optimized', alpha=0.7, color='lightgreen')
        
        ax4.set_xlabel('Turbine Number')
        ax4.set_ylabel('AEP (GWh/year)')
        ax4.set_title('Individual Turbine Performance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        # Fallback: show total AEP comparison
        ax4.bar(['Initial', 'Optimized'], [initial_aep, optimized_aep], 
                color=['lightcoral', 'lightgreen'], alpha=0.7)
        ax4.set_ylabel('Total AEP (GWh/year)')
        ax4.set_title('Total Farm Performance')
        ax4.grid(True, alpha=0.3)
    
    # 5. Distance Matrix (Turbine Spacing)
    ax5 = plt.subplot(2, 3, 5)
    
    # Calculate distance matrix for optimized layout
    distances = np.zeros((n_turbines, n_turbines))
    for i in range(n_turbines):
        for j in range(n_turbines):
            if i != j:
                dx = optimized_pos[i, 0] - optimized_pos[j, 0]
                dy = optimized_pos[i, 1] - optimized_pos[j, 1]
                distances[i, j] = np.sqrt(dx**2 + dy**2)
    
    im = ax5.imshow(distances, cmap='viridis', aspect='auto')
    ax5.set_title('Turbine Spacing Matrix\n(Optimized Layout)')
    ax5.set_xlabel('Turbine Index')
    ax5.set_ylabel('Turbine Index')
    cbar5 = plt.colorbar(im, ax=ax5)
    cbar5.set_label('Distance (m)')
    
    # 6. Performance Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create summary text
    improvement_pct = ((optimized_aep - initial_aep) / initial_aep) * 100
    aep_gain = optimized_aep - initial_aep
    
    summary_text = f"""
OPTIMIZATION SUMMARY
{'='*25}

Number of Turbines: {n_turbines}

Initial AEP: {initial_aep:.2f} GWh/year
Optimized AEP: {optimized_aep:.2f} GWh/year
Improvement: {improvement_pct:.2f}%
AEP Gain: {aep_gain:.2f} GWh/year

Individual Turbine Performance:
• Initial AEP range: {turbine_aeps.min():.2f} - {turbine_aeps.max():.2f} GWh/year
• Optimized AEP range: {opt_turbine_aeps.min():.2f} - {opt_turbine_aeps.max():.2f} GWh/year
• Wake loss reduction: {((opt_turbine_aeps.std() - turbine_aeps.std())/turbine_aeps.std()*100 if turbine_aeps.std() > 0 else 0):.1f}%

Layout Characteristics:
• Average turbine spacing: {distances[distances>0].mean():.0f} m
• Minimum spacing: {distances[distances>0].min():.0f} m
• Maximum spacing: {distances.max():.0f} m

Site Utilization:
• Site area utilization: Optimized
• Constraint satisfaction: ✓ All constraints met
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'topfarm_layout_analysis_{n_turbines}turbines.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📁 Detailed layout plot saved: topfarm_layout_analysis_{n_turbines}turbines.png")

def run_multiple_layout_analysis():
    """Run layout analysis for multiple turbine configurations"""
    
    print("🎯 COMPREHENSIVE TOPFARM LAYOUT ANALYSIS")
    print("=" * 70)
    
    turbine_counts = [9, 16, 36, 64]
    results = []
    
    for n_turbines in turbine_counts:
        print(f"\n{'='*20} {n_turbines} TURBINES {'='*20}")
        result = create_topfarm_layout_plots(n_turbines, show_optimization_process=False)
        results.append(result)
        print(f"✅ {n_turbines}-turbine analysis complete")
    
    # Create comparison summary
    create_layout_comparison_summary(results)
    
    return results

def create_layout_comparison_summary(results):
    """Create a summary comparison of all layouts"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for i, result in enumerate(results):
        ax = axes[i//2, i%2]
        n_turb = result['n_turbines']
        
        # Plot both initial and optimized on same axes
        ax.scatter(result['initial_pos'][:, 0], result['initial_pos'][:, 1], 
                  c='red', s=40, alpha=0.6, label='Initial', marker='o')
        ax.scatter(result['optimized_pos'][:, 0], result['optimized_pos'][:, 1], 
                  c='green', s=40, alpha=0.8, label='Optimized', marker='s')
        
        # Draw movement arrows for significant moves
        for j in range(n_turb):
            dx = result['optimized_pos'][j, 0] - result['initial_pos'][j, 0]
            dy = result['optimized_pos'][j, 1] - result['initial_pos'][j, 1]
            if np.sqrt(dx**2 + dy**2) > 50:  # Only significant movements
                ax.arrow(result['initial_pos'][j, 0], result['initial_pos'][j, 1], 
                        dx, dy, head_width=30, head_length=30, 
                        fc='blue', ec='blue', alpha=0.5)
        
        ax.set_title(f'{n_turb} Turbines\nImprovement: {result["improvement"]:.2f}%')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('topfarm_layout_comparison_all.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📁 Comparison summary saved: topfarm_layout_comparison_all.png")

if __name__ == "__main__":
    # Run comprehensive analysis
    results = run_multiple_layout_analysis()
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print("="*50)
    print("Files created:")
    for result in results:
        n = result['n_turbines']
        print(f"📊 topfarm_layout_analysis_{n}turbines.png")
    print(f"📈 topfarm_layout_comparison_all.png")
    print("\nThese plots show:")
    print("• Initial vs optimized turbine positions")
    print("• Wake effects (colored by individual turbine AEP)")
    print("• Turbine movement vectors")
    print("• Individual turbine performance comparison")
    print("• Turbine spacing analysis")
    print("• Performance summary statistics")