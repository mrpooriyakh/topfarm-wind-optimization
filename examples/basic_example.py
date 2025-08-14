#!/usr/bin/env python3
"""
Basic Wind Farm Optimization Example

This example demonstrates how to run a simple wind farm optimization
using the TopFarm framework.
"""

import sys
import os

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from single_farm_optimization import run_simple_optimization_study, create_visualizations

def main():
    """Run a basic optimization example"""
    
    print("🌟 Basic TopFarm Optimization Example")
    print("=" * 50)
    
    # Run optimization study
    results = run_simple_optimization_study()
    
    if results:
        # Create visualizations
        create_visualizations(results)
        
        print("\n✅ Example completed successfully!")
        print("Check the results/ folder for generated plots and data.")
    else:
        print("❌ Optimization failed!")

if __name__ == "__main__":
    main()
