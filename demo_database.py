"""
Demo script showing the ProgramDatabase interface as specified in the requirements.

This demonstrates the exact interface pattern:
parent_program, inspirations = database.sample()
prompt = prompt_sampler.build(parent_program, inspirations)
diff = llm.generate(prompt)
child_program = apply_diff(parent_program, diff)
results = evaluator.execute(child_program)
database.add(child_program, results)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evosim.core.database import ProgramDatabase, create_database


def demo_basic_usage():
    """Demonstrate basic database usage."""
    print("=== ProgramDatabase Demo ===\n")
    
    # Create database with different sampling strategies
    database = create_database("fitness", temperature=1.5)
    
    # Add some initial APL programs (simulating baseline and initial population)
    initial_apls = [
        {
            "apl": "actions.precombat=flask\nactions=berserking,if=buff.recklessness.up",
            "dps": 45000.0,
            "metadata": {"sim_time": 300, "iterations": 1000}
        },
        {
            "apl": "actions.precombat=flask,food\nactions=recklessness\nactions+=/rampage,if=rage>40",
            "dps": 47500.0,
            "metadata": {"sim_time": 300, "iterations": 1000}
        },
        {
            "apl": "actions.precombat=flask\nactions=recklessness,if=cooldown.rampage.ready",
            "dps": 46200.0,
            "metadata": {"sim_time": 300, "iterations": 1000}
        }
    ]
    
    # Add initial programs to database
    for i, apl_data in enumerate(initial_apls):
        program = database.add(
            apl_code=apl_data["apl"],
            evaluation_results={"dps": apl_data["dps"], **apl_data["metadata"]}
        )
        print(f"Added initial program {i+1}: {program.program_id} (DPS: {program.dps_score})")
    
    print(f"\nDatabase stats: {database.get_statistics()}")
    
    # Demonstrate the exact interface pattern from requirements
    print("\n=== Evolutionary Loop Simulation ===")
    
    for generation in range(3):
        print(f"\n--- Generation {generation + 1} ---")
        
        # Sample parent and inspirations (exact interface as required)
        parent_program, inspirations = database.sample()
        
        print(f"Selected parent: {parent_program.program_id} (DPS: {parent_program.dps_score})")
        print(f"Inspirations: {[f'{p.program_id} (DPS: {p.dps_score})' for p in inspirations]}")
        
        # Simulate the rest of the evolutionary loop
        # In real usage, these would be:
        # prompt = prompt_sampler.build(parent_program, inspirations)
        # diff = llm.generate(prompt)
        # child_program = apply_diff(parent_program, diff)
        # results = evaluator.execute(child_program)
        
        # For demo, simulate an improved child program
        simulated_child_apl = parent_program.apl_code + f"\n# Generated modification {generation+1}"
        simulated_dps_improvement = parent_program.dps_score * (1.02 + generation * 0.01)  # Slight improvement
        simulated_results = {
            "dps": simulated_dps_improvement,
            "sim_time": 300,
            "iterations": 1000,
            "generation": generation + 1
        }
        
        # Add child program to database (exact interface as required)  
        child_program = database.add(
            apl_code=simulated_child_apl,
            evaluation_results=simulated_results,
            parent_id=parent_program.program_id,
            diff_applied=f"# Simulated diff for generation {generation + 1}"
        )
        
        print(f"Created child: {child_program.program_id} (DPS: {child_program.dps_score:.1f})")
        
        database.advance_generation()
    
    # Show final statistics
    print(f"\n=== Final Results ===")
    print(f"Database stats: {database.get_statistics()}")
    
    best_programs = database.get_best_programs(3)
    print(f"\nTop 3 programs:")
    for i, program in enumerate(best_programs, 1):
        print(f"{i}. {program.program_id}: {program.dps_score:.1f} DPS (Gen {program.generation})")


def demo_sampling_strategies():
    """Demonstrate different sampling strategies."""
    print("\n\n=== Sampling Strategies Demo ===\n")
    
    # Create databases with different strategies
    strategies = {
        "fitness": create_database("fitness", temperature=2.0),
        "tournament": create_database("tournament", tournament_size=3),
        "diversity": create_database("diversity", diversity_weight=0.4)
    }
    
    # Add same test data to all databases
    test_data = [
        ("old_high_performer", 50000.0, 0),
        ("recent_medium", 48000.0, 5),
        ("old_medium", 47000.0, 1),
        ("recent_low", 45000.0, 4),
        ("very_recent", 46000.0, 6)
    ]
    
    for strategy_name, db in strategies.items():
        print(f"\n--- {strategy_name.title()} Strategy ---")
        
        for name, dps, gen in test_data:
            db.current_generation = gen
            db.add(f"actions=test_{name}", {"dps": dps})
            
        # Sample multiple times to show strategy behavior
        selections = []
        for _ in range(5):
            parent, inspirations = db.sample(num_inspirations=1)
            selections.append((parent.dps_score, parent.generation))
        
        print(f"Parent selections (DPS, Generation): {selections}")


if __name__ == "__main__":
    demo_basic_usage()
    demo_sampling_strategies()
