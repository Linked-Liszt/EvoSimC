# Project Proposal: EvoSimC - LLM-Driven Evolution of SimulationCraft APLs

Potential Subtitles:

How I Learned to Stop Worrying and Love the LLM-Generated APL

My LLM Parses Higher Than Your Main

Hallucinating Our Way to Higher Parses

It Just Works (Until SimC Segfaults)

## 1. Introduction & Motivation

This project, "EvoSimAPL," aims to construct a Python-based system for the automated evolution and optimization of SimulationCraft (SimC) Action Priority Lists (APLs) using Large Language Models (LLMs). Inspired by Google DeepMind's AlphaEvolve, which successfully applied LLM-driven evolutionary algorithms to complex coding and scientific discovery tasks, EvoSimAPL will adapt these principles to the domain of WoW combat optimization. SimC APLs, which define character combat logic through a programmatic, priority-ordered structure, are analogous to the code evolved by AlphaEvolve. By leveraging LLMs to propose targeted modifications (diffs) to APLs and using SimC's DPS output (via a Dockerized environment) as a fitness function, this project seeks to discover novel, high-performing APLs.

## 2. Project Objectives

1. To implement an evolutionary computation framework where an LLM generates modifications (diffs) to SimC APLs.
2. To develop a robust system for applying these LLM-generated diffs to parent APLs to create new candidate APLs.
3. To utilize a Dockerized SimC instance for consistent and reliable evaluation of APLs, with DPS as the primary fitness metric.
4. To establish a "Program Database" for storing and managing a diverse population of APLs, their scores, and evolutionary history, drawing inspiration from AlphaEvolve's evolutionary database concepts.
5. To demonstrate iterative improvement of APLs, leading to higher DPS for a chosen class/specialization.

## 3. Methodology

EvoSimAPL will employ an evolutionary algorithm orchestrated by a Python controller:

Initialization:

1. A human-provided baseline APL and a base SimC character profile will serve as the starting point.
2. A "Program Database" will be initialized. This database will store APLs (as full strings), their corresponding DPS scores, generation number, parent APL (if applicable), and the diff that led to their creation. It will be designed to maintain a population of solutions.

Evolutionary Loop (per generation):

1. **Parent & Inspiration Selection (from Program Database):**
   - Select a "parent program" (an existing APL) from the Program Database. Selection strategies will aim to balance exploitation (picking high-fitness APLs) and exploration (picking diverse or less-optimized APLs). This could involve tournament selection, fitness-proportionate selection, or selecting based on novelty/age.
   - Select one or more "inspiration programs" (other high-performing or structurally interesting APLs) from the database to provide richer context to the LLM.

2. **Prompt Construction (AlphaEvolve-Inspired for Diffs):**
   - A detailed prompt will be generated, instructing the LLM to act as an expert SimC APL optimizer.
   - The prompt will provide the parent_program (APL) and its DPS.
   - It will include the inspiration_programs and their DPS.
   - Crucially, it will instruct the LLM to propose changes to the parent_program in a specific diff format (e.g., AlphaEvolve's -<<<<<<<< SEARCH ... ======== ... >>>>>>> REPLACE format).
   - Contextual information about SimC APL syntax, common optimization goals, and examples of desired diffs will be included.

3. **Creative Generation (LLM-Generated Diffs):**
   - The prompt is sent to the LLM.
   - The LLM's primary output is expected to be a sequence of diff blocks representing suggested modifications to the parent APL.

4. **Diff Application:**
   - A Python function will parse the LLM-generated diffs.
   - These diffs will be applied to the parent_program to create a child_program (the new candidate APL). This step requires careful string manipulation and error handling for malformed or inapplicable diffs.

5. **Evaluation (SimC via Docker Fitness Function):**
   - The child_program (new APL) is combined with the base character profile.
   - This complete profile is then evaluated by executing SimC within a dedicated Docker container. The Docker environment ensures consistency.
   - The primary fitness score (DPS) is extracted from SimC's JSON output.
   - APLs resulting from invalid diffs or causing SimC errors will be heavily penalized (e.g., assigned a fitness of 0). AlphaEvolve's "evaluation cascade" (filtering faulty programs early) can be a source of inspiration here for future enhancements.

6. **Program Database Update & Evolution:**
   - The child_program, its DPS score, the diff applied, and its parent are added to the Program Database.
   - The database will implement strategies to manage population size and diversity, inspired by evolutionary algorithms (e.g., culling low-performers, ensuring uniqueness, potentially using a simplified version of MAP-Elites by binning APLs based on characteristics to maintain diversity). This aligns with AlphaEvolve's goal of continuously improving programs while exploring the search space.

## 4. Technical Stack

- Programming Language: Python 3.x
- Simulation Environment: SimulationCraft CLI running within a Docker container (e.g., simulationcraftorg/simc).
- LLM Interaction: API of a chosen Large Language Model (e.g., OpenAI GPT series, Gemini, Claude).
- Core Libraries:
  - docker (Python SDK for managing Docker container execution).
  - subprocess (potentially for interacting with Docker CLI if SDK is not fully used).
  - json (for SimC output).
  - LLM provider's Python client.
  - Libraries for string manipulation and potentially diff parsing/application (e.g., difflib for comparison, custom logic for application).
- Container Orchestration (for setup): Docker Compose can be used to define the SimC worker service.

## 5. Expected Outcomes & Success Criteria

1. A robust Python application capable of executing the advanced evolutionary loop, including LLM-generated diff application and Dockerized SimC evaluation.
2. Demonstrable and significant DPS improvement over the baseline APL for a chosen class/specialization, showcasing the system's ability to discover superior APLs.
3. Collection of effective diff-based mutations generated by the LLM, providing insights into APL optimization strategies.
4. A flexible framework that serves as a strong foundation for future research, potentially incorporating more sophisticated EC techniques from AlphaEvolve (e.g., multi-objective optimization, meta-prompt evolution, parallelized evaluation).

## 6. Key Challenges

1. LLM Prompt Engineering for Diffs: Guiding the LLM to consistently produce valid, meaningful, and correctly formatted diffs.
2. Robust Diff Application: Creating a reliable system to parse and apply LLM-generated diffs to APL strings, handling potential errors or ambiguities.
3. Maintaining APL Semantic Correctness: Ensuring that applied diffs result in APLs that are not only syntactically valid but also semantically sensible for SimC.
4. Balancing Exploration vs. Exploitation: Designing effective selection and population management strategies for the Program Database.

## Current file structure: 
```
evosim/
├── core/
│   ├── __init__.py
│   ├── controller.py          # Main evolutionary loop
│   ├── database.py            # Program database (APL storage/selection)
│   ├── prompt_sampler.py         # Sampling program that generates new APLs
│   └── evaluator.py           # SimC Docker evaluation
├── llm/
│   ├── __init__.py
│   ├── client.py              # LLM API wrapper
│   └── prompts.py             # Prompt templates and sampling
├── simc/
│   ├── __init__.py
│   ├── docker_runner.py       # Docker SimC execution
├── config/
│   ├── default.yaml           # Default configuration
│   └── prompts/
│       └── apl_evolution.txt  # Main evolution prompt template
├── main.py                    # Entry point
├── requirements.txt
└── README.md
```