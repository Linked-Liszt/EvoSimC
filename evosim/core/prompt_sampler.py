"""
Prompt sampler for generating rich prompts for APL evolution.

Based on AlphaEvolve's approach to prompt sampling, this module constructs
detailed prompts that include parent programs, inspiration programs, and
contextual information to guide LLM-generated APL improvements.
"""

import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from ..entities import APLProgram


class DiffApplicationError(Exception):
    """Exception raised when diff application fails."""
    pass


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""
    include_reasoning: bool = True
    include_evaluation_metadata: bool = True
    max_inspirations: int = 3
    include_dps_comparison: bool = True
    temperature_instructions: bool = True
    diff_format_instructions: bool = True


class PromptSampler:
    """
    Builds rich prompts for LLM-guided APL evolution.
    
    Inspired by AlphaEvolve's prompt sampling approach, this class constructs
    detailed prompts that include:
    - Parent program and its performance
    - High-performing inspiration programs 
    - Context about APL optimization
    - Instructions for generating diffs
    """
    
    def __init__(self, config: Optional[PromptConfig] = None):
        self.config = config or PromptConfig()
        
    def build(self, parent_program: APLProgram, 
              inspirations: List[APLProgram]) -> str:
        """
        Build a complete prompt for APL evolution.
        
        Args:
            parent_program: The current APL to be improved
            inspirations: High-performing APL programs for context
            
        Returns:
            Complete prompt string for the LLM
        """
        prompt_parts = [
            self._build_system_instructions(),
            self._build_context_section(),
            self._build_inspirations_section(inspirations),
            self._build_parent_section(parent_program),
            self._build_diff_instructions(),
            self._build_task_section()
        ]
        
        return "\n\n".join(prompt_parts)
    
    def apply_diff(self, original_code: str, llm_response: str) -> str:
        """
        Apply SEARCH/REPLACE diff blocks from LLM response to original code.
        
        Args:
            original_code: The original APL code to modify
            llm_response: LLM response containing SEARCH/REPLACE blocks
            
        Returns:
            Modified APL code with diffs applied
            
        Raises:
            DiffApplicationError: If diff parsing or application fails
        """
        # Extract all SEARCH/REPLACE blocks from the LLM response
        diff_blocks = self._extract_diff_blocks(llm_response)
        
        if not diff_blocks:
            raise DiffApplicationError("No valid SEARCH/REPLACE blocks found in LLM response")
        
        modified_code = original_code
        
        # Apply each diff block sequentially
        for i, (search_text, replace_text) in enumerate(diff_blocks):
            try:
                modified_code = self._apply_single_diff(modified_code, search_text, replace_text)
            except DiffApplicationError as e:
                raise DiffApplicationError(f"Failed to apply diff block {i+1}: {e}")
        
        return modified_code
    
    def _extract_diff_blocks(self, llm_response: str) -> List[Tuple[str, str]]:
        """
        Extract SEARCH/REPLACE blocks from LLM response.
        
        Expected format:
        ```
        -<<<<<<< SEARCH
        text to find
        =======
        text to replace with
        >>>>>>> REPLACE
        ```
        
        Returns:
            List of (search_text, replace_text) tuples
        """
        # Pattern to match SEARCH/REPLACE blocks
        pattern = r'<{7}\s*SEARCH\s*\n(.*?)\n={7}\s*\n(.*?)\n>{7}\s*REPLACE'
        
        matches = re.findall(pattern, llm_response, re.DOTALL | re.IGNORECASE)
        
        if not matches:
            # Try alternative pattern without code block markers
            pattern = r'-<<<<<<< SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>> REPLACE'
            matches = re.findall(pattern, llm_response, re.DOTALL | re.IGNORECASE)
        
        return [(search.strip(), replace.strip()) for search, replace in matches]
    
    def _apply_single_diff(self, code: str, search_text: str, replace_text: str) -> str:
        """
        Apply a single SEARCH/REPLACE diff to the code.
        
        Args:
            code: Current code state
            search_text: Text to find and replace
            replace_text: Text to replace with
            
        Returns:
            Modified code
            
        Raises:
            DiffApplicationError: If search text not found or multiple matches
        """
        if not search_text:
            raise DiffApplicationError("Empty search text in diff block")
        
        # Count occurrences of search text
        count = code.count(search_text)
        
        if count == 0:
            # Try fuzzy matching by normalizing whitespace
            normalized_code = self._normalize_whitespace(code)
            normalized_search = self._normalize_whitespace(search_text)
            
            if normalized_search in normalized_code:
                # Find the original text that matches when normalized
                lines = code.split('\n')
                search_lines = search_text.split('\n')
                
                for i in range(len(lines) - len(search_lines) + 1):
                    candidate_lines = lines[i:i + len(search_lines)]
                    candidate_text = '\n'.join(candidate_lines)
                    
                    if self._normalize_whitespace(candidate_text) == normalized_search:
                        # Replace the original text
                        before = '\n'.join(lines[:i])
                        after = '\n'.join(lines[i + len(search_lines):])
                        
                        if before and after:
                            return f"{before}\n{replace_text}\n{after}"
                        elif before:
                            return f"{before}\n{replace_text}"
                        elif after:
                            return f"{replace_text}\n{after}"
                        else:
                            return replace_text
            
            raise DiffApplicationError(f"Search text not found in code: '{search_text[:50]}...'")
        
        if count > 1:
            raise DiffApplicationError(f"Search text appears {count} times in code (must be unique): '{search_text[:50]}...'")
        
        # Perform the replacement
        return code.replace(search_text, replace_text, 1)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace for fuzzy matching."""
        # Replace multiple whitespace with single space, strip lines
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(line for line in lines if line)

    def _build_system_instructions(self) -> str:
        """Build the system instructions for the LLM."""
        return """Act as an expert SimulationCraft APL (Action Priority List) optimizer. Your task is to iteratively improve APL code to maximize DPS (Damage Per Second) output.

You are part of an evolutionary optimization system that continuously improves APL performance through targeted modifications. Your role is to analyze the current APL and propose specific, targeted improvements based on your expertise."""
    
    def _build_context_section(self) -> str:
        """Build context about APL optimization and SimulationCraft."""
        return """## Context: APL Optimization

Action Priority Lists (APLs) in SimulationCraft define the decision-making logic for character rotations. Key optimization principles:

- **Priority ordering**: Higher priority actions should be more impactful or situational
- **Resource management**: Efficiently use energy, rage, mana, or other resources  
- **Cooldown optimization**: Align powerful cooldowns with damage buffs
- **Conditional logic**: Use appropriate conditions (buff states, resource levels, cooldowns)
- **Target selection**: Optimize for single-target vs. multi-target scenarios

Common APL syntax:
- `actions.precombat=spell1,spell2` - Pre-combat setup
- `actions=spell,if=condition` - Main rotation with conditions
- `actions+=/spell,if=condition` - Additional priority items
- Conditions: `buff.name.up`, `cooldown.spell.ready`, `resource.current>X`, etc."""
    
    def _build_inspirations_section(self, inspirations: List[APLProgram]) -> str:
        """Build the inspirations section showing high-performing programs."""
        if not inspirations:
            return "## Inspirations\n\nNo inspiration programs available."
        
        section = "## High-Performing Programs (Inspirations)\n\n"
        section += "The following APL programs have performed well in recent evaluations:\n\n"
        
        for i, program in enumerate(inspirations[:self.config.max_inspirations], 1):
            section += f"### Inspiration {i} (DPS: {program.dps_score:,.1f})\n"
            section += f"```\n{program.apl_code}\n```\n"
            
            if self.config.include_reasoning and program.reasoning:
                section += f"**Previous reasoning**: {program.reasoning}\n"
            
            if self.config.include_evaluation_metadata and program.simc_result:
                simc_result = program.simc_result
                section += f"**Simulation**: Valid={simc_result.is_valid}"
                if hasattr(simc_result, 'errors') and simc_result.errors:
                    section += f", Errors: {len(simc_result.errors)}"
                section += "\n"
            
            section += "\n"
        
        return section.rstrip()
    
    def _build_parent_section(self, parent_program: APLProgram) -> str:
        """Build the parent program section."""
        section = f"## Current Program to Improve (DPS: {parent_program.dps_score:,.1f})\n\n"
        section += "Here is the current APL that needs optimization:\n\n"
        section += f"```\n{parent_program.apl_code}\n```\n"
        
        if self.config.include_dps_comparison and parent_program.simc_result:
            simc_result = parent_program.simc_result
            section += "\n**Current Performance**:\n"
            section += f"- DPS: {parent_program.dps_score:,.1f}\n"
            section += f"- Valid simulation: {simc_result.is_valid}\n"
            if simc_result.errors:
                section += f"- Errors: {len(simc_result.errors)}\n"
        
        if self.config.include_reasoning and parent_program.reasoning:
            section += f"\n**Previous reasoning**: {parent_program.reasoning}\n"
        
        return section
    
    def _build_diff_instructions(self) -> str:
        """Build instructions for the diff format."""
        if not self.config.diff_format_instructions:
            return ""
        
        return """## Instructions: Generating Improvements

Propose improvements to the current APL using the following SEARCH/REPLACE diff format:

```
-<<<<<<< SEARCH
# Exact text to find and replace (must match exactly)
=======  
# New text to replace it with
>>>>>>> REPLACE
```

**Important requirements:**
- The SEARCH block must match the existing code exactly (including whitespace)
- Make targeted, specific changes rather than rewriting everything
- You can propose multiple SEARCH/REPLACE blocks for different improvements
- Focus on meaningful optimizations that should increase DPS
- Ensure the resulting APL remains syntactically valid

**Example:**
```
-<<<<<<< SEARCH
actions=spell1
actions+=/spell2,if=buff.example.up
=======
actions=spell1,if=cooldown.important_spell.ready
actions+=/important_spell,if=cooldown.important_spell.ready
actions+=/spell2,if=buff.example.up
>>>>>>> REPLACE
```"""
    
    def _build_task_section(self) -> str:
        """Build the final task instructions."""
        base_task = """## Task

Analyze the current APL and propose specific improvements to increase DPS. Consider:

1. **Rotation optimization**: Improve the priority order of abilities
2. **Resource efficiency**: Better resource management (energy, rage, mana, etc.)
3. **Cooldown alignment**: Align powerful cooldowns with damage buffs
4. **Conditional logic**: Add or improve conditions for ability usage
5. **Missing optimizations**: Add important abilities or buffs that are missing"""

        if self.config.diff_format_instructions:
            base_task += """

Provide your reasoning for each change, then give the SEARCH/REPLACE blocks to implement the improvements."""
        else:
            base_task += """

Provide your reasoning for each proposed change."""
        
        if self.config.temperature_instructions:
            base_task += """

**Be creative but practical**: Consider both incremental improvements and more significant strategic changes. Don't be afraid to propose meaningful restructuring if it could lead to better performance."""
        
        return base_task


class AdaptivePromptSampler(PromptSampler):
    """
    Adaptive prompt sampler that modifies prompts based on evolutionary progress.
    
    Implements adaptive prompting strategies inspired by AlphaEvolve's
    dynamic prompt adjustment based on search progress.
    """
    
    def __init__(self, config: Optional[PromptConfig] = None):
        super().__init__(config)
        self.generation_history = []
    
    def update_generation_stats(self, generation: int, best_dps: float, 
                              avg_improvement: float):
        """Update statistics for adaptive prompting."""
        self.generation_history.append({
            'generation': generation,
            'best_dps': best_dps,
            'avg_improvement': avg_improvement
        })
    
    def build(self, parent_program: APLProgram, 
              inspirations: List[APLProgram]) -> str:
        """Build adaptive prompt based on evolutionary progress."""
        base_prompt = super().build(parent_program, inspirations)
        
        # Add adaptive instructions based on progress
        adaptive_section = self._build_adaptive_section()
        
        if adaptive_section:
            # Insert before task section
            prompt_parts = base_prompt.split("## Task")
            if len(prompt_parts) == 2:
                return prompt_parts[0] + adaptive_section + "\n\n## Task" + prompt_parts[1]
        
        return base_prompt
    
    def _build_adaptive_section(self) -> str:
        """Build adaptive instructions based on evolutionary progress."""
        if len(self.generation_history) < 3:
            return ""
        
        recent_improvements = [h['avg_improvement'] for h in self.generation_history[-3:]]
        avg_recent_improvement = sum(recent_improvements) / len(recent_improvements)
        
        section = "## Adaptive Strategy\n\n"
        
        if avg_recent_improvement < 0.01:  # Low improvement rate
            section += "**Recent progress has been slow**. Consider more significant changes:\n"
            section += "- Restructure the rotation order more substantially\n"
            section += "- Add missing abilities or buffs that might have been overlooked\n"
            section += "- Try more creative conditional logic\n"
            section += "- Consider alternative approaches to resource management\n\n"
        elif avg_recent_improvement > 0.05:  # High improvement rate
            section += "**Recent progress has been strong**. Continue with targeted refinements:\n"
            section += "- Make precise adjustments to conditions and thresholds\n"
            section += "- Fine-tune priority orders\n"
            section += "- Optimize timing of cooldown usage\n\n"
        else:
            section += "**Progress is steady**. Balance refinement with exploration:\n"
            section += "- Try both incremental improvements and moderate changes\n"
            section += "- Look for opportunities to optimize underperforming sections\n\n"
        
        return section


# Factory function for easy prompt sampler creation
def create_prompt_sampler(sampler_type: str = "basic", **kwargs) -> PromptSampler:
    """
    Factory function to create different types of prompt samplers.
    
    Args:
        sampler_type: One of "basic", "adaptive"
        **kwargs: Configuration arguments for the sampler
        
    Returns:
        Configured PromptSampler instance
    """
    if sampler_type == "basic":
        config = PromptConfig(**kwargs)
        return PromptSampler(config)
    elif sampler_type == "adaptive":
        config = PromptConfig(**kwargs)
        return AdaptivePromptSampler(config)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}. Available: 'basic', 'adaptive'")
