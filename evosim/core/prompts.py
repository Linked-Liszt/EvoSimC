
EVOKER_INFO_PROMPT = """

# Devastation Evoker DPS Spells (SimC Format)

Here's a list of core Devastation Evoker DPS spells, with their corresponding SimulationCraft (SimC) formatted names. SimC generally converts spaces and special characters to underscores and uses lowercase for spell names.

---

* **Fire Breath:** `fire_breath` 2.5s cast, 30s cooldown | Inhale, stoking your inner flame. Release to exhale, burning enemies in a cone in front of you for [((163.409%% of Spell power)) + ((39.4834%% of Spell power) * (10 + 0))] Fire damage, reduced beyond 5 targets. Empowering causes more of the damage to be dealt immediately instead of over time.  I:   Deals [(163.409%% of Spell power)] damage instantly and [(39.4834%% of Spell power) * (10 + 0)] over 20 sec. II:  Deals [(163.409%% of Spell power) + ((39.4834%% of Spell power) * 3)] damage instantly and [(39.4834%% of Spell power) * (7 + 0)] over 14 sec. III: Deals [(163.409%% of Spell power) + ((39.4834%% of Spell power) * 6)] damage instantly and [(39.4834%% of Spell power) * (4 + 0)] over 8 sec.
* **Disintegrate:** `disintegrate` 3s cast, 3 essence | Tear into an enemy with a blast of blue magic, inflicting (475.53%% of Spell power) Spellfrost damage over 3 sec
* **Eternity Surge:** 2.5s cast, 30s cooldown | Focus your energies to release a salvo of pure magic, dealing [(818.4%% of Spell power)] Spellfrost damage to an enemy. Damages additional enemies within 25 yds when empowered. (1=1 target, 2=2 targets, 3=3 targets)
* **Pyre:** `pyre` instant, 3 essence | Lob a ball of flame, dealing (209%% of Spell power) Fire damage to the target and nearby enemies.
* **Living Flame:** `living_flame` 2s cast, no cost | Send a flickering flame towards your target dealing (232.001%% of Spell power) Fire damage to an enemy
* **Azure Strike:** `azure_strike` instant (1 GCD) | Project intense energy onto 2 enemies, dealing (108.79%% of Spell power) Spellfrost damage to them.
* **Shattering Star:** `shattering_star` 20s cooldown | Exhale a bolt of concentrated power from your mouth for (230.56%% of Spell power) Spellfrost damage that cracks the [Eternity's Span: targets' / target's] defenses, increasing the damage they take from you by 20%% for 4 sec.
* **Deep Breath:** `deep_breath` 120s cooldown | Take in a deep breath and fly to the targeted location, spewing molten cinders dealing (331.43%% of Spell power) Volcanic damage to enemies in your path.
* **Dragonrage:** `dragonrage` 120s cooldown | Your main offensive cooldown Erupt with draconic fury and exhale Pyres at 3 enemies within 25 yds. For 18 sec, Essence Burst's chance to occur is increased to 100%%
* **Engulf:** `engluf' ~20s cooldown, hasted | Engulf your target in dragonflame, damaging them for (481%% of Spell power) Fire or healing them for (510%% of Spell power). For each of your periodic effects on the target, effectiveness is increased by 50%%.


Here's a list of the Devestation Evoker Talents that are active on the current build and relevant for DPS: 

They may effect the baseline spells above so be sure to factor them in when performing calculations.

* **Natural Convergence:**: Disintegrate channels 20%% faster.
* **Enkindled** Living Flame deals 6%% more damage and healing.
* **Blast Furnace** Fire Breath's damage over time lasts 4 sec longer.
* **Ruby Essence Burst** Your Living Flame has a 20%% chance to cause an Essence Burst, making your next Disintegrate or Pyre cost no Essence.
* **Azure Essence Burst** Azure Strike has a 15% chance to cause an Essence Burst, making your next Disintegrate or Pyre cost no Essence.
* **Dense Energy**: Pyre's Essence cost is reduced by 1.
* **Arcane Intensity**: Disintegrate's damage is increased by 16%%.
* **Ruby Embers**: Living Flame deals (26.5144%% of Spell power) damage over 12 sec to enemies. Stacks 3 times.
* **Animosity**: Casting an empower spell extends the duration of Dragonrage by 5 sec, up to a maximum of 20 sec.
* **Heat Wave**: Fire Breath deals 30%% more damage.
* **Eye of Infinity**: Eternity Surge deals 15%% more damage.
* **Catalyze**: While channeling Disintegrate your Fire Breath on the target deals damage 100% more often.
* **Tyranny**: During Deep Breath and Dragonrage you gain the maximum benefit of Mastery: Giantkiller regardless of targets' health.
* **Burnout**: Fire Breath damage has 16%% chance to cause your next Living Flame to be instant cast, stacking 2 times.
* **Font of Magic**: Your empower spells' maximum level is increased by 1, and they reach maximum empower level 20%% faster.
* **Azure Celerity**: Disintegrate ticks 1 additional time, but deals 10%% less damage.
* **Power Swell**: Casting an empower spell increases your Essence regeneration rate by 100% for 4 sec.
* **Scorching Embers**: Fire Breath causes enemies to take up to 40%% increased damage from your Red spells, increased based on its empower level.
* **Causality**: Disintegrate reduces the remaining cooldown of your empower spells by 0.50 sec each time it deals damage. Pyre reduces the remaining cooldown of your empower spells by 0.40 sec per enemy struck, up to 2.0 sec.
* **Scintillation**: Disintegrate has a 15%% chance each time it deals damage to launch a level 1 Eternity Surge at 40% power.
* **Iridescence**: Casting an empower spell increases the damage of your next 2 spells of the same color by 20% within 10 sec.
---


Here's some additional information that is useful for writing APLs for Devastation Evoker as inferred from previous runs:
To access essence info the standard way is to use the variable: 'essence.deficit' Don't use any other variables to check for essence. 
The shattering star debuff is called 'debuff.shattering_star_debuff.up'
To empower spells use empower_to like this: `actions+=/fire_breath,empower_to=3`
To access fire breath debuff: `dot.fire_breath_damage.remains`
For enkindle debuff: `dot.enkindle.remains`
For iridescence red buff: `buff.iridescence_red.up` and for blue: `buff.iridescence_blue.up`
"""



APL_INFO_PROMPT = """
Action Priority Lists (APLs) in SimulationCraft define the decision-making logic for character rotations. Below is relevant information from the SimulationCraft documentation:

A couple pieces of advice for writing actions lists:

1. Do not forget, it is priority-based, it's as simple as that!
2. You don not have to check if an action is ready before using it. If the action is not ready, it will simply not be used. This is a key difference from other systems where you might need to check cooldowns or resource availability before executing an action.
2a. Only create checks if you want different behavior based on the action's state. Like if you want to hold a spell until within a certain buff. 
3. Do not try to optimize your actions lists for computations performances. Just focus on correctly modeling the gameplay you want.




Below are advanced features of APLs that can help you write more complex and efficient action lists. Use these only if needed, as they can make your APL harder to read and maintain.

# Conditional expressions
Conditional expressions allow complex composition of conditions, and are usually added to action priority list through the "if" keyword, using the following syntax:
```
# Uses faerie fire if:
# * There are less than three stacks
# * And neither "expose armor" nor "sunder armor" are on the target
actions+=/faerie_fire,if=debuff.faerie_fire.stack<3&!(debuff.sunder_armor.up|debuff.expose_armor.up)
```

The same syntax is used for the _interrupt _if_ action modifier and the the _sec_ modifier for _wait\_fixed_ actions.

Player Expressions are also used for raid event player _if filter.

## Operators

### Simple example
Consider the following APL line:

```
actions+=/faerie_fire,if=debuff.faerie_fire.stack<3&!(debuff.sunder_armor.up|debuff.expose_armor.up)
```

Given that `&` means "AND", `!` means "NOT", and `|` means "OR", we can replace the symbols with their English equivalents:

```
debuff.faerie_fire.stack LESS THAN 3 AND NOT(debuff.sunder_armor.up OR debuff.expose_armor.up)
```

Therefore, this APL line says to cast Faerie Fire if there are less than three stacks of Faerie Fire and neither Sunder Armor nor Expose Armor are up.

### Operator precedence
To be able to read APL expressions, it is important to know the precedence rules (i.e. the order of operations). An operator with higher precedence binds more tightly to its operands: for example, `2+3*5` is evaluated as `2+(3*5)` because `*` has higher precedence than `+`. Similarly, `a&b|c&d` is evaluated as `(a&b)|(c&d)` because `&` has higher precedence than `|`.

Within the same precedence class, Simulationcraft always uses left-to-right associativity: for example, `4-5+6` is evaluated as `(4-5)+6` because `+`, `-` have the same precedence but the `4-5` appears first when reading left to right.

Here is the full list of operators in Simulationcraft, in order from highest to lowest precedence (i.e. entries listed first bind more tightly):

* Function calls: `floor()` `ceil()`
* Unary operators: `+` `-` `@` `!`
* Multiplication, division, and modulus: `*` `%` `%%`
* Addition and subtraction: `+` `-`
* Max and min: `<?` `>?`
* Comparison operators: `=` `!=` `<` `<=` `>` `>=` `~` `!~`
* Logical AND: `&`
* Logical XOR: `^`
* Logical OR: `|`

Note that this is very similar to the operator precedence in other programming languages such as C++ or Javascript.

### Arithmetic operators
All Simulationcraft expressions evaluate to double-precision floating-point numbers. The following operators perform floating-point arithmetic.

* `+` is the addition operator: `x+y` evaluates to the sum of `x` and `y`. As a unary operator, it is a no-op: `+x` evaluates to the same as `x`.
* `-` is the subtraction operator: `x-y` evaluates to `x` minus `y`. As a unary operator, it negates the operand: `-x` evaluates to the negation of `x`.
* `*` is the multiplication operator: `x*y` evaluates to the product of `x` and `y`.
* `%` is the division operator: `x%y` evaluates to the floating-point quotient of `x` by `y` (for example, `3%2` equals `1.5`). Note that `/` could not be used for division since it is already being used as the separator token.
* `%%` is the modulus operator: `x%%y` evaluates to the floating-point remainder when `x` is divided by `y` (for example, `9%%2.5` equals `1.5`). The result has the same sign as `x`.
* **`@`** is the absolute value operator: `@x` evaluates to the absolute value of `x`.
* **`<?`** is the max operator: `x<?y` evaluates to the greater of `x` and `y`.
* **`>?`** is the min operator: `x>?y` evaluates to the lesser of `x` and `y`.
* `floor()` is the floor function: `floor(x)` evaluates to the greatest integer value that is less than or equal to `x`. Note that this is different from truncation for negative operands: for example, `floor(-2.5)` equals `-3`.
* `ceil()` is the ceiling function: `ceil(x)` evaluates to the least integer value that is greater than or equal to `x`.

### Comparison operators
* `=` is the equality operator: `x=y` evaluates to `1` if `x` is equal to `y`, and `0` if they are not equal.
* `!=` is the inequality operator: `x!=y` evaluates to `1` if `x` is not equal to `y`, and `0` if they are equal.
* `<` `<=` `>` `>=` are the relational operators: these should be self-explanatory given the above.

### Logical operators
Logical operators work with truth values. Zero is the same as false; any nonzero value is considered truthy.

* `&` is the logical AND operator: `x&y` evaluates to `1` if both `x` and `y` are nonzero, and `0` otherwise.
* `|` is the logical OR operator: `x|y` evaluates to `1` if either `x` or `y` or both are nonzero, and `0` otherwise.
* `^` is the logical XOR operator: `x^y` evaluates to `1` if either `x` or `y`, but not both, are nonzero, and `0` otherwise.
* `!` is the logical NOT operator: `!x` evaluates to `1` if `x` is zero, and `0` otherwise.

### Important note on booleans
There is no distinct boolean type, as Simulationcraft expressions are all evaluated as floating-point numbers. It is common for APLs to coerce values between boolean and arithmetic contexts freely.
- In a boolean context, `0` is considered false, while anything nonzero is considered true. For example, `3&4` evaluates to `1` since `3` and `4` are being used in a boolean context (operands to `&`) and are both considered true. Logical operators and comparisons will always output either `0` or `1` regardless of their operands.
- Example: `guillotine,if=cooldown.demonic_strength.remains`. The condition is equivalent to `cooldown.demonic_strength.remains!=0` and it is synonymous with Demonic Strength being on cooldown: `!cooldown.demonic_strength.ready` means basically the same thing.
- On the flip side, logical `0` or `1` values can be used in an arithmetic context. For example, `(a==b)*c` will evaluate to `c` if `a` is equal to `b`, or `0` otherwise.
- Example: `variable,name=next_tyrant,op=set,value=14+talent.grimoire_felguard+talent.summon_vilefiend`. Since talent checks evaluate to `0` or `1` depending on whether the talent is learned, this logic starts with an initial timer of 14 and adds 1 for each of these talents learned.

### SpellQuery operators
These operators may only be used in SpellQuery.
* `~` is the string `in` operator: `s~t` evaluates to true if `s` is a substring of `t` (case-insensitive).
* `!~` is the string `not in` operator: `s!~t` evaluates to true if `s` is not a substring of `t` (case-insensitive).

"""

FULL_PROMPT = EVOKER_INFO_PROMPT + APL_INFO_PROMPT