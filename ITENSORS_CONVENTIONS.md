# ITensors MPS Conventions for Quantum Computing

## Summary
This document describes the established conventions for working with Matrix Product States (MPS) in ITensors.jl for quantum computing applications, specifically for implementing quantum adders and similar circuits.

## Bit Ordering Convention

### Our Convention (Established)
- **Site 1** = **Leftmost bit** (Most Significant Bit, MSB)
- **Site L** = **Rightmost bit** (Least Significant Bit, LSB)

Example: For state `|0101⟩`:
- Site 1 stores bit 0 (leftmost)
- Site 2 stores bit 1
- Site 3 stores bit 0  
- Site 4 stores bit 1 (rightmost)

### ITensors Native Convention
ITensors uses the **opposite** convention internally:
- Site 1 corresponds to the **rightmost bit** in binary position
- Site L corresponds to the **leftmost bit** in binary position

Example: `productMPS(sites, [1,1,1,2])` creates:
- Site 1 has bit 0, Site 2 has bit 0, Site 3 has bit 0, Site 4 has bit 1
- This represents position 8 in the state vector = `|1000⟩` in binary

## Implementation Pattern

### Creating MPS States
```julia
# Input: binary string like "0101"
string_vec = "0101"

# Direct mapping: site i gets bit at position i
vec = 1 .+ parse.(Int, [string_vec[i] for i in 1:L])

# Create MPS (vec uses 1-based indexing: 1=bit 0, 2=bit 1)
initial_state = productMPS(qubit_site, vec)
```

### Reading MPS States
```julia
function display_state(state)
    contracted = contract(state)
    state_vec = vec(Array(contracted, inds(contracted)...))
    loc = findall(x->abs(x)>1e-12, state_vec)
    if length(loc) > 0
        pos = loc[1] - 1  # Convert to 0-based indexing
        binary = string(pos, base=2, pad=L)
        return reverse(binary)  # Reverse to match our site 1 = leftmost convention
    else
        return "0"^L
    end
end
```

**Key insight**: The `reverse()` in `display_state` is necessary because ITensors' computational basis ordering is opposite to our site ordering.

## Addition Tensor Implementation

### Tensor Structure
The `create_addition_tensor_with_carry(shift_bit, s, s_prime, c_in, c_out)` creates a 4-index tensor:
- `s`: Input site index
- `s_prime`: Output site index (primed)
- `c_in`: Carry input index
- `c_out`: Carry output index

### Applying Addition to MPS
```julia
# Create addition tensor
T = create_addition_tensor_with_carry(shift_bit, qubit_site[i], prime(qubit_site[i]), carry_links[i+1], carry_links[i])

# Apply to MPS site
tmp_tensor = T * initial_state[i]
noprime!(tmp_tensor)

# Fix carry indices to specific values (e.g., 0 for both)
c_in_0 = onehot(inds(tmp_tensor)[2] => 1)   # Index 2 is c_in
c_out_0 = onehot(inds(tmp_tensor)[3] => 1)  # Index 3 is c_out
tmp_tensor = tmp_tensor * c_in_0 * c_out_0

# Direct replacement (NO SVD needed!)
final_state[i] = tmp_tensor
```

**Important**: After contracting with carry states, `tmp_tensor` has the correct indices `(site, link)` to directly replace the MPS tensor. **Do NOT use SVD** - it creates extra bond dimensions and corrupts the state.

## Truth Table for Binary Addition

The `create_addition_tensor_with_carry(shift_bit, ...)` implements:
```
input_bit + shift_bit + carry_in = output_bit + 2 × carry_out
```

For `shift_bit = 0` (identity operation):
```
|input, carry_in⟩ → |output, carry_out⟩
|0, 0⟩ → |0, 0⟩  ✓
|0, 1⟩ → |1, 0⟩  ✓
|1, 0⟩ → |1, 0⟩  ✓
|1, 1⟩ → |0, 1⟩  ✓
```

For `shift_bit = 1` (add 1):
```
|input, carry_in⟩ → |output, carry_out⟩
|0, 0⟩ → |1, 0⟩  ✓
|0, 1⟩ → |0, 1⟩  ✓
|1, 0⟩ → |0, 1⟩  ✓
|1, 1⟩ → |1, 1⟩  ✓
```

## Common Pitfalls

### ❌ Wrong: Using SVD after tensor contraction
```julia
# This creates extra bond dimensions and corrupts state
tmp_tensor = tmp_tensor * c0 * c2
U, S, V = svd(tmp_tensor, [inds(tmp_tensor)[1], inds(tmp_tensor)[2]], ...)
final_state[i] = U
final_state[i+1] = S * V  # WRONG!
```

### ✅ Correct: Direct replacement
```julia
tmp_tensor = tmp_tensor * c0 * c2
final_state[i] = tmp_tensor  # Correct!
```

### ❌ Wrong: Using `array()` function
```julia
# This may fail with "objects of type Vector{Int64} are not callable"
state_vec = vec(array(contract(state)))
```

### ✅ Correct: Using `Array()` with explicit indices
```julia
contracted = contract(state)
state_vec = vec(Array(contracted, inds(contracted)...))
```

### ❌ Wrong: Ignoring bit ordering
```julia
# This gives wrong results
vec = 1 .+ parse.(Int, [string_vec[ram_phy[i]] for i in 1:L])
```

### ✅ Correct: Direct mapping with our convention
```julia
# Site i stores string_vec[i]
vec = 1 .+ parse.(Int, [string_vec[i] for i in 1:L])
```

## Folding Convention

With `folded = false`:
- `ram_phy = [1, 2, 3, 4]` (identity mapping)
- `phy_ram = [1, 2, 3, 4]`

With `folded = true`:
- `ram_phy` may be permuted (e.g., `[1, 4, 2, 3]`)
- This scrambles physical vs logical bit positions
- **Recommendation**: Use `folded = false` for clarity, worry about folding later

## Working Example

See `benchmark_adder.jl` for a complete working example that:
1. Creates all 2^L computational basis states
2. Applies "add 0" operation (identity test)
3. Verifies output matches input for all states

```bash
julia benchmark_adder.jl
# Output should show identity: 0000=>0000, 0001=>0001, ..., 1111=>1111
```

## References

- Memory ID 4780900: adder_MPO optimization strategies
- `global_adder_passthrough.jl`: Contains `create_addition_tensor_with_carry` implementation
- `benchmark_adder.jl`: Working test script demonstrating correct usage

---

**Last Updated**: Session with debugging of ITensors bit ordering conventions
**Status**: ✅ Verified working with identity operation on all 16 states (L=4)

