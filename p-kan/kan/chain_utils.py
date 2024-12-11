# def chain_str(in_out_dim: int, num_electrons: int) -> str:
#     connection_idxs = list(range(in_out_dim))

#     # Get first, second, and third third of input indices
#     radial_indices = connection_idxs[0::3]
#     theta_indices = connection_idxs[1::3]
#     phi_indices = connection_idxs[2::3]

#     # Get output indices with remainders 0, 1, and 2 when divided by 3
#     radial_electrons = [i for i in range(in_out_dim) if i % 3 == 0]
#     theta_electrons = [i for i in range(in_out_dim) if i % 3 == 1]
#     phi_electrons = [i for i in range(in_out_dim) if i % 3 == 2]

#     # Create three chains
#     first_chain = f'{radial_electrons}->{radial_indices}'
#     second_chain = f'{theta_electrons}->{theta_indices}'
#     third_chain = f'{phi_electrons}->{phi_indices}'

#     return [first_chain, second_chain, third_chain]

# chains = chain_str(INPUT_DIM, NUM_ELECTRONS)

# # Set up all connections
# for chain in chains:
#     model.module(0, chain)
       
# # # Create chain for final layer where all intermediate nodes connect to output
# final_indices = list(range(OUTPUT_DIM))
# final_chain = f'{final_indices}->[0]'
# model.module(1, final_chain)

# radial_vars = [r'$r_{'+str(i)+'}$' for i in range(NUM_ELECTRONS)]
# theta_vars = [r'$\theta_{'+str(i)+'}$' for i in range(NUM_ELECTRONS)]
# phi_vars = [r'$\phi_{'+str(i)+'}$' for i in range(NUM_ELECTRONS)]

# input_vars = radial_vars + theta_vars + phi_vars