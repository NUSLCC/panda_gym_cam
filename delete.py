# Original list
original_list = ['[0.14538932,0.85397637,-0.20212984,-0.8922474,-0.1297952,-0.6646391,0.09693396]',
                 '[0.3952514,0.83730114,0.31910932,-0.82650536,0.10856915,-0.333211,0.07817602]',
                 '[0.06851804,0.7332262,-0.0658325,0.41486585,-0.10854077,0.66136456,-0.01851243]',
                 '[0.14641261,0.6595813,0.01736677,0.41137385,-0.08470821,0.638541,-0.02771848]',
                 '[0.08756363,0.5404841,-0.00434017,-0.07148671,-0.07634783,0.32750702,-0.0111832 ]']

# Multiply each element by 0.05 and convert back to string
result_list = []
for element in original_list:
    # Convert string to list of floats
    float_list = [float(num) for num in element.strip('[]').split(',')]

    # Multiply each float by 0.05
    multiplied_list = [num * 0.05 for num in float_list]

    # Convert back to string
    multiplied_string = '[' + ','.join(map(str, multiplied_list)) + ']'

    # Append to result list
    result_list.append(multiplied_string)

# Print the result
print(result_list)
