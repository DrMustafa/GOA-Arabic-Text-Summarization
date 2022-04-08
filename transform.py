
def binary_to_float(binary_list, minimum, maximum):
    #get the max value
    max_binary = 2**len(binary_list)-1

    #convert the binary to an integer
    integer = binary_to_int(binary_list, 0)

    #convert the integer to a floating point 
    floating_point = float(integer)/max_binary

    #scale the floating point from min to max
    scaled_floating_point = floating_point*maximum
    scaled_floating_point -= floating_point*minimum
    scaled_floating_point += minimum

    return scaled_floating_point
